import os
import shutil
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

# Core dependencies
import torch
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Document loaders for multiple formats
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    TextLoader
)

# Text splitters for contextual chunking
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    HeaderType
)

# Embeddings and reranking
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# For query expansion/reformulation
from langchain_openai import ChatOpenAI

# For metadata extraction
import fitz  # PyMuPDF
from datetime import datetime
import hashlib

# Environment variables
import dotenv
dotenv.load_dotenv()

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "gpt-3.5-turbo"  # Or use local model like "llama2" with Ollama
DB_DIR = "chroma_db"
DATA_DIR = "documents"

class AdvancedRAG:
    def __init__(self, data_dir: str = DATA_DIR, db_dir: str = DB_DIR, rebuild: bool = False):
        self.data_dir = data_dir
        self.db_dir = db_dir
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Initialize reranker
        self.reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
        
        # Initialize LLM
        self.llm = ChatOpenAI(temperature=0, model=LLM_MODEL)
        
        # Create or load vector store
        if rebuild and os.path.exists(db_dir):
            shutil.rmtree(db_dir)
            
        if not os.path.exists(db_dir):
            self.vectorstore = self._build_vectorstore()
        else:
            self.vectorstore = Chroma(
                persist_directory=db_dir,
                embedding_function=self.embeddings
            )
            
        # Create RAG chain
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
        
        self.query_processor = self._create_query_processor()
        self.rag_chain = self._create_rag_chain()
    
    def _build_vectorstore(self) -> Chroma:
        """Process documents and build vectorstore"""
        print("Processing documents and building vectorstore...")
        documents = self._load_documents()
        chunks = self._process_documents(documents)
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_dir
        )
        vectorstore.persist()
        print(f"Vectorstore built with {len(chunks)} chunks")
        return vectorstore
    
    def _load_documents(self) -> List[Document]:
        """Load documents from various formats"""
        documents = []
        for file_path in Path(self.data_dir).glob("**/*"):
            if file_path.is_file():
                try:
                    ext = file_path.suffix.lower()
                    if ext == ".pdf":
                        loader = PyPDFLoader(str(file_path))
                        docs = loader.load()
                        # Extract metadata
                        docs = self._extract_pdf_metadata(str(file_path), docs)
                    elif ext == ".docx":
                        loader = Docx2txtLoader(str(file_path))
                        docs = loader.load()
                    elif ext == ".pptx":
                        loader = UnstructuredPowerPointLoader(str(file_path))
                        docs = loader.load()
                    elif ext in [".xlsx", ".xls"]:
                        loader = UnstructuredExcelLoader(str(file_path))
                        docs = loader.load()
                    elif ext == ".csv":
                        loader = CSVLoader(str(file_path))
                        docs = loader.load()
                    elif ext in [".txt", ".md"]:
                        loader = TextLoader(str(file_path))
                        docs = loader.load()
                    else:
                        continue
                    
                    # Add basic metadata
                    for doc in docs:
                        if "source" not in doc.metadata:
                            doc.metadata["source"] = str(file_path)
                        doc.metadata["file_type"] = ext
                        doc.metadata["file_name"] = file_path.name
                        doc.metadata["ingestion_date"] = datetime.now().isoformat()
                        
                    documents.extend(docs)
                    print(f"Loaded {len(docs)} sections from {file_path}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        return documents
    
    def _extract_pdf_metadata(self, file_path: str, docs: List[Document]) -> List[Document]:
        """Extract additional metadata from PDF files"""
        try:
            pdf = fitz.open(file_path)
            
            # Extract document-level metadata
            metadata = pdf.metadata
            doc_metadata = {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "page_count": len(pdf),
                "creation_date": metadata.get("creationDate", "")
            }
            
            # Add metadata to each document section
            for i, doc in enumerate(docs):
                # Add document-level metadata
                for key, value in doc_metadata.items():
                    doc.metadata[key] = value
                
                # Add page number if available
                if "page" not in doc.metadata and i < len(pdf):
                    doc.metadata["page"] = i + 1
            
            pdf.close()
        except Exception as e:
            print(f"Error extracting PDF metadata from {file_path}: {e}")
        
        return docs
    
    def _process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents with contextual chunking and metadata enrichment"""
        # Header-aware chunking for structured documents
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                # Define headers to recognize
                HeaderType("##", "section"),
                HeaderType("###", "subsection"),
            ]
        )
        
        # Standard chunking for unstructured text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
        )
        
        # Process each document
        processed_chunks = []
        for doc in documents:
            # Generate a document ID for tracking parent relationships
            doc_id = hashlib.md5(f"{doc.metadata['source']}".encode()).hexdigest()
            
            # Try markdown splitting first for structure-aware chunks
            if any(header in doc.page_content for header in ["#", "##", "###"]):
                try:
                    md_chunks = markdown_splitter.split_text(doc.page_content)
                    for chunk in md_chunks:
                        chunk.metadata.update(doc.metadata)
                        chunk.metadata["doc_id"] = doc_id
                        chunk.metadata["chunking_method"] = "markdown"
                    processed_chunks.extend(md_chunks)
                    continue
                except Exception as e:
                    print(f"Markdown splitting failed, falling back to text splitter: {e}")
            
            # Fall back to standard chunking
            chunks = text_splitter.split_documents([doc])
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_{i}"
                chunk.metadata["chunk_id"] = chunk_id
                chunk.metadata["doc_id"] = doc_id
                chunk.metadata["chunking_method"] = "recursive"
                chunk.metadata["chunk_index"] = i
                chunk.metadata["total_chunks"] = len(chunks)
            
            processed_chunks.extend(chunks)
        
        return processed_chunks
    
    def _create_query_processor(self):
        """Create a query processor for query expansion/reformulation"""
        query_prompt = ChatPromptTemplate.from_template(
            """Given the following user question and conversation history, 
            reformulate the question to be more specific and detailed to improve retrieval.
            Focus on extracting specific terms, entities and concepts that should be searched for.
            
            Chat History:
            {chat_history}
            
            User Question:
            {question}
            
            Reformulated Query:"""
        )
        
        return query_prompt | self.llm | StrOutputParser()
    
    def _create_rag_chain(self):
        """Create the final RAG chain with reranking"""
        context_prompt = ChatPromptTemplate.from_template(
            """Answer the question based solely on the following context:
            
            {context}
            
            Question: {question}
            
            If the answer cannot be determined from the context, say "I don't have enough information to answer this question."
            Always cite the source documents by name when possible.
            """
        )
        
        def _rerank_docs(inputs):
            docs = inputs["docs"]
            question = inputs["question"]
            
            # Prepare passages for reranking
            passages = [doc.page_content for doc in docs]
            
            # Score passages with reranker
            if passages:
                pairs = [[question, passage] for passage in passages]
                scores = self.reranker.predict(pairs)
                
                # Sort by score
                scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
                
                # Take top 3-5 docs
                reranked_docs = [doc for doc, _ in scored_docs[:5]]
                return reranked_docs
            return []
        
        def _format_docs(docs):
            return "\n\n".join([f"Source: {doc.metadata.get('file_name', 'Unknown')}\n{doc.page_content}" for doc in docs])
        
        # Define the RAG chain
        rag_chain = (
            {
                "question": RunnablePassthrough(),
                "reformed_question": lambda x: self.query_processor.invoke({"question": x, "chat_history": ""}),
                "docs": lambda x: self.retriever.get_relevant_documents(x["reformed_question"])
            }
            | {
                "question": lambda x: x["question"],
                "docs": lambda x: _rerank_docs({"docs": x["docs"], "question": x["question"]})
            }
            | {
                "question": lambda x: x["question"],
                "context": lambda x: _format_docs(x["docs"])
            }
            | context_prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def query(self, question: str, chat_history: str = "") -> str:
        """Process query and return response"""
        # Update chat history if needed
        self.query_processor = self._create_query_processor()
        
        # Process query
        result = self.rag_chain.invoke(question)
        return result

def main():
    """Main entry point for the application"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced RAG System")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the vector database")
    parser.add_argument("--query", type=str, help="Run a specific query in non-interactive mode")
    args = parser.parse_args()
    
    # Initialize the RAG system
    rag = AdvancedRAG(rebuild=args.rebuild)
    
    if args.query:
        # Single query mode
        result = rag.query(args.query)
        print(f"\nQuery: {args.query}\n")
        print(f"Response: {result}\n")
    else:
        # Interactive mode
        print("Advanced RAG System initialized. Type 'exit' to quit.")
        chat_history = ""
        
        while True:
            query = input("You: ")
            if query.lower() == "exit":
                break
            
            response = rag.query(query, chat_history)
            print(f"Bot: {response}\n")
            
            # Update chat history
            chat_history += f"User: {query}\nAssistant: {response}\n"

if __name__ == "__main__":
    main()