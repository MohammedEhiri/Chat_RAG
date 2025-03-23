import os
import argparse

from typing import List, Dict, Optional
from pathlib import Path
import tempfile

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from pptx import Presentation
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import dotenv
dotenv.load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "gemma3"
DB_DIR = "vectore/chroma_pptx_db"
DATA_DIR = "documents"


class PPTXOptimizedRAG:
    def __init__(self, data_dir: str = DATA_DIR, db_dir: str = DB_DIR, rebuild: bool = False):
        self.data_dir = data_dir
        self.db_dir = db_dir
        
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        self.llm = OllamaLLM(model=LLM_MODEL)
        
        if rebuild and os.path.exists(db_dir):
            import shutil
            shutil.rmtree(db_dir)
            
        if not os.path.exists(db_dir):
            self.vectorstore = self._build_vectorstore()
        else:
            self.vectorstore = Chroma(
                persist_directory=db_dir,
                embedding_function=self.embeddings
            )
            
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 15, "fetch_k": 30}
        )
        
        self.rag_chain = self._create_rag_chain()
    
    def _build_vectorstore(self) -> Chroma:
        """Process documents and build vectorstore"""
        print("Processing PPTX documents and building vectorstore...")
        documents = self._process_pptx_files()
        
        if not documents:
            raise ValueError("Aucun contenu n'a été extrait des présentations. Vérifiez les fichiers PPTX.")
        
        print(f"Extracted {len(documents)} slides from presentations")
        
        chunks = self._create_chunks(documents)
        
        if not chunks:
            raise ValueError("Aucun chunk n'a été créé à partir des documents. Vérifiez le contenu des présentations.")
        
        print(f"Created {len(chunks)} chunks from slide content")
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_dir
        )
        print(f"Vectorstore built successfully with {len(chunks)} chunks")
        return vectorstore

    
    def _process_pptx_files(self) -> List[Document]:
        """Basic but reliable PPTX file processing"""
        documents = []
        
        for file_path in Path(self.data_dir).glob("**/*.pptx"):
            try:
                print(f"Processing presentation: {file_path}")
                presentation = Presentation(file_path)
                
                presentation_metadata = {
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "file_type": ".pptx",
                    "total_slides": len(presentation.slides)
                }
                
                for slide_idx, slide in enumerate(presentation.slides):
                    slide_num = slide_idx + 1
                    
                    all_text = self._extract_all_text(slide)
                    
                    if not all_text.strip():
                        print(f"  Skipping empty slide {slide_num}")
                        continue
                    
                    slide_content = f"Slide {slide_num}\n\n{all_text}"
                    
                    slide_metadata = {
                        **presentation_metadata,
                        "slide_number": slide_num
                    }
                    
                    doc = Document(
                        page_content=slide_content,
                        metadata=slide_metadata
                    )
                    documents.append(doc)
                    print(f"  Processed slide {slide_num} ({len(all_text)} chars)")
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        
        return documents
    
    def _extract_all_text(self, slide) -> str:
        """Extract all text from a slide without assumptions about structure"""
        text_parts = []
        
        for shape in slide.shapes:
            text = self._get_text_from_shape(shape)
            if text:
                text_parts.append(text)
            
            table_text = self._get_text_from_table(shape)
            if table_text:
                text_parts.append(table_text)
        
        return "\n".join(text_parts)
    
    def _get_text_from_shape(self, shape) -> str:
        """Extract text from a shape safely"""
        try:
            if hasattr(shape, "text"):
                return shape.text
            elif hasattr(shape, "text_frame") and hasattr(shape.text_frame, "text"):
                return shape.text_frame.text
        except:
            pass
        return ""
    
    def _get_text_from_table(self, shape) -> str:
        """Extract text from a table shape safely"""
        try:
            if hasattr(shape, "table"):
                rows = []
                for row in shape.table.rows:
                    cell_texts = []
                    for cell in row.cells:
                        if hasattr(cell, "text_frame") and hasattr(cell.text_frame, "text"):
                            cell_texts.append(cell.text_frame.text.strip())
                    if cell_texts:
                        rows.append(" | ".join(cell_texts))
                if rows:
                    return "Table:\n" + "\n".join(rows)
        except:
            pass
        return ""
    
    def _create_chunks(self, documents: List[Document]) -> List[Document]:
        """Create chunks from documents"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
        
        all_chunks = []
        
        for doc in documents:
            try:
                doc_chunks = splitter.split_documents([doc])
                
                for chunk in doc_chunks:
                    for key, value in doc.metadata.items():
                        chunk.metadata[key] = value
                
                all_chunks.extend(doc_chunks)
            except Exception as e:
                print(f"Error creating chunks for slide {doc.metadata.get('slide_number', 'unknown')}: {e}")
        
        return all_chunks
    
    def _create_rag_chain(self):
        """Create the RAG chain with presentation-specific context handling"""

        context_prompt = ChatPromptTemplate.from_template(
            """You are an expert in analyzing presentations and slide decks.
            
            Answer the question based on the following slides and content from presentations:
            
            {context}
            
            IMPORTANT GUIDELINES:
            1. Base your answer ONLY on the provided slides, not general knowledge
            2. If information is spread across multiple slides, synthesize it
            3. Mention which presentations and slides contain the information
            4. If the slides don't contain enough information, say so clearly
            5. Remember that presentations often have incomplete sentences and bullet points
            
            Question: {question}
            """
        )
        
        def _format_docs(docs):
            
            presentations = {}
            for doc in docs:
                source = doc.metadata.get("file_name", "Unknown")
                if source not in presentations:
                    presentations[source] = {}
                
                slide_num = doc.metadata.get("slide_number", 0)
                if slide_num not in presentations[source]:
                    presentations[source][slide_num] = []
                
                presentations[source][slide_num].append(doc.page_content)
            
            
            formatted = []
            for pres, slides in presentations.items():
                formatted.append(f"Presentation: {pres}")
                for slide_num in sorted(slides.keys()):
                    slide_content = "\n".join(slides[slide_num])
                    formatted.append(f"  Slide {slide_num}: {slide_content}")
                formatted.append("")
            
            return "\n".join(formatted)
        
        
        rag_chain = (
        {"context": lambda x: _format_docs(self.retriever.invoke(x)),
         "question": lambda x: x}
        | context_prompt
        | self.llm
        | StrOutputParser()
        )
        
        return rag_chain
    
    def query(self, question: str) -> str:
        """Process query and return response"""
        try:
            result = self.rag_chain.invoke(question)
            
            sources = self._get_sources(question)
            if sources:
                result += "\n\nSources:\n" + sources
                
            return result
        except Exception as e:
            return f"Erreur : {str(e)}"
    
    def _get_sources(self, question: str) -> str:
        """Get and format sources for a question"""
        try:
            docs = self.retriever.invoke(question)
            sources = {}
            
            for doc in docs:
                source = doc.metadata.get("file_name", "Inconnu")
                slide = doc.metadata.get("slide_number", "?")
                
                key = f"{source} (Slide {slide})"
                if key not in sources:
                    sources[key] = True
                    
            return "\n".join(list(sources.keys()))
        except Exception as e:
            print(f"Error retrieving sources: {e}")
            return ""







def main():
    """Main entry point for the application"""
    
    parser = argparse.ArgumentParser(description="PPTX-Optimized RAG System")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the vector database")
    parser.add_argument("--query", type=str, help="Run a specific query in non-interactive mode")
    args = parser.parse_args()
    
    rag = PPTXOptimizedRAG(rebuild=args.rebuild)
    
    if args.query:
        result = rag.query(args.query)
        print(f"\nQuery: {args.query}\n")
        print(f"Response: {result}\n")
    else:
        print("PPTX-Optimized RAG System initialized. Type 'exit' to quit.")
        
        while True:
            query = input("You: ")
            if query.lower() == "exit":
                break
            
            response = rag.query(query)
            print(f"Bot: {response}\n")







if __name__ == "__main__":
    main()