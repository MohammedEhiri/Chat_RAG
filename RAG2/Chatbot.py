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

# Ajout des imports pour PDF
from pypdf import PdfReader

import dotenv
dotenv.load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "gemma3"
DB_DIR = "vectore/chroma_docs_db"
DATA_DIR = "documents"


class DocumentRAG:
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
        print("Processing documents and building vectorstore...")
        
        # Traitement des fichiers PPTX
        pptx_documents = self._process_pptx_files()
        print(f"Extracted {len(pptx_documents)} slides from presentations")
        
        # Traitement des fichiers PDF
        pdf_documents = self._process_pdf_files()
        print(f"Extracted {len(pdf_documents)} pages from PDF documents")
        
        # Combiner tous les documents
        documents = pptx_documents + pdf_documents
        
        if not documents:
            raise ValueError("Aucun contenu n'a été extrait des documents. Vérifiez les fichiers.")
        
        chunks = self._create_chunks(documents)
        
        if not chunks:
            raise ValueError("Aucun chunk n'a été créé à partir des documents. Vérifiez le contenu.")
        
        print(f"Created {len(chunks)} chunks from document content")
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_dir
        )
        print(f"Vectorstore built successfully with {len(chunks)} chunks")
        return vectorstore

    def _process_pdf_files(self) -> List[Document]:
        """Process PDF files and extract content"""
        documents = []
        
        for file_path in Path(self.data_dir).glob("**/*.pdf"):
            try:
                print(f"Processing PDF: {file_path}")
                pdf = PdfReader(file_path)
                
                pdf_metadata = {
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "file_type": ".pdf",
                    "total_pages": len(pdf.pages)
                }
                
                for page_idx, page in enumerate(pdf.pages):
                    page_num = page_idx + 1
                    
                    text = page.extract_text()
                    
                    if not text.strip():
                        print(f"  Skipping empty page {page_num}")
                        continue
                    
                    page_content = f"Page {page_num}\n\n{text}"
                    
                    page_metadata = {
                        **pdf_metadata,
                        "page_number": page_num
                    }
                    
                    doc = Document(
                        page_content=page_content,
                        metadata=page_metadata
                    )
                    documents.append(doc)
                    print(f"  Processed page {page_num} ({len(text)} chars)")
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        
        return documents
    
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
                file_type = doc.metadata.get('file_type', 'unknown')
                if file_type == '.pptx':
                    item_num = doc.metadata.get('slide_number', 'unknown')
                    item_type = 'slide'
                else:
                    item_num = doc.metadata.get('page_number', 'unknown')
                    item_type = 'page'
                    
                print(f"Error creating chunks for {item_type} {item_num}: {e}")
        
        return all_chunks
    
    def _create_rag_chain(self):
        """Create the RAG chain with document-specific context handling"""

        context_prompt = ChatPromptTemplate.from_template(
            """
            Vous êtes un assistant factuel spécialisé dans l'analyse de documents tout en maintenant l'historique de la conversation.
            
            CONTEXTE RÉCUPÉRÉ :
            {context}
            
            Directives :
            1. Si des informations manquent dans le contexte, répondez : "Je n'ai pas suffisamment de données pour répondre précisément à cela."
            2. Citez des extraits pertinents du document pour appuyer vos réponses.
            3. Ne jamais inventer d'informations ni utiliser de connaissances externes.
            4. Maintenez un ton de conversation naturel.
            5. Faites référence à l'historique de la conversation lorsqu'il est pertinent.
            
            Question : {question}
            
            Réponse :
            """
        )
        
        def _format_docs(docs):
            
            documents = {}
            for doc in docs:
                source = doc.metadata.get("file_name", "Unknown")
                file_type = doc.metadata.get("file_type", "Unknown")
                
                if source not in documents:
                    documents[source] = {"type": file_type, "items": {}}
                
                if file_type == ".pptx":
                    item_num = doc.metadata.get("slide_number", 0)
                    item_type = "Slide"
                else:
                    item_num = doc.metadata.get("page_number", 0)
                    item_type = "Page"
                
                if item_num not in documents[source]["items"]:
                    documents[source]["items"][item_num] = []
                
                documents[source]["items"][item_num].append(doc.page_content)
            
            formatted = []
            for doc_name, doc_info in documents.items():
                doc_type = "Presentation" if doc_info["type"] == ".pptx" else "Document"
                formatted.append(f"{doc_type}: {doc_name}")
                
                for item_num in sorted(doc_info["items"].keys()):
                    item_type = "Slide" if doc_info["type"] == ".pptx" else "Page"
                    item_content = "\n".join(doc_info["items"][item_num])
                    formatted.append(f"  {item_type} {item_num}: {item_content}")
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
                file_type = doc.metadata.get("file_type", ".pptx")
                
                if file_type == ".pptx":
                    item_num = doc.metadata.get("slide_number", "?")
                    item_type = "Slide"
                else:
                    item_num = doc.metadata.get("page_number", "?")
                    item_type = "Page"
                
                key = f"{source} ({item_type} {item_num})"
                if key not in sources:
                    sources[key] = True
                    
            return "\n".join(list(sources.keys()))
        except Exception as e:
            print(f"Error retrieving sources: {e}")
            return ""


def main():
    """Main entry point for the application"""
    
    parser = argparse.ArgumentParser(description="Document RAG System")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the vector database")
    parser.add_argument("--query", type=str, help="Run a specific query in non-interactive mode")
    args = parser.parse_args()
    
    rag = DocumentRAG(rebuild=args.rebuild)
    
    if args.query:
        result = rag.query(args.query)
        print(f"\nQuery: {args.query}\n")
        print(f"Response: {result}\n")
    else:
        print("Document RAG System initialized. Type 'exit' to quit.")
        
        while True:
            query = input("You: ")
            if query.lower() == "exit":
                break
            
            response = rag.query(query)
            print(f"Bot: {response}\n")


if __name__ == "__main__":
    main()