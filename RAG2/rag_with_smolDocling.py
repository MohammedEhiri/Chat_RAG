import os
import sys
import argparse
import logging
import json
import datetime
import hashlib
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
import pdf2image
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError
from transformers import AutoProcessor, AutoModelForVision2Seq
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from pypdf import PdfReader
from pptx import Presentation
from docx import Document as WordDocument
import gradio as gr

# --- Configuration ---
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DOC_PROCESSING_MODEL = "ds4sd/SmolDocling-256M-preview"
LLM_MODEL = "gemma3:27b-it-q8_0"
DB_DIR = Path("vectore/chroma_rag_db")
DATA_DIR = Path("GO 2024")
CACHE_FILE = DB_DIR / "rag_cache_metadata.json"
RETRIEVER_K = 7
RETRIEVER_FETCH_K = 15

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocElementType(str):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    CODE = "code"
    FORMULA = "formula"
    HEADER = "header"

class SmolDoclingProcessor:
    def __init__(self, model_name: str = DOC_PROCESSING_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Loading SmolDocling processor on {self.device}...")
        
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(model_name).to(self.device)
            logging.info("SmolDocling processor loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load SmolDocling: {e}")
            raise

    def process_document_page(self, image: Image.Image) -> List[Dict]:
        """Process a single document page image into structured elements"""
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                output_scores=True,
                return_dict_in_generate=True
            )
            generated_text = self.processor.batch_decode(
                generated_ids.sequences, 
                skip_special_tokens=True
            )[0]
            return self._parse_doctags(generated_text)
        except Exception as e:
            logging.error(f"Error processing document page: {e}")
            return []

    def _parse_doctags(self, doctags_text: str) -> List[Dict]:
        """Parse DocTags output into structured elements"""
        elements = []
        current_element = None
        
        for line in doctags_text.split('\n'):
            if line.startswith('<') and line.endswith('>'):
                if current_element:
                    elements.append(current_element)
                
                tag_parts = line[1:-1].split()
                if len(tag_parts) < 2:
                    continue
                    
                element_type = tag_parts[0].lower()
                coordinates = list(map(float, tag_parts[1:5]))
                
                current_element = {
                    "type": element_type,
                    "bbox": coordinates,
                    "content": "",
                    "metadata": {}
                }
                
                if len(tag_parts) > 5:
                    for part in tag_parts[5:]:
                        if '=' in part:
                            key, val = part.split('=', 1)
                            current_element["metadata"][key] = val
            elif current_element:
                current_element["content"] += line + '\n'
        
        if current_element:
            elements.append(current_element)
            
        return elements

class DocumentRAG:
    def __init__(self, data_dir: Path = DATA_DIR, db_dir: Path = DB_DIR, rebuild: bool = False):
        self.data_dir = data_dir
        self.db_dir = db_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)

        try:
            logging.info("Initializing embedding model...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            )
            
            logging.info("Initializing LLM...")
            self.llm = Ollama(model=LLM_MODEL)
            
            logging.info("Initializing SmolDocling processor...")
            self.smol_docling = SmolDoclingProcessor()
        except Exception as e:
            logging.error(f"Model initialization failed: {e}")
            raise

        if rebuild and self.db_dir.exists():
            logging.info(f"Rebuild requested. Removing existing vector store: {self.db_dir}")
            import shutil
            shutil.rmtree(self.db_dir)
            if CACHE_FILE.exists():
                CACHE_FILE.unlink()
            self.db_dir.mkdir(parents=True, exist_ok=True)

        self.vectorstore = self._load_or_build_vectorstore()

        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": RETRIEVER_K, "fetch_k": RETRIEVER_FETCH_K}
        )

        self.rag_chain = self._create_rag_chain()

    def _get_current_file_metadata(self) -> Dict[str, float]:
        """Get modification times for all supported files in data_dir."""
        metadata = {}
        supported_extensions = {".pdf", ".pptx", ".docx"}
        try:
            for file_path in self.data_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    relative_path = file_path.relative_to(self.data_dir).as_posix()
                    metadata[relative_path] = file_path.stat().st_mtime
        except Exception as e:
            logging.error(f"Error scanning data directory {self.data_dir}: {e}")
        return metadata

    def _load_or_build_vectorstore(self) -> Chroma:
        """Load vector store if cache is valid, otherwise rebuild."""
        current_metadata = self._get_current_file_metadata()
        cache_valid = False

        if CACHE_FILE.exists() and self.db_dir.exists() and any(self.db_dir.iterdir()):
            try:
                with open(CACHE_FILE, 'r') as f:
                    cached_metadata = json.load(f)
                if current_metadata == cached_metadata:
                    logging.info("Cache metadata matches current files. Loading existing vector store.")
                    cache_valid = True
                else:
                    logging.info("File changes detected. Rebuilding vector store.")
                    import shutil
                    shutil.rmtree(self.db_dir)
                    self.db_dir.mkdir(parents=True, exist_ok=True)

            except Exception as e:
                logging.warning(f"Could not validate cache file {CACHE_FILE}. Rebuilding vector store. Error: {e}")
                if self.db_dir.exists():
                    import shutil
                    shutil.rmtree(self.db_dir)
                    self.db_dir.mkdir(parents=True, exist_ok=True)

        if cache_valid:
            try:
                return Chroma(
                    persist_directory=str(self.db_dir),
                    embedding_function=self.embeddings
                )
            except Exception as e:
                logging.error(f"Error loading vector store from {self.db_dir}, rebuilding. Error: {e}")

        logging.info("Building new vector store...")
        all_documents = []
        supported_extensions = {".pdf", ".pptx", ".docx"}
        file_paths_to_process = [
            fp for fp in self.data_dir.rglob("*")
            if fp.is_file() and fp.suffix.lower() in supported_extensions
        ]

        if not file_paths_to_process:
            logging.warning(f"No supported documents found in {self.data_dir}. Vector store will be empty.")
            vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=str(self.db_dir)
            )
            current_metadata = {}
            with open(CACHE_FILE, 'w') as f:
                json.dump(current_metadata, f)
            return vectorstore

        for file_path in file_paths_to_process:
            if file_path.suffix.lower() == ".pdf":
                all_documents.extend(self._process_pdf_files(file_path))
            elif file_path.suffix.lower() == ".pptx":
                all_documents.extend(self._process_pptx_files(file_path))
            elif file_path.suffix.lower() == ".docx":
                all_documents.extend(self._process_word_files(file_path))

        if not all_documents:
            raise ValueError(f"No content could be extracted from documents in {self.data_dir}.")

        logging.info(f"Extracted content from {len(file_paths_to_process)} files, resulting in {len(all_documents)} initial document pieces.")

        chunks = self._create_chunks_parallel(all_documents)

        if not chunks:
            raise ValueError("No chunks created after splitting. Check text splitter settings.")

        logging.info(f"Created {len(chunks)} chunks for vector store.")

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(self.db_dir)
        )
        logging.info(f"Vector store built successfully with {len(chunks)} chunks and persisted to {self.db_dir}")

        try:
            with open(CACHE_FILE, 'w') as f:
                json.dump(current_metadata, f)
            logging.info(f"Cache metadata saved to {CACHE_FILE}")
        except Exception as e:
            logging.error(f"Failed to save cache metadata to {CACHE_FILE}: {e}")

        return vectorstore

    def _process_pdf_files(self, file_path: Path) -> List[Document]:
        documents = []
        try:
            logging.info(f"Processing PDF: {file_path.name}")
            pdf = PdfReader(file_path)
            pdf_metadata_base = {
                "source": str(file_path.relative_to(self.data_dir)),
                "file_name": file_path.name,
                "file_type": ".pdf",
                "total_pages": len(pdf.pages)
            }

            # Convert PDF to images
            try:
                images = pdf2image.convert_from_path(
                    file_path,
                    dpi=300,
                    fmt='png',
                    thread_count=4,
                    poppler_path=None  # Set if Poppler is not in PATH
                )
            except (PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError) as e:
                logging.warning(f"pdf2image conversion failed: {e}")
                images = []

            if not images:
                logging.info("Falling back to traditional PDF text extraction")
                for page_idx, page in enumerate(pdf.pages):
                    page_num = page_idx + 1
                    text = page.extract_text() or ""
                    if not text.strip():
                        continue
                    page_metadata = {**pdf_metadata_base, "page_number": page_num}
                    doc = Document(page_content=text, metadata=page_metadata)
                    documents.append(doc)
                return documents

            # Process each page with SmolDocling
            for page_idx, (page, image) in enumerate(zip(pdf.pages, images)):
                page_num = page_idx + 1
                elements = self.smol_docling.process_document_page(image)
                if not elements:
                    logging.debug(f"No elements extracted from page {page_num}")
                    continue

                for element in elements:
                    element_type = element["type"]
                    content = element["content"].strip()
                    if not content:
                        continue

                    page_metadata = {
                        **pdf_metadata_base,
                        "page_number": page_num,
                        "element_type": element_type,
                        "element_bbox": element["bbox"],
                        **element.get("metadata", {})
                    }

                    # Format content based on element type
                    if element_type == "table":
                        content = f"TABLE:\n{content}"
                    elif element_type == "figure":
                        content = f"FIGURE: {element.get('metadata', {}).get('caption', '')}\n{content}"

                    doc = Document(
                        page_content=content,
                        metadata=page_metadata
                    )
                    documents.append(doc)

        except Exception as e:
            logging.error(f"Error processing PDF {file_path}: {e}", exc_info=True)
            # Final fallback to traditional text extraction
            try:
                logging.info("Attempting final fallback to basic PDF text extraction")
                pdf = PdfReader(file_path)
                for page_idx, page in enumerate(pdf.pages):
                    page_num = page_idx + 1
                    text = page.extract_text() or ""
                    if not text.strip():
                        continue
                    page_metadata = {**pdf_metadata_base, "page_number": page_num}
                    doc = Document(page_content=text, metadata=page_metadata)
                    documents.append(doc)
            except Exception as fallback_e:
                logging.error(f"Final fallback PDF processing also failed for {file_path}: {fallback_e}")

        return documents

    def _process_pptx_files(self, file_path: Path) -> List[Document]:
        documents = []
        try:
            logging.info(f"Processing presentation: {file_path.name}")
            presentation = Presentation(file_path)
            pptx_metadata_base = {
                "source": str(file_path.relative_to(self.data_dir)),
                "file_name": file_path.name,
                "file_type": ".pptx",
                "total_slides": len(presentation.slides)
            }
            for slide_idx, slide in enumerate(presentation.slides):
                slide_num = slide_idx + 1
                all_text = self._extract_all_text_from_slide(slide)
                if not all_text.strip():
                    continue
                slide_metadata = {**pptx_metadata_base, "slide_number": slide_num}
                doc = Document(page_content=all_text, metadata=slide_metadata)
                documents.append(doc)
        except Exception as e:
            logging.error(f"Error processing PPTX {file_path}: {e}", exc_info=True)
        return documents

    def _process_word_files(self, file_path: Path) -> List[Document]:
        documents = []
        try:
            logging.info(f"Processing Word document: {file_path.name}")
            doc = WordDocument(file_path)
            word_metadata = {
                "source": str(file_path.relative_to(self.data_dir)),
                "file_name": file_path.name,
                "file_type": ".docx",
            }
            full_text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text.strip())
            for table in doc.tables:
                table_rows = []
                for row in table.rows:
                    cell_texts = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if cell_texts:
                        table_rows.append(" | ".join(cell_texts))
                if table_rows:
                    full_text.append("Table:\n" + "\n".join(table_rows))
            combined_text = "\n\n".join(full_text).strip()
            if not combined_text:
                return documents
            doc_entry = Document(page_content=combined_text, metadata=word_metadata)
            documents.append(doc_entry)
        except Exception as e:
            logging.error(f"Error processing DOCX {file_path}: {e}", exc_info=True)
        return documents

    def _extract_all_text_from_slide(self, slide) -> str:
        text_parts = []
        try:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    text = shape.text.strip()
                    if text:
                        text_parts.append(text)
                elif shape.has_table:
                    table_rows = []
                    for row in shape.table.rows:
                        cell_texts = [cell.text_frame.text.strip() for cell in row.cells if cell.text_frame and cell.text_frame.text.strip()]
                        if cell_texts:
                            table_rows.append(" | ".join(cell_texts))
                    if table_rows:
                        text_parts.append("Table:\n" + "\n".join(table_rows))
        except Exception as e:
            logging.warning(f"Error extracting text from a shape/table on a slide: {e}")
        return "\n\n".join(text_parts).strip()

    def _create_chunks_parallel(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
            length_function=len,
        )
        try:
            return splitter.split_documents(documents)
        except Exception as e:
            logging.error(f"Error during document splitting: {e}", exc_info=True)
            raise

    def _create_rag_chain(self):
        context_prompt = ChatPromptTemplate.from_template(
            """
            You are an AI assistant tasked with answering questions based **exclusively** on the provided context and conversation history.

            CONVERSATION HISTORY:
            {chat_history}

            CONTEXT FROM DOCUMENTS:
            {context}

            Strict Guidelines:
            1. Base your answer **only** on the CONTEXT and CONVERSATION HISTORY.
            2. **Do not assume or invent information.** If the answer isn't in the context, explicitly state that the information isn't available.
            3. Answer directly to the question.
            4. **Cite sources** using the format `[Source X]` where X is the source number.
            5. If multiple sources confirm a point, cite them together, e.g., `[Source 1, Source 3]`.
            6. Keep a factual and professional tone.

            Question: {question}

            Answer (based only on context and history):
            """
        )
        return context_prompt | self.llm

    def query(self, question: str, chat_history: List[Tuple[str, str]] = []) -> Dict[str, Any]:
        try:
            docs = self.retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            chain_input = {"question": question, "chat_history": chat_history, "context": context}
            answer = self.rag_chain.invoke(chain_input)
            formatted_sources = self._format_source_citation(docs)
            return {"answer": answer, "sources": formatted_sources}
        except Exception as e:
            logging.error(f"Error during RAG query: {e}", exc_info=True)
            return {"answer": f"Error: {e}", "sources": ""}

    def _format_source_citation(self, docs: List[Document]) -> str:
        if not docs:
            return ""
        sources_seen = set()
        formatted_sources = []
        for i, doc in enumerate(docs):
            metadata = doc.metadata
            file_name = metadata.get("file_name", "Unknown Source")
            source_key = f"Source {i+1}: {file_name}"
            
            if metadata.get("file_type") == ".pdf":
                page_num = metadata.get("page_number", "?")
                source_key += f" (Page {page_num})"
                element_type = metadata.get("element_type")
                if element_type:
                    source_key += f", {element_type.upper()}"
            elif metadata.get("file_type") == ".pptx":
                slide_num = metadata.get("slide_number", "?")
                source_key += f" (Slide {slide_num})"
            
            unique_id = f"{source_key}_{hashlib.md5(doc.page_content.encode()).hexdigest()[:8]}"
            if unique_id not in sources_seen:
                formatted_sources.append(source_key)
                sources_seen.add(unique_id)
        return "\n\n**Sources Consulted:**\n" + "\n".join(f"- {s}" for s in formatted_sources)

def main():
    parser = argparse.ArgumentParser(description="Document RAG System with SmolDocling Integration")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of the vector database")
    parser.add_argument("--query", type=str, help="Run a specific query in non-interactive mode (outputs JSON)")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help=f"Directory containing documents (default: {DATA_DIR})")
    parser.add_argument("--db-dir", type=str, default=str(DB_DIR), help=f"Directory to store vector database (default: {DB_DIR})")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    db_path = Path(args.db_dir)

    try:
        logging.info(f"Initializing RAG system: Data='{data_path}', DB='{db_path}', Rebuild={args.rebuild}")
        rag = DocumentRAG(
            data_dir=data_path,
            db_dir=db_path,
            rebuild=args.rebuild,
        )
        logging.info("RAG system initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize RAG system: {e}", exc_info=True)
        return

    if args.query:
        result = rag.query(args.query)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    # Gradio Interface
    with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.blue)) as demo:
        gr.Markdown("# Document RAG Assistant")
        chatbot = gr.Chatbot(height=500)
        msg = gr.Textbox(label="Enter your question", placeholder="Type your question here...")
        clear = gr.Button("Clear Conversation")

        def respond(message, chat_history):
            response = rag.query(message)
            chat_history.append((message, response["answer"] + "\n\n" + response["sources"]))
            return "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue().launch()

if __name__ == "__main__":
    main()