import os
import argparse
import re
import json
import datetime
import time
import hashlib
import logging
import base64 # For encoding images for the multimodal LLM
from typing import List, Dict, Optional, Generator, Tuple, Any, AsyncGenerator
from pathlib import Path
from io import BytesIO
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor

# --- New/Changed Imports ---
import fitz # PyMuPDF for PDF processing
from PIL import Image # For image handling (optional, but good practice)
# --- End New/Changed Imports ---

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
# Use Ollama directly for multimodal, potentially simpler than wrapping if issues arise
from langchain_community.llms import Ollama
# Keep text embedding model separate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Keep docx/pptx libraries
from pptx import Presentation
from docx import Document as WordDocument
from docx.shared import Inches # To potentially check image size in docx


import gradio as gr
from gradio.themes.utils import colors, sizes
from gradio.themes import Base as ThemeBase

# --- Configuration ---
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "qwq:32b-q8_0" # Your main text LLM
# --- New Configuration ---
MULTIMODAL_LLM_MODEL = "llava:latest" # Or specify version, e.g., "llava:7b-v1.6"
PROCESS_IMAGES = False # Default to False, enable via command line
IMAGE_DESCRIPTION_PREFIX = "[Image Description: "
IMAGE_DESCRIPTION_SUFFIX = "]"
# --- End New Configuration ---

DB_DIR = Path("vectore/chroma_multimodal_rag_db") # New DB directory
DATA_DIR = Path("documents")
CACHE_FILE = DB_DIR / "rag_cache_metadata.json"
RETRIEVER_K = 7
RETRIEVER_FETCH_K = 15

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Document RAG Class ---
class MultimodalDocumentRAG: # Renamed class for clarity
    def __init__(self, data_dir: Path = DATA_DIR, db_dir: Path = DB_DIR, rebuild: bool = False,
                 sharepoint_config: Optional[Dict] = None, process_images: bool = PROCESS_IMAGES):
        self.data_dir = data_dir
        self.db_dir = db_dir
        self.sharepoint_config = sharepoint_config
        self.sharepoint_status = "Not Connected (Not Implemented)"
        self.downloaded_files_count = 0
        self.process_images = process_images # Store image processing flag

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)

        # TODO: Implement SharePoint File Sync Here
        if self.sharepoint_config:
            logging.warning("SharePoint integration is configured but not implemented.")
            self.sharepoint_status = "Configured (Sync Logic Missing)"

        # Text Embedding Model (remains the same)
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        # Main Text LLM
        self.llm = Ollama(model=LLM_MODEL)

        # --- New: Multimodal LLM for Image Description ---
        if self.process_images:
            try:
                logging.info(f"Initializing multimodal LLM: {MULTIMODAL_LLM_MODEL}")
                # Use the base Ollama class which handles multimodal better for some models
                self.multimodal_llm = Ollama(model=MULTIMODAL_LLM_MODEL)
                # Do a quick test invocation if possible (optional)
                # self.multimodal_llm.invoke("Describe this text: Hello")
                logging.info("Multimodal LLM initialized.")
            except Exception as e:
                logging.error(f"Failed to initialize multimodal LLM '{MULTIMODAL_LLM_MODEL}'. Disabling image processing. Error: {e}", exc_info=True)
                self.process_images = False # Disable if model fails
                self.multimodal_llm = None
        else:
            logging.info("Image processing is disabled.")
            self.multimodal_llm = None
        # --- End New ---


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

    # --- Caching and Loading (Keep _get_current_file_metadata, _load_or_build_vectorstore as before) ---
    def _get_current_file_metadata(self) -> Dict[str, float]:
        # (Code remains the same as previous version)
        metadata = {}
        supported_extensions = {".pdf", ".pptx", ".docx"}
        try:
            for file_path in self.data_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    try:
                        relative_path = file_path.relative_to(self.data_dir).as_posix()
                        metadata[relative_path] = file_path.stat().st_mtime
                    except Exception as e:
                         logging.warning(f"Could not get metadata for file {file_path}: {e}")
        except Exception as e:
            logging.error(f"Error scanning data directory {self.data_dir}: {e}")
        return metadata

    def _load_or_build_vectorstore(self) -> Chroma:
        # (Code remains largely the same, but calls updated processing methods)
        current_metadata = self._get_current_file_metadata()
        cache_valid = False

        if CACHE_FILE.exists() and self.db_dir.exists() and any(self.db_dir.iterdir()):
            try:
                with open(CACHE_FILE, 'r') as f:
                    cached_metadata = json.load(f)
                # Added check for process_images flag consistency in cache (optional but good)
                process_images_cached = cached_metadata.get("_process_images_flag", None)
                if current_metadata == cached_metadata.get("files", {}) and process_images_cached == self.process_images :
                    logging.info("Cache metadata matches current files and settings. Loading existing vector store.")
                    cache_valid = True
                else:
                    reason = "File changes detected" if current_metadata != cached_metadata.get("files", {}) else "Image processing setting changed"
                    logging.info(f"{reason}. Rebuilding vector store.")
                    logging.info(f"Removing outdated vector store: {self.db_dir}")
                    import shutil
                    shutil.rmtree(self.db_dir)
                    self.db_dir.mkdir(parents=True, exist_ok=True)

            except Exception as e:
                logging.warning(f"Could not validate cache file {CACHE_FILE}. Rebuilding vector store. Error: {e}")
                if self.db_dir.exists():
                     logging.info(f"Removing potentially corrupt vector store: {self.db_dir}")
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
             vectorstore = Chroma(embedding_function=self.embeddings, persist_directory=str(self.db_dir))
             cache_data = {"files": {}, "_process_images_flag": self.process_images}
             with open(CACHE_FILE, 'w') as f: json.dump(cache_data, f)
             return vectorstore

        # Process documents (Calls updated methods that handle images if enabled)
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
            raise ValueError("No chunks created after splitting.")

        logging.info(f"Created {len(chunks)} chunks for vector store.")

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(self.db_dir)
        )
        logging.info(f"Vector store built successfully with {len(chunks)} chunks and persisted to {self.db_dir}")

        try:
             # Save file metadata and image processing flag
             cache_data = {"files": current_metadata, "_process_images_flag": self.process_images}
             with open(CACHE_FILE, 'w') as f:
                json.dump(cache_data, f)
             logging.info(f"Cache metadata saved to {CACHE_FILE}")
        except Exception as e:
             logging.error(f"Failed to save cache metadata to {CACHE_FILE}: {e}")

        return vectorstore


    # --- New: Image Description Generation ---
    def _get_image_description(self, image_bytes: bytes, source_ref: str) -> Optional[str]:
        """Generates a text description for image bytes using the multimodal LLM."""
        if not self.process_images or not self.multimodal_llm:
            return None
        if not image_bytes:
            return None

        logging.debug(f"Generating description for image from: {source_ref}")
        try:
            # Encode image bytes as base64
            img_base64 = base64.b64encode(image_bytes).decode('utf-8')

            # Use the invoke method with image support (check Ollama Langchain docs for exact format)
            # Typically involves passing images in the input dictionary or message list
            # Example using structured input (adjust based on exact Langchain Ollama version/API)

            # Try simple invoke first if model handles base64 directly in prompt
            # response = self.multimodal_llm.invoke(f"Describe this image: [image data {img_base64}]")

            # More robust method using invoke with explicit image list:
            # Note: The exact API might vary slightly between Langchain versions.
            # Refer to Langchain's Ollama documentation for multimodal input.
            response = self.multimodal_llm.invoke(
                 "Describe the key elements in this image.", # Simple prompt
                 images=[img_base64] # Pass base64 string in the 'images' list
            )

            description = response.strip()
            logging.debug(f"Generated description for {source_ref} (Length: {len(description)}): {description[:100]}...")
            if not description or len(description) < 5: # Basic sanity check
                 logging.warning(f"Generated description for {source_ref} seems too short or empty.")
                 return None
            return description

        except Exception as e:
            logging.error(f"Failed to generate description for image from {source_ref}: {e}", exc_info=False) # Reduce noise, set True for debug
            return None

    # --- Updated File Processing Methods ---

    def _process_pdf_files(self, file_path: Path) -> List[Document]:
        """Processes PDF using PyMuPDF, extracts text and image descriptions."""
        documents = []
        try:
            logging.info(f"Processing PDF: {file_path.name}")
            doc = fitz.open(file_path)
            pdf_metadata_base = {
                "source": str(file_path.relative_to(self.data_dir)),
                "file_name": file_path.name,
                "file_type": ".pdf",
                "total_pages": len(doc)
            }

            for page_idx, page in enumerate(doc):
                page_num = page_idx + 1
                page_text = page.get_text("text", sort=True).strip() # Get text
                page_content_parts = [page_text] if page_text else []

                # Process images if enabled
                image_descriptions = []
                if self.process_images:
                    image_list = page.get_images(full=True)
                    logging.debug(f"  Page {page_num}: Found {len(image_list)} images.")
                    for img_index, img_info in enumerate(image_list):
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        # Simple filtering (optional): Skip very small images
                        # img = Image.open(BytesIO(image_bytes))
                        # if img.width < 50 or img.height < 50:
                        #     logging.debug(f"    Skipping small image {img_index+1} on page {page_num}")
                        #     continue

                        source_ref = f"{file_path.name} page {page_num} image {img_index+1}"
                        description = self._get_image_description(image_bytes, source_ref)
                        if description:
                            image_descriptions.append(f"{IMAGE_DESCRIPTION_PREFIX}{description}{IMAGE_DESCRIPTION_SUFFIX}")

                if image_descriptions:
                     page_content_parts.extend(image_descriptions)

                # Combine text and descriptions for the page
                combined_page_content = "\n\n".join(part for part in page_content_parts if part) # Join non-empty parts

                if not combined_page_content.strip():
                    logging.debug(f"  Skipping empty page {page_num} in {file_path.name}")
                    continue

                page_metadata = {**pdf_metadata_base, "page_number": page_num}
                doc_entry = Document(page_content=combined_page_content, metadata=page_metadata)
                documents.append(doc_entry)

            doc.close()
        except Exception as e:
            logging.error(f"Error processing PDF {file_path}: {e}", exc_info=True)
        return documents

    def _process_pptx_files(self, file_path: Path) -> List[Document]:
        """Processes PPTX, extracts text and image descriptions."""
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
                slide_text_parts = []

                # Extract text from shapes (as before)
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        text = shape.text.strip()
                        if text: slide_text_parts.append(text)
                    elif shape.has_table:
                         # (Table text extraction logic remains the same)
                         table_rows = []
                         for row in shape.table.rows:
                             cell_texts = [cell.text_frame.text.strip() for cell in row.cells if cell.text_frame and cell.text_frame.text.strip()]
                             if cell_texts: table_rows.append(" | ".join(cell_texts))
                         if table_rows: slide_text_parts.append("Table:\n" + "\n".join(table_rows))

                # Process images if enabled
                image_descriptions = []
                if self.process_images:
                    logging.debug(f"  Slide {slide_num}: Checking for images...")
                    img_index = 0
                    for shape in slide.shapes:
                        # Check if shape is a picture
                        if hasattr(shape, 'image'):
                            try:
                                image = shape.image
                                image_bytes = image.blob
                                image_ext = image.ext # e.g., 'png', 'jpg'

                                # Optional: Filter small images
                                # if shape.width < Inches(0.5) or shape.height < Inches(0.5):
                                #     logging.debug(f"    Skipping small image {img_index+1} on slide {slide_num}")
                                #     img_index += 1
                                #     continue

                                source_ref = f"{file_path.name} slide {slide_num} image {img_index+1}"
                                description = self._get_image_description(image_bytes, source_ref)
                                if description:
                                    image_descriptions.append(f"{IMAGE_DESCRIPTION_PREFIX}{description}{IMAGE_DESCRIPTION_SUFFIX}")
                                img_index += 1
                            except Exception as img_err:
                                logging.warning(f"    Failed to extract/process image {img_index+1} on slide {slide_num}: {img_err}")
                                img_index += 1


                # Combine text and descriptions for the slide
                combined_content_parts = slide_text_parts + image_descriptions
                combined_slide_content = "\n\n".join(part for part in combined_content_parts if part)

                if not combined_slide_content.strip():
                    logging.debug(f"  Skipping empty slide {slide_num} in {file_path.name}")
                    continue

                slide_metadata = {**pptx_metadata_base, "slide_number": slide_num}
                doc_entry = Document(page_content=combined_slide_content, metadata=slide_metadata)
                documents.append(doc_entry)

        except Exception as e:
            logging.error(f"Error processing PPTX {file_path}: {e}", exc_info=True)
        return documents

    def _process_word_files(self, file_path: Path) -> List[Document]:
        """Processes DOCX, extracts text and image descriptions."""
        documents = []
        try:
            logging.info(f"Processing Word document: {file_path.name}")
            doc = WordDocument(file_path)
            word_metadata = {
                "source": str(file_path.relative_to(self.data_dir)),
                "file_name": file_path.name,
                "file_type": ".docx",
            }

            content_parts = []

            # Extract text from paragraphs and tables (as before)
            for element in doc.element.body:
                 # Check element type (simplified - might need more robust check)
                 if element.tag.endswith('p'): # Paragraph
                     para_text = ""
                     for run in element.xpath('.//w:t'): # Get text runs
                        if run.text:
                             para_text += run.text
                     if para_text.strip():
                        content_parts.append(para_text.strip())
                 elif element.tag.endswith('tbl'): # Table
                     table_rows = []
                     for row in element.xpath('.//w:tr'):
                         cell_texts = [ ''.join(cell.xpath('.//w:t/text()')) for cell in row.xpath('.//w:tc')]
                         cell_texts_stripped = [ct.strip() for ct in cell_texts if ct.strip()]
                         if cell_texts_stripped:
                             table_rows.append(" | ".join(cell_texts_stripped))
                     if table_rows:
                        content_parts.append("Table:\n" + "\n".join(table_rows))


            # --- Extract Images from DOCX ---
            image_descriptions = []
            if self.process_images:
                logging.debug(f"  Document {file_path.name}: Checking for images...")
                img_index = 0
                try:
                    # Images are related parts. We need to find image relationships.
                    image_parts = [
                        part for rel_id, part in doc.part.related_parts.items()
                        if "image" in part.content_type
                    ]
                    logging.debug(f"    Found {len(image_parts)} potential image parts.")
                    for img_part in image_parts:
                         try:
                             image_bytes = img_part.blob
                             # Optionally filter by size (more complex as size isn't directly in part)
                             # img = Image.open(BytesIO(image_bytes))
                             # if img.width < 50 or img.height < 50: continue

                             source_ref = f"{file_path.name} doc image {img_index+1}"
                             description = self._get_image_description(image_bytes, source_ref)
                             if description:
                                 image_descriptions.append(f"{IMAGE_DESCRIPTION_PREFIX}{description}{IMAGE_DESCRIPTION_SUFFIX}")
                             img_index += 1
                         except Exception as img_err:
                              logging.warning(f"    Failed processing image part {img_index+1} in {file_path.name}: {img_err}")
                              img_index += 1
                except Exception as rel_err:
                     logging.error(f"  Error accessing relationships/parts in {file_path.name}: {rel_err}")

            if image_descriptions:
                 content_parts.extend(image_descriptions)
            # --- End Image Extraction ---


            combined_content = "\n\n".join(part for part in content_parts if part)

            if not combined_content.strip():
                logging.debug(f"  Skipping empty document: {file_path.name}")
                return documents

            # Treat whole doc as one 'Document' object before chunking
            doc_entry = Document(page_content=combined_content, metadata=word_metadata)
            documents.append(doc_entry)
            logging.debug(f"  Processed Word doc {file_path.name} ({len(combined_content)} chars)")

        except Exception as e:
            logging.error(f"Error processing DOCX {file_path}: {e}", exc_info=True)
        return documents


    # --- Chunking (Keep as before) ---
    def _create_chunks_parallel(self, documents: List[Document]) -> List[Document]:
        # (Code remains the same as previous version)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
            length_function=len,
        )
        all_chunks = []
        logging.info("Splitting documents into chunks...")
        try:
            all_chunks = splitter.split_documents(documents)
            logging.info(f"Finished splitting into {len(all_chunks)} chunks.")
        except Exception as e:
             logging.error(f"Error during document splitting: {e}", exc_info=True)
             raise
        return all_chunks


    # --- Chain Creation, Formatting, Querying (Keep as before) ---
    # These don't need modification as image info is now text
    def _format_chat_history(self, chat_history: List[Tuple[str, str]]) -> str:
         # (Code remains the same)
        if not chat_history: return "No conversation history yet."
        return "\n".join([f"Human: {q}\nAssistant: {a}" for q, a in chat_history])

    def _format_docs(self, docs: List[Document]) -> str:
         # (Code remains the same)
        if not docs: return "No relevant documents found."
        formatted_docs = []
        for i, doc in enumerate(docs):
            metadata = doc.metadata
            file_name = metadata.get("file_name", "Unknown Source")
            content = doc.page_content
            source_info = f"Source {i+1}: {file_name}"
            if metadata.get("file_type") == ".pdf": source_info += f" (Page {metadata.get('page_number', '?')})"
            elif metadata.get("file_type") == ".pptx": source_info += f" (Slide {metadata.get('slide_number', '?')})"
            formatted_docs.append(f"{source_info}\nContent: {content}\n---")
        return "\n".join(formatted_docs)

    def _create_rag_chain(self):
         # (Code remains the same, prompt implicitly handles image descriptions now)
         # Optional: Add a small note to the prompt that context *might* include image descriptions.
        context_prompt = ChatPromptTemplate.from_template(
            """
            Vous êtes un assistant IA chargé de répondre aux questions en vous basant **uniquement** sur le contexte fourni (qui peut inclure des descriptions d'images précédées de [Image Description: ...]) et l'historique de la conversation.
            {... rest of prompt is the same ...}
            HISTORIQUE DE LA CONVERSATION:
            {chat_history}

            CONTEXTE RÉCUPÉRÉ (Documents Pertinents):
            {context}

            Directives Strictes:
            1.  Basez votre réponse **exclusivement** sur le CONTEXTE RÉCUPÉRÉ et l'HISTORIQUE DE LA CONVERSATION.
            2.  **Ne supposez rien et n'inventez aucune information.** Si la réponse n'est pas dans le contexte, déclarez explicitement que l'information n'est pas disponible dans les documents fournis.
            3.  Répondez directement à la question posée.
            4.  **Citez vos sources** en utilisant le format `[Source X]` où X est le numéro de la source indiqué dans le CONTEXTE RÉCUPÉRÉ. Intégrez les citations de manière fluide dans votre réponse. Exemple: "Le graphique montre une tendance à la hausse [Source 1]."
            5.  Si plusieurs sources confirment un point, vous pouvez les citer ensemble, e.g., `[Source 1, Source 3]`.
            6.  Gardez un ton factuel et professionnel.
            7.  Si la question fait référence à la conversation précédente, tenez-en compte.

            Question: {question}

            Réponse (basée uniquement sur le contexte et l'historique):
            """
        )
        rag_chain_with_sources = RunnablePassthrough.assign(
            source_documents=lambda x: self.retriever.invoke(x["question"]),
            question=lambda x: x["question"],
            chat_history=lambda x: x.get("chat_history", [])
        ) | RunnablePassthrough.assign(
            context=lambda x: self._format_docs(x["source_documents"]),
            chat_history_str=lambda x: self._format_chat_history(x["chat_history"])
        ) | {
            "source_documents": lambda x: x["source_documents"],
            "answer": context_prompt | self.llm | StrOutputParser()
        }
        return rag_chain_with_sources

    def _format_source_citation(self, docs: List[Document]) -> str:
        # (Code remains the same)
        if not docs: return ""
        sources_seen = set()
        formatted_sources = []
        for i, doc in enumerate(docs):
            metadata = doc.metadata
            file_name = metadata.get("file_name", "Unknown Source")
            source_key = f"Source {i+1}: {file_name}"
            if metadata.get("file_type") == ".pdf": source_key += f" (Page {metadata.get('page_number', '?')})"
            elif metadata.get("file_type") == ".pptx": source_key += f" (Slide {metadata.get('slide_number', '?')})"
            unique_id = f"{source_key}_{metadata.get('chunk_id', id(doc))}"
            if unique_id not in sources_seen:
                formatted_sources.append(source_key)
                sources_seen.add(unique_id)
        if not formatted_sources: return ""
        return "\n\n**Sources Consultées:**\n" + "\n".join(f"- {s}" for s in formatted_sources)


    async def query_stream(self, question: str, chat_history: List[Tuple[str, str]]) -> AsyncGenerator[Dict[str, Any], None]:
        # (Code remains the same as previous corrected version)
        try:
            chain_input = {"question": question, "chat_history": chat_history}
            full_response_answer = ""
            streamed_sources = None
            logging.info(f"Invoking RAG chain stream for question: {question[:50]}...")
            async for chunk in self.rag_chain.astream(chain_input):
                 if "answer" in chunk:
                    full_response_answer += chunk["answer"]
                    yield {"type": "chunk", "content": full_response_answer}
                 if "source_documents" in chunk and chunk["source_documents"] and streamed_sources is None:
                     streamed_sources = chunk["source_documents"]
            logging.info("RAG chain stream finished.")
            formatted_sources = self._format_source_citation(streamed_sources or [])
            if formatted_sources:
                yield {"type": "sources", "content": formatted_sources}
        except ConnectionError as ce:
            logging.error(f"Connection Error during RAG query: {ce}", exc_info=False) # Less verbose
            yield {"type": "error", "content": f"Erreur de Connexion: Impossible de joindre le modèle LLM ({LLM_MODEL}). Vérifiez si Ollama est lancé. ({ce})"}
        except TimeoutError as te:
            logging.error(f"Timeout Error during RAG query: {te}", exc_info=False)
            yield {"type": "error", "content": f"Erreur de Timeout: La requête a pris trop de temps. ({te})"}
        except Exception as e:
            logging.error(f"Unexpected error during RAG async query: {e}", exc_info=True) # Keep True here
            yield {"type": "error", "content": f"Erreur Inattendue: {e}"}


    def query(self, question: str, chat_history: List[Tuple[str, str]]) -> Dict[str, Any]:
        # (Code remains the same)
        try:
            chain_input = {"question": question, "chat_history": chat_history}
            result = self.rag_chain.invoke(chain_input)
            answer = result.get("answer", "Erreur: Aucune réponse générée.")
            source_docs = result.get("source_documents", [])
            formatted_sources = self._format_source_citation(source_docs)
            return {"answer": answer, "sources": formatted_sources}
        except Exception as e:
            logging.error(f"Error during RAG query (non-streaming): {e}", exc_info=True)
            return {"answer": f"Erreur: {e}", "sources": ""}

    def get_sharepoint_status(self) -> str:
        # (Code remains the same)
        status = "## Statut SharePoint\n"
        status += f"- **Connexion:** {self.sharepoint_status}\n"
        if self.downloaded_files_count > 0: status += f"- **Documents synchronisés:** {self.downloaded_files_count}\n"
        status += f"- **Cache local:** `{self.data_dir}`"
        return status

# --- Gradio Theme (Keep as before) ---
class CustomTheme(ThemeBase):
     # (Code remains the same)
    def __init__(self):
        super().__init__(primary_hue=colors.blue, secondary_hue=colors.cyan, neutral_hue=colors.gray, spacing_size=sizes.spacing_md, radius_size=sizes.radius_md, text_size=sizes.text_md,)
    def set_styles(self):
        super().set_styles()
        self.styles.update({"button": {"padding": f"{sizes.spacing_sm} {sizes.spacing_md}","border_radius": sizes.radius_md,},"button_primary": {"background": f"linear-gradient(to right, {colors.blue[600]}, {colors.blue[500]})","color": colors.white,"_hover": {"background": f"linear-gradient(to right, {colors.blue[700]}, {colors.blue[600]})",}},"chatbot": {"border_radius": sizes.radius_lg,"box_shadow": f"0 2px 8px {colors.gray[200]}",}, "input": {"border_radius": sizes.radius_lg,"box_shadow": f"0 2px 8px {colors.gray[200]}",}})


# --- Main Application ---
def main():
    parser = argparse.ArgumentParser(description="Multimodal Document RAG System")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of the vector database")
    parser.add_argument("--query", type=str, help="Run a specific query in non-interactive mode (outputs JSON)")
    parser.add_argument("--sharepoint", action="store_true", help="Enable SharePoint integration (requires env vars)")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help=f"Directory containing documents (default: {DATA_DIR})")
    parser.add_argument("--db-dir", type=str, default=str(DB_DIR), help=f"Directory to store vector database (default: {DB_DIR})")
    # --- New Argument ---
    parser.add_argument("--process-images", action="store_true", default=PROCESS_IMAGES, help=f"Enable processing of images within documents using '{MULTIMODAL_LLM_MODEL}' (default: {PROCESS_IMAGES})")
    # --- End New Argument ---
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    db_path = Path(args.db_dir)

    sharepoint_config = None
    if args.sharepoint:
        # (SharePoint config loading remains the same)
        logging.info("SharePoint integration enabled via command line.")
        sharepoint_config = { # ... load from env vars ...
            }
        if not all([...]): # Check env vars
            logging.error("SharePoint env vars missing.")
            return

    try:
        logging.info(f"Initializing Multimodal RAG system: Data='{data_path}', DB='{db_path}', Rebuild={args.rebuild}, Process Images={args.process_images}")
        # Pass the process_images flag
        rag = MultimodalDocumentRAG(
            data_dir=data_path,
            db_dir=db_path,
            rebuild=args.rebuild,
            sharepoint_config=sharepoint_config,
            process_images=args.process_images # Pass the flag here
        )
        logging.info("Multimodal RAG system initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize RAG system: {e}", exc_info=True)
        return

    # --- CLI Query Mode (remains the same logic) ---
    if args.query:
        logging.info(f"Running single query: {args.query}")
        result = rag.query(args.query, chat_history=[])
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    # --- Gradio UI Mode (remains the same logic) ---
    logging.info("Starting Gradio interface...")
    custom_theme = CustomTheme()

    with gr.Blocks(theme=custom_theme, title="Assistant Documentaire Multimodal RAG") as demo:
        gr.Markdown("""
        # Assistant Documentaire Intelligent (Texte & Images)
        Posez vos questions sur vos documents. L'assistant répondra en se basant sur le texte et les descriptions des images trouvées, et citera ses sources.
        """)
        # Add note if image processing is enabled
        if rag.process_images:
             gr.Markdown(f"*Traitement des images activé (Modèle: `{MULTIMODAL_LLM_MODEL}`). L'indexation initiale peut être plus longue.*")


        if args.sharepoint:
            with gr.Accordion("Statut SharePoint", open=False):
                gr.Markdown(rag.get_sharepoint_status())

        chatbot = gr.Chatbot(label="Conversation", height=600, bubble_full_width=False, show_label=False)
        chat_history_state = gr.State([])

        with gr.Row():
            msg = gr.Textbox(label="Votre Question", placeholder="Posez votre question ici...", scale=7, container=False, autofocus=True, show_label=False)
            submit_btn = gr.Button("Envoyer", variant="primary", scale=1)
            clear_btn = gr.Button("Effacer", variant="secondary", scale=1)

        status = gr.Textbox(value="Prêt", label="Statut", interactive=False, max_lines=1)

        # --- Gradio Functions (user, bot) remain the same ---
        def user(message, history_list):
            # (Code remains the same)
            history_list.append((message, None))
            return "", history_list

        async def bot(history_list):
            # (Code remains the same as previous corrected version)
            if not history_list or history_list[-1][0] is None: yield history_list, gr.update(value="Erreur: Aucune question fournie."); return
            question = history_list[-1][0]
            context_history = history_list[:-1]
            yield history_list, gr.update(value="Recherche et génération en cours...")
            full_answer, sources_text, error_text = "", "", ""
            try:
                 async for update in rag.query_stream(question, context_history):
                     if update["type"] == "chunk":
                         full_answer = update["content"]
                         history_list[-1] = (question, full_answer + "▌")
                         yield history_list, gr.update(value="Génération...")
                     elif update["type"] == "sources":
                         sources_text = update["content"]
                         history_list[-1] = (question, full_answer + sources_text) # Append sources
                         yield history_list, gr.update(value="Finalisation...")
                     elif update["type"] == "error":
                         error_text = update["content"]
                         history_list[-1] = (question, error_text)
                         yield history_list, gr.update(value="Erreur"); return
                 # Final cleanup of cursor
                 final_text = history_list[-1][1]
                 if isinstance(final_text, str) and final_text.endswith("▌"):
                     history_list[-1] = (question, final_text[:-1])
                 if not error_text: yield history_list, gr.update(value="Prêt")
            except Exception as e:
                 logging.error(f"Error in Gradio bot async function: {e}", exc_info=True)
                 error_msg = f"Erreur inattendue dans l'interface: {e}"
                 current_answer = history_list[-1][1] if len(history_list[-1]) > 1 else ""
                 if isinstance(current_answer, str) and current_answer.endswith("▌"): history_list[-1] = (question, current_answer[:-1] + f"\n\n{error_msg}")
                 else: history_list[-1] = (question, f"{current_answer or ''}\n\n{error_msg}")
                 yield history_list, gr.update(value="Erreur Critique")

        # --- Gradio Event Wiring (remains the same) ---
        msg.submit(user, [msg, chat_history_state], [msg, chat_history_state], queue=False).then(bot, [chat_history_state], [chatbot, status])
        submit_btn.click(user, [msg, chat_history_state], [msg, chat_history_state], queue=False).then(bot, [chat_history_state], [chatbot, status])
        clear_btn.click(lambda: ([], []), None, [chatbot, chat_history_state], queue=False).then(lambda: "Prêt", None, status)

        # Add examples relevant to images
        gr.Examples(
            examples=[
                "Quels sont les principaux sujets abordés dans le document X ?",
                "Résume le rapport financier de l'année dernière.",
                "Quelle est la procédure pour demander des congés ?",
                "Y a-t-il des informations sur le projet 'Phoenix' ?",
                "Décris l'organigramme présenté dans la présentation Y.", # Image example
                "Que montre le graphique de la page 5 du rapport Z ?", # Image example
            ],
            inputs=msg,
            label="Exemples de Questions"
        )

    demo.queue().launch()


if __name__ == "__main__":
    # --- Update requirements suggestion ---
    # Consider creating a requirements.txt file with:
    # langchain langchain-chroma langchain-community langchain-core langchain-huggingface
    # sentence-transformers
    # PyMuPDF # Changed from pypdf
    # python-pptx
    # python-docx
    # Pillow # For image handling
    # ollama # Langchain Ollama integration
    # gradio
    # uvicorn httpx numpy
    # --- Add SharePoint library if implemented ---
    main()