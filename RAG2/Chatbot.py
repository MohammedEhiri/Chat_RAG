import os
import argparse
import re
from typing import List, Dict, Optional, Generator, Tuple
from pathlib import Path
from io import BytesIO
import tempfile
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pptx import Presentation
from docx import Document as WordDocument

import gradio as gr
from gradio.themes.utils import colors, sizes
from gradio.themes import Base as ThemeBase

from office365.sharepoint.client_context import ClientContext
from office365.sharepoint.files.file import File
from office365.runtime.auth.user_credential import UserCredential

import dotenv
dotenv.load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "gemma3"
DB_DIR = "vectore/chroma_docs_db"
DATA_DIR = "documents"


class SharePointClient:
    def __init__(self, site_url: str, username: str, password: str):
        self.site_url = site_url
        self.ctx = ClientContext(site_url).with_credentials(
            UserCredential(username, password)
        )

    def download_folder_contents(self, folder_url: str, local_path: str) -> List[Tuple[str, str]]:
        """Download all files from SharePoint folder recursively"""
        downloaded_files = []
        folder = self.ctx.web.get_folder_by_server_relative_url(folder_url)
        self.ctx.load(folder)
        self.ctx.execute_query()

        files = folder.files
        self.ctx.load(files)
        self.ctx.execute_query()

        # Ensure local directory exists
        os.makedirs(local_path, exist_ok=True)

        for file in files:
            try:
                # Download file content
                file_content = File.open_binary(self.ctx, file.serverRelativeUrl)
                
                # Save to local file
                file_path = os.path.join(local_path, file.name)
                with open(file_path, "wb") as f:
                    f.write(file_content.content)
                
                downloaded_files.append((file.name, file_path))
                print(f"Downloaded: {file.name}")
            except Exception as e:
                print(f"Error downloading {file.name}: {str(e)}")

        # Process subfolders recursively
        folders = folder.folders
        self.ctx.load(folders)
        self.ctx.execute_query()

        for subfolder in folders:
            subfolder_path = os.path.join(local_path, subfolder.name)
            downloaded_files.extend( self.download_folder_contents(subfolder.serverRelativeUrl, subfolder_path))
        
        return downloaded_files


class CustomTheme(ThemeBase):
    def __init__(self):
        super().__init__(
            primary_hue=colors.blue,
            secondary_hue=colors.cyan,
            neutral_hue=colors.gray,
            spacing_size=sizes.spacing_md,
            radius_size=sizes.radius_md,
            text_size=sizes.text_md,
        )
    
    def set_styles(self):
        super().set_styles()
        self.styles.update({
            "button": {
                "padding": f"{sizes.spacing_sm} {sizes.spacing_md}",
                "border_radius": sizes.radius_md,
            },
            "button_primary": {
                "background": f"linear-gradient(to right, {colors.blue[600]}, {colors.blue[500]})",
                "color": colors.white,
                "_hover": {
                    "background": f"linear-gradient(to right, {colors.blue[700]}, {colors.blue[600]})",
                }
            },
            "chatbot": {
                "border_radius": sizes.radius_lg,
                "box_shadow": f"0 2px 8px {colors.gray[200]}",
            },
            "input": {
                "border_radius": sizes.radius_lg,
                "box_shadow": f"0 2px 8px {colors.gray[200]}",
            }
        })


class DocumentRAG:
    def __init__(self, data_dir: str = DATA_DIR, db_dir: str = DB_DIR, rebuild: bool = False,
                 sharepoint_config: Optional[Dict] = None):
        self.data_dir = data_dir
        self.db_dir = db_dir
        
        # Download from SharePoint if configured
        if sharepoint_config:
            self._download_from_sharepoint(sharepoint_config)
        
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
    
    def _download_from_sharepoint(self, config: Dict):
        """Download documents from SharePoint"""
        print("Downloading documents from SharePoint...")
        
        # Create temp directory if using SharePoint
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        try:
            sp_client = SharePointClient(
                site_url=config["site_url"],
                username=config["username"],
                password=config["password"]
            )
            
            downloaded = sp_client.download_folder_contents(
                folder_url=config["folder_url"],
                local_path=self.data_dir
            )
            print(f"Downloaded {len(downloaded)} files from SharePoint")
        except Exception as e:
            print(f"Error downloading from SharePoint: {str(e)}")
            raise
    
    def _build_vectorstore(self) -> Chroma:
        """Process documents and build vectorstore"""
        print("Processing documents and building vectorstore...")
        
        # Process all document types
        pptx_documents = self._process_pptx_files()
        print(f"Extracted {len(pptx_documents)} slides from presentations")
        
        pdf_documents = self._process_pdf_files()
        print(f"Extracted {len(pdf_documents)} pages from PDF documents")
        
        word_documents = self._process_word_files()
        print(f"Extracted {len(word_documents)} sections from Word documents")
        
        # Combine all documents
        documents = pptx_documents + pdf_documents + word_documents
        
        if not documents:
            raise ValueError("No content extracted from documents. Please check your files.")
        
        chunks = self._create_chunks(documents)
        
        if not chunks:
            raise ValueError("No chunks created from document content. Please check the content.")
        
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
        """Process PowerPoint files and extract content"""
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
    
    def _process_word_files(self) -> List[Document]:
        """Process Word files and extract content"""
        documents = []
        
        for file_path in Path(self.data_dir).glob("**/*.docx"):
            try:
                print(f"Processing Word document: {file_path}")
                doc = WordDocument(file_path)
                
                word_metadata = {
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "file_type": ".docx",
                    "total_paragraphs": len(doc.paragraphs)
                }
                
                # Extract text from paragraphs
                full_text = []
                for para in doc.paragraphs:
                    if para.text.strip():
                        full_text.append(para.text)
                
                # Extract text from tables
                for table in doc.tables:
                    for row in table.rows:
                        for cell in row.cells:
                            if cell.text.strip():
                                full_text.append(cell.text)
                
                # Combine and clean text
                full_text = "\n".join(full_text)
                full_text = re.sub(r'\n\s*\n', '\n\n', full_text).strip()
                
                if not full_text:
                    print(f"  Skipping empty document: {file_path}")
                    continue
                
                # Create document with section headings
                doc_content = f"Document: {file_path.name}\n\n{full_text}"
                
                doc_entry = Document(
                    page_content=doc_content,
                    metadata=word_metadata
                )
                documents.append(doc_entry)
                print(f"  Processed document with {len(full_text)} characters")
                
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
                elif file_type == '.pdf':
                    item_num = doc.metadata.get('page_number', 'unknown')
                    item_type = 'page'
                else:
                    item_num = 'N/A'
                    item_type = 'section'
                    
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
                elif file_type == ".pdf":
                    item_num = doc.metadata.get("page_number", 0)
                    item_type = "Page"
                else:
                    item_num = "Section"
                    item_type = "Section"
                
                if item_num not in documents[source]["items"]:
                    documents[source]["items"][item_num] = []
                
                documents[source]["items"][item_num].append(doc.page_content)
            
            formatted = []
            for doc_name, doc_info in documents.items():
                if doc_info["type"] == ".pptx":
                    doc_type = "Presentation"
                elif doc_info["type"] == ".pdf":
                    doc_type = "PDF Document"
                else:
                    doc_type = "Word Document"
                
                formatted.append(f"{doc_type}: {doc_name}")
                
                for item_num in sorted(doc_info["items"].keys()):
                    if doc_info["type"] == ".pptx":
                        item_type = "Slide"
                    elif doc_info["type"] == ".pdf":
                        item_type = "Page"
                    else:
                        item_type = "Section"
                    
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
    
    def query_stream(self, question: str) -> Generator[str, None, None]:
        """Process query and stream response"""
        try:
            full_response = ""
            for chunk in self.rag_chain.stream(question):
                full_response += chunk
                yield full_response
            
            sources = self._get_sources(question)
            if sources:
                yield f"{full_response}\n\nSources:\n{sources}"
                
        except Exception as e:
            yield f"Erreur : {str(e)}"
    
    def query(self, question: str) -> str:
        """Process query and return complete response"""
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
                elif file_type == ".pdf":
                    item_num = doc.metadata.get("page_number", "?")
                    item_type = "Page"
                else:
                    item_num = "Section"
                    item_type = "Section"
                
                key = f"{source} ({item_type} {item_num})"
                if key not in sources:
                    sources[key] = True
                    
            return "\n".join(list(sources.keys()))
        except Exception as e:
            print(f"Error retrieving sources: {e}")
            return ""


def main():
    """Main entry point for the application"""
    
    parser = argparse.ArgumentParser(description="Document RAG System with SharePoint Integration")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the vector database")
    parser.add_argument("--query", type=str, help="Run a specific query in non-interactive mode")
    parser.add_argument("--sharepoint", action="store_true", help="Enable SharePoint integration")
    args = parser.parse_args()
    
    # Configure SharePoint if enabled
    sharepoint_config = None
    if args.sharepoint:
        sharepoint_config = {
            "site_url": os.getenv("SHAREPOINT_SITE_URL"),
            "username": os.getenv("SHAREPOINT_USERNAME"),
            "password": os.getenv("SHAREPOINT_PASSWORD"),
            "folder_url": os.getenv("SHAREPOINT_FOLDER_URL")
        }
    
    rag = DocumentRAG(
        rebuild=args.rebuild,
        sharepoint_config=sharepoint_config
    )
    
    if args.query:
        result = rag.query(args.query)
        print(f"\nQuery: {args.query}\n")
        print(f"Response: {result}\n")
    else:
        # Launch the improved Gradio interface
        custom_theme = CustomTheme()
        
        with gr.Blocks(theme=custom_theme, title="Assistant Documentaire RAG") as demo:
            # Header
            gr.Markdown("""
            # Assistant Documentaire RAG
            Posez des questions sur vos documents et obtenez des réponses précises basées sur leur contenu.
            """)
            
            # Chat interface
            chatbot = gr.Chatbot(
                height=600,
                bubble_full_width=False,
                avatar_images=(
                    "assets/user.png", 
                    "assets/assistant.png"
                )
            )
            
            # Input and controls
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Posez votre question ici...",
                    scale=7,
                    container=False,
                    autofocus=True
                )
                submit_btn = gr.Button("Envoyer", variant="primary", scale=1)
                clear_btn = gr.Button("Nouvelle conversation", variant="secondary", scale=1)
            
            # Additional controls
            with gr.Accordion("Options avancées", open=False):
                gr.Markdown("""
                **Sources des documents:**
                - Système de fichiers local
                - SharePoint (si configuré)
                
                **Formats supportés:**
                - Documents Word (.docx)
                - Présentations PowerPoint (.pptx)
                - Fichiers PDF (.pdf)
                
                **Fonctionnalités:**
                - Réponses basées uniquement sur le contenu des documents
                - Streaming des réponses en temps réel
                - Références aux sources originales
                """)
            
            # System status
            status = gr.Textbox(
                value="Système prêt à répondre à vos questions",
                interactive=False,
                label="Statut"
            )
            
            # Chat functions
            def user(message, chat_history):
                return "", chat_history + [[message, None]]
            
            def bot(chat_history):
                question = chat_history[-1][0]
                
                # Update status
                status.value = "🔍 Recherche dans les documents..."
                yield gr.update(), gr.update(), gr.update()
                
                try:
                    # Stream the response
                    full_response = ""
                    for chunk in rag.query_stream(question):
                        full_response = chunk
                        chat_history[-1][1] = full_response
                        yield "", chat_history, "Génération de la réponse..."
                    
                    # Add sources if available
                    if "Sources:" in full_response:
                        status.value = "Réponse générée avec sources"
                    else:
                        status.value = "Réponse générée"
                    
                    yield "", chat_history, status.value
                    
                except Exception as e:
                    status.value = f"Erreur: {str(e)}"
                    chat_history[-1][1] = f"Une erreur est survenue: {str(e)}"
                    yield "", chat_history, status.value
            
            # Event handlers
            msg.submit(
                user, [msg, chatbot], [msg, chatbot], queue=False
            ).then(
                bot, chatbot, [msg, chatbot, status]
            )
            
            submit_btn.click(
                user, [msg, chatbot], [msg, chatbot], queue=False
            ).then(
                bot, chatbot, [msg, chatbot, status]
            )
            
            clear_btn.click(
                lambda: ([], "Prêt pour une nouvelle conversation"),
                None, [chatbot, status], queue=False
            )
            
            # Examples
            gr.Examples(
                examples=[
                    "Quel est le sujet principal de ces documents?",
                    "Résumez les points clés de la présentation",
                    "Quelles sont les conclusions principales du rapport Word?",
                    "Trouvez toutes les références aux chiffres clés"
                ],
                inputs=msg,
                label="Exemples de questions"
            )

        demo.launch()




if __name__ == "__main__":
    main()