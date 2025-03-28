import streamlit as st
import os
import logging
import time
from typing import List, Dict, Optional, Any
# Import functionalities from other modules
from processor import process_documents
from vector_store import get_vector_store, add_chunks_to_vector_store, search_vector_store
from llm_interface import generate_response

# Configure basic logging
# Streamlit logs can be sometimes tricky, basic config helps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
DOCUMENTS_DIR = "documents"
VECTOR_STORE_PERSIST_DIR = "./data"
VECTOR_STORE_COLLECTION = "rag_collection"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5" # Good balance of performance and size
LLM_MODEL = "gemma3" # Or "llama3:8b", "phi3", etc. Make sure it's pulled in Ollama!
CHUNK_SIZE = 1000 # For chunk_by_title, this is max_characters, adjust as needed
CHUNK_OVERLAP = 100 # Less relevant for chunk_by_title but good practice for others
TOP_K_RESULTS = 5

# --- Helper Functions ---

@st.cache_resource(show_spinner="Initializing Vector Store...")
def initialize_store():
    """Initializes and caches the vector store and embedding function."""
    logger.info("Attempting to initialize vector store.")
    collection, _ = get_vector_store(
        persist_directory=VECTOR_STORE_PERSIST_DIR,
        collection_name=VECTOR_STORE_COLLECTION,
        embedding_model_name=EMBEDDING_MODEL
    )
    if collection is None:
        st.error("Failed to initialize Vector Store. Check logs and ChromaDB setup.")
        logger.error("Vector store initialization returned None.")
        return None
    logger.info("Vector store initialized successfully.")
    return collection

def get_document_list(doc_dir: str) -> List[str]:
    """Gets a list of supported documents from the specified directory."""
    supported_extensions = ['.pdf', '.docx', '.txt', '.md'] # Add more as needed by 'unstructured'
    files = []
    if os.path.isdir(doc_dir):
        for item in os.listdir(doc_dir):
            if os.path.isfile(os.path.join(doc_dir, item)) and \
               any(item.lower().endswith(ext) for ext in supported_extensions):
                files.append(os.path.join(doc_dir, item))
    return files

# --- Streamlit UI ---

st.set_page_config(page_title="Local RAG Q&A", layout="wide")
st.title("📄 Local RAG Document Q&A")
st.markdown(f"Ask questions about documents stored in the `{DOCUMENTS_DIR}` directory. Using Ollama (`{LLM_MODEL}`) locally.")

# Ensure documents directory exists
if not os.path.exists(DOCUMENTS_DIR):
    os.makedirs(DOCUMENTS_DIR)
    st.warning(f"Created documents directory: `{DOCUMENTS_DIR}`. Please add your files there.")

# --- Sidebar for Setup and Indexing ---
with st.sidebar:
    st.header("Setup & Indexing")

    # Initialize vector store (cached)
    collection = initialize_store()

    st.markdown("**1. Add Documents:**")
    st.write(f"Place your `.pdf`, `.docx`, `.txt` files into the `{DOCUMENTS_DIR}` folder.")

    st.markdown("**2. Index Documents:**")
    doc_files = get_document_list(DOCUMENTS_DIR)

    if not doc_files:
        st.info("No documents found in the `documents` directory.")
    else:
        st.write(f"Found {len(doc_files)} documents:")
        with st.expander("Show Documents", expanded=False):
            for f in doc_files:
                st.caption(os.path.basename(f))

    if st.button("Load & Index Documents", disabled=not doc_files or collection is None):
        if doc_files and collection is not None:
            try:
                with st.spinner(f"Processing {len(doc_files)} documents... This might take a while."):
                    start_time = time.time()
                    # Clear existing collection before adding new? Optional. Depends on workflow.
                    # logger.info(f"Clearing existing documents in collection '{VECTOR_STORE_COLLECTION}' before indexing.")
                    # collection.delete(ids=[]) # Deletes all, be careful! Might need enumeration if not supported

                    logger.info("Starting document processing and chunking...")
                    chunks = process_documents(
                        doc_files,
                        max_characters=CHUNK_SIZE # Only pass relevant args for chunk_by_title
                        # Remove overlap=CHUNK_OVERLAP
                    )
                    logger.info(f"Generated {len(chunks)} chunks.")

                    if chunks:
                        logger.info("Adding chunks to vector store...")
                        add_chunks_to_vector_store(collection, chunks)
                        end_time = time.time()
                        st.success(f"Successfully indexed {len(doc_files)} documents ({len(chunks)} chunks) in {end_time - start_time:.2f} seconds.")
                        logger.info("Indexing complete.")
                        st.session_state.indexed = True # Flag to indicate indexing happened
                    else:
                        st.warning("No chunks were generated from the documents. Check file content and logs.")
                        logger.warning("process_documents returned no chunks.")

            except Exception as e:
                st.error(f"Error during indexing: {e}")
                logger.exception("Exception during indexing process.")
        elif not doc_files:
            st.warning("No documents found to index.")
        else:
            st.error("Vector store not initialized. Cannot index.")

    # Display current collection count (simple way to see if indexing worked)
    if collection is not None:
        try:
            count = collection.count()
            st.metric("Chunks in Vector Store", count)
        except Exception as e:
            st.error(f"Could not get collection count: {e}")
            logger.error(f"Error getting collection count: {e}")
    else:
         st.warning("Vector store not ready.")

# --- Main Area for Q&A ---
st.header("Ask a Question")

query = st.text_input("Enter your question about the documents:", key="query_input")

# Check if indexing has happened (simple check)
has_been_indexed = collection is not None and collection.count() > 0 # or use st.session_state.indexed

if st.button("Get Answer", disabled=not query or not has_been_indexed):
    if not has_been_indexed:
        st.warning("Please index some documents first using the sidebar.")
    elif not query:
        st.warning("Please enter a question.")
    elif collection:
        with st.spinner("Searching relevant documents and generating answer..."):
            try:
                # 1. Retrieve relevant chunks
                logger.info(f"Searching vector store for query: {query}")
                retrieved_chunks = search_vector_store(collection, query, top_k=TOP_K_RESULTS)

                if not retrieved_chunks:
                    st.warning("Could not find relevant documents for your query.")
                    logger.warning("search_vector_store returned no chunks.")
                else:
                    # 2. Generate response using LLM
                    logger.info("Generating response with LLM.")
                    answer = generate_response(query, retrieved_chunks, llm_model_name=LLM_MODEL)

                    # 3. Display answer
                    st.subheader("Answer:")
                    st.markdown(answer) # Use markdown for better formatting potential

                    # 4. Display sources/context
                    with st.expander("Show Retrieved Context Chunks", expanded=False):
                        for i, chunk in enumerate(retrieved_chunks):
                            st.markdown(f"**Chunk {i+1} (Distance: {chunk['distance']:.4f})**")
                            metadata = chunk['metadata']
                            source_file = metadata.get('original_document', metadata.get('filename', 'Unknown'))
                            st.caption(f"Source: {os.path.basename(source_file)}, Page: {metadata.get('page_number', 'N/A')}")
                            st.text_area(label=f"Chunk {i+1} Text", value=chunk['text'], height=150, key=f"chunk_{i}")

            except Exception as e:
                st.error(f"An error occurred during query processing: {e}")
                logger.exception("Exception during query processing.")

elif not has_been_indexed and query:
     st.warning("The vector store is empty or not initialized. Please load and index documents using the sidebar button.")

st.markdown("---")
st.caption("Powered by Ollama, ChromaDB, SentenceTransformers, and Unstructured.")