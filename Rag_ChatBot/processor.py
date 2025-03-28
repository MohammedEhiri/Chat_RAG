import os
import logging
from typing import List, Dict, Any
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from unstructured.cleaners.core import clean

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Consider RecursiveCharacterTextSplitter if chunk_by_title isn't sufficient
# from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_elements_from_document(filepath: str) -> List[Any]:
    """
    Extracts elements (text, titles, images, tables etc.) from a document using unstructured.
    """
    try:
        logger.info(f"Processing document: {filepath}")
        # strategy="hi_res" potentially uses models for better layout analysis (slower)
        # Use model_name="yolox" or other detection models if needed for hi_res
        elements = partition(filename=filepath, strategy="fast") # Start with fast, maybe try "auto" or "hi_res" later
        return elements
    except Exception as e:
        logger.error(f"Error processing document {filepath}: {e}")
        return []

def chunk_elements(elements: List[Any], **chunking_kwargs) -> List[Dict[str, Any]]:
    """
    Chunks elements using unstructured's chunk_by_title or other strategies.
    Also includes basic cleaning. Adds metadata.
    """
    try:
        logger.info(f"Applying chunking with arguments: {chunking_kwargs}") # Log kwargs for debugging
        # Default kwargs can be defined here if needed, but will be overwritten by chunking_kwargs
        default_chunk_args = {
            "combine_text_under_n_chars": 200,
            "new_after_n_chars": 1800, 
            # Default max_characters if not provided via kwargs (optional)
             "max_characters": 2000 
        }
        # Merge defaults with passed kwargs, kwargs take precedence
        effective_chunk_args = {**default_chunk_args, **chunking_kwargs}

        # Explicitly remove 'overlap' if it exists, as chunk_by_title doesn't use it
        effective_chunk_args.pop('overlap', None)
        
        # REMOVE explicit max_characters from here - let it come from kwargs
        chunks = chunk_by_title(
            elements,
            **effective_chunk_args # Pass merged arguments
        )
        
        # --- The rest of the function remains the same ---
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            #...(existing code for cleaning and metadata)...
             text = clean(chunk.text, extra_whitespace=True)
             if not text.strip(): # Skip empty chunks early
                continue
                
             metadata = chunk.metadata.to_dict()
             # Add filename from metadata if exists, fallback safely
             filename = metadata.get('filename', 'unknown_file')
             metadata['chunk_id'] = f"{os.path.basename(filename)}-chunk-{i}"
             metadata['text_length'] = len(text)

             # Placeholder for advanced multimodal remains
             
             processed_chunks.append({
                "text": text,
                "metadata": metadata
             })

        logger.info(f"Generated {len(processed_chunks)} chunks from {len(elements)} elements.")
        return processed_chunks

    except Exception as e:
        logger.error(f"Error chunking elements: {e}", exc_info=True) # Added traceback logging
        return []

def process_documents(doc_paths: List[str], **chunking_kwargs) -> List[Dict[str, Any]]:
    """
    Loads, processes, and chunks a list of documents.
    """
    all_chunks = []
    for doc_path in doc_paths:
        if not os.path.exists(doc_path):
            logger.warning(f"Document not found: {doc_path}, skipping.")
            continue
        elements = extract_elements_from_document(doc_path)
        if elements:
            chunks = chunk_elements(elements, **chunking_kwargs)
            # Add original document path to metadata of each chunk from this doc
            for chunk in chunks:
                chunk['metadata']['original_document'] = doc_path
            all_chunks.extend(chunks)
    logger.info(f"Total chunks processed from all documents: {len(all_chunks)}")
    return all_chunks

# Example Usage (for testing this module standalone)
if __name__ == "__main__":
    # Create dummy files for testing
    os.makedirs("temp_docs", exist_ok=True)
    with open("temp_docs/test1.txt", "w") as f:
        f.write("This is the first section.\n\nIt contains some text.\n\nThis is the second section.\n\nMore text follows here.")
    # You would need a PDF library installed and a test PDF for this part
    # try:
    #     from reportlab.pdfgen import canvas
    #     from reportlab.lib.pagesizes import letter
    #     c = canvas.Canvas("temp_docs/test2.pdf", pagesize=letter)
    #     c.drawString(100, 750, "Page 1 Title")
    #     c.drawString(100, 700, "Some text on the first page.")
    #     c.showPage()
    #     c.drawString(100, 750, "Page 2 Title")
    #     c.drawString(100, 700, "Content for the second page.")
    #     c.save()
    #     logger.info("Created dummy PDF test2.pdf")
    # except ImportError:
    #     logger.warning("reportlab not installed, skipping PDF dummy creation.")
    #     # Create another txt as placeholder if pdf fails
    #     with open("temp_docs/test2.txt", "w") as f:
    #       f.write("Title on Page 1.\n\nContent page 1.\n\nTitle on Page 2.\n\nContent page 2.")


    doc_directory = "temp_docs"
    document_paths = [os.path.join(doc_directory, f) for f in os.listdir(doc_directory) if os.path.isfile(os.path.join(doc_directory, f))]

    if document_paths:
        chunks = process_documents(document_paths)
        for i, chunk in enumerate(chunks[:5]): # Print first 5 chunks
            print(f"--- Chunk {i+1} ---")
            print(f"Text: {chunk['text'][:100]}...") # Print start of text
            print(f"Metadata: {chunk['metadata']}")
            print("-" * 20)
    else:
        print("No documents found in temp_docs directory for testing.")

    # Clean up dummy files
    # import shutil
    # shutil.rmtree("temp_docs")