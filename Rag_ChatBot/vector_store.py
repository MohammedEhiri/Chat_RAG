import chromadb
from chromadb.utils import embedding_functions
import logging
# ---- ADD THIS LINE ----
from typing import List, Dict, Optional, Tuple, Any 
# -----------------------
import os # Make sure os is imported if you use it later (e.g., in main block)
import shutil # Make sure shutil is imported if you use it later (e.g., in main block)


logger = logging.getLogger(__name__)

# Global client and collection - keep as is for now
_client = None
_collection = None
_embedding_function = None

# --- Line 18 is below --- VVV
def get_vector_store(
    persist_directory: str = "./data",
    collection_name: str = "rag_documents",
    embedding_model_name: str = "all-MiniLM-L6-v2"
) -> Tuple[Optional[chromadb.Collection], Optional[Any]]: # <--- This line needs Tuple, Optional, Any
    """
    Initializes and returns a ChromaDB collection and the embedding function.
    Uses a persistent client.
    """
    global _client, _collection, _embedding_function # No changes needed here
    try:
        # The if condition `if _collection is None:` needs fixing as globals are not reliable across runs/modules easily
        # Better approach: Try to get the client/collection, if fails, create it.
        
        logger.info(f"Initializing ChromaDB client persistence directory: {persist_directory}")
        client = chromadb.PersistentClient(path=persist_directory) # Moved client initialization here

        logger.info(f"Loading embedding model: {embedding_model_name}")
        # Re-create embedding function instance - safer than relying on global state
        emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )

        logger.info(f"Getting or creating collection: {collection_name}")
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=emb_func, # Use the newly created function
            metadata={"hnsw:space": "cosine"} 
        )
        logger.info(f"Vector store initialized. Collection '{collection_name}' has {collection.count()} documents.")
        # Return the local variables instead of globals
        return collection, emb_func 
        
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}", exc_info=True) # Log traceback
        return None, None

# --- REST OF THE FILE (add_chunks_to_vector_store, search_vector_store, if __name__ == "__main__": ...) remains the same ---
# Make sure 'os' and 'shutil' are imported if you keep the __main__ block test code

def add_chunks_to_vector_store(collection: chromadb.Collection, chunks: List[Dict[str, Any]], batch_size: int = 100):
    """
    Adds processed chunks to the ChromaDB collection in batches.
    Expects chunks in the format [{'text': str, 'metadata': dict}, ...]
    Metadata must be flat dictionary with str/int/float values.
    """
    if not chunks:
        logger.warning("No chunks provided to add to the vector store.")
        return

    texts = [chunk['text'] for chunk in chunks]
     # Clean metadata for ChromaDB: ensure values are str, int, or float
    metadatas = []
    for chunk in chunks:
        cleaned_meta = {}
        for key, value in chunk['metadata'].items():
            if isinstance(value, (str, int, float, bool)): # Bool might be ok too
                 cleaned_meta[key] = value
            # Add specific handling if needed, e.g., convert lists/dicts to strings
            # elif isinstance(value, list):
            #     cleaned_meta[key] = ", ".join(map(str, value)) 
            else:
                cleaned_meta[key] = str(value) # Convert other types to string as a fallback
        metadatas.append(cleaned_meta)
        
    # Generate unique IDs for each chunk, crucial for updating/deleting
    ids = [chunk['metadata'].get('chunk_id', f"{chunk['metadata'].get('original_document','doc')}_{i}") for i, chunk in enumerate(chunks)]
    
    # Replace None IDs with placeholders if necessary
    ids = [id if id is not None else f"generated_id_{i}" for i, id in enumerate(ids)]


    logger.info(f"Adding {len(texts)} chunks to collection '{collection.name}'...")

    try:
        total_chunks = len(texts)
        added_count = 0
        failed_batches = 0
        for i in range(0, total_chunks, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            
            # Ensure no duplicate IDs within the batch itself (ChromaDB checks across collection)
            if len(batch_ids) != len(set(batch_ids)):
                logger.warning(f"Duplicate IDs detected within batch starting at index {i}. Skipping batch.")
                # Implement more granular handling here if needed (e.g. remove duplicates)
                failed_batches += 1
                continue # Skip this batch or handle duplicates

            logger.info(f"Adding batch {i // batch_size + 1} with {len(batch_ids)} chunks.")
            try:
                collection.add(
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                added_count += len(batch_ids)
            except Exception as batch_e:
                 logger.error(f"Error adding batch {i // batch_size + 1}: {batch_e}", exc_info=True)
                 failed_batches += 1


        if failed_batches == 0:
             logger.info(f"Successfully added {added_count} chunks to the collection.")
        else:
             logger.warning(f"Added {added_count} chunks, but {failed_batches} batches failed. Check logs.")
             
        logger.info(f"Collection '{collection.name}' now has {collection.count()} documents.")
    except Exception as e:
        logger.error(f"General error during document adding process: {e}", exc_info=True)


def search_vector_store(collection: chromadb.Collection, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Searches the vector store for chunks relevant to the query.
    Returns a list of retrieved chunks with metadata and distance.
    """
    if not query:
        logger.warning("Search query is empty.")
        return []
    try:
        logger.info(f"Performing vector search for query: '{query[:50]}...'")
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances'] # Request content, metadata, and distance
        )
        logger.info(f"Retrieved {len(results.get('ids', [[]])[0])} results.")

        # Format results nicely
        retrieved_chunks = []
        # Check if results dictionary and the inner lists are not None or empty
        if results and results.get('ids') and results['ids'][0]:
            ids = results['ids'][0]
            distances = results['distances'][0]
            metadatas = results['metadatas'][0]
            documents = results['documents'][0]
            
            # Ensure all lists have the same length
            if not (len(ids) == len(distances) == len(metadatas) == len(documents)):
                logger.error("Mismatch in lengths of results lists from ChromaDB query.")
                return [] # Return empty or handle partial results carefully

            for i, doc_id in enumerate(ids):
                retrieved_chunks.append({
                    "id": doc_id,
                    "text": documents[i],
                    "metadata": metadatas[i] if metadatas[i] is not None else {}, # Handle potential None metadata
                    "distance": distances[i]
                })
        return retrieved_chunks

    except Exception as e:
        logger.error(f"Error searching vector store: {e}", exc_info=True) # Log traceback
        return []

# Example Usage (for testing this module standalone)
if __name__ == "__main__":
    TEST_PERSIST_DIR = "./temp_chroma_data"
    TEST_COLLECTION = "test_collection"
    # Clean up previous test run if necessary
    
    if os.path.exists(TEST_PERSIST_DIR):
         logger.info(f"Removing existing test data directory: {TEST_PERSIST_DIR}")
         shutil.rmtree(TEST_PERSIST_DIR)

    collection, emb_fn = get_vector_store(persist_directory=TEST_PERSIST_DIR, collection_name=TEST_COLLECTION)

    if collection and emb_fn:
        print("Vector store initialized successfully.")
        print(f"Collection count: {collection.count()}")

        # Add dummy data
        dummy_chunks = [
            {"text": "The quick brown fox jumps over the lazy dog.", "metadata": {"source": "doc1", "chunk_id": "doc1-0"}},
            {"text": "Weather forecast for tomorrow is sunny.", "metadata": {"source": "doc2", "chunk_id": "doc2-0"}},
            {"text": "Local economies are showing signs of recovery.", "metadata": {"source": "doc1", "chunk_id": "doc1-1"}},
        ]
        add_chunks_to_vector_store(collection, dummy_chunks)
        print(f"Collection count after adding: {collection.count()}")

        # Search
        query = "What is the weather like?"
        results = search_vector_store(collection, query, top_k=2)

        print(f"\nSearch results for query: '{query}'")
        if results:
            for res in results:
                print(f"  Distance: {res['distance']:.4f}")
                print(f"  Text: {res['text']}")
                print(f"  Metadata: {res['metadata']}")
                print("-" * 10)
        else:
            print("No results found.")

        # Test adding data again (should update or handle duplicates based on ID)
        print("\nAdding slightly different data with same IDs:")
        more_dummy_chunks = [
            {"text": "The FAST brown fox jumps over the lazy dog.", "metadata": {"source": "doc1", "chunk_id": "doc1-0"}}, # Same ID doc1-0
            {"text": "An updated weather forecast: cloudy evening.", "metadata": {"source": "doc2", "chunk_id": "doc2-0"}}, # Same ID doc2-0
            {"text": "Stock markets reacted positively.", "metadata": {"source": "doc3", "chunk_id": "doc3-0"}}, # New ID
        ]
        add_chunks_to_vector_store(collection, more_dummy_chunks)
        print(f"Collection count after second add: {collection.count()}") # Should be 3 if upsert works

        # Search again
        query = "Tell me about the fox"
        results = search_vector_store(collection, query, top_k=2)
        print(f"\nSearch results for query: '{query}'")
        if results:
            for res in results:
                print(f"  Distance: {res['distance']:.4f}")
                print(f"  Text: {res['text']}") # Should show the updated "FAST" fox text if upsert worked
                print(f"  Metadata: {res['metadata']}")
                print("-" * 10)
        else:
            print("No results found.")


        # Clean up test data directory
        if os.path.exists(TEST_PERSIST_DIR):
             logger.info(f"Cleaning up test data directory: {TEST_PERSIST_DIR}")
             # Optional: Add error handling around rmtree
             try:
                 # Give a slight delay or ensure handles are closed if needed, especially on Windows
                 # time.sleep(0.1) 
                 shutil.rmtree(TEST_PERSIST_DIR)
             except OSError as e:
                 logger.error(f"Error removing directory {TEST_PERSIST_DIR}: {e}")
    else:
        print("Failed to initialize vector store for testing.")