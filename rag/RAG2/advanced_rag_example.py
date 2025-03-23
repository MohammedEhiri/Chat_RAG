# Example usage of enhanced RAG system

from advanced_rag import EnhancedRAGPipeline, RAGQueryEngine, create_hierarchical_chunks
from langchain_community.document_loaders import PyPDFDirectoryLoader


CHROMA_PATH = "chroma_DB"
DATA_PATH = "documents"

# Step 1: Ingest and process documents
def ingest_documents(data_path=DATA_PATH, db_path=CHROMA_PATH):
    # Load documents
    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()
    
    # Create hierarchical chunks with parent-child relationships
    hierarchical_chunks = create_hierarchical_chunks(documents)
    
    # Initialize RAG pipeline and store documents
    pipeline = EnhancedRAGPipeline(db_path=db_path)
    
    # Check for and remove duplicates
    chunk_ids = [doc.metadata["chunk_id"] for doc in hierarchical_chunks]
    unique_indices = []
    seen_ids = set()
    
    for i, chunk_id in enumerate(chunk_ids):
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            unique_indices.append(i)
    
    unique_chunks = [hierarchical_chunks[i] for i in unique_indices]
    unique_ids = [doc.metadata["chunk_id"] for doc in unique_chunks]
    
    # Add documents with their chunk_ids as document IDs
    pipeline.db.add_documents(
        documents=unique_chunks,
        ids=unique_ids
    )
    
    print(f"Ingested {len(unique_chunks)} chunks (removed {len(hierarchical_chunks) - len(unique_chunks)} duplicates)")
    return pipeline

# Step 2: Query the system
def query_system(query, pipeline=None):
    if pipeline is None:
        pipeline = EnhancedRAGPipeline()
    
    engine = RAGQueryEngine(pipeline)
    response, sources = engine.query(query)
    
    print(f"Query: {query}")
    print(f"Response: {response}")
    print(f"Sources: {sources}")
    
    return response, sources

if __name__ == "__main__":
    import sys
    
    # First ingest documents
    pipeline = ingest_documents()
    
    # If argument is provided, use it as query
    if len(sys.argv) > 1:
        query = sys.argv[1]
        query_system(query, pipeline)
    else:
        query_system("c'est quoi hallucination ?", pipeline)