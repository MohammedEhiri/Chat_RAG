import argparse
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_ollama import OllamaEmbeddings
from langchain.schema.document import Document
from typing import List, Dict, Any, Optional
import os
import sys
import numpy as np



CHROMA_PATH = "chroma_DB"
DATA_PATH = "documents"

PROMPT_TEMPLATE = """
You are a factual assistant specialized in analyzing documents while maintaining conversation history.

CONVERSATION HISTORY:
{history}

RETRIEVED CONTEXT:
{context}

Guidelines:
1. If information is missing from the context, respond: "I don't have sufficient data to answer this precisely"
2. Quote relevant document excerpts to support your answers
3. Never invent information or use external knowledge
4. Maintain a natural conversational tone
5. Reference the conversation history when relevant

Question: {question}

Answer:
"""

def initialize_rag():
    """Initialize or load the RAG pipeline"""
    import shutil
    
    # Always recreate the database to avoid dimension mismatch
    if os.path.exists(CHROMA_PATH):
        print("Removing existing database...")
        shutil.rmtree(CHROMA_PATH, ignore_errors=True)
    
    print("Initializing RAG system and ingesting documents...")
    pipeline = ingest_documents()
    
    # Add a check to ensure pipeline is not None
    if pipeline is None:
        raise ValueError("Failed to initialize RAG pipeline")
        
    return pipeline


# Parent-child chunk relationship tracker
class HierarchicalChunkManager:
    def __init__(self):
        self.parent_map = {}  # Maps child IDs to parent IDs
        
    def register_relationship(self, parent_id: str, child_id: str):
        self.parent_map[child_id] = parent_id
        
    def get_parent(self, child_id: str) -> Optional[str]:
        return self.parent_map.get(child_id)
    

# Enhanced text splitter with hierarchical chunks
def create_hierarchical_chunks(documents: List[Document]) -> List[Document]:
    # First create large chunks (parents)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
    )
    
    # Then create smaller chunks (children)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    
    hierarchical_manager = HierarchicalChunkManager()
    result_chunks = []
    
    for doc in documents:
        # Create parent chunks
        parent_chunks = parent_splitter.split_documents([doc])
        
        for i, parent in enumerate(parent_chunks):
            # Assign parent ID
            parent_id = f"{parent.metadata.get('source', 'unknown')}:parent:{i}"
            parent.metadata["chunk_id"] = parent_id
            parent.metadata["is_parent"] = True
            result_chunks.append(parent)
            
            # Create and process child chunks
            child_chunks = child_splitter.split_documents([parent])
            for j, child in enumerate(child_chunks):
                child_id = f"{parent_id}:child:{j}"
                child.metadata["chunk_id"] = child_id
                child.metadata["is_parent"] = False
                child.metadata["parent_id"] = parent_id
                hierarchical_manager.register_relationship(parent_id, child_id)
                result_chunks.append(child)
    
    return result_chunks


# 2. Advanced Embedding with metadata
class EnhancedEmbeddingFunction:
    def __init__(self):
        self.base_embedder = OllamaEmbeddings(model="nomic-embed-text")
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # For document embedding
        return self.base_embedder.embed_documents(texts)
        
    def embed_query(self, text: str) -> List[float]:
        # For query embedding
        return self.base_embedder.embed_query(text)
        
    def __call__(self, texts: List[str], metadata: List[Dict[str, Any]] = None) -> List[List[float]]:
        # Get base embeddings
        embeddings = self.embed_documents(texts)
        
        # If we have metadata, enhance embeddings
        if metadata:
            for i, meta in enumerate(metadata):
                if meta.get("is_parent") is False:
                    embeddings[i] = [e * 1.02 for e in embeddings[i]]
        
        return embeddings
    

# 3. Reranker for improved retrieval
class SimpleReranker:
    def __init__(self, db):
        self.db = db
        
    def rerank(self, query: str, initial_results: List[Document], k: int = 5):
        # Get parent documents for any child documents
        enhanced_results = []
        seen_parents = set()
        
        # First pass - collect child documents and their unique parents
        for doc, score in initial_results:
            enhanced_results.append((doc, score))
            
            # If this is a child document, add its parent
            if doc.metadata.get("is_parent") is False:
                parent_id = doc.metadata.get("parent_id")
                if parent_id and parent_id not in seen_parents:
                    seen_parents.add(parent_id)
                    
        # Second pass - fetch the parent documents
        for parent_id in seen_parents:
            parent_docs = self.db.get(where={"chunk_id": parent_id})
            if parent_docs["documents"]:
                # Create a Document object from the first match
                parent_doc = Document(
                    page_content=parent_docs["documents"][0],
                    metadata=parent_docs["metadatas"][0]
                )
                # Add parent with a slightly lower score to maintain original ranking
                enhanced_results.append((parent_doc, 0.95))  # Slightly lower score
                
        # Sort by score and return top k
        enhanced_results.sort(key=lambda x: x[1], reverse=True)
        return enhanced_results[:k]
    


class EnhancedRAGPipeline:
    def __init__(self, db_path="chroma"):
        self.embedding_function = EnhancedEmbeddingFunction()
        self.db = Chroma(
            persist_directory=db_path, 
            embedding_function=self.embedding_function,
            collection_metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 80, "hnsw:M": 8}  # Optimized storage
        )
        self.reranker = SimpleReranker(self.db)
        
    def query(self, query_text: str, k: int = 5):
        # 1. Small-to-big retrieval pattern
        # First get smaller, more specific chunks
        small_results = self.db.similarity_search_with_score(
            query_text, 
            k=k*2,  # Get more results initially
            filter={"is_parent": False}  # Only child chunks
        )
        
        # 2. Apply reranking to improve results
        reranked_results = self.reranker.rerank(query_text, small_results, k=k)
        
        # 3. Prepare context from reranked results
        context_parts = []
        sources = []
        
        for doc, score in reranked_results:
            # Add metadata annotation to identify document type
            prefix = "[PARENT CONTEXT] " if doc.metadata.get("is_parent") else "[SPECIFIC DETAIL] "
            context_parts.append(f"{prefix}{doc.page_content}")
            sources.append(doc.metadata.get("chunk_id", "unknown"))
            
        context = "\n\n---\n\n".join(context_parts)
        
        return context, sources
    

def ingest_documents(data_path=DATA_PATH, db_path=CHROMA_PATH):
    """Ingest documents with hierarchical chunking"""
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    
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

def query_rag(query_text, pipeline, history=""):
    """Query the RAG system with conversation history"""
    # Get context from the advanced RAG pipeline
    context, sources = pipeline.query(query_text)
    
    # Format prompt with context and history
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(
        context=context, 
        question=query_text,
        history=history
    )
    
    # Get response from LLM
    model = OllamaLLM(model="llama3.2")
    response_text = model.invoke(prompt)
    
    return response_text, sources

def handle_conversation_with_rag():
    """Main conversation loop with RAG integration"""
    # Initialize the RAG pipeline first
    pipeline = initialize_rag()
    
    history = ""
    print("Welcome to the RAG-enabled ChatBot! Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            break
            
        # Query RAG system with user input and history
        result, sources = query_rag(user_input, pipeline, history)
        
        # Print response with source information
        print(f"Bot: {result}")
        print(f"Sources: {', '.join(sources[:3])}...\n")
        
        # Update conversation history
        history += f"\nUser: {user_input}\nAI: {result}"

def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Single query mode")
    args = parser.parse_args()
    
    if args.query:
        # Single query mode
        pipeline = initialize_rag()
        response, sources = query_rag(args.query, pipeline)
        print(f"\nQuery: {args.query}\n")
        print(f"Response: {response}\n")
        print(f"Sources: {sources}")
    else:
        # Interactive conversation mode
        handle_conversation_with_rag()

if __name__ == "__main__":
    handle_conversation_with_rag()