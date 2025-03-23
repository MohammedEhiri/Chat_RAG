# Advanced RAG Pipeline Structure

# 1. Enhanced Data Ingestion
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

import numpy as np
from typing import List, Dict, Any, Optional
import os

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

# 4. Main RAG Query Pipeline
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

# 5. LLM Integration with Advanced Prompt
class RAGQueryEngine:
    def __init__(self, rag_pipeline, model_name="llama3"):
        self.rag_pipeline = rag_pipeline
        self.model = OllamaLLM(model="llama3.2")

        
    def query(self, query_text: str):
        context, sources = self.rag_pipeline.query(query_text)
        
        prompt = f"""You are an AI assistant that answers questions based on provided context.

CONTEXT:
{context}

RULES:
- Answer ONLY based on the context above
- If the context doesn't contain enough information, say "I don't have sufficient information"
- Consider both [PARENT CONTEXT] sections for broader understanding and [SPECIFIC DETAIL] sections for precise answers
- Cite specific parts of the context in your answer
- Structure complex answers with clear sections
- Do not use prior knowledge outside the provided context

QUESTION: {query_text}

ANSWER:
"""
        response = self.model.invoke(prompt)
        return response, sources