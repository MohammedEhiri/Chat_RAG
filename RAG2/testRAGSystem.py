import unittest
from rag_system import create_hierarchical_chunks, EnhancedRAGPipeline
from langchain.schema.document import Document

class TestRAGSystem(unittest.TestCase):
    
    def setUp(self):
        # Créer des documents de test
        self.test_docs = [
            Document(page_content="Ceci est un document de test sur l'intelligence artificielle.", 
                    metadata={"source": "test1.txt"}),
            Document(page_content="Les systèmes RAG combinent la récupération d'information avec la génération.", 
                    metadata={"source": "test2.txt"})
        ]
        
        # Initialiser le pipeline RAG
        self.pipeline = EnhancedRAGPipeline(db_path="test_chroma_db")
        
    def test_hierarchical_chunking(self):
        # Tester la création des chunks hiérarchiques
        chunks = create_hierarchical_chunks(self.test_docs)
        
        # Vérifier que nous avons des chunks parents et enfants
        self.assertTrue(any(chunk.metadata.get("is_parent") for chunk in chunks))
        self.assertTrue(any(not chunk.metadata.get("is_parent") for chunk in chunks))
        
    def test_document_retrieval(self):
        # Créer les chunks et les ajouter à la base de données
        chunks = create_hierarchical_chunks(self.test_docs)
        
        texts = [doc.page_content for doc in chunks]
        metadatas = [doc.metadata for doc in chunks]
        ids = [doc.metadata.get("chunk_id") for doc in chunks]
        
        self.pipeline.db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        
        # Tester la récupération
        query = "Qu'est-ce qu'un système RAG?"
        context, sources = self.pipeline.query(query)
        
        # Vérifier que le contexte contient des informations pertinentes
        self.assertIn("RAG", context)
        
    def test_query_relevance(self):
        # Tester des questions spécifiques
        test_queries = [
            "Expliquez l'intelligence artificielle",
            "Comment fonctionne la récupération d'information"
        ]
        
        chunks = create_hierarchical_chunks(self.test_docs)
        
        texts = [doc.page_content for doc in chunks]
        metadatas = [doc.metadata for doc in chunks]
        ids = [doc.metadata.get("chunk_id") for doc in chunks]
        
        self.pipeline.db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        
        for query in test_queries:
            context, _ = self.pipeline.query(query)
            # Vérifier que nous obtenons un résultat non vide
            self.assertTrue(len(context) > 0)
            
    def tearDown(self):
        # Nettoyer après les tests
        import shutil, os
        if os.path.exists("test_chroma_db"):
            shutil.rmtree("test_chroma_db")

if __name__ == "__main__":
    unittest.main()