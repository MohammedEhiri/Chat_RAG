from langchain_ollama import OllamaEmbeddings


def get_embedding_function():
    embeddings = OllamaEmbeddings(model="tazarov/all-minilm-l6-v2-f32")
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")

    return embeddings