import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from embedding_functions import get_embedding_function

CHROMA_PATH = "chroma_DB"

CHROMA_PATH = "chroma_DB"

PROMPT_TEMPLATE = """
You are a factual assistant specialized in analyzing documents. 
Always answer based ONLY on the following context from our knowledge base:

{context}

Guidelines:
1. If the information is missing or unclear in the context, respond: "I don't have sufficient data to answer this precisely"
2. Quote relevant document excerpts to support your answer
3. Never invent information or use external knowledge
4. Keep answers under 5 sentences

Question: {question}

Analysis:
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="The query text.")
    args = parser.parse_args()
    query_rag(args.query)

def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = OllamaLLM(model="llama3.2")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"\n\n\n Query : {query_text}\n\n\n Response: {response_text}\n\n\nSources: {sources}"
    print(formatted_response)
    return response_text

if __name__ == "__main__":
    main()





