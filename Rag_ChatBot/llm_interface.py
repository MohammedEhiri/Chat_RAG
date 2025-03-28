import ollama
import logging
# ---- ADD THIS LINE ----
from typing import List, Dict, Optional, Any 
# -----------------------

logger = logging.getLogger(__name__)

# --- VVV Line 7 is here - uses List, Dict, Any
def format_context(context_chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved chunks into a string for the LLM prompt."""
    context_str = ""
    for i, chunk in enumerate(context_chunks):
        # Safely get metadata, providing defaults
        metadata = chunk.get('metadata', {}) 
        source = metadata.get('original_document', metadata.get('source', 'Unknown Source'))
        page = metadata.get('page_number', None)
        
        source_info = f"Source: {source}" + (f", Page: {page}" if page else "")
        context_str += f"--- Context Chunk {i+1} ---\n"
        context_str += f"({source_info})\n"
        context_str += chunk.get('text', '') # Safely get text
        context_str += "\n-------------------------\n\n"
    return context_str

# --- VVV uses List, Dict, Any, Optional
def generate_response(
    query: str,
    context_chunks: List[Dict[str, Any]],
    llm_model_name: str = "mistral", 
    ollama_host: Optional[str] = None 
) -> str:
    """
    Generates a response from the local LLM using the retrieved context.
    """
    if not context_chunks:
        return "I couldn't find any relevant information in the documents to answer your question."

    formatted_context = format_context(context_chunks)

    # Strict Prompting - Instructing the LLM to use only the provided context
    prompt = f"""You are a helpful assistant tasked with answering questions based *only* on the provided context.
Your answer must be grounded in the information found in the text snippets below.
Do not use any external knowledge or information you might have.
If the answer cannot be found in the provided context, state clearly: "I cannot answer this question based on the provided documents."

CONTEXT:
{formatted_context}

QUESTION:
{query}

ANSWER (based *only* on the context provided above):
"""

    logger.info(f"Generating response using model: {llm_model_name}")
    # Consider logging less verbose prompt info normally, maybe just first/last lines or length
    # logger.debug(f"Prompt being sent to LLM (first 500 chars):\n{prompt[:500]}...") 

    try:
        # Initialize client inside the function for stateless operation unless shared state is managed
        client_args = {}
        if ollama_host:
            client_args['host'] = ollama_host
        
        # Use context manager for client if available, or ensure proper resource handling
        client = ollama.Client(**client_args)

        # Check if model exists locally? Optional but good for user feedback
        # Use client.list() maybe once at startup if feasible
        
        response = client.chat(
            model=llm_model_name,
            messages=[{'role': 'user', 'content': prompt}],
            # options={'temperature': 0.1} # Example: Lower temperature for more factual answers
        )

        # Make sure the response structure is as expected
        if 'message' in response and 'content' in response['message']:
            answer = response['message']['content']
            logger.info("LLM generation successful.")
            return answer.strip()
        else:
            logger.error(f"Unexpected response structure from Ollama: {response}")
            return "Error: Received an unexpected response structure from the language model."


    except ollama.ResponseError as e:
        logger.error(f"Ollama API error: {e.status_code} - {e.error}", exc_info=True) # Log traceback
        if e.status_code == 404:
             # Check if it's a model not found error vs. endpoint not found
             if "model" in e.error.lower() and f"'{llm_model_name}'" in e.error.lower():
                 return f"Error: The Ollama model '{llm_model_name}' was not found locally. Please ensure it is pulled (`ollama pull {llm_model_name}`) and Ollama is running."
             else:
                 return f"Error: Ollama API endpoint not found or unreachable ({e.status_code}). Is Ollama running?"
        # Handle other potential common errors like timeout, connection error if possible
        return f"Error generating response from Ollama: {e.error} (Status code: {e.status_code})"
        
    except Exception as e:
        # Catch potential network errors (requests.exceptions.ConnectionError) etc.
        logger.error(f"An unexpected error occurred while interfacing with Ollama: {e}", exc_info=True) # Log traceback
        return f"Error generating response: An unexpected error occurred. Check logs for details. ({type(e).__name__})"

# --- Example Usage Block remains the same ---
if __name__ == "__main__":
    # Make sure Ollama server is running and the model 'mistral' (or your default) is pulled.
    print("Testing LLM interface...")
    dummy_context = [
        {"text": "The company's Q3 revenue was $10 million, driven by strong sales in the North American market.", "metadata": {"source": "report.pdf", "page_number": 5}},
        {"text": "Future projections estimate a 15% growth in Q4, assuming market conditions remain stable.", "metadata": {"source": "report.pdf", "page_number": 6}},
    ]
    test_query = "What was the Q3 revenue and what are the projections for Q4?"

    # Assuming 'mistral' model is running on default localhost
    # Use ollama_host='http://your_ollama_ip:11434' if needed
    response = generate_response(test_query, dummy_context, llm_model_name="mistral")

    print("\n--- Test ---")
    print(f"Query: {test_query}")
    print(f"\nGenerated Response:\n{response}")
    print("------------")

    # Test case for info not in context
    test_query_no_context = "What is the CEO's name?"
    response_no_context = generate_response(test_query_no_context, dummy_context, llm_model_name="mistral")
    print(f"\nQuery: {test_query_no_context}")
    print(f"\nGenerated Response (No Context):\n{response_no_context}")
    print("------------")

     # Test case for no context chunks found
    test_query_empty_context = "Any question?"
    response_empty_context = generate_response(test_query_empty_context, [], llm_model_name="mistral")
    print(f"\nQuery: {test_query_empty_context}")
    print(f"\nGenerated Response (Empty Context):\n{response_empty_context}")
    print("------------")
    
    # Test connection error (assuming ollama isn't running on port 11435)
    print("\nTesting connection error:")
    try:
        response_conn_error = generate_response(test_query, dummy_context, llm_model_name="mistral", ollama_host="http://localhost:11435")
        print(f"\nGenerated Response (Connection Error Test):\n{response_conn_error}")
    except Exception as e:
        print(f"Caught expected exception: {e}")
    print("------------")

    # Test model not found error
    print("\nTesting model not found error:")
    non_existent_model = "non_existent_model_12345"
    response_model_error = generate_response(test_query, dummy_context, llm_model_name=non_existent_model)
    print(f"\nGenerated Response (Model Not Found Test):\n{response_model_error}")
    print("------------")