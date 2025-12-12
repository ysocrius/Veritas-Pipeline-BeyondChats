from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.utils.math import cosine_similarity
import numpy as np
from functools import lru_cache

# Load model once
# Using the same model as the main pipeline for fair comparison
MODEL_NAME = 'all-MiniLM-L6-v2'
_lc_embeddings = None

def get_embeddings_model():
    """Lazy-load the LangChain wrapper."""
    global _lc_embeddings
    if _lc_embeddings is None:
        _lc_embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    return _lc_embeddings

@lru_cache(maxsize=1000)
def _cached_encode_lc(text: str):
    """
    Cache embeddings using LangChain's interface.
    """
    model = get_embeddings_model()
    # embed_query is the LangChain equivalent for single text encoding
    return model.embed_query(text)

def score_relevance_lc(user_query: str, ai_response: str) -> float:
    """
    Computes semantic similarity using LangChain components.
    
    This is an alternative implementation to relevance.py to demonstrate
    proficiency with the LangChain ecosystem.
    """
    # Get embeddings (list of floats)
    query_vec = _cached_encode_lc(user_query)
    response_vec = _cached_encode_lc(ai_response)
    
    # LangChain's cosine_similarity expectation: list of lists
    # It returns a similarity matrix
    similarity_matrix = cosine_similarity([query_vec], [response_vec])
    
    # Return limits [0][0]
    return float(similarity_matrix[0][0])
