from sentence_transformers import SentenceTransformer, util
import numpy as np
from functools import lru_cache
import hashlib

# Load model once (global or singleton pattern preferable in prod)
# using a lightweight model for speed/cpu-friendliness
MODEL_NAME = 'all-MiniLM-L6-v2'
_model = None

def get_model():
    """Lazy-load the model to save memory when not needed."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

@lru_cache(maxsize=1000)
def _cached_encode(text: str):
    """
    Cache embeddings for repeated texts to improve performance at scale.
    Uses LRU cache to limit memory usage.
    """
    model = get_model()
    return model.encode(text, convert_to_tensor=True)

def score_relevance(user_query: str, ai_response: str) -> float:
    """
    Computes semantic similarity between query and response.
    Returns a score between 0.0 and 1.0.
    
    Uses caching to avoid re-computing embeddings for repeated queries/responses,
    which is critical for handling millions of daily conversations efficiently.
    """
    # Use cached encoding for better performance
    query_embedding = _cached_encode(user_query)
    response_embedding = _cached_encode(ai_response)
    
    cosine_score = util.cos_sim(query_embedding, response_embedding)
    return float(cosine_score[0][0])


