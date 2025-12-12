from sentence_transformers import CrossEncoder
from typing import List
from functools import lru_cache
import hashlib
from ..schemas import ContextChunk

# Load model once
MODEL_NAME = 'cross-encoder/nli-deberta-v3-small'
_model = None

def get_model():
    """Lazy-load the NLI model to save memory when not needed."""
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model

def _hash_text_pair(text1: str, text2: str) -> str:
    """Create a hash key for caching NLI predictions."""
    combined = f"{text1}||{text2}"
    return hashlib.md5(combined.encode()).hexdigest()

# Cache for NLI predictions to avoid recomputing for same context-response pairs
_nli_cache = {}

def score_groundedness(ai_response: str, context_chunks: List[ContextChunk]) -> float:
    """
    Checks if the AI response is supported by the provided context chunks.
    Uses an NLI model to predict entailment.
    
    Implements caching to avoid recomputing NLI for same context-response pairs,
    which significantly improves performance at scale.
    """
    if not context_chunks:
        return 0.0 # No context implies potential hallucination
    
    if not ai_response or not ai_response.strip():
        return 0.0 # Empty response

    model = get_model()
    
    # We want to check if ANY chunk supports the response.
    # Approach: Pair the response with each chunk as (Context, Response).
    # Predict: Entailment, Neutral, Contradiction.
    # If Entailment score is high for at least one chunk, we consider it grounded.
    
    max_entailment = 0.0
    
    for chunk in context_chunks:
        # Check cache first
        cache_key = _hash_text_pair(chunk.text, ai_response)
        
        if cache_key in _nli_cache:
            entailment_prob = _nli_cache[cache_key]
        else:
            # Compute NLI score
            # CrossEncoder output for nli-deberta-v3-small:
            # [contradiction_score, entailment_score, neutral_score]
            pair = [(chunk.text, ai_response)]
            scores = model.predict(pair, apply_softmax=True)
            entailment_prob = float(scores[0][1])  # Index 1 is Entailment
            
            # Cache the result
            _nli_cache[cache_key] = entailment_prob
        
        if entailment_prob > max_entailment:
            max_entailment = entailment_prob
            
    return float(max_entailment)
