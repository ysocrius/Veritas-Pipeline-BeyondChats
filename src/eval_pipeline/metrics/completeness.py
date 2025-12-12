from .relevance import score_relevance

def score_completeness(user_query: str, ai_response: str) -> float:
    """
    Estimates completeness.
    Ideally requires NLI or intent breakdown.
    For this assignment, we use a length/keyword heuristic combined with relevance.
    If relevance is high, we assume reasonable completeness for chat.
    """
    # Simple proxy: Relevance score is the baseline. 
    # If response is too short (< 20 chars) but query is long, penalize.
    rel = score_relevance(user_query, ai_response)
    if len(ai_response) < 20 and len(user_query) > 20:
        return rel * 0.5
    return rel
