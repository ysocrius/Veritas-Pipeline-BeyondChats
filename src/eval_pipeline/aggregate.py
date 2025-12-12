from pydantic import BaseModel
from typing import Optional
from .schemas import EvalInput
from .targeting import select_target_pair
from .metrics.relevance import score_relevance
from .metrics.completeness import score_completeness
from .metrics.groundedness import score_groundedness
from .profiling import LatencyProfiler, estimate_cost

class MetricScores(BaseModel):
    relevance: float
    completeness: float
    groundedness: float
    latency_ms: float
    estimated_cost: float

class EvalReport(BaseModel):
    status: str
    target_user_message: str
    target_ai_response: str
    scores: Optional[MetricScores] = None
    error: Optional[str] = None

def run_evaluation(data: EvalInput) -> EvalReport:
    """
    Orchestrates the evaluation pipeline.
    """
    profiler = LatencyProfiler()
    profiler.start()
    
    # 1. Select Target Pair
    target = select_target_pair(data.conversation, data.context)
    if not target:
        profiler.stop()
        return EvalReport(
            status="skipped",
            target_user_message="",
            target_ai_response="",
            error="Could not identify a valid User-AI pair with Context."
        )
        
    user_msg, ai_msg, context_key = target
    
    # 2. Get Context
    context_chunks = data.context.entries.get(context_key, [])
    
    # 3. Compute Metrics
    try:
        rel = score_relevance(user_msg.content, ai_msg.content)
        comp = score_completeness(user_msg.content, ai_msg.content)
        ground = score_groundedness(ai_msg.content, context_chunks)
        cost = estimate_cost(ai_msg.content) # Cost of response generation (proxy)
        
    except Exception as e:
        profiler.stop()
        return EvalReport(
            status="failed",
            target_user_message=user_msg.content,
            target_ai_response=ai_msg.content,
            error=str(e)
        )

    profiler.stop()
    
    scores = MetricScores(
        relevance=rel,
        completeness=comp,
        groundedness=ground,
        latency_ms=profiler.get_latency_ms(),
        estimated_cost=cost
    )
    
    return EvalReport(
        status="success",
        target_user_message=user_msg.content,
        target_ai_response=ai_msg.content,
        scores=scores
    )
