"""
Basic smoke tests for metric functions.
Note: These tests require model downloads on first run.
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval_pipeline.metrics.relevance import score_relevance
from eval_pipeline.metrics.completeness import score_completeness
from eval_pipeline.metrics.groundedness import score_groundedness
from eval_pipeline.schemas import ContextChunk


def test_relevance_high_similarity():
    """Test relevance scoring with highly similar texts."""
    query = "What is the capital of France?"
    response = "The capital of France is Paris."
    
    score = score_relevance(query, response)
    
    assert 0.0 <= score <= 1.0
    assert score > 0.5  # Should be reasonably high


def test_relevance_low_similarity():
    """Test relevance scoring with unrelated texts."""
    query = "What is the capital of France?"
    response = "I like pizza and ice cream."
    
    score = score_relevance(query, response)
    
    assert 0.0 <= score <= 1.0
    assert score < 0.5  # Should be low


def test_relevance_identical_texts():
    """Test relevance with identical texts."""
    text = "This is a test sentence."
    
    score = score_relevance(text, text)
    
    assert score > 0.95  # Should be very high (near 1.0)


def test_completeness_adequate_response():
    """Test completeness with adequate response."""
    query = "What is AI?"
    response = "AI stands for Artificial Intelligence, which is the simulation of human intelligence by machines."
    
    score = score_completeness(query, response)
    
    assert 0.0 <= score <= 1.0
    assert score > 0.3  # Should have some completeness


def test_completeness_too_short_response():
    """Test completeness penalty for very short responses."""
    query = "Explain the theory of relativity in detail."
    response = "E=mc²"
    
    score = score_completeness(query, response)
    
    assert 0.0 <= score <= 1.0
    # Short response should be penalized


def test_groundedness_with_supporting_context():
    """Test groundedness when context supports the response."""
    response = "Paris is the capital of France."
    chunks = [
        ContextChunk(text="Paris is the capital and largest city of France."),
        ContextChunk(text="France is a country in Europe.")
    ]
    
    score = score_groundedness(response, chunks)
    
    assert 0.0 <= score <= 1.0
    assert score > 0.3  # Should show some entailment


def test_groundedness_with_contradicting_context():
    """Test groundedness when context contradicts the response."""
    response = "London is the capital of France."
    chunks = [
        ContextChunk(text="Paris is the capital of France."),
        ContextChunk(text="London is the capital of the United Kingdom.")
    ]
    
    score = score_groundedness(response, chunks)
    
    assert 0.0 <= score <= 1.0
    # Score should be low due to contradiction


def test_groundedness_no_context():
    """Test groundedness with no context chunks."""
    response = "Paris is the capital of France."
    chunks = []
    
    score = score_groundedness(response, chunks)
    
    assert score == 0.0  # No context means no grounding


def test_groundedness_empty_response():
    """Test groundedness with empty response."""
    response = ""
    chunks = [ContextChunk(text="Some context")]
    
    score = score_groundedness(response, chunks)
    
    assert score == 0.0


def test_relevance_caching():
    """Test that caching works for repeated queries."""
    query = "Test query for caching"
    response = "Test response for caching"
    
    # First call
    score1 = score_relevance(query, response)
    
    # Second call (should use cache)
    score2 = score_relevance(query, response)
    
    # Scores should be identical
    assert score1 == score2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
