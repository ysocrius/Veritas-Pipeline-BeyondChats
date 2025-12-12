"""
Configuration settings for the evaluation pipeline.
Centralizes all configurable parameters for easy modification.
"""

# Model Configuration
RELEVANCE_MODEL = 'all-MiniLM-L6-v2'  # ~80MB, fast semantic similarity
GROUNDEDNESS_MODEL = 'cross-encoder/nli-deberta-v3-small'  # NLI for hallucination detection

# Cache Configuration
EMBEDDING_CACHE_SIZE = 1000  # LRU cache size for embeddings
NLI_CACHE_SIZE = 5000  # Cache size for NLI predictions

# Cost Estimation (adjust based on your model/provider)
COST_PER_1K_CHARS = 0.0001  # USD per 1000 characters

# Evaluation Thresholds (for pass/fail verdicts)
RELEVANCE_THRESHOLD = 0.7  # Minimum relevance score
COMPLETENESS_THRESHOLD = 0.6  # Minimum completeness score
GROUNDEDNESS_THRESHOLD = 0.5  # Minimum groundedness score

# Performance Settings
ENABLE_CACHING = True  # Enable/disable caching for performance
LAZY_MODEL_LOADING = True  # Load models only when needed

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
