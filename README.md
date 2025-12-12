# Veritas Pipeline - BeyondChats Eval

## Overview

This repository contains a robust, real-time evaluation pipeline for LLM-based chat systems. It evaluates AI responses against three key dimensions:
1.  **Response Relevance & Completeness**
2.  **Hallucination / Factual Accuracy (Groundedness)**
3.  **Safety & Toxicity (Guardrails)**
4.  **Latency & Costs**

## Local Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ysocrius/Veritas-Pipeline-BeyondChats.git
    cd Veritas-Pipeline-BeyondChats
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On Mac/Linux:
    # source venv/bin/activate
    
    pip install .
    # Or manually install requirements if preferred:
    # pip install pydantic sentence-transformers numpy scikit-learn transformers torch json5
    ```

3.  **Run the pipeline:**
    ```bash
    python scripts/run_eval.py --conversation "Sample Inputs/sample-chat-conversation-01.json" --context "Sample Inputs/sample_context_vectors-01.json" --output report_sample_01.json
    ```

## Architecture

The pipeline uses a modular architecture centered around a `Pipeline` class that orchestrates the flow of data through specialized evaluator components.

```
[Input: Chat JSON + Context JSON]
       |
       v
[Loader & Validator (Pydantic)]
       |
       v
[Target Selection] <--- Identifies the user query and AI response to evaluate
       |
       v
[Evaluator Engine]
    |-- Metric: Relevance (Cosine Similarity with MiniLM)
    |-- Metric: Groundedness (NLI with Cross-Encoder)
    |-- Metric: Latency & Cost (Deterministic profiling)
       |
       v
[Aggregator] <--- Compiles scores and determines pass/fail verdicts
       |
       v
[Output: JSON Report]
```

### Why this architecture?
*   **Modularity:** New metrics can be added as plugins without changing the core engine.
*   **Separation of Concerns:** Data validation is decoupled from evaluation logic.
*   **Scalability:** The stateless nature of the evaluators allows for easy parallelization.

## Design Philosophy

### Alignment with Assignment Goals
*   **Real-time Evaluation:** I implemented a high-performance pipeline prioritizing **Latency & Costs** alongside Relevance and Hallucination.
*   **Beyond Wrappers:** Instead of simply wrapping OpenAI's API (which is slow, expensive, and non-deterministic), I implemented a **Tiered Evaluation System** using specialized local models (`MiniLM` for relevance, `DeBERTa` for hallucination). This approach demonstrates a production-ready understanding of cost-effective scaling.

### Why No LangChain?
I deliberately chose **not** to use LangChain for this core pipeline, favoring a "pro-code" approach using `sentence-transformers` and `transformers` directly.
*   **Lower Latency:** Eliminates framework overhead for critical real-time paths.
*   **Zero Marginal Cost:** By running optimized local models, we avoid the per-token costs of external LLM APIs for the evaluation step itself.
*   **Control & Debuggability:** A custom `Pipeline` class provides granular control over the data flow and makes it easier to optimize specific bottlenecks without fighting framework abstractions.

## Scaling Strategy (Millions of Daily Conversations)

To ensure low latency and cost at scale:

1.  **Tier-1 vs Tier-2 Evaluation:** 
    *   **Tier-1 (100% of traffic):** Hash-based caching and lightweight regex/keyword checks.
    *   **Tier-2 (Sampled/Flagged):** Neural evaluation (this pipeline) runs on a 1-5% sample or on conversations flagged by user feedback.
2.  **Small, Specialized Models:** I use `all-MiniLM-L6-v2` (~80MB) for embeddings and `cross-encoder/nli-deberta-v3-small` for entailment instead of querying generic large LLMs (GPT-4), reducing inference cost by ~100x and latency to milliseconds.
3.  **Async/Queue-based Processing:** In production, this script would consume from a message queue (Kafka/RabbitMQ) rather than processing blocking HTTP requests, allowing for load smoothing.

## Technologies Used
*   **Python 3.9+**
*   **Pydantic:** for strict data validation.
*   **Sentence-Transformers:** for efficient semantic similarity.
*   **HuggingFace Transformers:** for NLI-based hallucination detection.
*   **json5:** for lenient JSON parsing (handles trailing commas).
*   **Pytest:** for automated testing.

## Running Tests

The project includes comprehensive tests to ensure correctness:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run specific test file
pytest tests/test_schemas.py

# Run with verbose output
pytest -v
```

## Understanding the Scores

The evaluation pipeline produces the following metrics:

### Relevance Score (0.0 - 1.0)
*   **What it measures:** Semantic similarity between user query and AI response
*   **Good score:** > 0.7 indicates the response addresses the query
*   **Low score:** < 0.5 suggests the response may be off-topic

### Completeness Score (0.0 - 1.0)
*   **What it measures:** Whether the response adequately addresses all aspects of the query
*   **Good score:** > 0.6 indicates sufficient coverage
*   **Note:** Very short responses are penalized

### Groundedness Score (0.0 - 1.0)
*   **What it measures:** How well the response is supported by the provided context
*   **Good score:** > 0.5 indicates strong factual support
*   **Low score:** < 0.3 suggests potential hallucination or unsupported claims
*   **Zero score:** No context provided or response contradicts all context

### Latency (milliseconds)
*   **What it measures:** Total evaluation time
*   **Note:** First run includes model download time (~1-2 minutes)
*   **Typical:** 100-500ms after models are cached

### Estimated Cost (USD)
*   **What it measures:** Approximate cost per evaluation
*   **Note:** Based on character count, not actual API calls
*   **My approach:** Uses local models (~$0.00001 per evaluation)

### Toxicity Score (0.0 - 1.0)
*   **What it measures:** Check for unsafe/toxic content using regex guardrails
*   **Good score:** 0.0 (Safe)
*   **Flagged:** > 0.5 (Unsafe)

## LLMOps & Reliability
To ensure production readiness as per the Job Description:
*   **CI/CD Pipeline:** Included `.github/workflows/test.yml` for automated testing on every push.
*   **Guardrails:** Integrated toxicity checks to prevent harmful outputs.


## Configuration

You can customize evaluation parameters in `src/eval_pipeline/config.py`:

```python
# Model selection
RELEVANCE_MODEL = 'all-MiniLM-L6-v2'
GROUNDEDNESS_MODEL = 'cross-encoder/nli-deberta-v3-small'

# Thresholds
RELEVANCE_THRESHOLD = 0.7
GROUNDEDNESS_THRESHOLD = 0.5

# Performance
ENABLE_CACHING = True
EMBEDDING_CACHE_SIZE = 1000
```

## Bonus: LangChain Experiments

While the main evaluation pipeline relies on optimized, "pro-code" `sentence-transformers` for maximum performance and minimum cost, I have included a demonstration of **Agentic & RAG capabilities using LangChain** in the `experiments/` directory.

To run the LangChain RAG demo:
1.  Navigate to `experiments/langchain_demo`
2.  Install requirements: `pip install -r requirements.txt`
3.  Set your OpenAI key (optional, script handles missing key gracefully):
    ```bash
    export OPENAI_API_KEY="sk-..."
    ```
4.  Run the demo:
    ```bash
    python basic_rag_demo.py
    ```
    *(This script demonstrates modern LCEL syntax, RAG architecture, and prompt engineering)*

### Metric Alternatives

For those interested in how the core metrics would look using LangChain's wrappers, I have provided an alternative implementation:
*   `src/eval_pipeline/metrics/relevance_lc.py`: Re-implements the semantic similarity metric using `langchain_community.embeddings.HuggingFaceEmbeddings`. This confirms that we can easily swap the underlying engine if needed, though we default to the lighter `sentence-transformers` for production efficiency.
