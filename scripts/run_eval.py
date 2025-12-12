import argparse
import sys
import json
from pathlib import Path

# Add src to path so we can import eval_pipeline
sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval_pipeline.loader import load_data
from eval_pipeline.aggregate import run_evaluation

def main():
    parser = argparse.ArgumentParser(
        description="Run LLM Evaluation Pipeline",
        epilog="Example: python run_eval.py --conversation data/conv.json --context data/ctx.json"
    )
    parser.add_argument("--conversation", type=str, required=True, 
                       help="Path to conversation JSON file")
    parser.add_argument("--context", type=str, required=True, 
                       help="Path to context vectors JSON file")
    parser.add_argument("--output", type=str, required=False, 
                       help="Path to output JSON report (default: report.json)", 
                       default="report.json")
    
    args = parser.parse_args()
    
    # Validate input files exist
    conv_path = Path(args.conversation)
    ctx_path = Path(args.context)
    
    if not conv_path.exists():
        print(f"Error: Conversation file not found: {args.conversation}")
        sys.exit(1)
    
    if not ctx_path.exists():
        print(f"Error: Context file not found: {args.context}")
        sys.exit(1)
    
    print(f"Loading data from:\n  Conversation: {args.conversation}\n  Context: {args.context}")
    print("\nNote: First run may take 1-2 minutes to download models (~200MB)")
    
    try:
        data = load_data(args.conversation, args.context)
        print("✓ Data loaded and validated successfully.")
    except ValueError as e:
        print(f"✗ Data validation error: {e}")
        print("\nTip: Check that your JSON files match the expected format.")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error loading data: {e}")
        sys.exit(1)
        
    print("\nRunning evaluation...")
    print("  - Computing relevance (semantic similarity)...")
    print("  - Computing completeness...")
    print("  - Computing groundedness (hallucination detection)...")
    print("  - Profiling latency and cost...")
    
    try:
        report = run_evaluation(data)
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        print("\nTip: Ensure you have sufficient memory (4GB+ recommended)")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Status: {report.status}")
    
    if report.status == "success" and report.scores:
        print(f"\nQuery: {report.target_user_message[:100]}...")
        print(f"Response: {report.target_ai_response[:100]}...")
        print("\nScores:")
        print(f"  Relevance:     {report.scores.relevance:.3f} {'✓' if report.scores.relevance > 0.7 else '⚠'}")
        print(f"  Completeness:  {report.scores.completeness:.3f} {'✓' if report.scores.completeness > 0.6 else '⚠'}")
        print(f"  Groundedness:  {report.scores.groundedness:.3f} {'✓' if report.scores.groundedness > 0.5 else '⚠'}")
        print(f"  Latency:       {report.scores.latency_ms:.2f} ms")
        print(f"  Est. Cost:     ${report.scores.estimated_cost:.8f}")
    elif report.error:
        print(f"\n✗ Error: {report.error}")
    
    output_path = Path(args.output)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report.model_dump_json(indent=2))
        print(f"\n✓ Report saved to: {output_path}")
    except Exception as e:
        print(f"\n✗ Failed to save report: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
