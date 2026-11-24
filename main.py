import json
from src.models import Document
from src.engine import Engine

def main():
    # Synthetic dataset
    corpus = [
        Document(doc_id="1", text="Company policy requires signed approvals."),
        Document(doc_id="2", text="The procedure for submittal involves a web form."),
        Document(doc_id="3", text="There is a hard limit of 5 days for review."),
        Document(doc_id="4", text="Irrelevant noise about coffee breaks.")
    ]

    engine = Engine(corpus, max_hops=3)
    
    # Run Benchmark
    query = "What is the policy procedure limit?"
    result = engine.run(query)

    # Calculate metrics
    metrics = {
        "success": result.is_sufficient,
        "hops_taken": result.hop_count,
        "raw_memory_size": len(result.raw_memory),
        "recall_met": result.is_sufficient # Simplified proxy
    }
    
    print(f"Orchestration complete. Hops: {result.hop_count}, Sufficient: {result.is_sufficient}")
    print("[LAB_RESULT_JSON]", json.dumps(metrics))

if __name__ == "__main__":
    main()