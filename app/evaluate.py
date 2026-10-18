import json
import math
import time
from pathlib import Path

from .config import get_settings
from .pipeline import RAGPipeline

def calc_mrr(retrieved_ids, relevant_ids):
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0

def calc_ndcg(retrieved_ids, relevant_ids, k=5):
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in relevant_ids:
            return 1.0 / math.log2(i + 2) # i+2 because rank is 1-indexed
    return 0.0

def main():
    settings = get_settings()
    pipeline = RAGPipeline(settings)
    
    dataset_path = Path("data/eval_dataset.json")
    if not dataset_path.exists():
        print(f"Error: {dataset_path} not found.")
        return
        
    questions = json.loads(dataset_path.read_text())
    
    # Pre-compute relevance mapping
    all_chunks = pipeline.store._collection.get(include=["documents"])
    chunk_map = dict(zip(all_chunks["ids"], all_chunks["documents"]))
    
    results = []
    latencies = []
    
    print(f"Running evaluation over {len(questions)} questions...")
    
    for q in questions:
        start = time.perf_counter()
        res = pipeline.ask(q["question"], top_k=5)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed * 1000)
        
        retrieved_ids = res["retrieved_chunk_ids"]
        
        # Determine relevant ids based on gold_snippet
        relevant_ids = set()
        if not q["expected_abstain"]:
            for cid, text in chunk_map.items():
                if q["gold_snippet"] in text:
                    relevant_ids.add(cid)
                    
        # Calculate Retrieval Metrics
        hit = 1.0 if any(rid in relevant_ids for rid in retrieved_ids) else 0.0
        mrr = calc_mrr(retrieved_ids, relevant_ids)
        ndcg = calc_ndcg(retrieved_ids, relevant_ids)
            
        # Calculate Answer Metrics
        abstain_correct = (res["abstained"] == q["expected_abstain"])
        
        results.append({
            "id": q["id"],
            "expected_abstain": q["expected_abstain"],
            "hit": hit,
            "mrr": mrr,
            "ndcg": ndcg,
            "abstain_correct": abstain_correct,
            "latency_ms": round(elapsed * 1000, 2),
            "tokens": res["usage"]["total_tokens"]
        })
        
        # Cleaner terminal output
        status = "Abstained (Correct)" if res["abstained"] and q["expected_abstain"] else "Answered"
        if res["abstained"] and not q["expected_abstain"]:
            status = "Abstained (FALSE NEGATIVE)"
            
        print(f"[{q['id']}] {status} | Hit: {hit:.0f} | MRR: {mrr:.2f} | NDCG: {ndcg:.2f}")
        
    # Aggregate - Only compute Retrieval metrics on answerable questions
    answerable_results = [r for r in results if not r["expected_abstain"]]
    n_answerable = len(answerable_results)
    
    hit_rate = sum(r["hit"] for r in answerable_results) / n_answerable if n_answerable else 0
    mrr = sum(r["mrr"] for r in answerable_results) / n_answerable if n_answerable else 0
    ndcg = sum(r["ndcg"] for r in answerable_results) / n_answerable if n_answerable else 0
    
    abstain_acc = sum(r["abstain_correct"] for r in results) / len(results)
    
    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.50)]
    p95 = latencies[int(len(latencies) * 0.95)]
    total_tokens = sum(r["tokens"] for r in results)
    
    # Output Report
    report = f"""# Evaluation Report

## Retrieval Metrics (Answerable Questions Only)
- **Hit Rate@5**: {hit_rate:.3f}
- **MRR@5**: {mrr:.3f}
- **nDCG@5**: {ndcg:.3f}

## Answer Metrics (All Questions)
- **Abstention Accuracy**: {abstain_acc:.3f} 

## Performance
- **p50 Latency**: {p50:.1f} ms
- **p95 Latency**: {p95:.1f} ms
- **Total Tokens Used**: {total_tokens}
"""
    
    print("\n" + report)
    Path("eval_report.md").write_text(report)
    print("Report saved to eval_report.md")

if __name__ == "__main__":
    main()