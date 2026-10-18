import json

def calculate_costs():
    # --- Assumptions ---
    # 384 dimensions * 4 bytes (float32) = 1536 bytes/vector
    # + ~400 bytes metadata (text, filename, etc) = ~2KB per vector
    bytes_per_vector = 2048 
    
    # Managed DB pricing (e.g., Pinecone/Qdrant Cloud average)
    # ~$25/mo base + $0.10 per GB/mo
    managed_base_cost = 25.0
    managed_gb_cost = 0.10
    
    # Groq API pricing for openai/gpt-oss-120b (approximate)
    # Input: ~$0.50 / 1M tokens; Output: ~$0.80 / 1M tokens
    # Average query in our eval used ~700 tokens (mostly input)
    groq_cost_per_1k_queries = 0.40 
    
    scales = [100_000, 1_000_000, 10_000_000]
    
    print(f"{'Vectors':>10} | {'Size (GB)':>10} | {'Managed DB/mo':>15} | {'Self-Hosted/mo':>15} | {'LLM Cost (1M queries)':>20}")
    print("-" * 80)
    
    for n in scales:
        size_gb = (n * bytes_per_vector) / (1024**3)
        
        # Managed cost: base + storage + 20% overhead for replication
        managed_cost = managed_base_cost + (size_gb * managed_gb_cost * 1.2)
        
        # Self-hosted cost: Just a single VM. 
        # (Small VM for 100K, larger 32GB RAM VM for 10M for the HNSW index)
        if n <= 1_000_000:
            self_hosted_cost = 40.0 # 8GB RAM VM
        else:
            self_hosted_cost = 240.0 # 32GB RAM VM
            
        # LLM cost
        llm_cost = (groq_cost_per_1k_queries / 1000) * 1_000_000 # 1M queries
        
        print(f"{n:>10,} | {size_gb:>10.2f} | ${managed_cost:>14.2f} | ${self_hosted_cost:>14.2f} | ${llm_cost:>19.2f}")

if __name__ == "__main__":
    calculate_costs()