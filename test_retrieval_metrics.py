
from evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k
)


retrieved_ids = [
    "chunk_002",
    "chunk_003",
    "chunk_001"
]

relevant_ids = [
    "chunk_001",
    "chunk_002"
]

print("Testing Retrieval Metrics...\n")

precision = precision_at_k(retrieved_ids, relevant_ids, 3)
recall = recall_at_k(retrieved_ids, relevant_ids, 3)
mrr = mean_reciprocal_rank(retrieved_ids, relevant_ids)
ndcg = ndcg_at_k(retrieved_ids, relevant_ids, 3)

print(f"Precision@3: {precision:.4f}")
print(f"Recall@3: {recall:.4f}")
print(f"MRR: {mrr:.4f}")
print(f"NDCG@3: {ndcg:.4f}")

assert abs(precision - 0.6667) < 0.001
assert recall == 1.0
assert abs(mrr - 1.0) < 0.001
assert abs(ndcg - 0.9197) < 0.001

print("\nAll retrieval metric tests passed!")
