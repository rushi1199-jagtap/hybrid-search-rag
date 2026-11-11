
import math


def precision_at_k(retrieved_ids, relevant_ids, k):
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    retrieved = retrieved_ids[:k]

    if not retrieved:
        return 0.0

    relevant = set(relevant_ids)

    hits = sum(
        1 for chunk_id in retrieved
        if chunk_id in relevant
    )

    return hits / k


def recall_at_k(retrieved_ids, relevant_ids, k):
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    retrieved = set(retrieved_ids[:k])

    hits = len(retrieved.intersection(relevant))

    return hits / len(relevant)


def mean_reciprocal_rank(retrieved_ids, relevant_ids):
    relevant = set(relevant_ids)

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant:
            return 1 / rank

    return 0.0


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    dcg = 0.0

    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant:
            dcg += 1 / math.log2(rank + 1)

    ideal_hits = min(len(relevant), k)

    idcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg if idcg else 0.0
