
import json
from pathlib import Path

from retrieval.hybrid_search import hybrid_search
from evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k
)


DATASET_PATH = Path(__file__).parent / "smoke_test_dataset.json"
K = 3


def load_evaluation_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Evaluation dataset must be a non-empty list.")

    return dataset


def evaluate_configuration(dataset, use_reranking):
    metric_results = {
        "precision": [],
        "recall": [],
        "mrr": [],
        "ndcg": []
    }

    configuration_name = (
        "WITH RERANKING"
        if use_reranking
        else "WITHOUT RERANKING"
    )

    print("\n" + "=" * 60)
    print(f"HYBRID SEARCH EVALUATION: {configuration_name}")
    print("=" * 60)

    for index, item in enumerate(dataset, start=1):
        query = item["query"]
        relevant_ids = item["relevant_chunk_ids"]
        expected_answerable = item["expected_answerable"]

        print(f"\nQuery {index}: {query}")

        results = hybrid_search(
            query=query,
            top_k=K,
            use_reranking=use_reranking
        )

        retrieved_ids = [
            result["chunk_id"]
            for result in results
        ]

        print(f"Retrieved chunks: {retrieved_ids}")
        print(f"Expected answerable: {expected_answerable}")

        if not relevant_ids:
            print("Retrieval metrics: SKIPPED (no relevant chunks)")
            continue

        precision = precision_at_k(
            retrieved_ids, relevant_ids, K
        )

        recall = recall_at_k(
            retrieved_ids, relevant_ids, K
        )

        mrr = mean_reciprocal_rank(
            retrieved_ids, relevant_ids
        )

        ndcg = ndcg_at_k(
            retrieved_ids, relevant_ids, K
        )

        metric_results["precision"].append(precision)
        metric_results["recall"].append(recall)
        metric_results["mrr"].append(mrr)
        metric_results["ndcg"].append(ndcg)

        print(f"Precision@{K}: {precision:.4f}")
        print(f"Recall@{K}: {recall:.4f}")
        print(f"MRR: {mrr:.4f}")
        print(f"NDCG@{K}: {ndcg:.4f}")

    print("\nAVERAGE METRICS")
    print("-" * 40)

    for metric, values in metric_results.items():
        if values:
            average = sum(values) / len(values)
            print(f"{metric.upper()}: {average:.4f}")
        else:
            print(f"{metric.upper()}: N/A")

    print(
        f"Evaluated answerable queries: "
        f"{len(metric_results['precision'])}"
    )

    return metric_results


def evaluate_retrieval():
    dataset = load_evaluation_dataset()

    print("\nHYBRID SEARCH: RERANKING COMPARISON")
    print(f"Dataset queries: {len(dataset)}")
    print(f"Evaluation cutoff K: {K}")

    without_reranking = evaluate_configuration(
        dataset,
        use_reranking=False
    )

    with_reranking = evaluate_configuration(
        dataset,
        use_reranking=True
    )

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    print(
        f"{'Metric':<15}"
        f"{'Without Reranking':<22}"
        f"{'With Reranking':<20}"
    )

    print("-" * 57)

    for metric in without_reranking:
        without_values = without_reranking[metric]
        with_values = with_reranking[metric]

        without_avg = (
            sum(without_values) / len(without_values)
            if without_values else 0.0
        )

        with_avg = (
            sum(with_values) / len(with_values)
            if with_values else 0.0
        )

        print(
            f"{metric.upper():<15}"
            f"{without_avg:<22.4f}"
            f"{with_avg:<20.4f}"
        )


if __name__ == "__main__":
    evaluate_retrieval()
