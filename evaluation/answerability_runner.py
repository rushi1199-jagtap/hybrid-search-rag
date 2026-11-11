
import json
from pathlib import Path

from generation.answer_pipeline import run_rag_pipeline


DATASET_PATH = Path(__file__).parent / "smoke_test_dataset.json"

TOP_K = 3


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Evaluation dataset must be a non-empty list.")

    return dataset


def evaluate_answerability():
    dataset = load_dataset()

    # Confusion matrix counters
    tp = 0  # Answerable and system answered
    tn = 0  # Unanswerable and system abstained
    fp = 0  # Unanswerable but system answered
    fn = 0  # Answerable but system abstained

    print("\nANSWERABILITY AND ABSTENTION EVALUATION")
    print("=" * 65)

    for index, item in enumerate(dataset, start=1):
        query = item["query"]
        expected_answerable = item["expected_answerable"]

        print(f"\nQuery {index}: {query}")
        print("-" * 65)

        # Execute the actual RAG pipeline
        response = run_rag_pipeline(
            query=query,
            top_k=TOP_K,
            use_reranking=True
        )

        predicted_answerable = not response["insufficient_evidence"]

        # Update confusion matrix
        if expected_answerable and predicted_answerable:
            tp += 1
            result = "CORRECTLY ANSWERED"

        elif not expected_answerable and not predicted_answerable:
            tn += 1
            result = "CORRECTLY ABSTAINED"

        elif not expected_answerable and predicted_answerable:
            fp += 1
            result = "FALSE ANSWER"

        else:
            fn += 1
            result = "MISSED ANSWER"

        print(f"Expected answerable : {expected_answerable}")
        print(f"Predicted answerable: {predicted_answerable}")
        print(f"Result              : {result}")
        print(f"Verification status : {response['verification']['overall_status']}")
        print(f"Retrieved chunks    : {len(response['retrieved_chunks'])}")

        print("\nGenerated response:")
        print(response["answer"])

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0.0

    answer_precision = tp / (tp + fp) if (tp + fp) else 0.0
    answer_recall = tp / (tp + fn) if (tp + fn) else 0.0

    abstention_precision = tn / (tn + fn) if (tn + fn) else 0.0
    abstention_recall = tn / (tn + fp) if (tn + fp) else 0.0

    print("\n\nFINAL EVALUATION RESULTS")
    print("=" * 65)

    print("\nConfusion Matrix:")
    print(f"True Positives  (TP): {tp}")
    print(f"True Negatives  (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")

    print("\nAnswerability Metrics:")
    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Answer Precision  : {answer_precision:.4f}")
    print(f"Answer Recall     : {answer_recall:.4f}")

    print("\nAbstention Metrics:")
    print(f"Abstention Precision: {abstention_precision:.4f}")
    print(f"Abstention Recall   : {abstention_recall:.4f}")

    print("\n" + "=" * 65)
    print("Evaluation completed.")
    print("Note: These are smoke-test results, not benchmark results.")


if __name__ == "__main__":
    evaluate_answerability()
