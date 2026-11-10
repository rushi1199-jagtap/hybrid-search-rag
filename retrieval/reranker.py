
import os
from dotenv import load_dotenv
from ingestion.embedding_generator import create_cohere_client

load_dotenv()

COHERE_RERANK_MODEL = os.getenv(
    "COHERE_RERANK_MODEL",
    "rerank-v3.5"
)


def rerank_results(query, results, top_n=5, client=None):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if not isinstance(results, list):
        raise ValueError("Results must be a list.")

    if not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be a positive integer.")

    if not results:
        return []

    for result in results:
        if not isinstance(result, dict) or not result.get("chunk_text"):
            raise ValueError("Every result must contain chunk_text.")

    if client is None:
        client = create_cohere_client()

    documents = [
        result["chunk_text"] for result in results
    ]

    response = client.rerank(
        model=COHERE_RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=min(top_n, len(documents))
    )

    reranked_results = []

    for rank, item in enumerate(response.results, start=1):
        original_result = results[item.index].copy()

        original_result["rerank_score"] = float(
            item.relevance_score
        )

        original_result["rerank_rank"] = rank

        reranked_results.append(original_result)

    return reranked_results
