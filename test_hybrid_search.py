
from retrieval.hybrid_search import hybrid_search


def test_hybrid_search():
    query = "How does deep learning work?"

    print("\nRunning hybrid search...")
    print(f"Query: {query}\n")

    results = hybrid_search(
        query=query,
        top_k=5,
        semantic_top_k=2,
        bm25_top_k=2
    )

    assert len(results) > 0

    for rank, result in enumerate(results, start=1):
        print(f"Rank: {rank}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"RRF Score: {result['rrf_score']:.6f}")
        print(f"Content: {result['chunk_text']}\n")

    print("Hybrid search test passed!")


if __name__ == "__main__":
    test_hybrid_search()
