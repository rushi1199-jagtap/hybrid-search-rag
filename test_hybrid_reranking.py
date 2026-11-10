
from retrieval.hybrid_search import hybrid_search


def test_hybrid_reranking():
    query = "How does deep learning work?"

    print("\nTesting Hybrid Search with Reranking...")

    results = hybrid_search(
        query=query,
        top_k=2,
        semantic_top_k=2,
        bm25_top_k=2,
        use_reranking=True
    )

    assert len(results) > 0

    for result in results:
        assert "rrf_score" in result
        assert "rerank_score" in result
        assert "rerank_rank" in result

        print(f"\nRank: {result['rerank_rank']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"RRF Score: {result['rrf_score']:.6f}")
        print(f"Rerank Score: {result['rerank_score']:.6f}")
        print(f"Content: {result['chunk_text']}")

    assert results[0]["chunk_id"] == "test_chunk_002"

    print("\nHybrid Search with Reranking test passed!")


if __name__ == "__main__":
    test_hybrid_reranking()
