
from retrieval.reranker import rerank_results


def test_reranker():
    query = "How does deep learning work?"

    sample_results = [
        {
            "chunk_id": "test_chunk_001",
            "chunk_text": "Machine learning enables computers to learn patterns from data.",
            "document_id": "test_document_001",
            "source": "sample.txt"
        },
        {
            "chunk_id": "test_chunk_002",
            "chunk_text": "Deep learning uses neural networks to solve complex problems.",
            "document_id": "test_document_001",
            "source": "sample.txt"
        }
    ]

    print("\nTesting Cohere Reranking...")
    print(f"Query: {query}\n")

    reranked_results = rerank_results(
        query=query,
        results=sample_results,
        top_n=2
    )

    assert len(reranked_results) == 2

    for result in reranked_results:
        assert "rerank_score" in result
        assert "rerank_rank" in result
        assert isinstance(result["rerank_score"], float)

        print(f"Rank: {result['rerank_rank']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Rerank Score: {result['rerank_score']:.6f}")
        print(f"Content: {result['chunk_text']}\n")

    assert reranked_results[0]["chunk_id"] == "test_chunk_002"

    print("Cohere Reranking test passed!")


if __name__ == "__main__":
    test_reranker()
