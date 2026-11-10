
from retrieval.vector_search import semantic_search


def test_semantic_search():

    query = "How does deep learning work?"

    print(f"Search query: {query}\n")

    results = semantic_search(
        query=query,
        top_k=2
    )

    assert len(results) > 0

    print("Search results:\n")

    for index, result in enumerate(results, start=1):

        assert result["chunk_id"]
        assert isinstance(result["score"], (int, float))
        assert result["chunk_text"]

        print(f"Result {index}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['chunk_text']}")
        print(f"Source: {result['source']}")
        print("-" * 50)

    print("\nSemantic search test passed!")


if __name__ == "__main__":
    test_semantic_search()
