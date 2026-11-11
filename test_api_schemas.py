
from pydantic import ValidationError

from api.schemas import RAGRequest, RAGResponse


def test_rag_request():
    request = RAGRequest(
        query="  What is deep learning?  ",
        top_k=5,
        semantic_top_k=10,
        bm25_top_k=10,
        use_reranking=True
    )

    assert request.query == "What is deep learning?"
    assert request.top_k == 5
    assert request.use_reranking is True

    print("RAG request schema test passed!")


def test_invalid_request():
    try:
        RAGRequest(query="   ")
        raise AssertionError("Empty query should be rejected.")

    except ValidationError:
        print("Invalid request validation test passed!")

    try:
        RAGRequest(
            query="What is AI?",
            top_k=0
        )
        raise AssertionError("Invalid top_k should be rejected.")

    except ValidationError:
        print("Invalid top_k validation test passed!")


def test_rag_response():
    response_data = {
        "query": "What is deep learning?",
        "answer": "Deep learning uses neural networks.",
        "citations": [],
        "insufficient_evidence": False,
        "verification": {
            "overall_status": "FULLY_VERIFIED",
            "total_claims": 0,
            "status_counts": {},
            "claim_results": []
        },
        "retrieved_chunks": []
    }

    response = RAGResponse(**response_data)

    assert response.query == "What is deep learning?"
    assert response.verification.overall_status == "FULLY_VERIFIED"

    print("RAG response schema test passed!")


if __name__ == "__main__":
    test_rag_request()
    test_invalid_request()
    test_rag_response()

    print("\nAll API schema tests passed!")
