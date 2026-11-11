
from fastapi.testclient import TestClient

from api.main import app
from api.routes import run_rag_pipeline


client = TestClient(app)


def mock_rag_pipeline(**kwargs):
    return {
        "query": kwargs["query"],
        "answer": "Deep learning uses neural networks. [Evidence 1]",
        "citations": [
            {
                "evidence_id": "Evidence 1",
                "chunk_id": "chunk_001",
                "document_id": "doc_001",
                "source": "AI_notes.pdf",
                "page_number": 2
            }
        ],
        "insufficient_evidence": False,
        "verification": {
            "overall_status": "FULLY_VERIFIED",
            "total_claims": 1,
            "status_counts": {
                "VERIFIED": 1
            },
            "claim_results": [
                {
                    "claim_id": "claim_001",
                    "claim_text": "Deep learning uses neural networks.",
                    "status": "VERIFIED",
                    "citation_valid": True,
                    "mapped_citations": [],
                    "invalid_evidence_ids": [],
                    "semantic_verification": {
                        "claim_id": "claim_001",
                        "claim_text": "Deep learning uses neural networks.",
                        "verdict": "SUPPORTED",
                        "reason": "The evidence supports the claim.",
                        "verified_evidence_ids": ["Evidence 1"]
                    },
                    "overall_status": "VERIFIED"
                }
            ]
        },
        "retrieved_chunks": [
            {
                "chunk_id": "chunk_001",
                "document_id": "doc_001",
                "chunk_text": "Deep learning uses neural networks.",
                "score": 0.90,
                "rrf_score": 0.03,
                "rerank_score": 0.95,
                "rerank_rank": 1,
                "source": "AI_notes.pdf",
                "file_type": "pdf",
                "page_number": 2,
                "chunk_index": 0
            }
        ]
    }


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Hybrid Search RAG API is running."

    print("Root endpoint test passed!")


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

    print("Health endpoint test passed!")


def test_ask_endpoint(monkeypatch):
    monkeypatch.setattr(
        "api.routes.run_rag_pipeline",
        mock_rag_pipeline
    )

    response = client.post(
        "/api/v1/ask",
        json={
            "query": "What is deep learning?",
            "top_k": 5,
            "semantic_top_k": 10,
            "bm25_top_k": 10,
            "use_reranking": True
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "What is deep learning?"
    assert data["verification"]["overall_status"] == "FULLY_VERIFIED"
    assert data["verification"]["total_claims"] == 1
    assert len(data["retrieved_chunks"]) == 1

    print("Ask endpoint test passed!")


def test_invalid_request():
    response = client.post(
        "/api/v1/ask",
        json={
            "query": "   ",
            "top_k": 5
        }
    )

    assert response.status_code == 422

    print("Invalid request test passed!")


def test_not_found_endpoint():
    response = client.get("/invalid-endpoint")

    assert response.status_code == 404

    print("404 endpoint test passed!")


if __name__ == "__main__":
    test_root_endpoint()
    test_health_endpoint()

    from pytest import MonkeyPatch

    monkeypatch = MonkeyPatch()

    try:
        test_ask_endpoint(monkeypatch)
    finally:
        monkeypatch.undo()

    test_invalid_request()
    test_not_found_endpoint()

    print("\nAll API tests passed!")
