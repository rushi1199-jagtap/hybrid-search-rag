
from types import SimpleNamespace

from generation.verification.verification_engine import verify_claims


class MockInteractions:
    def create(self, **kwargs):
        return SimpleNamespace(
            output_text=(
                '{"verdict": "SUPPORTED", '
                '"reason": "The supplied evidence directly supports the claim."}'
            )
        )


class MockGeminiClient:
    def __init__(self):
        self.interactions = MockInteractions()
        self.closed = False

    def close(self):
        self.closed = True


def test_verification_engine():
    mapped_claims = [
        {
            "claim_id": "claim_001",
            "claim_text": "Deep learning uses neural networks.",
            "has_citation": True,
            "citation_valid": True,
            "status": "VALID_CITATION",
            "mapped_citations": [
                {
                    "evidence_id": "Evidence 1",
                    "citation": {
                        "chunk_id": "test_chunk_002",
                        "document_id": "test_document_001",
                        "source": "sample.txt",
                        "page_number": None
                    },
                    "evidence": {
                        "chunk_id": "test_chunk_002",
                        "document_id": "test_document_001",
                        "source": "sample.txt",
                        "page_number": None,
                        "chunk_text": (
                            "Deep learning uses neural networks "
                            "to solve complex problems."
                        )
                    }
                }
            ],
            "invalid_evidence_ids": []
        },
        {
            "claim_id": "claim_002",
            "claim_text": "This claim has an invalid citation.",
            "has_citation": True,
            "citation_valid": False,
            "status": "INVALID_CITATION",
            "mapped_citations": [],
            "invalid_evidence_ids": ["Evidence 2"]
        },
        {
            "claim_id": "claim_003",
            "claim_text": "This claim has no citation.",
            "has_citation": False,
            "citation_valid": False,
            "status": "MISSING_CITATION",
            "mapped_citations": [],
            "invalid_evidence_ids": []
        }
    ]

    mock_client = MockGeminiClient()

    print("\nTesting verification engine...")

    results = verify_claims(
        mapped_claims=mapped_claims,
        client=mock_client
    )

    assert len(results) == 3

    assert results[0]["overall_status"] == "VERIFIED"
    assert results[0]["semantic_verification"]["verdict"] == "SUPPORTED"

    assert results[1]["overall_status"] == "INVALID_CITATION"
    assert results[1]["semantic_verification"] is None

    assert results[2]["overall_status"] == "MISSING_CITATION"
    assert results[2]["semantic_verification"] is None

    assert mock_client.closed is False

    print("\nVerification Results:")

    for result in results:
        print(
            result["claim_id"],
            "|",
            result["overall_status"]
        )

    print("\nVerification engine test passed!")


if __name__ == "__main__":
    test_verification_engine()
