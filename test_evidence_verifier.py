
from generation.verification.evidence_verifier import verify_claim


def test_evidence_verifier():
    claim = {
        "claim_id": "claim_001",
        "claim_text": "Deep learning uses neural networks.",
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
        ]
    }

    print("\nTesting Gemini evidence verification...")

    result = verify_claim(claim)

    assert isinstance(result, dict)
    assert result["claim_id"] == "claim_001"
    assert result["verdict"] in [
        "SUPPORTED",
        "CONTRADICTED",
        "INSUFFICIENT_EVIDENCE"
    ]
    assert isinstance(result["reason"], str)
    assert len(result["reason"]) > 0

    print("\nClaim:", result["claim_text"])
    print("Verdict:", result["verdict"])
    print("Reason:", result["reason"])
    print("Verified Evidence:", result["verified_evidence_ids"])

    print("\nEvidence verifier test passed!")


if __name__ == "__main__":
    test_evidence_verifier()
