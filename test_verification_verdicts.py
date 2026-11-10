
import json
from types import SimpleNamespace

from generation.verification.verification_engine import verify_claims


class FakeInteractions:
    def __init__(self):
        self.responses = [
            {
                "verdict": "SUPPORTED",
                "reason": "The evidence directly supports the claim."
            },
            {
                "verdict": "CONTRADICTED",
                "reason": "The evidence directly conflicts with the claim."
            },
            {
                "verdict": "INSUFFICIENT_EVIDENCE",
                "reason": "The evidence does not establish the claim."
            }
        ]

    def create(self, **kwargs):
        response = self.responses.pop(0)
        return SimpleNamespace(output_text=json.dumps(response))


class FakeClient:
    def __init__(self):
        self.interactions = FakeInteractions()


def test_verification_verdicts():
    claims = []

    for index in range(1, 4):
        claims.append({
            "claim_id": f"claim_{index:03d}",
            "claim_text": f"Test claim number {index}.",
            "status": "VALID_CITATION",
            "has_citation": True,
            "citation_valid": True,
            "mapped_citations": [
                {
                    "evidence_id": "Evidence 1",
                    "evidence": {
                        "chunk_id": "test_chunk_001",
                        "document_id": "test_document_001",
                        "source": "sample.txt",
                        "page_number": 1,
                        "chunk_text": "Sample evidence for verification."
                    }
                }
            ]
        })

    client = FakeClient()

    results = verify_claims(
        mapped_claims=claims,
        client=client
    )

    expected_verdicts = [
        "SUPPORTED",
        "CONTRADICTED",
        "INSUFFICIENT_EVIDENCE"
    ]

    expected_statuses = [
        "VERIFIED",
        "CONTRADICTED",
        "INSUFFICIENT_EVIDENCE"
    ]

    for result, expected_verdict, expected_status in zip(
        results,
        expected_verdicts,
        expected_statuses
    ):
        assert result["semantic_verification"]["verdict"] == expected_verdict
        assert result["overall_status"] == expected_status

        print(
            result["claim_id"],
            "|",
            expected_verdict,
            "|",
            result["overall_status"]
        )

    print("\nAll three semantic verification verdicts passed!")


if __name__ == "__main__":
    test_verification_verdicts()
