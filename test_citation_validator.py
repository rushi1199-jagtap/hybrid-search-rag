
from generation.verification.citation_validator import validate_citations


def test_citation_validator():
    mapped_claims = [
        {
            "claim_id": "claim_001",
            "claim_text": "Valid citation claim.",
            "has_citation": True,
            "citation_valid": True,
            "mapped_citations": [{"evidence_id": "Evidence 1"}],
            "invalid_evidence_ids": []
        },
        {
            "claim_id": "claim_002",
            "claim_text": "Invalid citation claim.",
            "has_citation": True,
            "citation_valid": False,
            "mapped_citations": [],
            "invalid_evidence_ids": ["Evidence 2"]
        },
        {
            "claim_id": "claim_003",
            "claim_text": "Uncited claim.",
            "has_citation": False,
            "citation_valid": False,
            "mapped_citations": [],
            "invalid_evidence_ids": []
        }
    ]

    results = validate_citations(mapped_claims)

    assert results[0]["status"] == "VALID_CITATION"
    assert results[1]["status"] == "INVALID_CITATION"
    assert results[2]["status"] == "MISSING_CITATION"

    print("\nCitation Validation Results:")

    for result in results:
        print(
            result["claim_id"],
            "|",
            result["status"]
        )

    print("\nCitation validator test passed!")


if __name__ == "__main__":
    test_citation_validator()
