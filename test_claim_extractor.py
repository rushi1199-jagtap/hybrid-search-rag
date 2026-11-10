
from generation.verification.claim_extractor import extract_claims


def test_claim_extractor():
    answer = (
        "Deep learning uses neural networks [Evidence 1]. "
        "Neural networks contain multiple layers [Evidence 2]. "
        "They are used to solve complex problems."
    )

    claims = extract_claims(answer)

    assert len(claims) == 3

    assert claims[0]["claim_id"] == "claim_001"
    assert claims[0]["evidence_ids"] == ["Evidence 1"]
    assert claims[0]["has_citation"] is True

    assert claims[1]["evidence_ids"] == ["Evidence 2"]
    assert claims[1]["has_citation"] is True

    assert claims[2]["evidence_ids"] == []
    assert claims[2]["has_citation"] is False

    print("\nExtracted Claims:")

    for claim in claims:
        print(claim)

    print("\nClaim extractor test passed!")


if __name__ == "__main__":
    test_claim_extractor()
