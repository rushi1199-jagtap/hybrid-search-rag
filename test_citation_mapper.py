
from generation.verification.citation_mapper import map_claim_citations


def test_citation_mapper():
    claims = [
        {
            "claim_id": "claim_001",
            "claim_text": "Deep learning uses neural networks.",
            "evidence_ids": ["Evidence 1"],
            "has_citation": True
        },
        {
            "claim_id": "claim_002",
            "claim_text": "Neural networks contain multiple layers.",
            "evidence_ids": ["Evidence 2"],
            "has_citation": True
        },
        {
            "claim_id": "claim_003",
            "claim_text": "They solve complex problems.",
            "evidence_ids": ["Evidence 3"],
            "has_citation": True
        },
        {
            "claim_id": "claim_004",
            "claim_text": "This is an uncited claim.",
            "evidence_ids": [],
            "has_citation": False
        }
    ]

    citations = [
        {
            "evidence_id": "Evidence 1",
            "chunk_id": "test_chunk_002",
            "document_id": "test_document_001",
            "source": "sample.txt",
            "page_number": None
        },
        {
            "evidence_id": "Evidence 2",
            "chunk_id": "wrong_chunk_id",
            "document_id": "test_document_001",
            "source": "sample.txt",
            "page_number": None
        }
    ]

    retrieved_chunks = [
        {
            "chunk_id": "test_chunk_002",
            "document_id": "test_document_001",
            "source": "sample.txt",
            "page_number": None,
            "chunk_text": "Deep learning uses neural networks to solve complex problems."
        },
        {
            "chunk_id": "test_chunk_001",
            "document_id": "test_document_001",
            "source": "sample.txt",
            "page_number": None,
            "chunk_text": "Machine learning enables computers to learn patterns from data."
        }
    ]

    results = map_claim_citations(
        claims=claims,
        citations=citations,
        retrieved_chunks=retrieved_chunks
    )

    assert results[0]["citation_valid"] is True
    assert results[0]["mapped_citations"][0]["evidence"]["chunk_id"] == "test_chunk_002"

    assert results[1]["citation_valid"] is False
    assert results[1]["invalid_evidence_ids"] == ["Evidence 2"]

    assert results[2]["citation_valid"] is False
    assert results[2]["invalid_evidence_ids"] == ["Evidence 3"]

    assert results[3]["citation_valid"] is False

    print("\nCitation Mapping Results:")

    for result in results:
        print(
            result["claim_id"],
            "| Citation valid:",
            result["citation_valid"],
            "| Invalid evidence:",
            result["invalid_evidence_ids"]
        )

    print("\nCitation mapper test passed!")


if __name__ == "__main__":
    test_citation_mapper()


import pytest


@pytest.mark.parametrize(
    "field,wrong_value",
    [
        ("document_id", "wrong_document_id"),
        ("source", "wrong_source.pdf"),
        ("page_number", 99)
    ]
)
def test_citation_mapper_rejects_metadata_mismatch(field, wrong_value):
    retrieved_chunk = {
        "chunk_id": "test_chunk_001",
        "document_id": "test_document_001",
        "source": "sample.txt",
        "page_number": 1,
        "chunk_text": "Machine learning enables computers to learn patterns."
    }

    citation = {
        "evidence_id": "Evidence 1",
        "chunk_id": "test_chunk_001",
        "document_id": "test_document_001",
        "source": "sample.txt",
        "page_number": 1
    }

    # Tamper with one citation metadata field.
    citation[field] = wrong_value

    results = map_claim_citations(
        claims=[
            {
                "claim_id": "claim_001",
                "claim_text": "Machine learning enables computers to learn patterns.",
                "evidence_ids": ["Evidence 1"],
                "has_citation": True
            }
        ],
        citations=[citation],
        retrieved_chunks=[retrieved_chunk]
    )

    assert results[0]["citation_valid"] is False
    assert results[0]["mapped_citations"] == []
    assert results[0]["invalid_evidence_ids"] == ["Evidence 1"]
