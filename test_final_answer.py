
from generation.verification.final_answer import build_final_answer


def test_final_answer():
    generated_response = {
        "answer": (
            "Deep learning uses neural networks [Evidence 1]. "
            "This claim has no citation."
        ),
        "citations": [
            {
                "evidence_id": "Evidence 1",
                "chunk_id": "test_chunk_002",
                "document_id": "test_document_001",
                "source": "sample.txt",
                "page_number": None
            }
        ],
        "insufficient_evidence": False
    }

    verification_results = [
        {
            "claim_id": "claim_001",
            "claim_text": "Deep learning uses neural networks.",
            "overall_status": "VERIFIED"
        },
        {
            "claim_id": "claim_002",
            "claim_text": "This claim has no citation.",
            "overall_status": "MISSING_CITATION"
        }
    ]

    result = build_final_answer(
        generated_response,
        verification_results
    )

    assert result["answer"] == generated_response["answer"]
    assert result["citations"] == generated_response["citations"]

    assert result["verification"]["overall_status"] == "PARTIALLY_VERIFIED"
    assert result["verification"]["total_claims"] == 2

    assert result["verification"]["status_counts"]["VERIFIED"] == 1
    assert result["verification"]["status_counts"]["MISSING_CITATION"] == 1

    print("\nFinal Answer:")
    print(result["answer"])

    print("\nVerification Summary:")
    print(result["verification"]["overall_status"])

    print("\nStatus Counts:")

    for status, count in result["verification"]["status_counts"].items():
        print(status, ":", count)

    print("\nFinal answer builder test passed!")


if __name__ == "__main__":
    test_final_answer()
