
def build_final_answer(generated_response, verification_results):
    if not isinstance(generated_response, dict):
        raise ValueError("Generated response must be a dictionary.")

    if not isinstance(verification_results, list):
        raise ValueError("Verification results must be a list.")

    required_fields = ["answer", "citations", "insufficient_evidence"]

    for field in required_fields:
        if field not in generated_response:
            raise ValueError(f"Missing generated response field: {field}")

    status_counts = {
        "VERIFIED": 0,
        "CONTRADICTED": 0,
        "INSUFFICIENT_EVIDENCE": 0,
        "INVALID_CITATION": 0,
        "MISSING_CITATION": 0
    }

    for result in verification_results:
        status = result.get("overall_status")

        if status not in status_counts:
            raise ValueError(f"Unknown verification status: {status}")

        status_counts[status] += 1

    total_claims = len(verification_results)

    if total_claims == 0:
        overall_status = "NO_CLAIMS"
    elif status_counts["CONTRADICTED"] > 0:
        overall_status = "CONTAINS_CONTRADICTIONS"
    elif (
        status_counts["INVALID_CITATION"] > 0
        or status_counts["MISSING_CITATION"] > 0
        or status_counts["INSUFFICIENT_EVIDENCE"] > 0
    ):
        overall_status = "PARTIALLY_VERIFIED"
    elif status_counts["VERIFIED"] == total_claims:
        overall_status = "FULLY_VERIFIED"
    else:
        overall_status = "NOT_FULLY_VERIFIED"

    return {
        "answer": generated_response["answer"],
        "citations": generated_response["citations"],
        "insufficient_evidence": generated_response["insufficient_evidence"],
        "verification": {
            "overall_status": overall_status,
            "total_claims": total_claims,
            "status_counts": status_counts,
            "claim_results": verification_results
        }
    }
