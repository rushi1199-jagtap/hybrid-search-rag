
from generation.verification.citation_validator import validate_citations
from generation.verification.evidence_verifier import verify_claim
from generation.gemini_client import create_gemini_client


def verify_claims(mapped_claims, client=None):
    if not isinstance(mapped_claims, list):
        raise ValueError("Mapped claims must be a list.")

    citation_results = validate_citations(mapped_claims)

    valid_claims = [
        result for result in citation_results
        if result["status"] == "VALID_CITATION"
    ]

    owns_client = client is None
    if owns_client and valid_claims:
        client = create_gemini_client()

    try:
        verification_results = []

        for result in citation_results:
            if result["status"] != "VALID_CITATION":
                verification_results.append({
                    **result,
                    "semantic_verification": None,
                    "overall_status": result["status"]
                })
                continue

            semantic_result = verify_claim(
                claim=result,
                client=client
            )

            verdict = semantic_result["verdict"]

            if verdict == "SUPPORTED":
                overall_status = "VERIFIED"
            elif verdict == "CONTRADICTED":
                overall_status = "CONTRADICTED"
            else:
                overall_status = "INSUFFICIENT_EVIDENCE"

            verification_results.append({
                **result,
                "semantic_verification": semantic_result,
                "overall_status": overall_status
            })

        return verification_results

    finally:
        if owns_client and client is not None:
            client.close()
