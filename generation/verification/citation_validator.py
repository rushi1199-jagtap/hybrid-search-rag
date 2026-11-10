
def validate_citations(mapped_claims):
    if not isinstance(mapped_claims, list):
        raise ValueError("Mapped claims must be a list.")

    validation_results = []

    for claim in mapped_claims:
        if not isinstance(claim, dict):
            raise ValueError("Every mapped claim must be a dictionary.")

        citation_valid = claim.get("citation_valid", False)
        has_citation = claim.get("has_citation", False)

        if not has_citation:
            status = "MISSING_CITATION"
        elif not citation_valid:
            status = "INVALID_CITATION"
        else:
            status = "VALID_CITATION"

        validation_results.append({
            "claim_id": claim.get("claim_id"),
            "claim_text": claim.get("claim_text"),
            "status": status,
            "citation_valid": citation_valid,
            "mapped_citations": claim.get("mapped_citations", []),
            "invalid_evidence_ids": claim.get("invalid_evidence_ids", [])
        })

    return validation_results
