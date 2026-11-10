
def map_claim_citations(claims, citations, retrieved_chunks):
    if not isinstance(claims, list):
        raise ValueError("Claims must be a list.")

    if not isinstance(citations, list):
        raise ValueError("Citations must be a list.")

    if not isinstance(retrieved_chunks, list):
        raise ValueError("Retrieved chunks must be a list.")

    evidence_map = {
        f"Evidence {index}": chunk
        for index, chunk in enumerate(retrieved_chunks, start=1)
    }

    citation_map = {}

    for citation in citations:
        if not isinstance(citation, dict):
            raise ValueError("Every citation must be a dictionary.")

        evidence_id = citation.get("evidence_id")

        if not evidence_id:
            continue

        citation_map.setdefault(evidence_id, []).append(citation)

    mapped_claims = []

    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError("Every claim must be a dictionary.")

        evidence_ids = claim.get("evidence_ids", [])
        mapped_citations = []
        invalid_evidence_ids = []

        for evidence_id in evidence_ids:
            chunk = evidence_map.get(evidence_id)
            matching_citations = citation_map.get(evidence_id, [])

            if chunk is None or not matching_citations:
                invalid_evidence_ids.append(evidence_id)
                continue

            valid_citation = None

            for citation in matching_citations:
                metadata_matches = (
                    citation.get("chunk_id") == chunk.get("chunk_id")
                    and citation.get("document_id") == chunk.get("document_id")
                    and citation.get("source") == chunk.get("source")
                    and citation.get("page_number") == chunk.get("page_number")
                )

                if metadata_matches:
                    valid_citation = citation.copy()
                    break

            if valid_citation is None:
                invalid_evidence_ids.append(evidence_id)
            else:
                mapped_citations.append({
                    "evidence_id": evidence_id,
                    "citation": valid_citation,
                    "evidence": chunk.copy()
                })

        mapped_claims.append({
            **claim,
            "mapped_citations": mapped_citations,
            "invalid_evidence_ids": invalid_evidence_ids,
            "citation_valid": (
                bool(evidence_ids)
                and len(invalid_evidence_ids) == 0
                and len(mapped_citations) > 0
            )
        })

    return mapped_claims
