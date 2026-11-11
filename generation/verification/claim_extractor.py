
import re


EVIDENCE_PATTERN = re.compile(r"\[Evidence\s+\d+\]")


def extract_claims(answer):
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Answer must be a non-empty string.")

    answer = answer.strip()

    answer = re.sub(
        r"([.!?])\s*((?:\[Evidence\s+\d+\]\s*)+)",
        lambda match: (
            " "
            + match.group(2).strip()
            + match.group(1)
            + " "
        ),
        answer
    )

    sentences = re.split(r"(?<=[.!?])\s+", answer)

    claims = []

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        evidence_ids = EVIDENCE_PATTERN.findall(sentence)

        evidence_ids = list(dict.fromkeys(
            evidence_id[1:-1] for evidence_id in evidence_ids
        ))

        claim_text = EVIDENCE_PATTERN.sub("", sentence)
        claim_text = re.sub(r"\s+([.,!?;:])", r"\1", claim_text)
        claim_text = re.sub(r"\s+", " ", claim_text).strip()

        if not claim_text:
            continue

        claims.append({
            "claim_id": f"claim_{len(claims) + 1:03d}",
            "claim_text": claim_text,
            "evidence_ids": evidence_ids,
            "has_citation": bool(evidence_ids)
        })

    return claims
