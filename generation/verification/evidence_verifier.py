
import json
from dotenv import load_dotenv

from generation.gemini_client import create_gemini_client


load_dotenv()


VERIFICATION_SYSTEM_PROMPT = """
You are a strict evidence verification system.

Your task is to evaluate whether the supplied document evidence supports
or contradicts a factual claim.

Use ONLY the supplied evidence.

Return exactly one of these verdicts:

1. SUPPORTED:
The evidence directly supports the claim.

2. CONTRADICTED:
The evidence directly conflicts with the claim.

3. INSUFFICIENT_EVIDENCE:
The evidence does not provide enough information to establish
or contradict the claim.

Rules:
- Do not use outside knowledge.
- Do not assume facts that are not explicitly present.
- Do not treat related information as proof.
- If evidence is ambiguous, choose INSUFFICIENT_EVIDENCE.
- If evidence contains both supporting and contradicting information,
  choose INSUFFICIENT_EVIDENCE and explain the conflict.
- Explain the reasoning using the supplied evidence.
- Do not invent quotations.

Return valid JSON with exactly these fields:

{
    "verdict": "SUPPORTED",
    "reason": "Explanation based only on the supplied evidence."
}

Do not wrap the JSON in Markdown code fences.
"""


def verify_claim(claim, client=None):
    if not isinstance(claim, dict):
        raise ValueError("Claim must be a dictionary.")

    claim_text = claim.get("claim_text")

    if not isinstance(claim_text, str) or not claim_text.strip():
        raise ValueError("Claim text must be a non-empty string.")

    if claim.get("status") != "VALID_CITATION":
        raise ValueError(
            "Only claims with valid citations can be semantically verified."
        )

    mapped_citations = claim.get("mapped_citations", [])

    if not mapped_citations:
        raise ValueError("No mapped evidence is available for verification.")

    evidence_parts = []

    for item in mapped_citations:
        evidence = item.get("evidence", {})

        evidence_parts.append(
            f"Evidence ID: {item.get('evidence_id')}\n"
            f"Chunk ID: {evidence.get('chunk_id')}\n"
            f"Document ID: {evidence.get('document_id')}\n"
            f"Source: {evidence.get('source')}\n"
            f"Page Number: {evidence.get('page_number')}\n"
            f"Content: {evidence.get('chunk_text')}"
        )

    evidence_context = "\n\n".join(evidence_parts)

    user_prompt = f"""
Claim:
{claim_text}

Document Evidence:
{evidence_context}

Determine whether the evidence supports, contradicts, or is insufficient
to establish the claim.

Return valid JSON only.
"""

    owns_client = client is None

    if owns_client:
        client = create_gemini_client()

    try:
        interaction = client.interactions.create(
            model=__import__("os").getenv(
                "GEMINI_MODEL",
                "gemini-3.5-flash-lite"
            ),
            system_instruction=VERIFICATION_SYSTEM_PROMPT,
            input=user_prompt
        )

        response_text = interaction.output_text

        if not response_text or not response_text.strip():
            raise ValueError("Gemini returned an empty verification response.")

        response_text = response_text.strip()

        if response_text.startswith("```"):
            lines = response_text.splitlines()
            lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            response_text = "\n".join(lines).strip()

        try:
            verification = json.loads(response_text)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Gemini returned invalid verification JSON: {error}"
            ) from error

        if not isinstance(verification, dict):
            raise ValueError("Verification response must be a JSON object.")

        required_fields = ["verdict", "reason"]

        for field in required_fields:
            if field not in verification:
                raise ValueError(
                    f"Verification response is missing field: {field}"
                )

        allowed_verdicts = {
            "SUPPORTED",
            "CONTRADICTED",
            "INSUFFICIENT_EVIDENCE"
        }

        if verification["verdict"] not in allowed_verdicts:
            raise ValueError("Gemini returned an invalid verification verdict.")

        if not isinstance(verification["reason"], str):
            raise ValueError("Verification reason must be a string.")

        return {
            "claim_id": claim.get("claim_id"),
            "claim_text": claim_text,
            "verdict": verification["verdict"],
            "reason": verification["reason"],
            "verified_evidence_ids": [
                item.get("evidence_id")
                for item in mapped_citations
            ]
        }

    finally:
        if owns_client:
            client.close()
