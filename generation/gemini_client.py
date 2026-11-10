
import os
import json
from dotenv import load_dotenv
from google import genai

from generation.prompt_templates import SYSTEM_PROMPT, build_user_prompt

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


def create_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from .env")

    return genai.Client(api_key=api_key)


def parse_gemini_response(response_text):
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError("Gemini returned an empty response.")

    response_text = response_text.strip()

    if response_text.startswith("```"):
        lines = response_text.splitlines()

        if lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        response_text = "\n".join(lines).strip()

    try:
        answer_data = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Gemini returned invalid JSON: {error}"
        ) from error

    if not isinstance(answer_data, dict):
        raise ValueError("Gemini response must be a JSON object.")

    required_fields = [
        "answer",
        "citations",
        "insufficient_evidence"
    ]

    for field in required_fields:
        if field not in answer_data:
            raise ValueError(
                f"Gemini response is missing field: {field}"
            )

    if not isinstance(answer_data["answer"], str):
        raise ValueError("Answer must be a string.")

    if not isinstance(answer_data["citations"], list):
        raise ValueError("Citations must be a list.")

    if not isinstance(answer_data["insufficient_evidence"], bool):
        raise ValueError(
            "insufficient_evidence must be a boolean."
        )

    for citation in answer_data["citations"]:
        if not isinstance(citation, dict):
            raise ValueError("Every citation must be a JSON object.")

    return answer_data


def generate_answer(query, context, client=None):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if not isinstance(context, str) or not context.strip():
        raise ValueError("Context must be a non-empty string.")

    owns_client = client is None

    if owns_client:
        client = create_gemini_client()

    try:
        user_prompt = build_user_prompt(query, context)

        interaction = client.interactions.create(
            model=GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
            input=user_prompt
        )

        response_text = interaction.output_text

        answer_data = parse_gemini_response(response_text)

        return answer_data

    finally:
        if owns_client:
            client.close()
