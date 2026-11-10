
import re
from copy import deepcopy


def clean_text(text: str) -> str:
    """
    Clean extracted document text while preserving
    meaningful paragraph and line boundaries.
    """

    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    # 1. Normalize Windows and Mac line endings to \n
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # 2. Remove null bytes and unwanted control characters.
    # Preserve newline and tab characters.
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    # 3. Replace non-breaking spaces with normal spaces.
    text = text.replace("\u00a0", " ")

    # 4. Remove unnecessary spaces and tabs within each line.
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = re.sub(r"[ \t]+", " ", line)
        line = line.strip()
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # 5. Replace three or more consecutive newlines
    # with a maximum of two newlines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 6. Remove leading and trailing whitespace.
    return text.strip()


def clean_record(record: dict) -> dict:
    """
    Clean one document record without modifying
    the original extracted text.
    """

    cleaned_record = deepcopy(record)

    original_text = record.get("text", "")

    cleaned_record["cleaned_text"] = clean_text(original_text)

    return cleaned_record


def clean_records(records: list[dict]) -> list[dict]:
    """
    Clean multiple document records.
    """

    return [clean_record(record) for record in records]
