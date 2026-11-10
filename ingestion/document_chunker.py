
import hashlib


def find_split_position(
    text: str,
    start: int,
    target: int,
    chunk_size: int
) -> int:
    """
    Find a natural splitting point near the target position.

    Priority:
    1. Paragraph boundary
    2. Line boundary
    3. Sentence boundary
    4. Word boundary

    If no suitable boundary exists, use the target position.
    """

    # Search the final 30% of the chunk for a natural boundary.
    search_start = start + int(chunk_size * 0.7)
    search_area = text[search_start:target]

    separators = ["\n\n", "\n", ". ", "? ", "! ", " "]

    for separator in separators:
        position = search_area.rfind(separator)

        if position != -1:
            return search_start + position + len(separator)

    return target


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> list[dict]:
    """
    Split text into overlapping chunks.

    Returns chunks containing:
    - chunk_text
    - start_char
    - end_char
    """

    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        target = min(start + chunk_size, text_length)

        if target == text_length:
            end = text_length
        else:
            end = find_split_position(
                text=text,
                start=start,
                target=target,
                chunk_size=chunk_size
            )

        chunk_content = text[start:end].strip()

        if chunk_content:
            chunks.append({
                "chunk_text": chunk_content,
                "start_char": start,
                "end_char": end
            })

        if end >= text_length:
            break

        # Move forward while retaining the requested overlap.
        next_start = max(start + 1, end - chunk_overlap)

        # Avoid starting the next chunk in the middle of a word.
        while (
            next_start < end
            and not text[next_start - 1].isspace()
            and not text[next_start].isspace()
        ):
            next_start += 1

        start = next_start

    return chunks


def generate_chunk_id(
    document_id: str,
    page_number,
    chunk_index: int,
    start_char: int,
    end_char: int
) -> str:
    """Generate a deterministic ID for each chunk."""

    unique_value = (
        f"{document_id}:{page_number}:"
        f"{chunk_index}:{start_char}:{end_char}"
    )

    return hashlib.sha256(
        unique_value.encode("utf-8")
    ).hexdigest()[:16]


def chunk_records(
    records: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> list[dict]:
    """
    Convert cleaned document records into chunk records.

    Each chunk retains its original document metadata.
    """

    all_chunks = []

    for record in records:
        cleaned_text = record.get("cleaned_text")

        if cleaned_text is None:
            raise ValueError(
                f"Missing cleaned_text for {record.get('source', 'unknown document')}. "
                "Clean the records before chunking."
            )

        text_chunks = chunk_text(
            text=cleaned_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        for index, chunk in enumerate(text_chunks):
            chunk_id = generate_chunk_id(
                document_id=record["document_id"],
                page_number=record.get("page_number"),
                chunk_index=index,
                start_char=chunk["start_char"],
                end_char=chunk["end_char"]
            )

            chunk_record = {
                "chunk_id": chunk_id,
                "document_id": record["document_id"],
                "source": record["source"],
                "file_type": record["file_type"],
                "page_number": record.get("page_number"),
                "chunk_index": index,
                "chunk_text": chunk["chunk_text"],
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"]
            }

            all_chunks.append(chunk_record)

    return all_chunks
