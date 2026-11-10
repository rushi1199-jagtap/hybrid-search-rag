
def build_context(results):
    if not isinstance(results, list):
        raise ValueError("Results must be a list.")

    if not results:
        return "No relevant document evidence was retrieved."

    context_parts = []

    for index, result in enumerate(results, start=1):
        if not isinstance(result, dict):
            raise ValueError("Each result must be a dictionary.")

        required_fields = ["chunk_id", "chunk_text", "document_id"]

        for field in required_fields:
            if not result.get(field):
                raise ValueError(f"Missing required field: {field}")

        source = result.get("source") or "Unknown source"
        page_number = result.get("page_number")
        chunk_id = result["chunk_id"]
        document_id = result["document_id"]

        metadata = [
            f"Document ID: {document_id}",
            f"Source: {source}",
            f"Chunk ID: {chunk_id}"
        ]

        if page_number is not None:
            metadata.append(f"Page Number: {page_number}")

        context_parts.append(
            f"[Evidence {index}]\n"
            f"{' | '.join(metadata)}\n"
            f"Content:\n{result['chunk_text']}"
        )

    return "\n\n".join(context_parts)
