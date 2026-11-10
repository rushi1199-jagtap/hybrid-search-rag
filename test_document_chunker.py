
from ingestion.document_loader import load_document
from ingestion.document_cleaner import clean_records
from ingestion.document_chunker import chunk_text, chunk_records


def test_chunk_text():
    print("\nTesting chunk_text()...")

    sample_text = (
        "Retrieval-Augmented Generation combines retrieval "
        "with text generation. "
        "It retrieves relevant information from external documents. "
        "The language model uses this information to generate answers. "
    ) * 15

    chunks = chunk_text(
        text=sample_text,
        chunk_size=200,
        chunk_overlap=40
    )

    assert len(chunks) > 1

    for index, chunk in enumerate(chunks):
        assert chunk["chunk_text"].strip()
        assert chunk["start_char"] < chunk["end_char"]
        assert len(chunk["chunk_text"]) <= 200

        print(f"\nChunk {index + 1}")
        print(f"Start: {chunk['start_char']}")
        print(f"End: {chunk['end_char']}")
        print(f"Length: {len(chunk['chunk_text'])}")
        print(f"Text: {chunk['chunk_text'][:100]}...")

    print("\nchunk_text() passed!")


def test_chunk_records():
    print("\nTesting chunk_records()...")

    files = [
        "sample.txt",
        "sample.docx",
        "sample.pdf"
    ]

    for filename in files:
        records = load_document(
            f"data/sample_documents/{filename}"
        )

        cleaned_records = clean_records(records)

        chunks = chunk_records(
            cleaned_records,
            chunk_size=500,
            chunk_overlap=100
        )

        assert len(chunks) > 0

        for chunk in chunks:
            assert chunk["chunk_id"]
            assert chunk["document_id"]
            assert chunk["source"] == filename
            assert chunk["chunk_text"].strip()

        chunk_ids = [chunk["chunk_id"] for chunk in chunks]

        assert len(chunk_ids) == len(set(chunk_ids))

        print(f"\nFile: {filename}")
        print(f"Total chunks: {len(chunks)}")

        for chunk in chunks:
            print(f"Chunk ID: {chunk['chunk_id']}")
            print(f"Page: {chunk['page_number']}")
            print(f"Chunk text: {chunk['chunk_text'][:120]}")
            print("-" * 40)

    print("\nchunk_records() passed!")


def test_invalid_settings():
    print("\nTesting invalid chunk settings...")

    try:
        chunk_text("Some text", chunk_size=100, chunk_overlap=100)
        raise AssertionError("Expected ValueError was not raised.")

    except ValueError:
        print("Invalid overlap correctly rejected.")

    try:
        chunk_text("Some text", chunk_size=0, chunk_overlap=0)
        raise AssertionError("Expected ValueError was not raised.")

    except ValueError:
        print("Invalid chunk size correctly rejected.")

    print("\nInvalid settings tests passed!")


if __name__ == "__main__":
    test_chunk_text()
    test_chunk_records()
    test_invalid_settings()

    print("\nAll document chunker tests passed!")
