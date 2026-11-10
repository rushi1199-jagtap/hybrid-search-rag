
from ingestion.embedding_generator import embed_chunk_records
from ingestion.vector_store import (
    get_pinecone_index,
    upsert_chunk_records,
    get_index_stats
)


def test_vector_storage():

    sample_chunks = [
        {
            "chunk_id": "test_chunk_001",
            "document_id": "test_document_001",
            "source": "sample.txt",
            "file_type": "txt",
            "page_number": None,
            "chunk_index": 0,
            "chunk_text": (
                "Machine learning enables computers "
                "to learn patterns from data."
            ),
            "start_char": 0,
            "end_char": 74
        },
        {
            "chunk_id": "test_chunk_002",
            "document_id": "test_document_001",
            "source": "sample.txt",
            "file_type": "txt",
            "page_number": None,
            "chunk_index": 1,
            "chunk_text": (
                "Deep learning uses neural networks "
                "to solve complex problems."
            ),
            "start_char": 75,
            "end_char": 142
        }
    ]

    print("Generating Cohere embeddings...")

    embedded_chunks = embed_chunk_records(sample_chunks)

    print("\nConnecting to Pinecone...")

    index = get_pinecone_index()

    print("\nUploading vectors...")

    uploaded_count = upsert_chunk_records(
        embedded_chunks,
        index=index
    )

    assert uploaded_count == 2

    print("\nChecking Pinecone statistics...")

    stats = get_index_stats(index)

    print(f"Total vectors: {stats.total_vector_count}")

    assert stats.total_vector_count >= 2

    print("\nVector storage test passed!")


if __name__ == "__main__":
    test_vector_storage()
