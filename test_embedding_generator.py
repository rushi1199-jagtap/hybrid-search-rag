
from ingestion.embedding_generator import (
    generate_embeddings,
    embed_query
)


def test_document_embeddings():
    sample_documents = [
        "Machine learning enables computers to learn from data.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand text."
    ]

    print("Generating document embeddings...")

    embeddings = generate_embeddings(
        texts=sample_documents,
        input_type="search_document"
    )

    assert len(embeddings) == len(sample_documents)

    for index, embedding in enumerate(embeddings):
        assert len(embedding) == 1024

        print(
            f"Document {index + 1}: "
            f"{len(embedding)} dimensions"
        )

    print("\nDocument embedding test passed!")


def test_query_embedding():
    query = "What is deep learning?"

    print("\nGenerating query embedding...")

    embedding = embed_query(query)

    assert len(embedding) == 1024

    print(f"Query embedding dimensions: {len(embedding)}")
    print("Query embedding test passed!")


if __name__ == "__main__":
    test_document_embeddings()
    test_query_embedding()
