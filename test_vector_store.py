
from ingestion.vector_store import (
    create_pinecone_client,
    create_index,
    get_pinecone_index
)


def test_pinecone_connection():
    print("Connecting to Pinecone...")

    client = create_pinecone_client()

    print("Pinecone connection successful!")

    return client


def test_create_index(client):
    print("\nChecking Pinecone index...")

    index_name = create_index(client)

    assert index_name == "hybrid-search-rag"

    print(f"Index name: {index_name}")


def test_get_index(client):
    print("\nGetting Pinecone index client...")

    index = get_pinecone_index(client)

    stats = index.describe_index_stats()

    print("Pinecone index client obtained successfully!")
    print(f"Index statistics: {stats}")


if __name__ == "__main__":
    pinecone_client = test_pinecone_connection()

    test_create_index(pinecone_client)

    test_get_index(pinecone_client)

    print("\nAll Pinecone tests passed!")
