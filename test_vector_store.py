
import pytest

from ingestion.vector_store import (
    create_pinecone_client,
    create_index,
    get_pinecone_index
)


@pytest.fixture(scope="module")
def client():
    return create_pinecone_client()


def test_pinecone_connection(client):
    assert client is not None


def test_create_index(client):
    index_name = create_index(client)

    assert index_name == "hybrid-search-rag"


def test_get_index(client):
    index = get_pinecone_index(client)

    assert index is not None

    stats = index.describe_index_stats()

    assert stats is not None
