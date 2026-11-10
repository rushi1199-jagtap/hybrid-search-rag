
import os

from dotenv import load_dotenv
from pinecone import Pinecone


load_dotenv()

PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "hybrid-search-rag"
)

PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

EMBEDDING_DIMENSION = int(
    os.getenv("EMBEDDING_DIMENSION", "1024")
)


def create_pinecone_client():
    """Create an authenticated Pinecone client."""

    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        raise ValueError(
            "PINECONE_API_KEY is missing from the .env file."
        )

    return Pinecone(api_key=api_key)


def create_index(client=None):
    """
    Create the Pinecone vector index if it doesn't exist.

    Returns the index name.
    """

    if client is None:
        client = create_pinecone_client()

    existing_indexes = [
        index.name for index in client.indexes.list()
    ]

    if PINECONE_INDEX_NAME in existing_indexes:
        print(
            f"Index '{PINECONE_INDEX_NAME}' already exists."
        )
        return PINECONE_INDEX_NAME

    print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")

    client.indexes.create(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        deployment={
            "deployment_type": "managed",
            "cloud": PINECONE_CLOUD,
            "region": PINECONE_REGION
        }
    )

    print("Pinecone index created successfully!")

    return PINECONE_INDEX_NAME


def get_pinecone_index(client=None):
    """Return the data-plane index client."""

    if client is None:
        client = create_pinecone_client()

    index_name = create_index(client)

    return client.Index(index_name)


def upsert_chunk_records(
    chunk_records,
    index=None,
    batch_size=100
):
    """
    Store embedded document chunks in Pinecone.

    Each record must contain:
        chunk_id
        embedding
        chunk_text
        document_id
        source
        file_type
        page_number
        chunk_index
    """

    if not chunk_records:
        return 0

    if not 1 <= batch_size <= 1000:
        raise ValueError("batch_size must be between 1 and 1000.")

    if index is None:
        index = get_pinecone_index()

    vectors = []

    for record in chunk_records:
        required_fields = [
            "chunk_id",
            "embedding",
            "chunk_text",
            "document_id"
        ]

        for field in required_fields:
            if field not in record:
                raise ValueError(
                    f"Missing required field: {field}"
                )

        embedding = record["embedding"]

        if len(embedding) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Invalid embedding dimension for "
                f"{record['chunk_id']}."
            )

        metadata = {
            "chunk_text": record["chunk_text"],
            "document_id": record["document_id"]
        }

        # Add optional metadata only when values are available.
        optional_fields = [
            "source",
            "file_type",
            "page_number",
            "chunk_index",
            "start_char",
            "end_char"
        ]

        for field in optional_fields:
            value = record.get(field)

            if value is not None:
                metadata[field] = value

        vectors.append({
            "id": str(record["chunk_id"]),
            "values": embedding,
            "metadata": metadata
        })

    total_upserted = 0

    for start in range(0, len(vectors), batch_size):
        batch = vectors[start:start + batch_size]

        response = index.upsert(vectors=batch)

        total_upserted += response.upserted_count

        print(
            f"Uploaded {total_upserted}/"
            f"{len(vectors)} vectors"
        )

    return total_upserted


def get_index_stats(index=None):
    """Return Pinecone index statistics."""

    if index is None:
        index = get_pinecone_index()

    return index.describe_index_stats()
