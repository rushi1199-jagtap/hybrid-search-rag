
import os
import math

import cohere
from dotenv import load_dotenv


load_dotenv()

COHERE_EMBED_MODEL = os.getenv(
    "COHERE_EMBED_MODEL",
    "embed-v4.0"
)

EMBEDDING_DIMENSION = int(
    os.getenv("EMBEDDING_DIMENSION", "1024")
)

DEFAULT_BATCH_SIZE = 32
MAX_BATCH_SIZE = 96


def create_cohere_client():
    """Create and return an authenticated Cohere client."""

    api_key = os.getenv("COHERE_API_KEY")

    if not api_key:
        raise ValueError(
            "COHERE_API_KEY is missing from the .env file."
        )

    return cohere.ClientV2(api_key=api_key)


def generate_embeddings(
    texts,
    input_type="search_document",
    batch_size=DEFAULT_BATCH_SIZE,
    client=None
):
    """
    Generate embeddings for a list of texts.

    input_type:
        search_document -> document chunks
        search_query    -> user search queries
    """

    if input_type not in ("search_document", "search_query"):
        raise ValueError(
            "input_type must be 'search_document' or 'search_query'."
        )

    if not isinstance(texts, list) or not texts:
        raise ValueError("texts must be a non-empty list.")

    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(
            f"batch_size must be between 1 and {MAX_BATCH_SIZE}."
        )

    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Invalid or empty text at index {index}."
            )

    if client is None:
        client = create_cohere_client()

    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        response = client.embed(
            model=COHERE_EMBED_MODEL,
            texts=batch,
            input_type=input_type,
            embedding_types=["float"],
            output_dimension=EMBEDDING_DIMENSION
        )

        batch_embeddings = response.embeddings.float

        if len(batch_embeddings) != len(batch):
            raise ValueError(
                "Cohere returned an unexpected number of embeddings."
            )

        for embedding in batch_embeddings:
            vector = [float(value) for value in embedding]

            if len(vector) != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Expected {EMBEDDING_DIMENSION} dimensions, "
                    f"received {len(vector)}."
                )

            if not all(math.isfinite(value) for value in vector):
                raise ValueError(
                    "Embedding contains invalid numeric values."
                )

            all_embeddings.append(vector)

    return all_embeddings


def embed_query(query, client=None):
    """Generate an embedding for a user's search query."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty.")

    embeddings = generate_embeddings(
        texts=[query],
        input_type="search_query",
        batch_size=1,
        client=client
    )

    return embeddings[0]


def embed_chunk_records(
    chunk_records,
    batch_size=DEFAULT_BATCH_SIZE,
    client=None
):
    """
    Generate embeddings for document chunks.

    Returns copies of the original records with an additional
    'embedding' field. Original records remain unchanged.
    """

    if not chunk_records:
        return []

    texts = []

    for index, record in enumerate(chunk_records):
        if "chunk_text" not in record:
            raise ValueError(
                f"chunk_text missing from record at index {index}."
            )

        texts.append(record["chunk_text"])

    embeddings = generate_embeddings(
        texts=texts,
        input_type="search_document",
        batch_size=batch_size,
        client=client
    )

    embedded_records = []

    for record, embedding in zip(chunk_records, embeddings):
        embedded_record = record.copy()
        embedded_record["embedding"] = embedding

        embedded_records.append(embedded_record)

    return embedded_records
