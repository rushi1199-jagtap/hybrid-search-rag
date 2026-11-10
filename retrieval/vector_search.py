
from ingestion.embedding_generator import embed_query
from ingestion.vector_store import get_pinecone_index


def semantic_search(
    query,
    top_k=5,
    index=None
):
    """
    Search Pinecone using a natural-language query.

    Returns the most similar document chunks with
    their metadata and similarity scores.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty.")

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    # Step 1: Generate query embedding using Cohere.
    query_embedding = embed_query(query)

    # Step 2: Connect to Pinecone.
    if index is None:
        index = get_pinecone_index()

    # Step 3: Search for similar vectors.
    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    # Step 4: Extract search results.
    results = []

    for match in response.matches:
        metadata = match.metadata or {}

        results.append({
            "chunk_id": match.id,
            "score": match.score,
            "chunk_text": metadata.get("chunk_text", ""),
            "document_id": metadata.get("document_id"),
            "source": metadata.get("source"),
            "file_type": metadata.get("file_type"),
            "page_number": metadata.get("page_number"),
            "chunk_index": metadata.get("chunk_index")
        })

    return results
