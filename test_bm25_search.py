
from retrieval.bm25_search import build_bm25_index, save_bm25_index

sample_chunks = [
    {
        "chunk_id": "test_chunk_001",
        "document_id": "test_document_001",
        "source": "sample.txt",
        "file_type": "txt",
        "page_number": None,
        "chunk_index": 0,
        "chunk_text": "Machine learning enables computers to learn patterns from data."
    },
    {
        "chunk_id": "test_chunk_002",
        "document_id": "test_document_001",
        "source": "sample.txt",
        "file_type": "txt",
        "page_number": None,
        "chunk_index": 1,
        "chunk_text": "Deep learning uses neural networks to solve complex problems."
    }
]

index_data = build_bm25_index(sample_chunks)
save_bm25_index(index_data)

print("\nBM25 index rebuilt with matching Pinecone chunks!")
