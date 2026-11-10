
from retrieval.reciprocal_rank_fusion import (
    reciprocal_rank_fusion
)


def test_rrf():

    semantic_results = [
        {"chunk_id": "A", "chunk_text": "Chunk A"},
        {"chunk_id": "B", "chunk_text": "Chunk B"},
        {"chunk_id": "C", "chunk_text": "Chunk C"}
    ]

    bm25_results = [
        {"chunk_id": "B", "chunk_text": "Chunk B"},
        {"chunk_id": "C", "chunk_text": "Chunk C"},
        {"chunk_id": "A", "chunk_text": "Chunk A"}
    ]

    print("Combining semantic and BM25 results...")

    fused_results = reciprocal_rank_fusion(
        result_lists=[
            semantic_results,
            bm25_results
        ],
        k=60,
        top_k=3
    )

    assert len(fused_results) == 3

    assert fused_results[0]["chunk_id"] == "B"
    assert fused_results[1]["chunk_id"] == "A"
    assert fused_results[2]["chunk_id"] == "C"

    for rank, result in enumerate(fused_results, start=1):

        print(f"\nRank: {rank}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"RRF Score: {result['rrf_score']:.6f}")

    print("\nRRF test passed!")


if __name__ == "__main__":
    test_rrf()
