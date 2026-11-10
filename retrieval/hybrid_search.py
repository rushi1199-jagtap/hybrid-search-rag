
from retrieval.vector_search import semantic_search
from retrieval.bm25_search import bm25_search
from retrieval.reciprocal_rank_fusion import reciprocal_rank_fusion
from retrieval.reranker import rerank_results


def hybrid_search(
    query,
    top_k=5,
    semantic_top_k=10,
    bm25_top_k=10,
    use_reranking=True
):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if not all(
        isinstance(value, int) and value > 0
        for value in [top_k, semantic_top_k, bm25_top_k]
    ):
        raise ValueError("All top_k values must be positive integers.")

    if not isinstance(use_reranking, bool):
        raise ValueError("use_reranking must be a boolean.")

    semantic_results = semantic_search(
        query=query,
        top_k=semantic_top_k
    )

    keyword_results = bm25_search(
        query=query,
        top_k=bm25_top_k
    )

    fused_results = reciprocal_rank_fusion(
        result_lists=[semantic_results, keyword_results],
        k=60,
        top_k=max(top_k, 10) if use_reranking else top_k
    )

    if use_reranking and fused_results:
        final_results = rerank_results(
            query=query,
            results=fused_results,
            top_n=top_k
        )
    else:
        final_results = fused_results[:top_k]

    return final_results
