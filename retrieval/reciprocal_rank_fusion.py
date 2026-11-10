
def reciprocal_rank_fusion(
    result_lists,
    k=60,
    top_k=5
):
    """
    Combine multiple ranked result lists using RRF.
    """

    if not isinstance(result_lists, list):
        raise ValueError("result_lists must be a list.")

    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    fused_results = {}

    for results in result_lists:

        seen_in_list = set()

        for rank, result in enumerate(results, start=1):

            chunk_id = result.get("chunk_id")

            if not chunk_id:
                raise ValueError("Every result must have a chunk_id.")

            if chunk_id in seen_in_list:
                continue

            seen_in_list.add(chunk_id)

            if chunk_id not in fused_results:
                fused_results[chunk_id] = {
                    **result,
                    "rrf_score": 0.0
                }

            fused_results[chunk_id]["rrf_score"] += (
                1.0 / (k + rank)
            )

    ranked_results = sorted(
        fused_results.values(),
        key=lambda item: item["rrf_score"],
        reverse=True
    )

    return ranked_results[:top_k]
