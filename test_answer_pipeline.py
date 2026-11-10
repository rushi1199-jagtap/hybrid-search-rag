
from generation.answer_pipeline import run_rag_pipeline


def test_answer_pipeline():
    query = "How does deep learning work?"

    print("\nTesting end-to-end RAG pipeline with claim verification...")

    result = run_rag_pipeline(
        query=query,
        top_k=5,
        semantic_top_k=2,
        bm25_top_k=2,
        use_reranking=True
    )

    print("\nGenerated Answer:")
    print(result["answer"])

    print("\nCitations:")
    for citation in result["citations"]:
        print(citation)

    print("\nRetrieved Chunks:")
    for chunk in result["retrieved_chunks"]:
        print(
            chunk["chunk_id"],
            "|",
            chunk.get("rerank_score", chunk.get("score"))
        )

    print("\nInsufficient Evidence:")
    print(result["insufficient_evidence"])

    verification = result["verification"]

    print("\nVerification Summary:")
    print(verification["overall_status"])

    print("\nTotal Claims:")
    print(verification["total_claims"])

    print("\nStatus Counts:")
    for status, count in verification["status_counts"].items():
        print(status, ":", count)

    print("\nIndividual Claim Verification:")

    for claim in verification["claim_results"]:
        print("\nClaim ID:", claim["claim_id"])
        print("Claim:", claim["claim_text"])
        print("Citation Status:", claim["status"])
        print("Overall Status:", claim["overall_status"])

        semantic_result = claim.get("semantic_verification")

        if semantic_result:
            print("Semantic Verdict:", semantic_result["verdict"])
            print("Reason:", semantic_result["reason"])

    assert isinstance(result["answer"], str)
    assert isinstance(result["citations"], list)
    assert isinstance(result["retrieved_chunks"], list)
    assert isinstance(verification, dict)
    assert "overall_status" in verification
    assert "claim_results" in verification
    assert "status_counts" in verification

    print("\nEnd-to-end RAG pipeline with verification test passed!")


if __name__ == "__main__":
    test_answer_pipeline()
