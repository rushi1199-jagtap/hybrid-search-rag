
from unittest.mock import patch

from generation.answer_pipeline import run_rag_pipeline


@patch("generation.answer_pipeline.generate_answer")
@patch("generation.answer_pipeline.hybrid_search")
def test_empty_retrieval_abstains_without_calling_gemini(
    mock_hybrid_search,
    mock_generate_answer
):
    mock_hybrid_search.return_value = []

    result = run_rag_pipeline(
        query="What is quantum computing?"
    )

    assert result["insufficient_evidence"] is True

    assert result["answer"] == (
        "The available documents do not contain sufficient information "
        "to answer this question."
    )

    assert result["citations"] == []
    assert result["retrieved_chunks"] == []

    assert result["verification"]["overall_status"] == (
        "ABSTAINED_INSUFFICIENT_EVIDENCE"
    )

    assert result["verification"]["total_claims"] == 0
    mock_generate_answer.assert_not_called()


if __name__ == "__main__":
    test_empty_retrieval_abstains_without_calling_gemini()
    print("Empty retrieval test passed!")
