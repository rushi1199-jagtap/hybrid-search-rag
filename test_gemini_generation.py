
from generation.gemini_client import generate_answer
from generation.context_builder import build_context


def test_gemini_generation():
    sample_results = [
        {
            "chunk_id": "test_chunk_002",
            "document_id": "test_document_001",
            "source": "sample.txt",
            "file_type": "txt",
            "page_number": None,
            "chunk_index": 1,
            "chunk_text": "Deep learning uses neural networks to solve complex problems."
        },
        {
            "chunk_id": "test_chunk_001",
            "document_id": "test_document_001",
            "source": "sample.txt",
            "file_type": "txt",
            "page_number": None,
            "chunk_index": 0,
            "chunk_text": "Machine learning enables computers to learn patterns from data."
        }
    ]

    query = "What does deep learning use?"
    context = build_context(sample_results)

    print("\nTesting Gemini answer generation...")

    answer = generate_answer(query, context)

    assert isinstance(answer, dict)
    assert isinstance(answer["answer"], str)
    assert isinstance(answer["citations"], list)
    assert isinstance(answer["insufficient_evidence"], bool)

    print("\nGemini answer:")
    print(answer["answer"])

    print("\nCitations:")
    for citation in answer["citations"]:
        print(citation)

    print("\nInsufficient evidence:", answer["insufficient_evidence"])
    print("\nGemini answer generation test passed!")


if __name__ == "__main__":
    test_gemini_generation()

