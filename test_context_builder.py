
from generation.context_builder import build_context


def test_context_builder():
    sample_results = [
        {
            "chunk_id": "chunk_001",
            "document_id": "doc_001",
            "source": "AI_notes.pdf",
            "page_number": 2,
            "chunk_text": "Deep learning uses neural networks to solve complex problems."
        },
        {
            "chunk_id": "chunk_002",
            "document_id": "doc_001",
            "source": "AI_notes.pdf",
            "page_number": 3,
            "chunk_text": "Neural networks contain multiple interconnected layers."
        }
    ]

    print("\nTesting Context Builder...")

    context = build_context(sample_results)

    assert "chunk_001" in context
    assert "chunk_002" in context
    assert "AI_notes.pdf" in context
    assert "Page Number: 2" in context
    assert "Page Number: 3" in context
    assert "Deep learning" in context
    assert "interconnected layers" in context

    print("\nGenerated Context:\n")
    print(context)

    print("\nContext Builder test passed!")


def test_empty_results():
    context = build_context([])

    assert context == "No relevant document evidence was retrieved."

    print("Empty results test passed!")


if __name__ == "__main__":
    test_context_builder()
    test_empty_results()
