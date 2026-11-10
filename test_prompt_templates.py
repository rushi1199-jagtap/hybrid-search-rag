
from generation.prompt_templates import SYSTEM_PROMPT, build_user_prompt


def test_system_prompt():
    assert "ONLY the evidence" in SYSTEM_PROMPT
    assert "Evidence ID" in SYSTEM_PROMPT
    assert "insufficient_evidence" in SYSTEM_PROMPT
    assert "conflicting information" in SYSTEM_PROMPT

    print("System prompt test passed!")


def test_user_prompt():
    query = "What is deep learning?"

    context = """
    [Evidence 1]
    Document ID: doc_001
    Source: AI_notes.pdf
    Chunk ID: chunk_001
    Page Number: 2
    Content:
    Deep learning uses neural networks to solve complex problems.
    """

    prompt = build_user_prompt(query, context)

    assert query in prompt
    assert context in prompt
    assert "Retrieved Document Evidence:" in prompt
    assert "valid JSON" in prompt

    print("\nGenerated User Prompt:\n")
    print(prompt)

    print("\nUser prompt test passed!")


def test_invalid_inputs():
    invalid_inputs = [
        ("", "some context"),
        ("valid query", ""),
        (None, "some context"),
        ("valid query", None)
    ]

    for query, context in invalid_inputs:
        try:
            build_user_prompt(query, context)
            raise AssertionError("Expected ValueError was not raised.")
        except ValueError:
            pass

    print("Invalid input tests passed!")


if __name__ == "__main__":
    test_system_prompt()
    test_user_prompt()
    test_invalid_inputs()
