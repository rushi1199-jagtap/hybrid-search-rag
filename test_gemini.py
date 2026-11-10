import os
from dotenv import load_dotenv
from google import genai


def test_gemini_connection():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from .env")

    if not model_name:
        raise ValueError("GEMINI_MODEL is missing from .env")

    client = genai.Client(api_key=api_key)

    try:
        print(f"Connecting to Gemini using {model_name}...")
        print("Using Google's Interactions API.\n")

        interaction = client.interactions.create(
            model=model_name,
            input="Explain Retrieval-Augmented Generation (RAG) in exactly two sentences."
        )

        print("Gemini connection successful!")
        print(f"Model: {model_name}")
        print("\nGemini response:")
        print(interaction.output_text)

    except Exception as error:
        print(f"\nGemini API error: {type(error).__name__}")
        print(error)

    finally:
        client.close()


if __name__ == "__main__":
    test_gemini_connection()