
from ingestion.document_loader import load_document
from ingestion.document_cleaner import (
    clean_text,
    clean_record,
    clean_records
)


def test_clean_text():
    messy_text = (
        "  Company Employee Handbook  \r\n"
        "\r\n\r\n"
        "1. Working Hours   \n"
        "Employees must work  8 hours per day.\t\n"
        "\n\n\n"
        "2. Leave Policy"
    )

    cleaned = clean_text(messy_text)

    print("\nTesting clean_text():")
    print(cleaned)

    assert "  " not in cleaned
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned.startswith("Company Employee Handbook")

    print("\nclean_text() passed!")


def test_clean_record():
    record = {
        "document_id": "test123",
        "source": "sample.txt",
        "file_type": "txt",
        "page_number": None,
        "text": "  Hello   World  \n\n\nThis is a test.  "
    }

    cleaned_record = clean_record(record)

    print("\nTesting clean_record():")
    print(cleaned_record)

    assert cleaned_record["cleaned_text"] == (
        "Hello World\n\nThis is a test."
    )

    # Original text must remain unchanged.
    assert cleaned_record["text"] == record["text"]

    print("\nclean_record() passed!")


def test_clean_real_document():
    records = load_document("data/sample_documents/sample.pdf")

    cleaned_records = clean_records(records)

    print("\nTesting with actual PDF:")
    for record in cleaned_records:
        print(f"Source: {record['source']}")
        print(f"Page: {record['page_number']}")
        print(f"Cleaned text:\n{record['cleaned_text']}")

        assert record["cleaned_text"].strip()

    print("\nReal document cleaning passed!")


if __name__ == "__main__":
    test_clean_text()
    test_clean_record()
    test_clean_real_document()

    print("\nAll document cleaner tests passed!")
