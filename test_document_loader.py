
from pathlib import Path

from ingestion.document_loader import load_document


SAMPLE_DIR = Path("data/sample_documents")


def main():
    for filename in ["sample.txt", "sample.docx", "sample.pdf"]:
        file_path = SAMPLE_DIR / filename

        print("\n" + "=" * 60)
        print(f"Testing: {filename}")
        print("=" * 60)

        try:
            records = load_document(file_path)

            for record in records:
                print(f"Document ID : {record['document_id']}")
                print(f"Source      : {record['source']}")
                print(f"File Type   : {record['file_type']}")
                print(f"Page Number : {record['page_number']}")
                print(f"Text        :\n{record['text']}")

        except Exception as error:
            print(f"ERROR: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
