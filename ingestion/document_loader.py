
from pathlib import Path
from hashlib import sha256

import pymupdf
from docx import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl


def generate_document_id(file_path: Path) -> str:
    """
    Generate a stable document ID using the file's content.
    The same file content produces the same ID.
    """
    file_hash = sha256(file_path.read_bytes()).hexdigest()
    return file_hash[:16]


def load_pdf(file_path: Path) -> list[dict]:
    """Extract text from a PDF, preserving individual page numbers."""
    records = []
    document_id = generate_document_id(file_path)

    with pymupdf.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            records.append({
                "document_id": document_id,
                "source": file_path.name,
                "file_type": "pdf",
                "page_number": page_number,
                "text": text
            })

    return records


def load_docx(file_path: Path) -> list[dict]:
    """Extract paragraphs and tables from a DOCX in document order."""
    records = []
    document_id = generate_document_id(file_path)
    document = DocxDocument(file_path)

    content_parts = []

    for element in document.element.body.iterchildren():
        if isinstance(element, CT_P):
            paragraph = Paragraph(element, document)
            text = paragraph.text.strip()

            if text:
                content_parts.append(text)

        elif isinstance(element, CT_Tbl):
            table = Table(element, document)

            for row in table.rows:
                row_text = " | ".join(
                    cell.text.strip() for cell in row.cells
                )

                if row_text.strip():
                    content_parts.append(row_text)

    full_text = "\n".join(content_parts).strip()

    records.append({
        "document_id": document_id,
        "source": file_path.name,
        "file_type": "docx",
        "page_number": None,
        "text": full_text
    })

    return records


def load_txt(file_path: Path) -> list[dict]:
    """Read a TXT file using UTF-8 encoding."""
    document_id = generate_document_id(file_path)
    text = file_path.read_text(encoding="utf-8-sig").strip()

    return [{
        "document_id": document_id,
        "source": file_path.name,
        "file_type": "txt",
        "page_number": None,
        "text": text
    }]


def load_document(file_path: str | Path) -> list[dict]:
    """
    Load a PDF, DOCX, or TXT file.

    Returns a list of records containing extracted text
    and document metadata.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Document not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Expected a file, but received: {file_path}"
        )

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        records = load_pdf(file_path)

    elif extension == ".docx":
        records = load_docx(file_path)

    elif extension == ".txt":
        records = load_txt(file_path)

    else:
        raise ValueError(
            f"Unsupported file format: {extension}. "
            "Supported formats: PDF, DOCX, TXT."
        )

    if not any(record["text"] for record in records):
        raise ValueError(
            f"No extractable text found in: {file_path.name}. "
            "The PDF may be scanned or image-only and require OCR."
        )

    return records
