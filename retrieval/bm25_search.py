
import pickle
import re

from pathlib import Path
from rank_bm25 import BM25Okapi


INDEX_PATH = (
    Path(__file__).resolve().parent.parent
    / "storage"
    / "bm25_index"
    / "bm25_index.pkl"
)


def tokenize_text(text):
    """Convert text into lowercase word tokens."""

    if not isinstance(text, str):
        raise ValueError("Text must be a string.")

    return re.findall(r"\b\w+\b", text.lower())


def build_bm25_index(chunk_records):
    """
    Build a BM25 index from document chunk records.

    Each record must contain chunk_id and chunk_text.
    """

    if not chunk_records:
        raise ValueError("No chunk records provided.")

    tokenized_corpus = []
    valid_records = []

    for index, record in enumerate(chunk_records):

        if "chunk_id" not in record:
            raise ValueError(
                f"chunk_id missing from record {index}."
            )

        if "chunk_text" not in record:
            raise ValueError(
                f"chunk_text missing from record {index}."
            )

        tokens = tokenize_text(record["chunk_text"])

        if not tokens:
            raise ValueError(
                f"Chunk {record['chunk_id']} contains no searchable text."
            )

        tokenized_corpus.append(tokens)
        valid_records.append(record.copy())

    bm25 = BM25Okapi(tokenized_corpus)

    return {
        "bm25": bm25,
        "records": valid_records,
        "tokenized_corpus": tokenized_corpus
    }


def save_bm25_index(index_data, file_path=INDEX_PATH):
    """Save the BM25 index and its records to disk."""

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("wb") as file:
        pickle.dump(index_data, file)

    print(f"BM25 index saved to: {file_path}")


def load_bm25_index(file_path=INDEX_PATH):
    """Load a previously saved BM25 index."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"BM25 index not found: {file_path}"
        )

    with file_path.open("rb") as file:
        return pickle.load(file)


def bm25_search(query, top_k=5, index_data=None):
    """Search the BM25 index using keyword relevance."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty.")

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    if index_data is None:
        index_data = load_bm25_index()

    query_tokens = tokenize_text(query)

    if not query_tokens:
        raise ValueError("Query contains no searchable terms.")

    bm25 = index_data["bm25"]
    records = index_data["records"]

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = []

    for index in ranked_indices:

        record = records[index]

        results.append({
            "chunk_id": record["chunk_id"],
            "score": float(scores[index]),
            "chunk_text": record["chunk_text"],
            "document_id": record.get("document_id"),
            "source": record.get("source"),
            "file_type": record.get("file_type"),
            "page_number": record.get("page_number"),
            "chunk_index": record.get("chunk_index")
        })

    return results
