from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


@dataclass
class RAGDocument:
    """Single document that can later be embedded and stored in FAISS."""

    text: str
    metadata: dict[str, Any]


def load_json_files(data_dir: str | Path = DATA_DIR) -> list[dict[str, Any]]:
    """
    Step 1: Load every JSON file from the data folder.

    Expected folder shape:
        data/
          AAPL/
            overview.json
            financial_summary.json
            earnings_highlights.json
            news_1.json
            news_2.json
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data folder not found: {data_path}")

    json_records: list[dict[str, Any]] = []

    # Each ticker folder becomes a small batch of source documents.
    for json_file in sorted(data_path.glob("*/*.json")):
        with json_file.open("r", encoding="utf-8") as file:
            record = json.load(file)

        record["_file_path"] = str(json_file)
        json_records.append(record)

    return json_records


def json_to_rag_document(record: dict[str, Any]) -> RAGDocument:
    """
    Step 2: Convert one JSON record into a RAG-ready document.

    The text is what will be embedded later.
    The metadata is what lets us identify where a retrieved result came from.
    """
    required_fields = [
        "ticker",
        "company",
        "sector",
        "doc_type",
        "title",
        "date",
        "source_url",
        "text",
    ]

    missing_fields = [field for field in required_fields if field not in record]
    if missing_fields:
        file_path = record.get("_file_path", "unknown file")
        raise ValueError(f"{file_path} is missing fields: {missing_fields}")

    # This text block is what the retriever will actually index and search.
    text = (
        f"Title: {record['title']}\n"
        f"Company: {record['company']} ({record['ticker']})\n"
        f"Sector: {record['sector']}\n"
        f"Document type: {record['doc_type']}\n"
        f"Date: {record['date']}\n\n"
        f"{record['text']}"
    )

    # Metadata stays lightweight so results are easy to trace back later.
    metadata = {
        "ticker": record["ticker"],
        "company": record["company"],
        "sector": record["sector"],
        "doc_type": record["doc_type"],
        "title": record["title"],
        "date": record["date"],
        "source_url": record["source_url"],
        "file_path": record.get("_file_path"),
    }

    return RAGDocument(text=text, metadata=metadata)


def load_rag_documents(data_dir: str | Path = DATA_DIR) -> list[RAGDocument]:
    """Load all JSON files and convert them into RAG-ready documents."""
    records = load_json_files(data_dir)
    return [json_to_rag_document(record) for record in records]


if __name__ == "__main__":
    documents = load_rag_documents(DATA_DIR)

    print(f"Loaded {len(documents)} RAG documents")
    print("\nExample document:")
    print(documents[0].text[:500])
    print("\nExample metadata:")
    print(json.dumps(documents[0].metadata, indent=2))
