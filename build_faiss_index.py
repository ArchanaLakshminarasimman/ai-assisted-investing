from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rag_loader import load_rag_documents


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = Path("rag_index")
FAISS_INDEX_FILE = "documents.faiss"
METADATA_FILE = "metadata.json"
DOCUMENTS_FILE = "documents.json"
INDEX_INFO_FILE = "index_info.json"


def import_faiss():
    try:
        import faiss
    except ImportError as error:
        raise SystemExit(
            "Missing FAISS. Install dependencies with:\n"
            "  python3 -m pip install -r requirements-rag.txt"
        ) from error

    return faiss


def import_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise SystemExit(
            "Missing sentence-transformers. Install dependencies with:\n"
            "  python3 -m pip install -r requirements-rag.txt"
        ) from error

    return SentenceTransformer


SentenceTransformer = import_sentence_transformer()


def load_sentence_transformer_model(
    model_name: str,
    allow_download: bool = False,
):
    """
    Load the embedding model from local cache first.

    This keeps the default build/retrieval path stable in offline environments
    while still allowing an explicit first-time download when needed.
    """
    try:
        # Try the local cache first so routine runs stay offline-friendly.
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception as local_error:
        if not allow_download:
            raise RuntimeError(
                f"Model '{model_name}' is not available in the local cache.\n"
                "Download it once on an internet-enabled machine, or rerun with "
                "allow_download=True."
            ) from local_error

    try:
        return SentenceTransformer(model_name, local_files_only=False)
    except Exception as download_error:
        raise RuntimeError(
            f"Unable to load model '{model_name}'. "
            "Check your internet connection or confirm the model is cached locally."
        ) from download_error


def create_embeddings(
    texts: list[str],
    model_name: str = MODEL_NAME,
    allow_download: bool = False,
) -> np.ndarray:
    """
    Create normalized document embeddings.

    Normalized vectors let FAISS inner product behave like cosine similarity.
    """
    model = load_sentence_transformer_model(
        model_name,
        allow_download=allow_download,
    )

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray):
    """Build a FAISS index from document embeddings."""
    faiss = import_faiss()

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array")

    embedding_dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dimension)
    index.add(embeddings)

    return index


def save_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def build_and_save_index(
    data_dir: str | Path = "data",
    index_dir: str | Path = INDEX_DIR,
    model_name: str = MODEL_NAME,
    allow_download: bool = False,
) -> None:
    """
    Full Person 4 build step:
    1. Load RAG documents
    2. Create embeddings
    3. Build FAISS index
    4. Save FAISS index, metadata, and document text
    """
    faiss = import_faiss()

    documents = load_rag_documents(data_dir)
    if not documents:
        raise ValueError(f"No documents found in {data_dir}")

    texts = [document.text for document in documents]
    metadata = []
    stored_documents = []

    # Keep a stable id so search results can point back to the saved payload.
    for document_id, document in enumerate(documents):
        document_metadata = {
            "document_id": document_id,
            **document.metadata,
        }

        metadata.append(document_metadata)
        stored_documents.append(
            {
                "document_id": document_id,
                "text": document.text,
                "metadata": document_metadata,
            }
        )

    embeddings = create_embeddings(
        texts,
        model_name=model_name,
        allow_download=allow_download,
    )
    index = build_faiss_index(embeddings)

    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)

    # Save the whole bundle together so retrieval can boot from disk later.
    faiss.write_index(index, str(index_path / FAISS_INDEX_FILE))
    save_json(index_path / METADATA_FILE, metadata)
    save_json(index_path / DOCUMENTS_FILE, stored_documents)
    save_json(
        index_path / INDEX_INFO_FILE,
        {
            "model_name": model_name,
            "document_count": len(documents),
            "embedding_dimension": int(embeddings.shape[1]),
            "faiss_metric": "inner_product_on_normalized_embeddings",
            "faiss_index_file": FAISS_INDEX_FILE,
            "metadata_file": METADATA_FILE,
            "documents_file": DOCUMENTS_FILE,
        },
    )

    print(f"Loaded documents: {len(documents)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"FAISS vectors stored: {index.ntotal}")
    print(f"Saved FAISS index to: {index_path / FAISS_INDEX_FILE}")
    print(f"Saved metadata to: {index_path / METADATA_FILE}")
    print(f"Saved documents to: {index_path / DOCUMENTS_FILE}")


if __name__ == "__main__":
    build_and_save_index()
