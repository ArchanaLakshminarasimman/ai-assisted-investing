from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from build_faiss_index import (
    DOCUMENTS_FILE,
    FAISS_INDEX_FILE,
    INDEX_INFO_FILE,
    INDEX_DIR,
    MODEL_NAME,
    load_sentence_transformer_model,
)


def import_faiss():
    try:
        import faiss
    except ImportError as error:
        raise SystemExit(
            "Missing FAISS. Install dependencies with:\n"
            "  python3 -m pip install -r requirements-rag.txt"
        ) from error

    return faiss


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


class FAISSRetriever:
    """Load a saved FAISS index and retrieve the most relevant documents."""

    def __init__(
        self,
        index_dir: str | Path = INDEX_DIR,
        model_name: str | None = None,
        local_files_only: bool = True,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / FAISS_INDEX_FILE
        self.documents_path = self.index_dir / DOCUMENTS_FILE
        self.index_info_path = self.index_dir / INDEX_INFO_FILE

        # Fail early if the saved retrieval bundle is incomplete.
        self._validate_index_files()

        self.index_info = load_json(self.index_info_path)
        self.model_name = model_name or self.index_info.get("model_name", MODEL_NAME)
        self.model = load_sentence_transformer_model(
            self.model_name,
            allow_download=not local_files_only,
        )

        faiss = import_faiss()
        self.index = faiss.read_index(str(self.index_path))
        self.documents = load_json(self.documents_path)

        # The vectors and stored documents should always stay in sync.
        if self.index.ntotal != len(self.documents):
            raise ValueError(
                "FAISS index size does not match saved documents: "
                f"{self.index.ntotal} vectors vs {len(self.documents)} documents"
            )

    def _validate_index_files(self) -> None:
        missing_paths = [
            path
            for path in [self.index_path, self.documents_path, self.index_info_path]
            if not path.exists()
        ]

        if missing_paths:
            missing = "\n".join(f"  - {path}" for path in missing_paths)
            raise FileNotFoundError(
                "Missing FAISS retrieval files:\n"
                f"{missing}\n\n"
                "Build them first with:\n"
                "  python3 build_faiss_index.py"
            )

    def embed_query(self, query: str) -> np.ndarray:
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return query_embedding.astype("float32")

    def _matches_filters(
        self,
        metadata: dict[str, Any],
        filters: dict[str, Any] | None,
    ) -> bool:
        if not filters:
            return True

        for key, expected in filters.items():
            actual = metadata.get(key)

            if callable(expected):
                if not expected(actual):
                    return False
                continue

            if isinstance(expected, (list, tuple, set, frozenset)):
                if actual not in expected:
                    return False
                continue

            if actual != expected:
                return False

        return True

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        candidate_pool_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search FAISS and return matching documents.

        Return format:
            [
              {
                "rank": 1,
                "score": 0.82,
                "text": "...",
                "metadata": {...}
              }
            ]
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if candidate_pool_size is not None and candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be greater than 0")

        query_embedding = self.embed_query(query)
        search_size = top_k
        if filters:
            search_size = self.index.ntotal
            if candidate_pool_size is not None:
                search_size = min(
                    self.index.ntotal,
                    max(top_k, candidate_pool_size),
                )

        scores, indexes = self.index.search(query_embedding, search_size)

        results = []
        for rank, (score, index) in enumerate(zip(scores[0], indexes[0]), start=1):
            # FAISS pads with -1 when there are fewer real matches than requested.
            if index == -1:
                continue

            document = self.documents[int(index)]
            if not self._matches_filters(document["metadata"], filters):
                continue

            results.append(
                {
                    "rank": len(results) + 1,
                    "score": float(score),
                    "text": document["text"],
                    "metadata": document["metadata"],
                }
            )
            if len(results) == top_k:
                break

        return results

    def search_by_ticker(
        self,
        ticker: str,
        query: str,
        top_k: int = 5,
        candidate_pool_size: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.search(
            query=query,
            top_k=top_k,
            filters={"ticker": ticker.upper()},
            candidate_pool_size=candidate_pool_size,
        )


if __name__ == "__main__":
    retriever = FAISSRetriever()
    query = "AI growth in Microsoft cloud business"
    results = retriever.search(query, top_k=3)

    print(f"Query: {query}")
    for result in results:
        metadata = result["metadata"]
        print(
            f"{result['rank']}. {metadata['ticker']} | "
            f"{metadata['doc_type']} | "
            f"{metadata['title']} | "
            f"score={result['score']:.4f}"
        )
