from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag_loader import RAGDocument, load_rag_documents


def build_retrieval_index(documents: list[RAGDocument]):
    """
    Build a simple local retrieval index for testing.

    This uses TF-IDF instead of embeddings/FAISS so you can test retrieval
    without installing extra packages. Later, replace this part with
    sentence-transformer/OpenAI embeddings + FAISS.
    """
    document_texts = [document.text for document in documents]

    vectorizer = TfidfVectorizer(stop_words="english")
    document_vectors = vectorizer.fit_transform(document_texts)

    return vectorizer, document_vectors


def retrieve_documents(
    query: str,
    documents: list[RAGDocument],
    vectorizer: TfidfVectorizer,
    document_vectors,
    top_k: int = 5,
):
    """Return the top matching documents for a user query."""
    query_vector = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, document_vectors).ravel()

    top_indexes = similarity_scores.argsort()[::-1][:top_k]

    results = []
    for index in top_indexes:
        results.append(
            {
                "score": float(similarity_scores[index]),
                "text": documents[index].text,
                "metadata": documents[index].metadata,
            }
        )

    return results


def print_retrieval_results(query: str, results: list[dict]) -> None:
    """Print retrieval results in a readable format."""
    print("=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]

        print(f"\nResult #{rank}")
        print(f"Score: {result['score']:.4f}")
        print(f"Ticker: {metadata['ticker']}")
        print(f"Company: {metadata['company']}")
        print(f"Sector: {metadata['sector']}")
        print(f"Document type: {metadata['doc_type']}")
        print(f"Title: {metadata['title']}")
        print(f"Date: {metadata['date']}")
        print(f"Source: {metadata['source_url']}")
        print(f"Preview: {result['text'][:300]}...")


def main() -> None:
    documents = load_rag_documents("data")
    vectorizer, document_vectors = build_retrieval_index(documents)

    test_queries = [
        "AI growth in Microsoft cloud business",
        "Apple services revenue and margins",
        "healthcare company risks and earnings",
        "bank earnings performance and investment banking",
        "NVIDIA data center AI demand",
    ]

    print(f"Loaded {len(documents)} documents for retrieval testing")

    for query in test_queries:
        results = retrieve_documents(
            query=query,
            documents=documents,
            vectorizer=vectorizer,
            document_vectors=document_vectors,
            top_k=3,
        )
        print_retrieval_results(query, results)


if __name__ == "__main__":
    main()
