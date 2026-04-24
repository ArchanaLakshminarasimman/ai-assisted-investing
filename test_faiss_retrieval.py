from __future__ import annotations

from faiss_retriever import FAISSRetriever


TEST_QUERIES = [
    "AI growth in Microsoft cloud business",
    "Apple services revenue and margins",
    "healthcare company risks and earnings",
    "bank earnings performance and investment banking",
    "NVIDIA data center AI demand",
]


def print_results(query: str, results: list[dict]) -> None:
    print("=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    for result in results:
        metadata = result["metadata"]
        print(f"\nResult #{result['rank']}")
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
    retriever = FAISSRetriever("rag_index", local_files_only=True)

    print(f"Loaded FAISS index with {retriever.index.ntotal} documents")

    # These queries give us a quick sanity check across a few sectors.
    for query in TEST_QUERIES:
        results = retriever.search(query, top_k=3)
        print_results(query, results)


if __name__ == "__main__":
    main()
