from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from faiss_retriever import FAISSRetriever
from rag_insights import (
    build_insight_text,
    build_stock_query,
    extract_gemini_text,
    extract_ollama_text,
    generate_insights,
    load_ranked_stocks,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
INDEX_DIR = PROJECT_ROOT / "rag_index"


class TestRagInsights(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sample_rankings = EXAMPLES_DIR / "mock_rankings.json"
        cls.retriever = FAISSRetriever(INDEX_DIR, local_files_only=True)

    def test_load_ranked_stocks_from_json(self) -> None:
        stocks = load_ranked_stocks(self.sample_rankings)

        self.assertEqual(len(stocks), 3)
        self.assertEqual(stocks[0].ticker, "MSFT")
        self.assertEqual(stocks[0].rank, 1)
        self.assertAlmostEqual(stocks[0].score or 0.0, 0.941, places=3)

    def test_load_ranked_stocks_from_person3_csv_uses_latest_snapshot(self) -> None:
        csv_payload = """Date,Ticker,Sector,Rank,Score
2026-03-20,V,Finance,1,0.9424
2026-03-20,UNH,Healthcare,2,0.5675
2026-02-28,MSFT,Technology,1,0.7777
2026-02-28,NVDA,Technology,2,0.6666
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            ranking_file = Path(temp_dir) / "ranked_holdings.csv"
            ranking_file.write_text(csv_payload, encoding="utf-8")

            stocks = load_ranked_stocks(ranking_file)

        self.assertEqual(len(stocks), 2)
        self.assertEqual([stock.ticker for stock in stocks], ["V", "UNH"])
        self.assertEqual([stock.rank for stock in stocks], [1, 2])
        self.assertTrue(all(stock.date == "2026-03-20" for stock in stocks))

    def test_build_stock_query_includes_signal_summary(self) -> None:
        stock = load_ranked_stocks(self.sample_rankings)[0]
        query = build_stock_query(stock)

        self.assertIn("MSFT", query)
        self.assertIn("Strong cloud and AI demand", query)

    def test_search_by_ticker_keeps_results_scoped(self) -> None:
        results = self.retriever.search_by_ticker(
            ticker="MSFT",
            query="Microsoft cloud AI growth risks",
            top_k=3,
            candidate_pool_size=self.retriever.index.ntotal,
        )

        self.assertTrue(results)
        self.assertTrue(all(result["metadata"]["ticker"] == "MSFT" for result in results))

    def test_generate_template_insights(self) -> None:
        stocks = load_ranked_stocks(self.sample_rankings)[:2]
        insights = generate_insights(
            stocks=stocks,
            retriever=self.retriever,
            top_k=3,
            generator="template",
        )

        self.assertEqual(len(insights), 2)
        self.assertTrue(insights[0].evidence)
        self.assertTrue(insights[0].key_points)
        self.assertIn("MSFT", insights[0].insight)

    def test_extract_ollama_text_from_chat_response(self) -> None:
        response_payload = {
            "message": {
                "role": "assistant",
                "content": "Microsoft ranks highly because of cloud growth and AI demand.",
            }
        }

        self.assertEqual(
            extract_ollama_text(response_payload),
            "Microsoft ranks highly because of cloud growth and AI demand.",
        )

    def test_extract_gemini_text_from_generate_content_response(self) -> None:
        response_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "Microsoft ranks highly because of cloud growth and AI demand.",
                            }
                        ]
                    }
                }
            ]
        }

        self.assertEqual(
            extract_gemini_text(response_payload),
            "Microsoft ranks highly because of cloud growth and AI demand.",
        )

    def test_build_insight_text_uses_ollama_generator(self) -> None:
        stock = load_ranked_stocks(self.sample_rankings)[0]

        with patch("rag_insights.generate_with_ollama", return_value="mocked llama output") as mock_generate:
            insight_text = build_insight_text(
                generator="ollama",
                stock=stock,
                evidence=[],
                key_points=[],
                risk_points=[],
                model="llama3.1:8b",
            )

        self.assertEqual(insight_text, "mocked llama output")
        mock_generate.assert_called_once()

    def test_build_insight_text_uses_gemini_generator(self) -> None:
        stock = load_ranked_stocks(self.sample_rankings)[0]

        with patch("rag_insights.generate_with_gemini", return_value="mocked gemini output") as mock_generate:
            insight_text = build_insight_text(
                generator="gemini",
                stock=stock,
                evidence=[],
                key_points=[],
                risk_points=[],
                model="gemini-2.5-flash",
            )

        self.assertEqual(insight_text, "mocked gemini output")
        mock_generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
