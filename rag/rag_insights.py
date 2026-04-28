from __future__ import annotations

import argparse
import csv
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from faiss_retriever import FAISSRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RANKING_FILE = PROJECT_ROOT / "xgboost_backtesting_outputs" / "ranking" / "ranked_holdings.csv"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "rag_index"

JSON_LIST_KEYS = ("rankings", "stocks", "results", "data", "items")
TICKER_KEYS = ("ticker", "symbol", "stock", "stock_ticker")
RANK_KEYS = ("rank", "ranking", "position")
SCORE_KEYS = ("score", "final_score", "combined_score", "model_score")
COMPANY_KEYS = ("company", "company_name", "name")
SECTOR_KEYS = ("sector", "industry_group")
DATE_KEYS = ("date", "rebalance_date", "as_of_date")
SIGNAL_KEYS = (
    "signal_summary",
    "reason",
    "thesis",
    "summary",
    "why",
    "notes",
    "explanation",
)
RISK_KEYWORDS = (
    "risk",
    "risks",
    "pressure",
    "pressures",
    "competition",
    "competitive",
    "regulatory",
    "cyclical",
    "volatile",
    "volatility",
    "uncertainty",
    "decline",
    "headwind",
    "slowdown",
    "patent",
    "exposure",
)
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


@dataclass
class RankedStock:
    ticker: str
    rank: int | None = None
    score: float | None = None
    company: str | None = None
    sector: str | None = None
    signal_summary: str | None = None
    date: str | None = None


@dataclass
class EvidenceItem:
    retrieval_rank: int
    retrieval_score: float
    ticker: str
    company: str | None
    sector: str | None
    doc_type: str
    title: str
    date: str
    source_url: str
    snippet: str
    full_text: str


@dataclass
class StockInsight:
    ticker: str
    company: str | None
    sector: str | None
    rank: int | None
    score: float | None
    signal_summary: str | None
    date: str | None
    retrieval_query: str
    generator: str
    insight: str
    key_points: list[str]
    risk_points: list[str]
    evidence: list[EvidenceItem]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    return text or None


def normalize_record_keys(record: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key).strip().lower(): value
        for key, value in record.items()
    }


def first_present(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if normalize_text(value) is not None:
            return value

    return None


def parse_int(value: Any) -> int | None:
    text = normalize_text(value)
    if text is None:
        return None

    try:
        return int(float(text))
    except ValueError:
        return None


def parse_float(value: Any) -> float | None:
    text = normalize_text(value)
    if text is None:
        return None

    try:
        return float(text)
    except ValueError:
        return None


def parse_date_text(value: Any) -> datetime | None:
    text = normalize_text(value)
    if text is None:
        return None

    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    return None


def filter_to_latest_snapshot(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not records:
        return records

    dated_records: list[tuple[datetime, dict[str, Any]]] = []
    for record in records:
        parsed_date = parse_date_text(first_present(record, DATE_KEYS))
        if parsed_date is None:
            continue

        dated_records.append((parsed_date, record))

    if not dated_records:
        return records

    latest_date = max(parsed_date for parsed_date, _ in dated_records)
    latest_records = [
        record
        for parsed_date, record in dated_records
        if parsed_date == latest_date
    ]
    latest_records.sort(
        key=lambda record: parse_int(first_present(record, RANK_KEYS)) or 10**9
    )
    return latest_records


def load_ranking_records(path: str | Path) -> list[dict[str, Any]]:
    ranking_path = Path(path)
    suffix = ranking_path.suffix.lower()

    if suffix == ".json":
        payload = load_json(ranking_path)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in JSON_LIST_KEYS:
                records = payload.get(key)
                if isinstance(records, list):
                    return records

        raise ValueError(
            f"Could not find a ranking list inside JSON file: {ranking_path}"
        )

    if suffix in {".csv", ".tsv"}:
        delimiter = "," if suffix == ".csv" else "\t"
        with ranking_path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file, delimiter=delimiter))

    raise ValueError(
        f"Unsupported ranking file format: {ranking_path}. Use JSON, CSV, or TSV."
    )


def normalize_ranked_stock(record: dict[str, Any], default_rank: int) -> RankedStock:
    ticker = normalize_text(first_present(record, TICKER_KEYS))
    if ticker is None:
        raise ValueError(f"Ranking record is missing a ticker: {record}")

    return RankedStock(
        ticker=ticker.upper(),
        rank=parse_int(first_present(record, RANK_KEYS)) or default_rank,
        score=parse_float(first_present(record, SCORE_KEYS)),
        company=normalize_text(first_present(record, COMPANY_KEYS)),
        sector=normalize_text(first_present(record, SECTOR_KEYS)),
        signal_summary=normalize_text(first_present(record, SIGNAL_KEYS)),
        date=normalize_text(first_present(record, DATE_KEYS)),
    )


def load_ranked_stocks(path: str | Path) -> list[RankedStock]:
    records = [
        normalize_record_keys(record)
        for record in load_ranking_records(path)
    ]
    records = filter_to_latest_snapshot(records)
    return [
        normalize_ranked_stock(record, default_rank=index)
        for index, record in enumerate(records, start=1)
    ]


def build_manual_rankings(tickers: list[str]) -> list[RankedStock]:
    return [
        RankedStock(ticker=ticker.strip().upper(), rank=index)
        for index, ticker in enumerate(tickers, start=1)
        if ticker.strip()
    ]


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_body(text: str) -> str:
    _, separator, remainder = text.partition("\n\n")
    body = remainder if separator else text
    return collapse_whitespace(body)


def split_sentences(text: str) -> list[str]:
    clean_text = collapse_whitespace(text)
    if not clean_text:
        return []

    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", clean_text)
        if sentence.strip()
    ]


def build_snippet(text: str, max_sentences: int = 2) -> str:
    sentences = split_sentences(extract_body(text))
    return " ".join(sentences[:max_sentences])


def unique_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_items: list[str] = []

    for item in items:
        normalized = collapse_whitespace(item)
        if not normalized:
            continue
        if normalized in seen:
            continue

        seen.add(normalized)
        unique_items.append(normalized)

    return unique_items


def build_stock_query(stock: RankedStock) -> str:
    subject = stock.company or stock.ticker
    query_parts = [
        f"{subject} {stock.ticker} why this stock ranks highly",
        "recent earnings growth profitability outlook risks",
    ]

    if stock.signal_summary:
        query_parts.append(stock.signal_summary)

    return ". ".join(query_parts)


def build_evidence_items(
    retriever: FAISSRetriever,
    stock: RankedStock,
    top_k: int,
) -> tuple[str, list[EvidenceItem]]:
    query = build_stock_query(stock)
    results = retriever.search_by_ticker(
        ticker=stock.ticker,
        query=query,
        top_k=top_k,
        candidate_pool_size=retriever.index.ntotal,
    )

    if not results:
        results = retriever.search_by_ticker(
            ticker=stock.ticker,
            query=stock.ticker,
            top_k=top_k,
            candidate_pool_size=retriever.index.ntotal,
        )

    evidence_items: list[EvidenceItem] = []
    for result in results:
        metadata = result["metadata"]
        stock.company = stock.company or metadata.get("company")
        stock.sector = stock.sector or metadata.get("sector")
        evidence_items.append(
            EvidenceItem(
                retrieval_rank=result["rank"],
                retrieval_score=result["score"],
                ticker=metadata.get("ticker", stock.ticker),
                company=metadata.get("company"),
                sector=metadata.get("sector"),
                doc_type=metadata.get("doc_type", "unknown"),
                title=metadata.get("title", "Untitled"),
                date=metadata.get("date", "unknown"),
                source_url=metadata.get("source_url", ""),
                snippet=build_snippet(result["text"]),
                full_text=result["text"],
            )
        )

    return query, evidence_items


def build_key_points(evidence: list[EvidenceItem], limit: int = 3) -> list[str]:
    points = []
    for item in evidence:
        sentences = split_sentences(item.snippet)
        if sentences:
            first_point = sentences[0]
            if len(first_point) < 40 and len(sentences) > 1:
                first_point = f"{first_point} {sentences[1]}"

            points.append(first_point)

    return unique_preserving_order(points)[:limit]


def build_risk_points(evidence: list[EvidenceItem], limit: int = 2) -> list[str]:
    risk_sentences: list[str] = []

    for item in evidence:
        for sentence in split_sentences(extract_body(item.full_text)):
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in RISK_KEYWORDS):
                risk_sentences.append(sentence)

    return unique_preserving_order(risk_sentences)[:limit]


def join_as_clause(points: list[str]) -> str:
    return "; ".join(point.rstrip(". ") for point in points if point)


def build_template_insight(
    stock: RankedStock,
    key_points: list[str],
    risk_points: list[str],
) -> str:
    company_label = stock.company or stock.ticker
    opening = f"{company_label} ({stock.ticker})"
    if stock.rank is not None and stock.score is not None:
        opening = f"{opening} ranks #{stock.rank} with score {stock.score:.4f}."
    elif stock.rank is not None:
        opening = f"{opening} ranks #{stock.rank}."
    elif stock.score is not None:
        opening = f"{opening} has score {stock.score:.4f}."
    else:
        opening = f"{opening} is supported by retrieved company evidence."

    parts = [opening]

    if stock.signal_summary:
        parts.append(
            "Upstream ranking signal: "
            f"{stock.signal_summary.rstrip('. ')}."
        )

    if key_points:
        parts.append(
            "Retrieved documents support the ranking through "
            f"{join_as_clause(key_points[:2])}."
        )

    if risk_points:
        parts.append(
            "Important risk context includes "
            f"{join_as_clause(risk_points[:2])}."
        )

    return " ".join(parts)


def build_generation_system_prompt() -> str:
    return (
        "You generate concise stock-ranking explanations. "
        "Use only the provided ranking info and retrieved evidence. "
        "Write one short paragraph, mention at least one upside driver "
        "and one risk, and do not invent facts."
    )


def build_generation_user_payload(
    stock: RankedStock,
    evidence: list[EvidenceItem],
    key_points: list[str],
    risk_points: list[str],
) -> dict[str, Any]:
    return {
        "ticker": stock.ticker,
        "company": stock.company,
        "sector": stock.sector,
        "rank": stock.rank,
        "score": stock.score,
        "signal_summary": stock.signal_summary,
        "key_points": key_points,
        "risk_points": risk_points,
        "evidence": [
            {
                "doc_type": item.doc_type,
                "title": item.title,
                "date": item.date,
                "source_url": item.source_url,
                "snippet": item.snippet,
            }
            for item in evidence
        ],
    }


def build_llm_messages(
    stock: RankedStock,
    evidence: list[EvidenceItem],
    key_points: list[str],
    risk_points: list[str],
) -> list[dict[str, str]]:
    prompt_payload = build_generation_user_payload(
        stock=stock,
        evidence=evidence,
        key_points=key_points,
        risk_points=risk_points,
    )

    return [
        {
            "role": "system",
            "content": build_generation_system_prompt(),
        },
        {
            "role": "user",
            "content": json.dumps(prompt_payload, indent=2),
        },
    ]


def extract_chat_completion_text(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices", [])
    if not choices:
        raise RuntimeError("OpenAI response did not include any choices.")

    message = choices[0].get("message", {})
    content = message.get("content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))

        return "".join(text_parts).strip()

    raise RuntimeError("OpenAI response content was not in a recognized format.")


def extract_ollama_text(response_payload: dict[str, Any]) -> str:
    response_text = response_payload.get("response")
    if isinstance(response_text, str) and response_text.strip():
        return response_text.strip()

    message = response_payload.get("message", {})
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()

    raise RuntimeError("Ollama response did not include generated text.")


def extract_gemini_text(response_payload: dict[str, Any]) -> str:
    candidates = response_payload.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini response did not include any candidates.")

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    text_parts = []

    for part in parts:
        text = part.get("text")
        if isinstance(text, str):
            text_parts.append(text)

    combined = "".join(text_parts).strip()
    if combined:
        return combined

    raise RuntimeError("Gemini response did not include generated text parts.")


def generate_with_openai(
    stock: RankedStock,
    evidence: list[EvidenceItem],
    key_points: list[str],
    risk_points: list[str],
    model: str | None,
) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Use --generator template or export an API key."
        )

    model_name = model or os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL

    payload = {
        "model": model_name,
        "messages": build_llm_messages(stock, evidence, key_points, risk_points),
        "temperature": 0.2,
    }

    request = urllib.request.Request(
        url="https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response_payload = json.load(response)
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            "OpenAI request failed with "
            f"status {error.code}: {error_body}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"OpenAI request failed: {error.reason}") from error

    return extract_chat_completion_text(response_payload)


def generate_with_gemini(
    stock: RankedStock,
    evidence: list[EvidenceItem],
    key_points: list[str],
    risk_points: list[str],
    model: str | None,
) -> str:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Export it before using --generator gemini."
        )

    model_name = model or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    prompt_payload = build_generation_user_payload(
        stock=stock,
        evidence=evidence,
        key_points=key_points,
        risk_points=risk_points,
    )
    request_payload = {
        "systemInstruction": {
            "parts": [
                {
                    "text": build_generation_system_prompt(),
                }
            ]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": json.dumps(prompt_payload, indent=2),
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
        },
    }

    request = urllib.request.Request(
        url=(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent"
        ),
        data=json.dumps(request_payload).encode("utf-8"),
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_payload = json.load(response)
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            "Gemini request failed with "
            f"status {error.code}: {error_body}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Gemini request failed: {error.reason}") from error

    return extract_gemini_text(response_payload)


def generate_with_ollama(
    stock: RankedStock,
    evidence: list[EvidenceItem],
    key_points: list[str],
    risk_points: list[str],
    model: str | None,
) -> str:
    model_name = model or os.environ.get("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL

    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    payload = {
        "model": model_name,
        "messages": build_llm_messages(stock, evidence, key_points, risk_points),
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }

    request = urllib.request.Request(
        url=f"{ollama_base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_payload = json.load(response)
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            "Ollama request failed with "
            f"status {error.code}: {error_body}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            "Ollama request failed. Make sure the Ollama app/server is running "
            f"and the model '{model_name}' is installed. Details: {error.reason}"
        ) from error

    return extract_ollama_text(response_payload)


def build_insight_text(
    generator: str,
    stock: RankedStock,
    evidence: list[EvidenceItem],
    key_points: list[str],
    risk_points: list[str],
    model: str | None = None,
) -> str:
    if generator == "gemini":
        return generate_with_gemini(
            stock=stock,
            evidence=evidence,
            key_points=key_points,
            risk_points=risk_points,
            model=model,
        )

    if generator == "ollama":
        return generate_with_ollama(
            stock=stock,
            evidence=evidence,
            key_points=key_points,
            risk_points=risk_points,
            model=model,
        )

    if generator == "openai":
        return generate_with_openai(
            stock=stock,
            evidence=evidence,
            key_points=key_points,
            risk_points=risk_points,
            model=model,
        )

    return build_template_insight(stock, key_points, risk_points)


def generate_insights(
    stocks: list[RankedStock],
    retriever: FAISSRetriever,
    top_k: int = 3,
    generator: str = "template",
    model: str | None = None,
) -> list[StockInsight]:
    insights: list[StockInsight] = []

    for stock in stocks:
        query, evidence = build_evidence_items(
            retriever=retriever,
            stock=stock,
            top_k=top_k,
        )

        if not evidence:
            raise ValueError(f"No FAISS evidence found for ticker {stock.ticker}")

        key_points = build_key_points(evidence)
        risk_points = build_risk_points(evidence)
        insight_text = build_insight_text(
            generator=generator,
            stock=stock,
            evidence=evidence,
            key_points=key_points,
            risk_points=risk_points,
            model=model,
        )

        insights.append(
            StockInsight(
                ticker=stock.ticker,
                company=stock.company,
                sector=stock.sector,
                rank=stock.rank,
                score=stock.score,
                signal_summary=stock.signal_summary,
                date=stock.date,
                retrieval_query=query,
                generator=generator,
                insight=insight_text,
                key_points=key_points,
                risk_points=risk_points,
                evidence=evidence,
            )
        )

    return insights


def evidence_to_dict(item: EvidenceItem) -> dict[str, Any]:
    return {
        "retrieval_rank": item.retrieval_rank,
        "retrieval_score": item.retrieval_score,
        "ticker": item.ticker,
        "company": item.company,
        "sector": item.sector,
        "doc_type": item.doc_type,
        "title": item.title,
        "date": item.date,
        "source_url": item.source_url,
        "snippet": item.snippet,
    }


def insight_to_dict(item: StockInsight) -> dict[str, Any]:
    return {
        "ticker": item.ticker,
        "company": item.company,
        "sector": item.sector,
        "rank": item.rank,
        "score": item.score,
        "signal_summary": item.signal_summary,
        "date": item.date,
        "retrieval_query": item.retrieval_query,
        "generator": item.generator,
        "insight": item.insight,
        "key_points": item.key_points,
        "risk_points": item.risk_points,
        "evidence": [evidence_to_dict(evidence_item) for evidence_item in item.evidence],
    }


def build_output_payload(
    insights: list[StockInsight],
    input_source: str,
    generator: str,
    index_dir: str | Path,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_source": input_source,
        "generator": generator,
        "index_dir": str(index_dir),
        "insights": [insight_to_dict(insight) for insight in insights],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Person 5 pipeline: load rankings, query FAISS, and generate stock insights."
    )
    parser.add_argument(
        "--ranking-file",
        help=(
            "Path to Person 3 output file (JSON, CSV, or TSV). "
            f"Defaults to {DEFAULT_RANKING_FILE} when no --ticker values are given."
        ),
    )
    parser.add_argument(
        "--ticker",
        dest="tickers",
        action="append",
        default=[],
        help="Manual ticker input for working ahead before Person 3 hands over files.",
    )
    parser.add_argument(
        "--index-dir",
        default=str(DEFAULT_INDEX_DIR),
        help="Directory containing the saved FAISS index bundle.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many retrieved documents to use per stock.",
    )
    parser.add_argument(
        "--generator",
        choices=("template", "openai", "ollama", "gemini"),
        default="openai",
        help="Use template mode, OpenAI, Ollama, or Gemini. Default: OpenAI.",
    )
    parser.add_argument(
        "--model",
        help=(
            "Model name for --generator openai, --generator ollama, or "
            "--generator gemini. Defaults: gpt-4.1-mini for OpenAI, "
            "llama3.1:8b for Ollama, gemini-2.5-flash for Gemini."
        ),
    )
    parser.add_argument(
        "--output-file",
        help="Optional path to save the final insight JSON.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.ranking_file:
        ranked_stocks = load_ranked_stocks(args.ranking_file)
        input_source = str(args.ranking_file)
    elif not args.tickers:
        if not DEFAULT_RANKING_FILE.exists():
            raise SystemExit(
                "No ranking input provided, and the default ranking file was not found: "
                f"{DEFAULT_RANKING_FILE}"
            )

        ranked_stocks = load_ranked_stocks(DEFAULT_RANKING_FILE)
        input_source = str(DEFAULT_RANKING_FILE)
    else:
        ranked_stocks = build_manual_rankings(args.tickers)
        input_source = "manual_tickers"

    if not ranked_stocks:
        raise SystemExit("No ranked stocks were loaded.")

    retriever = FAISSRetriever(index_dir=args.index_dir, local_files_only=True)
    insights = generate_insights(
        stocks=ranked_stocks,
        retriever=retriever,
        top_k=args.top_k,
        generator=args.generator,
        model=args.model,
    )

    payload = build_output_payload(
        insights=insights,
        input_source=input_source,
        generator=args.generator,
        index_dir=args.index_dir,
    )

    if args.output_file:
        output_path = Path(args.output_file)
        save_json(output_path, payload)
        print(f"Saved {len(insights)} insights to {output_path}")
        return

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
