# RAG

This folder contains the retrieval-augmented generation pipeline for turning ranked stocks into evidence-backed narrative insights.

At a high level, the flow is:

```text
data/*.json
    -> rag_loader.py
    -> build_faiss_index.py
    -> rag_index/
    -> faiss_retriever.py
    -> rag_insights.py
    -> JSON insight output
```

## Files In This Folder

- [`rag_loader.py`](/Users/archana/Documents/ai-assisted-investing/rag/rag_loader.py): loads ticker JSON documents from `data/` and converts them into RAG-ready text + metadata.
- [`build_faiss_index.py`](/Users/archana/Documents/ai-assisted-investing/rag/build_faiss_index.py): embeds documents with `sentence-transformers` and saves a FAISS index bundle to `rag_index/`.
- [`faiss_retriever.py`](/Users/archana/Documents/ai-assisted-investing/rag/faiss_retriever.py): loads the saved FAISS bundle and runs semantic search.
- [`rag_insights.py`](/Users/archana/Documents/ai-assisted-investing/rag/rag_insights.py): reads ranked stocks, retrieves ticker-specific evidence, and generates short investment-style explanations.
- [`test_retrieval.py`](/Users/archana/Documents/ai-assisted-investing/rag/test_retrieval.py): simple TF-IDF retrieval sanity check that does not require FAISS.
- [`test_faiss_retrieval.py`](/Users/archana/Documents/ai-assisted-investing/rag/test_faiss_retrieval.py): sample FAISS retrieval queries against the saved index.
- [`test_rag_insights.py`](/Users/archana/Documents/ai-assisted-investing/rag/test_rag_insights.py): unit tests for ranking ingestion, retrieval scoping, and generator helpers.

## What The Pipeline Does

### 1. Load source documents

`rag_loader.py` expects the `data/` folder to be organized by ticker, with JSON files such as:

```text
data/
  AAPL/
    overview.json
    financial_summary.json
    earnings_highlights.json
    news_1.json
    news_2.json
```

Each JSON record is expected to contain:

- `ticker`
- `company`
- `sector`
- `doc_type`
- `title`
- `date`
- `source_url`
- `text`

The loader converts each record into:

- a retrieval text block used for embeddings and search
- lightweight metadata used for filtering and traceability

### 2. Build the FAISS bundle

Run:

```bash
python3 rag/build_faiss_index.py
```

This script:

- loads all JSON documents from `data/`
- embeds them with `sentence-transformers/all-MiniLM-L6-v2`
- normalizes the vectors
- builds a FAISS inner-product index
- saves the index and supporting JSON files to `rag_index/`

Important detail: the build path tries to load the embedding model from the local cache first. That keeps routine runs offline-friendly, but it means the model must already be available locally unless you explicitly use the helper with `allow_download=True`.

### 3. Retrieve evidence

`faiss_retriever.py` loads the saved bundle from `rag_index/` and exposes:

- `search(query, top_k=...)`
- `search_by_ticker(ticker, query, top_k=...)`

The retriever validates that:

- `documents.faiss` exists
- `documents.json` exists
- `index_info.json` exists
- the number of stored FAISS vectors matches the number of saved documents

### 4. Generate stock insights

Run the default insight generator:

```bash
python3 rag/rag_insights.py --generator template --ticker MSFT --ticker NVDA
```

By default, `rag_insights.py` can take rankings from:

- `--ranking-file path/to/file.csv`
- repeated `--ticker` inputs for manual testing
- the default file `xgboost_backtesting_outputs/ranking/ranked_holdings.csv` if no other input is provided

Supported ranking formats:

- JSON
- CSV
- TSV

When a ranking file contains multiple dates, the script keeps only the latest snapshot before generating insights.

## Generator Options

`rag_insights.py` supports four generator modes:

- `template`: no API call, uses retrieved evidence to assemble a deterministic paragraph
- `openai`: calls the OpenAI Chat Completions API
- `ollama`: calls a local Ollama server
- `gemini`: calls the Google Gemini API

Current defaults in the script:

- default generator: `openai`
- default OpenAI model: `gpt-4.1-mini`
- default Ollama model: `llama3.1:8b`
- default Gemini model: `gemini-2.5-flash`

### Environment Variables

- OpenAI: `OPENAI_API_KEY`, optional `OPENAI_MODEL`
- Gemini: `GEMINI_API_KEY` or `GOOGLE_API_KEY`, optional `GEMINI_MODEL`
- Ollama: optional `OLLAMA_BASE_URL`, optional `OLLAMA_MODEL`

## Useful Commands

### Build the FAISS index

```bash
python3 rag/build_faiss_index.py
```

### Run quick TF-IDF retrieval checks

```bash
python3 rag/test_retrieval.py
```

### Run quick FAISS retrieval checks

```bash
python3 rag/test_faiss_retrieval.py
```

### Generate insights from the latest ranked holdings file

```bash
python3 rag/rag_insights.py --generator template
```

### Generate insights from a specific backtest output and save JSON

```bash
python3 rag/rag_insights.py --ranking-file xgboost_backtesting_outputs/ranking/ranked_holdings.csv --generator template --output-file xgboost_backtesting_outputs/insights/all_sectors_insights.json
```

## Output Shape

`rag_insights.py` prints or saves a JSON payload containing:

- `generated_at`
- `input_source`
- `generator`
- `index_dir`
- `insights`

Each insight includes:

- stock identity and ranking fields
- the retrieval query used
- the final generated paragraph
- extracted `key_points`
- extracted `risk_points`
- evidence items with title, date, source URL, snippet, and retrieval score

## Practical Notes

- `build_faiss_index.py` and `faiss_retriever.py` are written as script-first modules, so the documented workflow is to run them as files rather than import them as a package.
- The pipeline is ticker-filtered during retrieval, which helps keep evidence scoped to the ranked stock instead of the full corpus.
- `test_retrieval.py` is useful when you want a lightweight retrieval smoke test without FAISS.
- If you rebuild the index after changing `data/`, the contents of `rag_index/` should be treated as regenerated artifacts.
