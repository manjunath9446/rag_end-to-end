# Problem 1: Cost-Efficient RAG Application

## Overview

This repository contains a complete, production-minded
Retrieval-Augmented Generation (RAG) service. It is designed to prove
that a **self-hosted, low-cost vector architecture** is a credible and
highly performant alternative to expensive, fully-managed vector
databases that charge for always-on pods.

The system ingests PDF, HTML, and Markdown documents, chunks and embeds
them into a local ChromaDB instance, and uses an LLM (via Groq) to
generate answers with strict, verifiable citations.

## Architecture & Data Flow

1.  **Ingestion**: Files are parsed (with an OCR fallback for scanned
    PDFs). Text is chunked using a sentence-aware splitter (default: 512
    chars, 64 overlap).
2.  **Idempotency**: Document IDs are generated using SHA-256 hashes of
    the filename and content. Re-ingesting an identical file triggers a
    `delete_by_doc` followed by an `upsert`, preventing duplicate
    vectors.
3.  **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384
    dimensions). Vectors are L2-normalized.
4.  **Storage**: **ChromaDB** (embedded, persistent) using cosine
    similarity. Supports exact-match metadata filtering.
5.  **Retrieval**: Top-k chunks are retrieved via semantic search.
6.  **Generation**: `openai/gpt-oss-120b` via Groq (OpenAI-compatible
    API) generates an answer.
7.  **Citations & Abstention**: The LLM is forced to cite `[chunk_id]`.
    The backend verifies these IDs against the retrieved set, stripping
    hallucinations. If the context is insufficient, the system strictly
    abstains.

## Technology Choices & Justification

-   **Vector Store (ChromaDB)**: Chosen because it can run embedded
    locally, eliminating network latency and per-vector storage costs.
    It supports HNSW indexing out of the box and persists to disk.
-   **Embeddings (MiniLM)**: A free, fast, and highly capable
    384-dimensional model that runs efficiently on CPU without requiring
    a GPU.
-   **LLM (Groq)**: Groq provides blazing-fast inference for open-weight
    models via an OpenAI-compatible API, making it perfect for a
    cost-efficient stack.

## Setup & Configuration

1.  Create and activate a virtual environment:

    ``` bash
    python -m venv .venv
    source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
    ```

2.  Install dependencies:

    ``` bash
    pip install -r requirements.txt
    pip install -e .
    ```

3.  Configure environment variables in a `.env` file:

    ``` env
    RAG_DATA_DIR=./data
    RAG_COLLECTION=documents
    RAG_CHUNK_SIZE=512
    RAG_CHUNK_OVERLAP=64
    RAG_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
    RAG_TOP_K=5

    # LLM Config (Groq)
    RAG_GROK_API_KEY=gsk_your_groq_key_here
    RAG_GROK_BASE_URL=https://api.groq.com/openai/v1
    RAG_GROK_MODEL=openai/gpt-oss-120b
    ```

## How to Run

### 1. The Streamlit UI (Recommended)

Launch the visual interface to batch-upload documents and chat with the
corpus:

``` bash
streamlit run app/streamlit_app.py
```

### 2. The FastAPI Service

Start the REST API:

``` bash
uvicorn app.api:app --reload --port 8000
```

**Endpoints:** - `POST /upload`: Upload a PDF, MD, or HTML file. -
`POST /retrieve`:
`{"query": "How much is the Growth plan?", "top_k": 3}` - `POST /ask`:
`{"question": "What is the SLA?"}`

### 3. The CLI

``` bash
python -m app.cli upload data/samples/acme-docs.md
python -m app.cli ask "What encryption is used at rest?"
```

------------------------------------------------------------------------

## Evaluation Methodology & Results

We evaluated the system using a fixed dataset (`data/eval_dataset.json`)
containing **20 questions**: 15 answerable questions (with gold snippets
to identify relevant chunks) and 5 unanswerable "trap" questions to test
abstention.

To reproduce these numbers, run:

``` bash
python -m app.evaluate
```

### Retrieval Metrics (Answerable Questions Only)

-   **Hit Rate@5**: `1.000` (A relevant chunk was retrieved in the top 5
    for 100% of questions)
-   **MRR@5**: `0.856` (The correct chunk was usually ranked #1)
-   **nDCG@5**: `0.893`
-   **Context Precision**: `0.34` (Because we retrieve 5 chunks, and
    usually only 1-2 contain the specific answer).

### Answer Metrics (All Questions)

-   **Abstention Accuracy**: `1.000` (The system perfectly answered all
    15 real questions and correctly refused all 5 hallucination traps).
-   **Hallucinations**: `0`

### Performance

-   **p50 Latency**: `1035.2 ms` (Retrieval + LLM Generation)
-   **p95 Latency**: `8350.5 ms` (Reflects LLM API generation variance)
-   **Token Usage**: \~700 tokens per query.

------------------------------------------------------------------------

## Cost Analysis

**Assumptions:** - 384-dim float32 vector + metadata ≈ 2KB per vector. -
Managed DB pricing reflects a typical Serverless/Storage model (\~\$25
base + \$0.10/GB). - Self-hosted reflects a standard Cloud VM (8GB RAM
for \<1M, 32GB RAM for 10M to hold the HNSW index in memory). - LLM
inference is billed at \~\$0.40 per 1,000 queries.

  --------------------------------------------------------------------------
  Vectors   Size (GB)  Managed DB/mo Self-Hosted VM/mo LLM Cost (1M queries)
  --------- ---------- ------------- ----------------- ---------------------
  100k      0.19 GB    \$25.02       \$40.00           \$400.00

  1M        1.91 GB    \$25.23       \$40.00           \$400.00

  10M       19.07 GB   \$27.29       \$240.00          \$400.00
  --------------------------------------------------------------------------

**Takeaway**: LLM inference dominates the cost at low-to-medium scales.
However, if you compare this to an *always-on managed pod* (which
requires dedicated RAM, e.g., \$300+/mo at 10M vectors), self-hosting
becomes significantly cheaper at scale.

------------------------------------------------------------------------

## Discussion & Trade-offs

### Was retrieval or generation the weak link?

**Generation** is the weak link, specifically regarding latency and
cost. - **Retrieval** is practically perfect (Hit Rate 100%, MRR 0.856)
and instantaneous (\<50ms). ChromaDB + MiniLM is an incredibly robust
combination. - **Generation** incurred a p95 latency of 8.3 seconds and
will cost \$400/month at 1M queries. This proves the premise of the
assignment: the vector DB is *not* the bottleneck; the LLM is.

### When would you switch back to a managed DB?

We would switch back to a managed vector DB (like Pinecone or Qdrant
Cloud) if: 1. **Scale exceeds 10M vectors**: Managing an HNSW index on a
single VM becomes an operational burden (index rebuilds, memory limits).
2. **High write velocity**: If documents are being ingested 24/7 at a
high rate, SQLite/ChromaDB's write locks become a bottleneck compared to
a distributed managed system. 3. **Multi-region needs**: If the
application requires active-active replication across continents.

### Known Limitations

-   **Scanned PDFs**: Handled via Tesseract OCR, which adds \~2 seconds
    per page.
-   **No Cross-Encoder Reranker**: We rely solely on bi-encoder cosine
    similarity. Adding a cross-encoder (e.g., `ms-marco-MiniLM`) would
    improve nDCG but double retrieval latency.
-   **Strict Abstention**: Because the LLM is strictly prompted to
    abstain if the context is insufficient, occasionally a very broad
    question might trigger a false abstention if the exact keyword isn't
    retrieved.
