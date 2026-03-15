<div align="center">

#  Production-Grade RAG Pipeline

### Hybrid Retrieval · Cross-Encoder Reranking · RAGAS Evaluation

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.10.x-blueviolet)](https://llamaindex.ai)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC382D?logo=qdrant)](https://qdrant.tech)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai)](https://openai.com)
[![Cohere](https://img.shields.io/badge/Cohere-Rerank_v3-39594D)](https://cohere.com)

*A production-ready Retrieval-Augmented Generation system engineered for complex technical domains — dense legal filings, compliance handbooks, and technical documentation — with provable improvements in retrieval quality through rigorous evaluation.*

[Architecture](#-architecture) · [Quick Start](#-quick-start) · [How It Works](#-how-it-works) · [Evaluation](#-evaluation--metrics) · [UI](#-interactive-ui) · [API](#-api-reference)

</div>

---

##  What Problem Does This Solve?

**Naive RAG is broken for production use.** Simple vector-similarity search fails on complex documents because:

| Problem | Why It Happens | Impact |
|---------|---------------|--------|
| **Semantic gaps** | Dense embeddings miss keyword-critical queries (legal terms, part numbers) | Relevant chunks are never retrieved |
| **Precision dilution** | Top-k results include loosely-related chunks | LLM hallucinates from noisy context |
| **No measurability** | No framework to quantify retrieval quality | Can't prove the system works |

This pipeline solves all three by combining **hybrid retrieval** (dense + sparse), **cross-encoder reranking** (Cohere Rerank v3), and a **RAGAS evaluation harness** that quantifies the improvement at every layer.

### The Engineering Goal

> Demonstrate, with numbers, that Context Precision increases from **~0.60 (naive)** to **>0.80 (optimized)** through each architectural decision.

---

##  Architecture

The system is organized into **5 clearly separated layers**, each with a specific responsibility:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INTERACTIVE UI                               │
│   Query Engine  │  Ingestion  │  Evaluation  │  Settings            │
└────────┬────────┴──────┬──────┴──────┬───────┴──────────────────────┘
         │               │             │
         ▼               ▼             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FastAPI BACKEND                                │
│   POST /query  │  POST /ingest  │  GET /stats  │  POST /evaluate   │
└────────┬────────┴──────┬─────────┴──────┬───────┴──────────────────┘
         │               │                │
    ┌────▼────┐    ┌─────▼─────┐   ┌──────▼──────┐
    │ LAYER 4 │    │  LAYER 1  │   │  LAYER 5    │
    │ Generate │    │  Ingest   │   │  Evaluate   │
    └────┬────┘    └─────┬─────┘   └──────┬──────┘
         │               │                │
    ┌────▼────┐    ┌─────▼─────┐   ┌──────▼──────┐
    │ LAYER 3 │    │ Semantic   │   │   RAGAS     │
    │ Rerank  │    │ Chunker    │   │  Harness    │
    │ (Cohere)│    │ (p95)      │   │ (3 metrics) │
    └────┬────┘    └─────┬─────┘   └─────────────┘
         │               │
    ┌────▼────────────────▼────┐
    │       LAYER 2            │
    │   Hybrid Retrieval       │
    │                          │
    │  ┌──────────┐ ┌────────┐ │
    │  │  Qdrant  │ │  BM25  │ │
    │  │ (Dense)  │ │(Sparse)│ │
    │  └────┬─────┘ └───┬────┘ │
    │       └───┬───────┘      │
    │     RRF Fusion           │
    └──────────────────────────┘
```

### Layer-by-Layer Breakdown

| Layer | Component | What It Does |
|-------|-----------|-------------|
| **Layer 1 — Ingestion** | `SemanticSplitterNodeParser` | Splits documents at natural topic boundaries (percentile = 95) rather than fixed token windows. Embeds with `text-embedding-3-small` (1536-d) and stores in Qdrant. |
| **Layer 2 — Hybrid Retrieval** | `QueryFusionRetriever` | Fires dense (Qdrant) + sparse (BM25) retrievers in parallel, each returning top-20 candidates. Merges via **Reciprocal Rank Fusion**: `score(doc) = Σ 1/(60 + rank_i)` |
| **Layer 3 — Reranking** | `CohereRerank` (v3.0) | Cross-encoder jointly scores all 20 `(query, chunk)` pairs. Keeps only the top 5 — this is the **primary driver** of Context Precision gains. |
| **Layer 4 — Generation** | `GPT-4o-mini` | Receives the 5 highest-quality chunks + user query. Synthesizes a grounded answer with source citations. |
| **Layer 5 — Evaluation** | RAGAS | Generates 50 synthetic QA pairs (GPT-4o), then scores both naive and optimized pipelines on Faithfulness, Context Precision, and Answer Relevancy. |

---

## 🛠 Tech Stack

| Layer | Technology | Why This Choice |
|-------|-----------|----------------|
| **Framework** | LlamaIndex `0.10.x` | Production-grade abstractions for indexing, retrieval, and query engines with composable post-processors |
| **Vector DB** | Qdrant | Sub-millisecond similarity search, native filtering, horizontal scaling — runs locally via Docker or managed cloud |
| **Dense Embeddings** | OpenAI `text-embedding-3-small` | 1536-d vectors, strong semantic understanding at low cost (~$0.02/1M tokens) |
| **Sparse Retrieval** | BM25 | Lexical matching ensures keyword-critical queries (exact terms, IDs, acronyms) are never missed |
| **Fusion** | Reciprocal Rank Fusion | Rank-based merge is parameter-free and score-agnostic — works even when BM25 and dense scores are on different scales |
| **Reranker** | Cohere Rerank v3 | Cross-encoder architecture jointly attends to query and document, providing much higher precision than bi-encoder similarity |
| **LLM** | OpenAI GPT-4o-mini | Fast, cost-effective generation with strong grounding capabilities |
| **Evaluation** | RAGAS | Industry-standard framework for evaluating RAG pipelines with reference-free and reference-based metrics |
| **API** | FastAPI | Async-native, automatic OpenAPI docs, built-in validation via Pydantic |
| **Frontend** | Vanilla HTML/CSS/JS | Zero build step, no framework overhead — ships as a single `index.html` served by FastAPI |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker (for Qdrant)
- OpenAI API key
- Cohere API key

### 1. Clone & Install

```bash
git clone https://github.com/ayushsoni02/Production-RAG-Pipeline.git
cd Production-RAG-Pipeline
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
export COHERE_API_KEY="..."
```

### 3. Start Qdrant

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

### 4. Launch the Server

```bash
python -m uvicorn rag_pipeline.api.main:app --reload --port 8000
```

### 5. Open the UI

Navigate to **http://localhost:8000** — you'll see the 4-panel interactive dashboard.

### 6. Upload Documents & Query

1. Switch to the **Ingestion** panel → drag & drop your PDF/TXT/HTML/MD files
2. Switch to the **Query Engine** panel → type a question → hit **Run Query**
3. The pipeline trace shows each step with latency, and retrieved chunks are displayed with relevance scores

---

## ⚙️ How It Works

### End-to-End Flow

```
User uploads docs ──→ Semantic chunking ──→ Qdrant + BM25 index
                                                    │
User asks question ──→ Dense retriever (top-20) ────┤
                       BM25 retriever  (top-20) ────┤
                                                    ▼
                                           RRF Fusion (20 candidates)
                                                    │
                                                    ▼
                                         Cohere Rerank (top-5)
                                                    │
                                                    ▼
                                         GPT-4o-mini Generation
                                                    │
                                                    ▼
                                    Answer + Sources + Pipeline Trace
```

### Why Hybrid Retrieval?

Dense embeddings excel at **semantic similarity** but fail on **exact-match queries** (legal statute numbers, product IDs, acronyms). BM25 is the opposite — great at keyword matching, poor at semantic understanding.

By running both in parallel and merging via RRF, the system gets the best of both worlds:

- **Dense** catches semantically related chunks even when wording differs
- **BM25** catches keyword-critical matches that embeddings would rank poorly
- **RRF** is score-agnostic — it only cares about rank position, so it works even though BM25 and cosine scores are on completely different scales

### Why Cross-Encoder Reranking?

The initial retrievers use **bi-encoder** architecture — query and document are embedded independently, then compared. This is fast but imprecise.

Cohere Rerank v3 uses a **cross-encoder** — it jointly processes `(query, document)` pairs, allowing deep token-level attention. This is slower but dramatically more precise, which is why it's applied only to the top-20 candidates rather than the full corpus.

**This single step is the primary driver of Context Precision gains.**

### Why Semantic Chunking Over Fixed-Size?

Fixed-size chunking (e.g., 512 tokens) arbitrarily cuts text mid-paragraph or mid-thought. Semantic chunking uses embedding similarity between consecutive sentences to detect natural topic boundaries.

With `breakpoint_percentile_threshold=95`, the splitter only breaks when the semantic shift between adjacent sentences is in the top 5% — preserving coherent information units.

---

##  Evaluation & Metrics

### RAGAS Framework

The evaluation harness uses [RAGAS](https://docs.ragas.io/) to score both the **naive** (dense-only, no reranking) and **optimized** (hybrid + reranking) pipelines on three metrics:

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **Context Precision** | Are the retrieved chunks actually relevant to the question? | Directly measures retrieval quality — noise in context → hallucinations |
| **Faithfulness** | Is the generated answer grounded in the retrieved context? | Catches LLM hallucinations — every claim should be traceable to a source |
| **Answer Relevancy** | Does the answer actually address the question asked? | Catches tangential or partial answers |

### Target Metrics

| Metric | Naive RAG | Optimized | Target |
|--------|-----------|-----------|--------|
| Context Precision | ~0.60 | >0.80 | **≥ 0.80** |
| Faithfulness | ~0.71 | >0.88 | ≥ 0.85 |
| Answer Relevancy | ~0.73 | >0.85 | ≥ 0.85 |

### How the Golden Dataset Works

1. **GPT-4o** (stronger model) generates 50 synthetic QA pairs from document chunks
2. Both pipelines answer every question independently
3. **GPT-4o-mini** acts as the RAGAS critic (deliberately different from the generation LLM to avoid self-grading bias)
4. Metrics are computed and the delta analysis quantifies the value of each optimization

---

## Interactive UI

The frontend is a 4-panel single-page app with a dark glassmorphic design:

| Panel | Features |
|-------|----------|
| **⚡ Query Engine** | Textarea for questions, mode selector (Hybrid+Rerank / Naive), candidates-k slider (5–40), rerank top-n slider (1–10), pipeline trace with per-step latency, chunk cards with score bars, answer with inline citations |
| **📄 Ingestion** | Drag-and-drop file upload, indexed document list, collection stats (total chunks, documents, embedding dimensions), re-index and test-set generation buttons |
| **📊 Evaluation** | RAGAS metric cards with delta indicators, naive vs optimized comparison table, configurable sample count, live evaluation log |
| **⚙️ Settings** | Toggles for BM25/Reranker/Query Expansion, model dropdowns (LLM, embedding, reranker), chunking strategy selector |

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | `GET` | Serves the interactive UI |
| `/query` | `POST` | Run a query through the pipeline |
| `/ingest` | `POST` | Upload and ingest documents (multipart) |
| `/stats` | `GET` | Collection statistics |
| `/evaluate` | `POST` | Run RAGAS evaluation harness |
| `/generate-testset` | `POST` | Generate synthetic golden dataset |

### `POST /query`

```json
// Request
{
  "question": "What are the data retention requirements?",
  "mode": "optimized",   // or "naive"
  "top_k": 20,
  "rerank_n": 5
}

// Response
{
  "answer": "According to the compliance handbook...",
  "source_nodes": [
    {
      "rank": 1,
      "score": 0.9234,
      "text": "Section 4.2 specifies that...",
      "source": "compliance_handbook.pdf",
      "token_count": 187
    }
  ],
  "latency_ms": 1247.3,
  "steps": [
    { "name": "total_pipeline", "latency_ms": 1247.3 }
  ]
}
```

### `POST /ingest`

```bash
curl -X POST http://localhost:8000/ingest \
  -F "files=@document.pdf" \
  -F "files=@handbook.txt"
```

---

## 📁 Project Structure

```
Production-RAG-Pipeline/
├── data/                           # Place your documents here
├── rag_pipeline/
│   ├── config.py                   # All configuration & env vars
│   ├── ingestion.py                # Layer 1: Semantic chunking → Qdrant + BM25
│   ├── retriever.py                # Layer 2: Hybrid retrieval with RRF fusion
│   ├── query_engine.py             # Layer 3-4: Reranking + LLM generation
│   ├── evaluation/
│   │   ├── generate_testset.py     # Synthetic golden dataset (GPT-4o)
│   │   └── ragas_harness.py        # Naive vs Optimized RAGAS scoring
│   ├── api/
│   │   └── main.py                 # FastAPI server (5 endpoints)
│   └── ui/
│       └── index.html              # 4-panel interactive frontend
└── requirements.txt
```

---

## ⚡ Configuration

All configuration is centralized in [`config.py`](rag_pipeline/config.py) and read from environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *required* | OpenAI API key |
| `COHERE_API_KEY` | *required* | Cohere API key |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | `None` | Qdrant API key (optional for local) |
| `COLLECTION_NAME` | `rag_hybrid_prod` | Qdrant collection name |
| `EMBED_MODEL_ID` | `text-embedding-3-small` | Embedding model |
| `LLM_MODEL_ID` | `gpt-4o-mini` | Generation LLM |
| `SEMANTIC_THRESHOLD` | `95` | Percentile breakpoint for chunking |
| `HYBRID_TOP_K` | `20` | Candidates before reranking |
| `RERANK_TOP_N` | `5` | Final chunks sent to LLM |

---

## Why This Project?

Most RAG tutorials stop at "embed → retrieve → generate." This is fine for demos, but **production systems demand more:**

1. **Measurability** — You can't ship what you can't measure. RAGAS gives us a quantitative framework to prove each optimization actually helps.

2. **Hybrid retrieval** — Real-world queries are a mix of semantic intent and exact keyword matches. A single retriever can't handle both.

3. **Reranking** — The difference between "somewhat relevant" and "exactly what the user needs" is the difference between a useful system and a hallucination machine.

4. **Transparency** — The pipeline trace shows exactly what happened at every step, making debugging and optimization tractable.

This project exists to demonstrate that **production RAG is an engineering discipline** — not just an API call — and that each architectural decision can be validated with data.

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

<div align="center">


</div>
