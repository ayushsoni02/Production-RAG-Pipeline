"""
Central configuration for the RAG pipeline.
All secrets read from environment variables — never hardcoded.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

# ── API keys ──────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
COHERE_API_KEY: str = os.environ.get("COHERE_API_KEY", "")

# ── Qdrant ────────────────────────────────────────────────────────────
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME: str = "rag_hybrid_prod"

# ── Embedding ─────────────────────────────────────────────────────────
EMBED_MODEL_ID: str = "text-embedding-3-small"
VECTOR_SIZE: int = 1536

# ── LLM ───────────────────────────────────────────────────────────────
LLM_MODEL_ID: str = "gpt-4o-mini"
GENERATOR_LLM_ID: str = "gpt-4o"          # stronger model for testset gen

# ── Semantic chunking ─────────────────────────────────────────────────
SEMANTIC_THRESHOLD: int = 95               # percentile breakpoint

# ── Retrieval ─────────────────────────────────────────────────────────
HYBRID_TOP_K: int = 20                     # candidates before reranking
RERANK_TOP_N: int = 5                      # final chunks sent to LLM
NAIVE_TOP_K: int = 3                       # naive mode: dense-only top-k

# ── Reranker ──────────────────────────────────────────────────────────
RERANK_MODEL: str = "rerank-english-v3.0"

# ── Paths ─────────────────────────────────────────────────────────────
DATA_DIR: str = os.path.join(os.path.dirname(__file__), "..", "data")
GOLDEN_DATASET_PATH: str = os.path.join(
    os.path.dirname(__file__), "evaluation", "golden_dataset.json"
)
