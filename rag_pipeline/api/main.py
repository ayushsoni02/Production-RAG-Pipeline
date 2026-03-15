"""
FastAPI server — production RAG pipeline endpoints.

Endpoints:
  POST /query     → question + mode → answer, sources, latency
  POST /ingest    → file upload → chunk count, status
  GET  /stats     → collection stats
  POST /evaluate  → RAGAS harness → naive vs optimized scores
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_pipeline.config import (
    COLLECTION_NAME,
    DATA_DIR,
    HYBRID_TOP_K,
    RERANK_TOP_N,
    QDRANT_URL,
    QDRANT_API_KEY,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Global pipeline state ─────────────────────────────────────────────

_state: dict[str, Any] = {
    "nodes": [],
    "bm25_retriever": None,
    "qdrant_index": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: try to initialise the pipeline from existing Qdrant data."""
    logger.info("FastAPI starting — checking existing Qdrant collection…")
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        info = client.get_collection(COLLECTION_NAME)
        logger.info(
            "Found existing collection '%s' with %d points",
            COLLECTION_NAME, info.points_count,
        )
    except Exception:
        logger.info("No existing Qdrant collection — ingest documents first.")
    yield
    logger.info("FastAPI shutting down")


# ── App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Production RAG Pipeline",
    description="Hybrid retrieval + cross-encoder reranking + RAGAS evaluation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the UI
UI_DIR = os.path.join(os.path.dirname(__file__), "..", "ui")


@app.get("/")
async def serve_ui():
    """Serve the interactive frontend."""
    index_path = os.path.join(UI_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "UI not found — place index.html in rag_pipeline/ui/"}


# ── Request / Response models ────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    mode: str = Field(default="optimized", pattern="^(optimized|naive)$")
    top_k: int = Field(default=HYBRID_TOP_K, ge=5, le=40)
    rerank_n: int = Field(default=RERANK_TOP_N, ge=1, le=10)


class QueryResponse(BaseModel):
    answer: str
    source_nodes: list[dict]
    latency_ms: float
    steps: list[dict] = []


class IngestResponse(BaseModel):
    chunks_created: int
    documents_loaded: int
    status: str


class StatsResponse(BaseModel):
    total_chunks: int
    documents: int
    collection: str


class EvalRequest(BaseModel):
    n_samples: int = Field(default=20, ge=1, le=100)


class EvalResponse(BaseModel):
    naive: dict
    optimized: dict
    delta: dict


# ── POST /query ──────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """Run a query through the RAG pipeline."""
    from rag_pipeline.retriever import build_hybrid_retriever, build_naive_retriever
    from rag_pipeline.query_engine import build_query_engine, query_with_trace

    if _state["bm25_retriever"] is None and req.mode == "optimized":
        # Try to build from existing nodes if we have them
        if not _state["nodes"]:
            raise HTTPException(
                status_code=400,
                detail="No documents ingested yet. Upload documents first.",
            )

    t0 = time.perf_counter()

    if req.mode == "optimized":
        retriever = build_hybrid_retriever(
            _state["bm25_retriever"],
            top_k=req.top_k,
        )
        engine = build_query_engine(
            retriever,
            use_reranker=True,
            rerank_top_n=req.rerank_n,
        )
    else:
        retriever = build_naive_retriever(top_k=req.top_k)
        engine = build_query_engine(retriever, use_reranker=False)

    result = query_with_trace(engine, req.question)
    return QueryResponse(**result)


# ── POST /ingest ─────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(files: list[UploadFile] = File(...)):
    """Upload and ingest documents."""
    from rag_pipeline.ingestion import run_ingestion

    os.makedirs(DATA_DIR, exist_ok=True)

    # Save uploaded files to data dir
    saved_paths = []
    for f in files:
        dest = os.path.join(DATA_DIR, f.filename)
        with open(dest, "wb") as out:
            content = await f.read()
            out.write(content)
        saved_paths.append(dest)
        logger.info("Saved uploaded file: %s", dest)

    # Run full ingestion
    result = run_ingestion(DATA_DIR)

    # Update global state
    _state["nodes"] = result["nodes"]
    _state["bm25_retriever"] = result["bm25_retriever"]
    _state["qdrant_index"] = result["qdrant_index"]

    return IngestResponse(
        chunks_created=result["chunks_created"],
        documents_loaded=result["documents_loaded"],
        status="success",
    )


# ── GET /stats ───────────────────────────────────────────────────────

@app.get("/stats", response_model=StatsResponse)
async def stats_endpoint():
    """Return collection statistics."""
    total_chunks = 0
    documents = 0

    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        info = client.get_collection(COLLECTION_NAME)
        total_chunks = info.points_count
    except Exception:
        pass

    # Count files in data dir
    if os.path.isdir(DATA_DIR):
        documents = len([
            f for f in os.listdir(DATA_DIR)
            if os.path.isfile(os.path.join(DATA_DIR, f))
        ])

    return StatsResponse(
        total_chunks=total_chunks,
        documents=documents,
        collection=COLLECTION_NAME,
    )


# ── POST /evaluate ───────────────────────────────────────────────────

@app.post("/evaluate", response_model=EvalResponse)
async def evaluate_endpoint(req: EvalRequest):
    """Run RAGAS evaluation harness."""
    from rag_pipeline.retriever import build_hybrid_retriever, build_naive_retriever
    from rag_pipeline.query_engine import build_query_engine
    from rag_pipeline.evaluation.ragas_harness import run_ragas_harness

    if not _state["bm25_retriever"] or not _state["nodes"]:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested. Ingest documents first.",
        )

    # Build both engines
    hybrid_ret = build_hybrid_retriever(_state["bm25_retriever"])
    optimized_engine = build_query_engine(hybrid_ret, use_reranker=True)

    naive_ret = build_naive_retriever()
    naive_engine = build_query_engine(naive_ret, use_reranker=False)

    result = run_ragas_harness(
        naive_engine=naive_engine,
        optimized_engine=optimized_engine,
        n_samples=req.n_samples,
    )
    return EvalResponse(**result)


# ── POST /generate-testset ───────────────────────────────────────────

@app.post("/generate-testset")
async def generate_testset_endpoint(n_samples: int = 50):
    """Generate a synthetic golden dataset."""
    from rag_pipeline.evaluation.generate_testset import generate_golden_dataset

    dataset = generate_golden_dataset(n_samples=n_samples)
    return {
        "status": "success",
        "samples_generated": len(dataset),
    }
