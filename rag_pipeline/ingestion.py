"""
Layer 1 — INGESTION
Load documents → semantic chunking → embed → Qdrant + BM25 index.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import BaseNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from rag_pipeline.config import (
    COLLECTION_NAME,
    EMBED_MODEL_ID,
    OPENAI_API_KEY,
    QDRANT_API_KEY,
    QDRANT_URL,
    SEMANTIC_THRESHOLD,
    VECTOR_SIZE,
    DATA_DIR,
)

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────

def _get_qdrant_client() -> QdrantClient:
    """Create a Qdrant client with optional API key."""
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )


def _get_embed_model() -> OpenAIEmbedding:
    """Return the configured OpenAI embedding model."""
    return OpenAIEmbedding(
        model=EMBED_MODEL_ID,
        api_key=OPENAI_API_KEY,
    )


# ── Core ingestion ───────────────────────────────────────────────────

def load_documents(data_dir: str | None = None) -> list[Any]:
    """Load raw documents from disk using LlamaIndex SimpleDirectoryReader."""
    directory = data_dir or DATA_DIR
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
        logger.warning("Data directory %s was empty — created it.", directory)
        return []

    t0 = time.perf_counter()
    reader = SimpleDirectoryReader(
        input_dir=directory,
        recursive=True,
        required_exts=[".pdf", ".txt", ".html", ".md"],
    )
    documents = reader.load_data()
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        "Loaded %d document(s) from %s in %.1f ms",
        len(documents), directory, elapsed,
    )
    return documents


def semantic_chunk(documents: list[Any], embed_model: OpenAIEmbedding | None = None) -> list[BaseNode]:
    """Split documents into semantically-coherent chunks."""
    embed_model = embed_model or _get_embed_model()

    t0 = time.perf_counter()
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=SEMANTIC_THRESHOLD,
        embed_model=embed_model,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        "Semantic chunking produced %d nodes in %.1f ms", len(nodes), elapsed,
    )
    return nodes


def index_to_qdrant(
    nodes: list[BaseNode],
    embed_model: OpenAIEmbedding | None = None,
) -> VectorStoreIndex:
    """Embed nodes and persist to Qdrant."""
    embed_model = embed_model or _get_embed_model()
    client = _get_qdrant_client()

    t0 = time.perf_counter()

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        "Indexed %d nodes to Qdrant collection '%s' in %.1f ms",
        len(nodes), COLLECTION_NAME, elapsed,
    )
    return index


def build_bm25_index(nodes: list[BaseNode]):
    """Build an in-memory BM25 retriever over the same node set."""
    from llama_index.retrievers.bm25 import BM25Retriever

    t0 = time.perf_counter()
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=20,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("Built BM25 index over %d nodes in %.1f ms", len(nodes), elapsed)
    return bm25_retriever


# ── Full pipeline ────────────────────────────────────────────────────

def run_ingestion(data_dir: str | None = None) -> dict:
    """
    End-to-end ingestion: load → chunk → Qdrant + BM25.
    Returns dict with nodes, bm25_retriever, qdrant_index, and stats.
    """
    logger.info("=== Starting ingestion pipeline ===")
    overall_t0 = time.perf_counter()

    embed_model = _get_embed_model()

    # Step 1: load documents
    documents = load_documents(data_dir)
    if not documents:
        return {
            "nodes": [],
            "bm25_retriever": None,
            "qdrant_index": None,
            "chunks_created": 0,
            "documents_loaded": 0,
        }

    # Step 2: semantic chunking
    nodes = semantic_chunk(documents, embed_model)

    # Step 3: index to Qdrant
    qdrant_index = index_to_qdrant(nodes, embed_model)

    # Step 4: build BM25 index
    bm25_retriever = build_bm25_index(nodes)

    overall_elapsed = (time.perf_counter() - overall_t0) * 1000
    logger.info(
        "=== Ingestion complete: %d docs → %d chunks in %.1f ms ===",
        len(documents), len(nodes), overall_elapsed,
    )

    return {
        "nodes": nodes,
        "bm25_retriever": bm25_retriever,
        "qdrant_index": qdrant_index,
        "chunks_created": len(nodes),
        "documents_loaded": len(documents),
    }
