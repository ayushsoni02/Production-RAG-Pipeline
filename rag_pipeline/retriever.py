"""
Layer 2 — HYBRID RETRIEVAL
BM25 (sparse) + Qdrant (dense) → Reciprocal Rank Fusion.
"""

import logging
import time

from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import BaseNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from rag_pipeline.config import (
    COLLECTION_NAME,
    EMBED_MODEL_ID,
    HYBRID_TOP_K,
    OPENAI_API_KEY,
    QDRANT_API_KEY,
    QDRANT_URL,
    NAIVE_TOP_K,
)

logger = logging.getLogger(__name__)


# ── Dense retriever from existing Qdrant collection ──────────────────

def _build_qdrant_retriever(top_k: int = HYBRID_TOP_K):
    """
    Build a VectorStoreIndex retriever that queries an
    *already-populated* Qdrant collection.
    """
    from llama_index.core import VectorStoreIndex

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
    )
    embed_model = OpenAIEmbedding(model=EMBED_MODEL_ID, api_key=OPENAI_API_KEY)

    # Build an index handle on the existing store (no re-ingestion)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    return index.as_retriever(similarity_top_k=top_k)


# ── Hybrid fusion retriever ─────────────────────────────────────────

def build_hybrid_retriever(
    bm25_retriever: BM25Retriever,
    top_k: int = HYBRID_TOP_K,
) -> QueryFusionRetriever:
    """
    Combine BM25 (sparse) + Qdrant (dense) via Reciprocal Rank Fusion.

    RRF formula per document:  score(doc) = Σ  1 / (60 + rank_i(doc))
    """
    dense_retriever = _build_qdrant_retriever(top_k=top_k)

    fusion_retriever = QueryFusionRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        mode="reciprocal_rerank",         # RRF
        similarity_top_k=top_k,
        num_queries=1,                    # no query expansion by default
    )
    logger.info(
        "Built hybrid fusion retriever  (dense top-%d + BM25 top-%d → RRF top-%d)",
        top_k, bm25_retriever._similarity_top_k, top_k,
    )
    return fusion_retriever


def build_naive_retriever(top_k: int = NAIVE_TOP_K):
    """
    Naive dense-only retriever for baseline comparison.
    Returns top-k results from Qdrant with no BM25 and no fusion.
    """
    retriever = _build_qdrant_retriever(top_k=top_k)
    logger.info("Built naive dense-only retriever (top-%d)", top_k)
    return retriever
