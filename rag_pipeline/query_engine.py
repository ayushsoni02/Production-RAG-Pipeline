"""
Layer 3+4 — RERANKING & GENERATION
CohereRerank post-processor → GPT-4o-mini answer synthesis.
"""

import logging
import time

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.openai import OpenAI
from llama_index.postprocessor.cohere_rerank import CohereRerank

from rag_pipeline.config import (
    COHERE_API_KEY,
    LLM_MODEL_ID,
    OPENAI_API_KEY,
    RERANK_MODEL,
    RERANK_TOP_N,
)

logger = logging.getLogger(__name__)


def _build_llm():
    """Instantiate the generation LLM."""
    return OpenAI(model=LLM_MODEL_ID, api_key=OPENAI_API_KEY)


def _build_reranker(top_n: int = RERANK_TOP_N) -> CohereRerank:
    """Instantiate Cohere cross-encoder reranker."""
    return CohereRerank(
        api_key=COHERE_API_KEY,
        model=RERANK_MODEL,
        top_n=top_n,
    )


def build_query_engine(
    retriever,
    *,
    use_reranker: bool = True,
    rerank_top_n: int = RERANK_TOP_N,
) -> RetrieverQueryEngine:
    """
    Build a full query engine:
      retriever → (optional) CohereRerank → GPT-4o-mini synthesis.
    """
    llm = _build_llm()
    synth = get_response_synthesizer(llm=llm, response_mode="compact")

    node_postprocessors = []
    if use_reranker:
        node_postprocessors.append(_build_reranker(top_n=rerank_top_n))

    engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synth,
        node_postprocessors=node_postprocessors,
    )
    label = "optimized (reranker ON)" if use_reranker else "naive (reranker OFF)"
    logger.info("Built %s query engine", label)
    return engine


def query_with_trace(engine: RetrieverQueryEngine, question: str) -> dict:
    """
    Run a query and return a detailed trace dict:
      { answer, source_nodes, latency_ms, steps[] }
    """
    steps: list[dict] = []

    # ── Retrieve ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    response = engine.query(question)
    total_ms = (time.perf_counter() - t0) * 1000

    steps.append({"name": "total_pipeline", "latency_ms": round(total_ms, 1)})

    # ── Source nodes ──────────────────────────────────────────────
    source_nodes = []
    for i, node in enumerate(response.source_nodes):
        source_nodes.append({
            "rank": i + 1,
            "score": round(node.score, 4) if node.score else None,
            "text": node.node.get_content()[:500],
            "source": node.node.metadata.get("file_name", "unknown"),
            "token_count": len(node.node.get_content().split()),
        })

    return {
        "answer": str(response),
        "source_nodes": source_nodes,
        "latency_ms": round(total_ms, 1),
        "steps": steps,
    }
