"""
Evaluation — RAGAS harness.
Compares Naive RAG vs Optimized RAG using RAGAS metrics.
Uses GPT-4o-mini as the RAGAS critic (separate from the generation LLM
to avoid the model grading its own outputs).
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    faithfulness,
)

from rag_pipeline.config import (
    GOLDEN_DATASET_PATH,
    LLM_MODEL_ID,
    OPENAI_API_KEY,
    EMBED_MODEL_ID,
)

logger = logging.getLogger(__name__)


def _load_golden_dataset(path: str | None = None) -> list[dict]:
    """Load the golden QA dataset from disk."""
    path = path or GOLDEN_DATASET_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Golden dataset not found at {path}. "
            "Run generate_testset.py first."
        )
    with open(path) as f:
        return json.load(f)


def _run_pipeline_on_dataset(engine, dataset: list[dict]) -> list[dict]:
    """
    Run a query engine against every question in the golden dataset.
    Returns list of dicts with keys expected by RAGAS.
    """
    results = []
    for i, qa in enumerate(dataset):
        question = qa["question"]
        try:
            response = engine.query(question)
            contexts = [
                node.node.get_content()
                for node in response.source_nodes
            ]
            results.append({
                "question": question,
                "answer": str(response),
                "contexts": contexts,
                "ground_truth": qa["ground_truth"],
            })
        except Exception as e:
            logger.warning("Query %d failed: %s", i, e)
            results.append({
                "question": question,
                "answer": "",
                "contexts": [],
                "ground_truth": qa["ground_truth"],
            })
        if (i + 1) % 10 == 0:
            logger.info("Processed %d/%d queries", i + 1, len(dataset))
    return results


def _evaluate_with_ragas(results: list[dict]) -> dict:
    """
    Score a set of results using RAGAS.
    Returns dict of metric_name → float score.
    """
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    # Build RAGAS-compatible Dataset
    ds = Dataset.from_dict({
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })

    # RAGAS critic uses GPT-4o-mini (separate from generation LLM)
    critic_llm = ChatOpenAI(
        model=LLM_MODEL_ID,
        openai_api_key=OPENAI_API_KEY,
    )
    critic_embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL_ID,
        openai_api_key=OPENAI_API_KEY,
    )

    scores = evaluate(
        dataset=ds,
        metrics=[faithfulness, context_precision, answer_relevancy],
        llm=critic_llm,
        embeddings=critic_embeddings,
    )

    return {k: round(v, 4) for k, v in scores.items() if isinstance(v, (int, float))}


def run_ragas_harness(
    naive_engine,
    optimized_engine,
    n_samples: int | None = None,
    golden_path: str | None = None,
) -> dict:
    """
    Full RAGAS comparison harness.

    Returns:
        {
          "naive":     { "faithfulness": ..., "context_precision": ..., ... },
          "optimized": { ... },
          "delta":     { ... },
        }
    """
    logger.info("=== Starting RAGAS evaluation harness ===")
    overall_t0 = time.perf_counter()

    dataset = _load_golden_dataset(golden_path)
    if n_samples and n_samples < len(dataset):
        dataset = dataset[:n_samples]

    logger.info("Loaded golden dataset with %d QA pairs", len(dataset))

    # ── Run naive pipeline ──
    logger.info("--- Running NAIVE pipeline ---")
    naive_results = _run_pipeline_on_dataset(naive_engine, dataset)
    naive_scores = _evaluate_with_ragas(naive_results)
    logger.info("Naive scores: %s", naive_scores)

    # ── Run optimized pipeline ──
    logger.info("--- Running OPTIMIZED pipeline ---")
    opt_results = _run_pipeline_on_dataset(optimized_engine, dataset)
    opt_scores = _evaluate_with_ragas(opt_results)
    logger.info("Optimized scores: %s", opt_scores)

    # ── Compute deltas ──
    delta = {}
    for key in opt_scores:
        if key in naive_scores:
            delta[key] = round(opt_scores[key] - naive_scores[key], 4)

    elapsed = (time.perf_counter() - overall_t0) * 1000
    logger.info("=== RAGAS harness complete in %.1f ms ===", elapsed)

    result = {
        "naive": naive_scores,
        "optimized": opt_scores,
        "delta": delta,
    }
    logger.info("Final comparison:\n%s", json.dumps(result, indent=2))
    return result
