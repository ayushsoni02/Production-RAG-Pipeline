"""
Evaluation — Synthetic golden dataset generator.
Uses a stronger LLM (GPT-4o) to generate QA pairs from document chunks.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from rag_pipeline.config import (
    DATA_DIR,
    EMBED_MODEL_ID,
    GENERATOR_LLM_ID,
    GOLDEN_DATASET_PATH,
    OPENAI_API_KEY,
    SEMANTIC_THRESHOLD,
)

logger = logging.getLogger(__name__)

# ── Prompt template for QA pair generation ────────────────────────────

QA_GENERATION_PROMPT = """You are an expert question-answer generator.
Given the following text chunk from a document, generate a question and
its corresponding answer. The question should be specific, non-trivial,
and answerable solely from the provided text.

TEXT CHUNK:
{chunk_text}

Respond in valid JSON with exactly two keys:
{{"question": "...", "answer": "..."}}
"""


def _generate_qa_from_chunk(llm: OpenAI, chunk_text: str) -> dict | None:
    """Generate a single QA pair from a text chunk using the LLM."""
    try:
        prompt = QA_GENERATION_PROMPT.format(chunk_text=chunk_text[:3000])
        response = llm.complete(prompt)
        text = str(response).strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        return json.loads(text)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Failed to parse QA from chunk: %s", e)
        return None


def generate_golden_dataset(
    n_samples: int = 50,
    data_dir: str | None = None,
    output_path: str | None = None,
) -> list[dict]:
    """
    Generate a synthetic golden dataset of QA pairs.

    Steps:
      1. Load documents and chunk them
      2. Sample n_samples chunks
      3. Use GPT-4o to generate a QA pair per chunk
      4. Save to golden_dataset.json
    """
    output_path = output_path or GOLDEN_DATASET_PATH
    data_directory = data_dir or DATA_DIR

    logger.info("=== Generating golden dataset (%d samples) ===", n_samples)
    t0 = time.perf_counter()

    # Load and chunk
    reader = SimpleDirectoryReader(
        input_dir=data_directory,
        recursive=True,
        required_exts=[".pdf", ".txt", ".html", ".md"],
    )
    documents = reader.load_data()

    embed_model = OpenAIEmbedding(model=EMBED_MODEL_ID, api_key=OPENAI_API_KEY)
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=SEMANTIC_THRESHOLD,
        embed_model=embed_model,
    )
    nodes = splitter.get_nodes_from_documents(documents)

    if not nodes:
        logger.warning("No nodes found — cannot generate testset.")
        return []

    # Sample chunks (evenly spaced if we have more than n_samples)
    import random
    random.seed(42)
    sample_nodes = random.sample(nodes, min(n_samples, len(nodes)))

    # Generate QA pairs using GPT-4o (stronger model)
    llm = OpenAI(model=GENERATOR_LLM_ID, api_key=OPENAI_API_KEY)

    dataset: list[dict] = []
    for i, node in enumerate(sample_nodes):
        chunk_text = node.get_content()
        if len(chunk_text.strip()) < 50:
            continue

        qa = _generate_qa_from_chunk(llm, chunk_text)
        if qa and "question" in qa and "answer" in qa:
            dataset.append({
                "question": qa["question"],
                "ground_truth": qa["answer"],
                "context": chunk_text,
                "source": node.metadata.get("file_name", "unknown"),
            })
            logger.info("Generated QA %d/%d", len(dataset), n_samples)

        if len(dataset) >= n_samples:
            break

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        "=== Golden dataset: %d QA pairs saved to %s in %.1f ms ===",
        len(dataset), output_path, elapsed,
    )
    return dataset
