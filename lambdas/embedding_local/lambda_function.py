"""
Local Embedding Module — BAAI/bge-m3
=====================================
Drop-in replacement for lambdas/embedding/lambda_function.py.

Uses BAAI/bge-m3 locally (CPU or GPU) for both dense + learned-sparse
embeddings.  All texts in a chunk are encoded in a single batched
model.encode() call — no Bedrock calls, no throttling.

Key differences vs Bedrock path
---------------------------------
  Dense  : BGE-M3 1024-dim cosine vectors  (same dims as Titan v2 default)
  Sparse : BGE-M3 learned lexical weights  (vs simple TF from Bedrock path)
           — higher quality; captures semantic importance, not just frequency
  Batched: all texts per chunk in one model call  → much faster locally

Interface (identical to lambdas/embedding/lambda_function.py)
--------------------------------------------------------------
  _process_document(chunk, *, lambda_ctx=None, cold_start=False) -> dict
  _process_ci(ci) -> dict
  handler(event, context) -> dict

Install
-------
  pip install FlagEmbedding
  # model weights (~1.5 GB) are downloaded on first use to HF_HOME or ~/.cache/huggingface
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_NAME       = os.environ.get("LOCAL_EMBEDDING_MODEL", "BAAI/bge-m3")
INDEX_LAMBDA_ARN = os.environ.get("INDEX_LAMBDA_ARN", "")

# Truncate input at character level before encoding (BGE-M3 max ≈ 8 192 tokens)
_MAX_INPUT_CHARS = 25_000

_model       = None
_model_lock  = threading.Lock()
_COLD_START: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Model loader (lazy, cached, thread-safe)
# ─────────────────────────────────────────────────────────────────────────────

def _get_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError:
            raise ImportError(
                "FlagEmbedding is not installed.\n"
                "Run:  pip install FlagEmbedding"
            )
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("[LocalEmbedding] Loading %s on %s …", MODEL_NAME, device)
        t0 = time.monotonic()
        _model = BGEM3FlagModel(
            MODEL_NAME,
            use_fp16=True,          # faster on both CPU and GPU
            device=device,
        )
        logger.info("[LocalEmbedding] Model loaded in %.1fs", time.monotonic() - t0)
    return _model


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    global _COLD_START
    cold_start  = _COLD_START
    _COLD_START = False

    source_type = event.get("source_type", "document")

    if source_type == "ci":
        result = _process_ci(event)
    else:
        result = _process_document(event, lambda_ctx=context, cold_start=cold_start)

    if INDEX_LAMBDA_ARN:
        import boto3
        boto3.client("lambda").invoke(
            FunctionName   = INDEX_LAMBDA_ARN,
            InvocationType = "Event",
            Payload        = json.dumps(result).encode(),
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CI path
# ─────────────────────────────────────────────────────────────────────────────

def _process_ci(ci: dict) -> dict:
    text   = (ci.get("knownCI", "") or ci["normalization"]["normalized_text"])[:_MAX_INPUT_CHARS]
    tokens = ci["normalization"]["tokens"]

    model  = _get_model()
    output = model.encode([text], return_dense=True, return_sparse=True,
                          return_colbert_vecs=False)

    dense_vector  = output["dense_vecs"][0].tolist()
    sparse_vector = _lexical_weights_to_dict(output["lexical_weights"][0])

    return {
        **ci,
        "embedding": {
            "dense_vector":  dense_vector,
            "sparse_vector": sparse_vector,
            "model":         MODEL_NAME,
            "dimensions":    len(dense_vector),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Document path
# ─────────────────────────────────────────────────────────────────────────────

def _process_document(
    chunk: dict,
    *,
    lambda_ctx: Any = None,
    cold_start: bool = False,
) -> dict:
    text         = chunk["normalization"]["normalized_text"]
    tokens       = chunk["normalization"]["tokens"]
    heading_text = chunk.get("heading_embedding_text", "")
    objects      = chunk.get("extraction", {}).get("objects", [])
    doc_id       = chunk.get("document_id", "")
    chunk_id     = chunk.get("chunk_id", "")

    t0 = time.monotonic()
    model = _get_model()

    # ── Collect all texts to encode in one batch ──────────────────────────────
    # Slot 0: chunk text  (always — gives dense + sparse)
    # Slot 1: heading text (if present — dense only assigned later)
    # Slot 2…N: object texts
    # Slot N+1…: sentence span texts

    slot_texts: list[str] = [text[:_MAX_INPUT_CHARS]]
    heading_slot: int | None = None
    if heading_text:
        heading_slot = len(slot_texts)
        slot_texts.append(heading_text[:_MAX_INPUT_CHARS])

    obj_slots:  list[tuple[int, int]]       = []   # (slot_idx, obj_idx)
    sent_slots: list[tuple[int, int, int]]  = []   # (slot_idx, obj_idx, span_idx)

    for obj_idx, obj in enumerate(objects):
        if obj.get("text") and not obj.get("embedding"):
            slot_idx = len(slot_texts)
            slot_texts.append(_object_embedding_text(obj)[:_MAX_INPUT_CHARS])
            obj_slots.append((slot_idx, obj_idx))

        for span_idx, span in enumerate(obj.get("display_spans", [])):
            if (span.get("type") == "sentence"
                    and span.get("text")
                    and not span.get("embedding")):
                slot_idx = len(slot_texts)
                slot_texts.append(_sentence_embedding_text(obj, span)[:_MAX_INPUT_CHARS])
                sent_slots.append((slot_idx, obj_idx, span_idx))

    n_embed_requests = len(slot_texts)

    # ── Single batched encode ─────────────────────────────────────────────────
    output = model.encode(
        slot_texts,
        batch_size=32,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    dense_vecs      = output["dense_vecs"]        # ndarray (n, 1024)
    lexical_weights = output["lexical_weights"]   # list of {term: weight}

    # ── Assign results ────────────────────────────────────────────────────────
    dense_vector         = dense_vecs[0].tolist()
    sparse_vector        = _lexical_weights_to_dict(lexical_weights[0])
    heading_dense_vector = dense_vecs[heading_slot].tolist() if heading_slot is not None else []

    for slot_idx, obj_idx in obj_slots:
        objects[obj_idx]["embedding"] = dense_vecs[slot_idx].tolist()

    for slot_idx, obj_idx, span_idx in sent_slots:
        objects[obj_idx]["display_spans"][span_idx]["embedding"] = dense_vecs[slot_idx].tolist()

    embedding_time = time.monotonic() - t0
    logger.info(
        "[ChunkSummary] chunk=%s doc=%s objects=%d sentences=%d "
        "embed_requests=%d model=%s embedding_time=%.3fs cold_start=%s",
        chunk_id, doc_id, len(obj_slots), len(sent_slots),
        n_embed_requests, MODEL_NAME, embedding_time, cold_start,
    )

    if objects and "extraction" in chunk:
        chunk = {**chunk, "extraction": {**chunk["extraction"], "objects": objects}}

    return {
        **chunk,
        "embedding": {
            "dense_vector":         dense_vector,
            "heading_dense_vector": heading_dense_vector,
            "sparse_vector":        sparse_vector,
            "model":                MODEL_NAME,
            "dimensions":           len(dense_vector),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lexical_weights_to_dict(lw: Any) -> dict[str, float]:
    """Convert BGE-M3 lexical_weights output to {term: float} dict."""
    if isinstance(lw, dict):
        return {str(k): round(float(v), 6) for k, v in lw.items() if float(v) > 0}
    # FlagEmbedding sometimes returns a defaultdict or similar
    return {str(k): round(float(v), 6) for k, v in dict(lw).items() if float(v) > 0}


def _object_embedding_text(obj: dict) -> str:
    """Heading breadcrumb + object text — keeps vectors section-aware."""
    parts: list[str] = []
    heading_path = obj.get("heading_path")
    if heading_path:
        parts.extend(heading_path if isinstance(heading_path, list) else [heading_path])
    elif obj.get("section"):
        parts.append(obj["section"])
    parts.append(obj["text"])
    return "\n\n".join(filter(None, parts))


def _sentence_embedding_text(obj: dict, span: dict) -> str:
    """Heading breadcrumb + sentence text — section-aware sentence vectors."""
    parts: list[str] = []
    heading_path = obj.get("heading_path")
    if heading_path:
        parts.extend(heading_path if isinstance(heading_path, list) else [heading_path])
    elif obj.get("section"):
        parts.append(obj["section"])
    parts.append(span["text"])
    return "\n\n".join(filter(None, parts))
