"""
Embedding Lambda  (Unified — CI and Document)
==============================================
Routes on ``event["source_type"]``:
  "ci"       → embeds ci["knownCI"] text (dense + sparse)
  "document" → embeds chunk text + heading text + per-object + per-sentence

Both paths use the same Bedrock Titan model, the same sparse TF computation,
and the same stopword list.

Fan-out
-------
  Both paths → INDEX_LAMBDA_ARN
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

INDEX_LAMBDA_ARN = os.environ.get("INDEX_LAMBDA_ARN", "")
EMBEDDING_MODEL  = os.environ.get("EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
EMBEDDING_MAX_WORKERS = max(1, int(os.environ.get("EMBEDDING_MAX_WORKERS", "1")))

# Titan Embed supports ~8 192 tokens; truncate at character level to be safe
_MAX_INPUT_CHARS = 25_000

# Retry config for Bedrock throttling (full-jitter exponential backoff)
_EMBED_MAX_RETRIES = 8       # outer retries on top of boto3's 4 built-in attempts
_EMBED_BACKOFF_BASE = 2.0   # seconds
_EMBED_BACKOFF_CAP  = 60.0  # maximum jitter window (seconds)
_THROTTLE_CODES = frozenset({
    "ThrottlingException", "TooManyRequestsException",
    "ServiceUnavailableException", "RequestLimitExceeded",
    "ModelErrorException",   # Bedrock transient 500 — retry recommended by AWS
})

# Clinical stopwords — carry no discriminative weight in sparse matching
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "as", "is", "was", "are", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "that", "this", "which", "who", "whom",
    "it", "its", "from", "not", "no", "also", "each", "based", "per",
})

# ─── lazy AWS clients ─────────────────────────────────────────────────────────
_aws: dict = {}

def _get(service: str):
    if service not in _aws:
        import boto3
        _aws[service] = boto3.client(service)
    return _aws[service]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    source_type = event.get("source_type", "document")

    if source_type == "ci":
        ci_id = event.get("id", "unknown")
        logger.info("[Embedding] start source=ci ci_id=%s model=%s", ci_id, EMBEDDING_MODEL)
        try:
            result = _process_ci(event)
        except Exception as exc:
            logger.error("[Embedding] failed source=ci ci_id=%s error=%s", ci_id, exc)
            raise
        logger.info("[Embedding] done source=ci ci_id=%s dimensions=%d",
                    ci_id, result["embedding"]["dimensions"])

    else:
        chunk_id = event.get("chunk_id", "unknown")
        logger.info("[Embedding] start source=document chunk_id=%s model=%s", chunk_id, EMBEDDING_MODEL)
        try:
            result = _process_document(event)
        except Exception as exc:
            logger.error("[Embedding] failed source=document chunk_id=%s error=%s", chunk_id, exc)
            raise
        logger.info("[Embedding] done source=document chunk_id=%s dimensions=%d",
                    chunk_id, result["embedding"]["dimensions"])

    if INDEX_LAMBDA_ARN:
        _get("lambda").invoke(
            FunctionName   = INDEX_LAMBDA_ARN,
            InvocationType = "Event",
            Payload        = json.dumps(result).encode(),
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CI path
# ─────────────────────────────────────────────────────────────────────────────

def _process_ci(ci: dict) -> dict:
    # Use original casing — preserves acronyms and drug names (RP2D, JNJ-64407564)
    original_text = ci.get("knownCI", "") or ci["normalization"]["normalized_text"]
    tokens        = ci["normalization"]["tokens"]
    dense_vector  = _generate_dense_embedding(original_text)
    sparse_vector = _generate_sparse_embedding(tokens)

    return {
        **ci,
        "embedding": {
            "dense_vector":  dense_vector,
            "sparse_vector": sparse_vector,
            "model":         EMBEDDING_MODEL,
            "dimensions":    len(dense_vector),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Document path
# ─────────────────────────────────────────────────────────────────────────────

def _process_document(chunk: dict) -> dict:
    text    = chunk["normalization"]["normalized_text"]
    tokens  = chunk["normalization"]["tokens"]
    heading_text = chunk.get("heading_embedding_text", "")
    sparse_vector = _generate_sparse_embedding(tokens)

    # Submit ALL embeddings (chunk + heading + objects + sentences) in parallel
    objects = chunk.get("extraction", {}).get("objects", [])

    # task tag: ("chunk"|"heading"|"obj"|"span", obj_idx, span_idx, future)
    tasks: list[tuple[str, int, int | None, Any]] = []
    with ThreadPoolExecutor(max_workers=EMBEDDING_MAX_WORKERS) as pool:
        chunk_fut   = pool.submit(_generate_dense_embedding, text)
        heading_fut = pool.submit(_generate_dense_embedding, heading_text) if heading_text else None

        for obj_idx, obj in enumerate(objects):
            if obj.get("text") and not obj.get("embedding"):
                fut = pool.submit(_generate_dense_embedding, _object_embedding_text(obj))
                tasks.append(("obj", obj_idx, None, fut))

            for span_idx, span in enumerate(obj.get("display_spans", [])):
                if span.get("type") == "sentence" and span.get("text") and not span.get("embedding"):
                    fut = pool.submit(
                        _generate_dense_embedding,
                        _sentence_embedding_text(obj, span),
                    )
                    tasks.append(("span", obj_idx, span_idx, fut))

        dense_vector         = chunk_fut.result()
        heading_dense_vector = heading_fut.result() if heading_fut else []

        for kind, obj_idx, span_idx, fut in tasks:
            embedding = fut.result()
            if kind == "obj":
                objects[obj_idx]["embedding"] = embedding
            else:
                assert span_idx is not None
                objects[obj_idx]["display_spans"][span_idx]["embedding"] = embedding

    if objects and "extraction" in chunk:
        chunk = {**chunk, "extraction": {**chunk["extraction"], "objects": objects}}

    return {
        **chunk,
        "embedding": {
            "dense_vector":         dense_vector,
            "heading_dense_vector": heading_dense_vector,
            "sparse_vector":        sparse_vector,
            "model":                EMBEDDING_MODEL,
            "dimensions":           len(dense_vector),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Shared embedding logic
# ─────────────────────────────────────────────────────────────────────────────

def _generate_dense_embedding(text: str) -> list[float]:
    import botocore.exceptions
    payload = json.dumps({"inputText": text[:_MAX_INPUT_CHARS]}).encode()
    last_exc: Exception | None = None
    for attempt in range(_EMBED_MAX_RETRIES):
        try:
            response = _get("bedrock-runtime").invoke_model(
                modelId     = EMBEDDING_MODEL,
                contentType = "application/json",
                accept      = "application/json",
                body        = payload,
            )
            return json.loads(response["body"].read())["embedding"]
        except botocore.exceptions.ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code not in _THROTTLE_CODES:
                raise
            last_exc = exc
            # Full-jitter exponential backoff (AWS recommended pattern)
            window = min(_EMBED_BACKOFF_CAP, _EMBED_BACKOFF_BASE * (2 ** attempt))
            delay  = random.uniform(0, window)
            logger.warning(
                "[Embedding] throttled attempt=%d/%d code=%s msg=%s retrying in %.1fs",
                attempt + 1, _EMBED_MAX_RETRIES, code,
                exc.response.get("Error", {}).get("Message", ""),
                delay,
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def _generate_sparse_embedding(tokens: list[str]) -> dict[str, float]:
    filtered = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
    if not filtered:
        filtered = tokens
    tf    = Counter(filtered)
    total = max(sum(tf.values()), 1)
    return {term: round(count / total, 6) for term, count in tf.items()}


def _object_embedding_text(obj: dict) -> str:
    """Heading breadcrumb + paragraph text — keeps paragraph vectors section-aware."""
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
