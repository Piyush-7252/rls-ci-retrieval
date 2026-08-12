"""
Embedding Lambda — Cohere Embed  (secondary model for A/B benchmarking)
=======================================================================
Drop-in replacement for lambdas/embedding/lambda_function.py (Titan).

Loaded by chunk_worker when EMBEDDING_PROVIDER=cohere.
Stores results in a separate index (e.g. semantic-objects-cohere) by
pointing INDEX_LAMBDA_ARN at an index-lambda with that index configured.

Key differences vs Titan
-------------------------
  API     : batch up to 96 texts per Bedrock call  (vs single-text Titan)
  Format  : {"texts":[...], "input_type":"search_document"}
  Response: {"embeddings":[[...], ...]}
  CI path : uses input_type="search_query" (explicit query/document distinction)
  Sparse  : same TF-based token weights as Titan

Telemetry
---------
  Emits identical [ChunkSummary] log fields as the Titan lambda so
  embedding_doc_summary.py works with --doc filtering unchanged.
  bedrock_calls = number of batch API calls (fewer than embed_requests due to batching).
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

INDEX_LAMBDA_ARN  = os.environ.get("INDEX_LAMBDA_ARN", "")  # unused when loaded by chunk_worker
EMBEDDING_MODEL   = os.environ.get("EMBEDDING_MODEL", "cohere.embed-v4:0")
EMBEDDING_DEBUG   = os.environ.get("EMBEDDING_DEBUG", "").lower() in ("1", "true", "yes")

_COHERE_BATCH_SIZE = int(os.environ.get("COHERE_BATCH_SIZE", "96"))
_MAX_INPUT_CHARS   = 25_000

_EMBED_MAX_RETRIES  = 8
_EMBED_BACKOFF_BASE = 2.0
_EMBED_BACKOFF_CAP  = 60.0
_THROTTLE_CODES = frozenset({
    "ThrottlingException", "TooManyRequestsException",
    "ServiceUnavailableException", "RequestLimitExceeded",
    "ModelErrorException",
})

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "as", "is", "was", "are", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "that", "this", "which", "who", "whom",
    "it", "its", "from", "not", "no", "also", "each", "based", "per",
})

_COLD_START: bool = True
_aws: dict = {}


def _get(service: str):
    if service not in _aws:
        import boto3
        _aws[service] = boto3.client(service)
    return _aws[service]


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry — identical field names as Titan lambda for embedding_doc_summary compat
# ─────────────────────────────────────────────────────────────────────────────

class _EmbedStats:
    def __init__(self) -> None:
        self.calls            = 0   # Bedrock batch API calls
        self.success          = 0
        self.throttles        = 0
        self.extra_attempts   = 0
        self.total_backoff_ms = 0
        self.latency_ms_samples: list[int] = []

    def record_success(self, attempt: int, latency_ms: int = 0) -> None:
        self.calls          += 1
        self.success        += 1
        self.extra_attempts += attempt
        if latency_ms > 0:
            self.latency_ms_samples.append(latency_ms)

    def record_call_failure(self, attempt: int) -> None:
        self.calls          += 1
        self.extra_attempts += attempt

    def record_throttle(self, backoff_ms: int) -> None:
        self.throttles        += 1
        self.total_backoff_ms += backoff_ms


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

    # Indexing is owned by chunk_worker (_run_chunk calls idx._process_document).
    # INDEX_LAMBDA_ARN is intentionally not invoked here to avoid double-indexing.
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CI path — uses search_query input type for accurate query-side matching
# ─────────────────────────────────────────────────────────────────────────────

def _process_ci(ci: dict) -> dict:
    text   = ci.get("knownCI", "") or ci["normalization"]["normalized_text"]
    tokens = ci["normalization"]["tokens"]
    stats  = _EmbedStats()
    vecs   = _embed_texts([text[:_MAX_INPUT_CHARS]], input_type="search_query",
                          stats=stats)
    return {
        **ci,
        "embedding": {
            "dense_vector":  vecs[0],
            "sparse_vector": _sparse(tokens),
            "model":         EMBEDDING_MODEL,
            "dimensions":    len(vecs[0]),
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

    # Collect all texts into ordered slots: chunk, heading, objects, sentences
    slot_texts:   list[str]                = [text[:_MAX_INPUT_CHARS]]
    heading_slot: int | None               = None
    obj_slots:    list[tuple[int, int]]    = []   # (slot_idx, obj_idx)
    sent_slots:   list[tuple[int, int, int]] = [] # (slot_idx, obj_idx, span_idx)

    if heading_text:
        heading_slot = len(slot_texts)
        slot_texts.append(heading_text[:_MAX_INPUT_CHARS])

    for obj_idx, obj in enumerate(objects):
        if obj.get("text") and not obj.get("embedding"):
            slot_idx = len(slot_texts)
            slot_texts.append(_object_text(obj)[:_MAX_INPUT_CHARS])
            obj_slots.append((slot_idx, obj_idx))
        for span_idx, span in enumerate(obj.get("display_spans", [])):
            if (span.get("type") == "sentence"
                    and span.get("text")
                    and not span.get("embedding")):
                slot_idx = len(slot_texts)
                slot_texts.append(_sentence_text(obj, span)[:_MAX_INPUT_CHARS])
                sent_slots.append((slot_idx, obj_idx, span_idx))

    n_embed_requests = len(slot_texts)
    n_objects        = len(obj_slots)
    n_sentences      = len(sent_slots)

    stats = _EmbedStats()
    t0    = time.monotonic()

    all_vecs = _embed_texts(slot_texts, input_type="search_document", stats=stats,
                            doc_id=doc_id, chunk_id=chunk_id, lambda_ctx=lambda_ctx)

    embedding_time = time.monotonic() - t0

    dense_vector         = all_vecs[0]
    sparse_vector        = _sparse(tokens)
    heading_dense_vector = all_vecs[heading_slot] if heading_slot is not None else []

    for slot_idx, obj_idx in obj_slots:
        objects[obj_idx]["embedding"] = all_vecs[slot_idx]
    for slot_idx, obj_idx, span_idx in sent_slots:
        objects[obj_idx]["display_spans"][span_idx]["embedding"] = all_vecs[slot_idx]

    # ── Telemetry — same [ChunkSummary] fields as Titan lambda ───────────────
    throttle_rate = f"{100*stats.throttles/max(stats.calls,1):.1f}%"
    avg_attempts  = f"{1 + stats.extra_attempts/max(stats.calls,1):.2f}"
    _lat = sorted(stats.latency_ms_samples)

    def _pct(p: float) -> int:
        if not _lat: return 0
        idx = (p / 100) * (len(_lat) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(_lat) - 1)
        return int(_lat[lo] + (_lat[hi] - _lat[lo]) * (idx - lo))

    queue_wait_ms   = int(chunk.get("_queue_wait_ms", -1))
    memory_limit_mb = int(chunk.get("_memory_limit_mb", 0)
                          or getattr(lambda_ctx, "memory_limit_in_mb", 0) or 0)

    logger.info(
        "[ChunkSummary] chunk=%s doc=%s objects=%d sentences=%d "
        "embed_requests=%d bedrock_calls=%d bedrock_success=%d "
        "bedrock_throttles=%d bedrock_extra_attempts=%d total_backoff_ms=%d "
        "avg_attempts_per_call=%s throttle_rate=%s embedding_time=%.3fs cold_start=%s "
        "bedrock_lat_avg_ms=%d bedrock_lat_p95_ms=%d bedrock_lat_max_ms=%d "
        "peak_inflight=%d max_workers=%d memory_limit_mb=%d queue_wait_ms=%d "
        "embedding_provider=cohere embedding_model=%s",
        chunk_id, doc_id, n_objects, n_sentences,
        n_embed_requests, stats.calls, stats.success,
        stats.throttles, stats.extra_attempts, stats.total_backoff_ms,
        avg_attempts, throttle_rate, embedding_time, cold_start,
        int(sum(_lat)/len(_lat)) if _lat else 0,
        _pct(95), _lat[-1] if _lat else 0,
        1, 1, memory_limit_mb, queue_wait_ms,
        EMBEDDING_MODEL,
    )

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
# Cohere batch embedding with retry / backoff
# ─────────────────────────────────────────────────────────────────────────────

def _embed_texts(
    texts: list[str],
    *,
    input_type: str,
    stats: _EmbedStats,
    doc_id: str = "",
    chunk_id: str = "",
    lambda_ctx: Any = None,
) -> list[list[float]]:
    """Embed all texts via Cohere, issuing batches of up to _COHERE_BATCH_SIZE."""
    result: list[list[float] | None] = [None] * len(texts)
    for batch_start in range(0, len(texts), _COHERE_BATCH_SIZE):
        batch = texts[batch_start: batch_start + _COHERE_BATCH_SIZE]
        vecs  = _embed_batch_with_retry(
            batch, input_type=input_type, stats=stats,
            doc_id=doc_id, chunk_id=chunk_id, lambda_ctx=lambda_ctx,
            batch_num=batch_start // _COHERE_BATCH_SIZE,
        )
        for i, v in enumerate(vecs):
            result[batch_start + i] = v
    return result  # type: ignore[return-value]


def _embed_batch_with_retry(
    texts: list[str],
    *,
    input_type: str,
    stats: _EmbedStats,
    doc_id: str,
    chunk_id: str,
    lambda_ctx: Any,
    batch_num: int,
) -> list[list[float]]:
    import botocore.exceptions

    payload   = json.dumps({
        "texts": texts,
        "input_type": input_type,
        "truncate": "END",
        "embedding_types": ["float"],  # required for embed-v4:0; ignored by v3
    }).encode()
    last_exc: Exception | None = None
    request_id = getattr(lambda_ctx, "aws_request_id", "") if lambda_ctx else ""

    for attempt in range(_EMBED_MAX_RETRIES):
        t0 = time.monotonic()
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            response   = _get("bedrock-runtime").invoke_model(
                modelId=EMBEDDING_MODEL, contentType="application/json",
                accept="application/json", body=payload,
            )
            latency_ms = int((time.monotonic() - t0) * 1000)
            body       = json.loads(response["body"].read())
            raw        = body.get("embeddings", [])
            if isinstance(raw, dict):          # cohere v4: {"embeddings": {"float": [...]}}
                raw = raw.get("float", [])
            stats.record_success(attempt, latency_ms=latency_ms)
            if EMBEDDING_DEBUG:
                logger.info(
                    "[CohereEmbedding] ts=%s doc=%s chunk=%s request_id=%s "
                    "batch=%d attempt=%d status=SUCCESS latency_ms=%d texts=%d",
                    ts, doc_id, chunk_id, request_id, batch_num, attempt + 1,
                    latency_ms, len(texts),
                )
            return [list(v) for v in raw]
        except botocore.exceptions.ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code not in _THROTTLE_CODES:
                stats.record_call_failure(attempt)
                raise
            last_exc = exc
            window   = min(_EMBED_BACKOFF_CAP, _EMBED_BACKOFF_BASE * (2 ** attempt))
            delay    = random.uniform(0, window)
            delay_ms = int(delay * 1000)
            stats.record_throttle(delay_ms)
            logger.warning(
                "[CohereEmbedding] ts=%s doc=%s chunk=%s batch=%d attempt=%d/%d "
                "status=THROTTLED error=%s retry_after_ms=%d",
                ts, doc_id, chunk_id, batch_num, attempt + 1, _EMBED_MAX_RETRIES,
                code, delay_ms,
            )
            time.sleep(delay)

    stats.record_call_failure(_EMBED_MAX_RETRIES)
    raise last_exc  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — identical to Titan lambda
# ─────────────────────────────────────────────────────────────────────────────

def _sparse(tokens: list[str]) -> dict[str, float]:
    filtered = [t for t in tokens if t not in _STOPWORDS and len(t) > 1] or tokens
    tf    = Counter(filtered)
    total = max(sum(tf.values()), 1)
    return {term: round(count / total, 6) for term, count in tf.items()}


def _object_text(obj: dict) -> str:
    parts: list[str] = []
    hp = obj.get("heading_path")
    if hp:
        parts.extend(hp if isinstance(hp, list) else [hp])
    elif obj.get("section"):
        parts.append(obj["section"])
    parts.append(obj["text"])
    return "\n\n".join(filter(None, parts))


def _sentence_text(obj: dict, span: dict) -> str:
    parts: list[str] = []
    hp = obj.get("heading_path")
    if hp:
        parts.extend(hp if isinstance(hp, list) else [hp])
    elif obj.get("section"):
        parts.append(obj["section"])
    parts.append(span["text"])
    return "\n\n".join(filter(None, parts))
