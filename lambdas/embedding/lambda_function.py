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
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

INDEX_LAMBDA_ARN = os.environ.get("INDEX_LAMBDA_ARN", "")
EMBEDDING_MODEL  = os.environ.get("EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
EMBEDDING_MAX_WORKERS = max(1, int(os.environ.get("EMBEDDING_MAX_WORKERS", "1")))
# Set EMBEDDING_DEBUG=true to log every successful Bedrock call.
# Off by default — at 27k chunks × 150 embeddings that's 4M+ log lines.
EMBEDDING_DEBUG = os.environ.get("EMBEDDING_DEBUG", "").lower() in ("1", "true", "yes")

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

# Cold-start flag — True only for the first invocation in this execution environment.
# Reset to False after the first handler() call so subsequent warm invocations are
# distinguishable in logs without any external tooling.
_COLD_START: bool = True

# Clinical stopwords — carry no discriminative weight in sparse matching
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "as", "is", "was", "are", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "that", "this", "which", "who", "whom",
    "it", "its", "from", "not", "no", "also", "each", "based", "per",
})

# ─────────────────────────────────────────────────────────────────────────────
# Per-chunk Bedrock telemetry accumulator
# ─────────────────────────────────────────────────────────────────────────────

class _EmbedStats:
    """Thread-safe accumulator for per-chunk Bedrock embedding telemetry."""
    def __init__(self) -> None:
        self._lock     = threading.Lock()
        self.calls          = 0   # completed _generate_dense_embedding invocations
        self.success         = 0
        self.throttles       = 0   # total throttle events (across all retries)
        self.extra_attempts  = 0   # sum of extra attempts beyond the first (attempt index)
        self.total_backoff_ms = 0  # total wall-clock ms spent in throttle backoff sleeps
        self.active          = 0   # Bedrock requests currently in-flight (not sleeping)

    def enter(self) -> None:           # call immediately before invoke_model
        with self._lock:
            self.active += 1

    def exit_active(self) -> None:     # call immediately after invoke_model returns/throws
        with self._lock:
            self.active -= 1

    def record_success(self, attempt: int) -> None:
        with self._lock:
            self.calls          += 1
            self.success        += 1
            self.extra_attempts += attempt

    def record_call_failure(self, attempt: int) -> None:
        with self._lock:
            self.calls          += 1
            self.extra_attempts += attempt

    def record_throttle(self, backoff_ms: int) -> None:
        with self._lock:
            self.throttles        += 1
            self.total_backoff_ms += backoff_ms


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
    global _COLD_START
    cold_start  = _COLD_START
    _COLD_START = False

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
            result = _process_document(event, lambda_ctx=context, cold_start=cold_start)
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
    dense_vector  = _generate_dense_embedding(
        original_text,
        doc_id=str(ci.get("id", "")),
        chunk_id="ci",
    )
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

def _process_document(chunk: dict, *, lambda_ctx: Any = None, cold_start: bool = False) -> dict:
    text         = chunk["normalization"]["normalized_text"]
    tokens       = chunk["normalization"]["tokens"]
    heading_text = chunk.get("heading_embedding_text", "")
    sparse_vector = _generate_sparse_embedding(tokens)

    doc_id   = chunk.get("document_id", "")
    chunk_id = chunk.get("chunk_id", "")
    objects  = chunk.get("extraction", {}).get("objects", [])

    n_sentences = sum(
        1 for obj in objects
        for span in obj.get("display_spans", [])
        if span.get("type") == "sentence" and span.get("text") and not span.get("embedding")
    )
    n_objects = sum(1 for obj in objects if obj.get("text") and not obj.get("embedding"))
    # chunk + optional heading + objects + sentences
    n_embed_requests = 1 + (1 if heading_text else 0) + n_objects + n_sentences

    # Bug guard: lambda_ctx must be present on the document path.
    # If it's None here, the invocation wiring has changed — log loudly.
    if lambda_ctx is None:
        logger.error(
            "[BUG] Missing lambda_ctx for document embedding doc=%s chunk=%s",
            doc_id, chunk_id,
        )

    stats = _EmbedStats()
    t_chunk_start = time.monotonic()

    _kw = dict(doc_id=doc_id, chunk_id=chunk_id, stats=stats, lambda_ctx=lambda_ctx)

    # task tag: ("chunk"|"heading"|"obj"|"span", obj_idx, span_idx, future)
    tasks: list[tuple[str, int, int | None, Any]] = []
    with ThreadPoolExecutor(max_workers=EMBEDDING_MAX_WORKERS) as pool:
        chunk_fut   = pool.submit(_generate_dense_embedding, text, **_kw)
        heading_fut = pool.submit(_generate_dense_embedding, heading_text, **_kw) if heading_text else None

        for obj_idx, obj in enumerate(objects):
            if obj.get("text") and not obj.get("embedding"):
                fut = pool.submit(_generate_dense_embedding, _object_embedding_text(obj), **_kw)
                tasks.append(("obj", obj_idx, None, fut))

            for span_idx, span in enumerate(obj.get("display_spans", [])):
                if span.get("type") == "sentence" and span.get("text") and not span.get("embedding"):
                    fut = pool.submit(
                        _generate_dense_embedding,
                        _sentence_embedding_text(obj, span),
                        **_kw,
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

    embedding_time = time.monotonic() - t_chunk_start
    throttle_rate  = f"{100*stats.throttles/max(stats.calls,1):.1f}%"
    avg_attempts   = f"{1 + stats.extra_attempts/max(stats.calls,1):.2f}"
    logger.info(
        "[ChunkSummary] chunk=%s doc=%s objects=%d sentences=%d "
        "embed_requests=%d bedrock_calls=%d bedrock_success=%d "
        "bedrock_throttles=%d bedrock_extra_attempts=%d total_backoff_ms=%d "
        "avg_attempts_per_call=%s throttle_rate=%s embedding_time=%.3fs cold_start=%s",
        chunk_id, doc_id, n_objects, n_sentences,
        n_embed_requests, stats.calls, stats.success,
        stats.throttles, stats.extra_attempts, stats.total_backoff_ms,
        avg_attempts, throttle_rate, embedding_time, cold_start,
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
# Shared embedding logic
# ─────────────────────────────────────────────────────────────────────────────

def _generate_dense_embedding(
    text: str,
    *,
    doc_id: str = "",
    chunk_id: str = "",
    stats: "_EmbedStats | None" = None,
    lambda_ctx: Any = None,
) -> list[float]:
    import botocore.exceptions
    payload    = json.dumps({"inputText": text[:_MAX_INPUT_CHARS]}).encode()
    last_exc: Exception | None = None
    request_id = getattr(lambda_ctx, "aws_request_id", "") or os.environ.get("AWS_LAMBDA_LOG_STREAM_NAME", "")

    for attempt in range(_EMBED_MAX_RETRIES):
        t0            = time.monotonic()
        ts            = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        remaining_ms  = lambda_ctx.get_remaining_time_in_millis() if lambda_ctx is not None else None
        if stats is not None:
            stats.enter()   # track only while invoke_model is in-flight
        local_threads = stats.active if stats is not None else -1
        try:
            response = _get("bedrock-runtime").invoke_model(
                modelId     = EMBEDDING_MODEL,
                contentType = "application/json",
                accept      = "application/json",
                body        = payload,
            )
            latency_ms = int((time.monotonic() - t0) * 1000)
            result     = json.loads(response["body"].read())["embedding"]
            if stats is not None:
                stats.exit_active()
                stats.record_success(attempt)
            if EMBEDDING_DEBUG:
                logger.info(
                    "[BedrockEmbedding] ts=%s doc=%s chunk=%s request_id=%s attempt=%d/%d "
                    "status=SUCCESS latency_ms=%d vectors=%d "
                    "remaining_lambda_ms=%s local_embedding_threads=%d",
                    ts, doc_id, chunk_id, request_id, attempt + 1, _EMBED_MAX_RETRIES,
                    latency_ms, len(result), remaining_ms, local_threads,
                )
            return result
        except botocore.exceptions.ClientError as exc:
            latency_ms = int((time.monotonic() - t0) * 1000)
            if stats is not None:
                stats.exit_active()
            code = exc.response.get("Error", {}).get("Code", "")
            msg  = exc.response.get("Error", {}).get("Message", "")
            if code not in _THROTTLE_CODES:
                if stats is not None:
                    stats.record_call_failure(attempt)
                raise
            last_exc = exc
            # Full-jitter exponential backoff (AWS recommended pattern)
            window    = min(_EMBED_BACKOFF_CAP, _EMBED_BACKOFF_BASE * (2 ** attempt))
            delay     = random.uniform(0, window)
            delay_ms  = int(delay * 1000)
            if stats is not None:
                stats.record_throttle(delay_ms)
            logger.warning(
                "[BedrockEmbedding] ts=%s doc=%s chunk=%s request_id=%s attempt=%d/%d "
                "status=THROTTLED error=%s msg=%s latency_ms=%d retry_after_ms=%d "
                "remaining_lambda_ms=%s local_embedding_threads=%d",
                ts, doc_id, chunk_id, request_id, attempt + 1, _EMBED_MAX_RETRIES,
                code, msg, latency_ms, delay_ms, remaining_ms, local_threads,
            )
            time.sleep(delay)
    if stats is not None:
        stats.record_call_failure(_EMBED_MAX_RETRIES)
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
