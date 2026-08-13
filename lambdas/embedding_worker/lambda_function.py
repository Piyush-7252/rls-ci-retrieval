"""
Embedding Worker — Stage 2: GPU API + S3 Embedding Artifact
============================================================
Reads enriched-chunk pointers from the embedding queue, calls the GPU
embedding API for all texts, saves the embedded artifact to S3, then
pushes a pointer to the index queue.

Lambda concurrency is decoupled from GPU latency — the Lambda returns as
soon as it enqueues the index pointer; it does not wait for OpenSearch.

Message in (from embedding queue):
  {"document_id": "...", "chunk_id": "...",
   "enriched_s3_bucket": "...", "enriched_s3_key": "..."}

Message out (to index queue):
  {"document_id": "...", "chunk_id": "...",
   "embedding_s3_bucket": "...", "embedding_s3_key": "..."}

S3 embedding artifact:
  {ARTIFACT_BUCKET}/{doc_id}/embedding/{EMBEDDING_MODEL}/{chunk_id}.json
  (full embedded chunk — vectors attached to objects/sentences)

Embedding API contract (POST {EMBEDDING_API_URL})
--------------------------------------------------
  Request:
    {"request_id": "...", "document_id": "...", "chunk_id": "...",
     "texts": [...], "input_type": "search_document", "truncate": true}
  Response:
    {"embeddings": [[float,...], ...], "model": "...", "dimensions": N,
     "gpu_queue_wait_ms": N, "gpu_batch_size": N,
     "gpu_inference_ms": N, "gpu_total_request_ms": N}

Notify event:
  {"event": "chunk_embedded", "status": "EMBEDDED", gpu_metrics, ...}

Env vars:
  Required:
    EMBEDDING_API_URL   GPU embedding service  (e.g. http://host:8080/embed)
    ARTIFACT_BUCKET     S3 bucket for embedding artifacts
    INDEX_QUEUE_URL     SQS URL for the index stage queue

  Optional:
    EMBEDDING_API_KEY          Bearer token                  (default: "")
    EMBEDDING_API_TIMEOUT      HTTP timeout seconds          (default: 120)
    EMBEDDING_MODEL            Tag stored with vectors       (default: "gpu-embed")
    NOTIFY_SERVER_URL                                        (default: "" = off)
    NOTIFY_SERVER_TIMEOUT      Notify POST timeout seconds   (default: 5)
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import Counter
from typing import Any

import boto3
import requests as http_requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ─── env ──────────────────────────────────────────────────────────────────────

EMBEDDING_API_URL     = os.environ.get("EMBEDDING_API_URL", "")
EMBEDDING_API_KEY     = os.environ.get("EMBEDDING_API_KEY", "")
EMBEDDING_API_TIMEOUT = int(os.environ.get("EMBEDDING_API_TIMEOUT", "120"))
EMBEDDING_MODEL       = os.environ.get("EMBEDDING_MODEL", "gpu-embed")
EMBEDDING_VERSION     = os.environ.get("EMBEDDING_VERSION", "1")  # bump when preprocessing/pooling changes
ARTIFACT_BUCKET       = os.environ.get("ARTIFACT_BUCKET", "")
INDEX_QUEUE_URL       = os.environ.get("INDEX_QUEUE_URL", "")
NOTIFY_SERVER_URL     = os.environ.get("NOTIFY_SERVER_URL", "")
NOTIFY_SERVER_TIMEOUT = int(os.environ.get("NOTIFY_SERVER_TIMEOUT", "5"))

# ─── AWS client cache ─────────────────────────────────────────────────────────

_aws: dict[str, Any] = {}


def _get(service: str):
    if service not in _aws:
        _aws[service] = boto3.client(service)
    return _aws[service]


# ─── text helpers ─────────────────────────────────────────────────────────────

_MAX_INPUT_CHARS = 25_000

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "as", "is", "was", "are", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "that", "this", "which", "who", "whom",
    "it", "its", "from", "not", "no", "also", "each", "based", "per",
})


def _sparse(tokens: list[str]) -> dict[str, float]:
    filtered = [t for t in tokens if t not in _STOPWORDS and len(t) > 1] or tokens
    tf = Counter(filtered)
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


# ─── S3 helpers ───────────────────────────────────────────────────────────────

def _embedding_key(doc_id: str, chunk_id: str) -> str:
    return f"{doc_id}/embedding/{EMBEDDING_MODEL}/v{EMBEDDING_VERSION}/{chunk_id}.json"


def _load_enriched_from_s3(bucket: str, key: str) -> dict:
    obj = _get("s3").get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())


def _save_embedded_to_s3(doc_id: str, chunk_id: str, chunk: dict) -> str:
    key = _embedding_key(doc_id, chunk_id)
    _get("s3").put_object(
        Bucket=ARTIFACT_BUCKET,
        Key=key,
        Body=json.dumps(chunk, default=str).encode(),
        ContentType="application/json",
    )
    return key


def _embedding_exists(doc_id: str, chunk_id: str) -> bool:
    """Return True if this model's embedding artifact already exists in S3 (idempotency)."""
    if not ARTIFACT_BUCKET:
        return False
    try:
        _get("s3").head_object(Bucket=ARTIFACT_BUCKET, Key=_embedding_key(doc_id, chunk_id))
        return True
    except Exception:
        return False


# ─── GPU embedding API ────────────────────────────────────────────────────────

def _embed(
    texts: list[str],
    input_type: str,
    *,
    doc_id: str = "",
    chunk_id: str = "",
) -> tuple[list[list[float]], dict]:
    """Returns (vectors, gpu_metrics)."""
    if not EMBEDDING_API_URL:
        raise RuntimeError("EMBEDDING_API_URL is not configured")

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if EMBEDDING_API_KEY:
        headers["Authorization"] = f"Bearer {EMBEDDING_API_KEY}"

    t_req = time.perf_counter()
    resp = http_requests.post(
        EMBEDDING_API_URL,
        json={
            "request_id":  str(uuid.uuid4()),
            "document_id": doc_id,
            "chunk_id":    chunk_id,
            "texts":       texts,
            "input_type":  input_type,
            "truncate":    True,
        },
        headers=headers,
        timeout=EMBEDDING_API_TIMEOUT,
    )
    wall_ms = int((time.perf_counter() - t_req) * 1000)
    resp.raise_for_status()
    data = resp.json()

    vecs = data.get("embeddings", data) if isinstance(data, dict) else data
    if not isinstance(vecs, list) or len(vecs) != len(texts):
        raise ValueError(
            f"Embedding API: expected {len(texts)} vectors, got "
            f"{len(vecs) if isinstance(vecs, list) else type(vecs).__name__}"
        )

    d = data if isinstance(data, dict) else {}
    gpu_metrics = {
        "gpu_queue_wait_ms":    int(d.get("gpu_queue_wait_ms",    -1)),
        "gpu_batch_size":       int(d.get("gpu_batch_size",       len(texts))),
        "gpu_inference_ms":     int(d.get("gpu_inference_ms",     -1)),
        "gpu_total_request_ms": int(d.get("gpu_total_request_ms", wall_ms)),
    }
    return vecs, gpu_metrics


# ─── index queue dispatch ─────────────────────────────────────────────────────

def _push_to_index_queue(doc_id: str, chunk_id: str, bucket: str, key: str) -> None:
    if not INDEX_QUEUE_URL:
        raise RuntimeError("INDEX_QUEUE_URL is not configured")
    _get("sqs").send_message(
        QueueUrl=INDEX_QUEUE_URL,
        MessageBody=json.dumps({
            "document_id":         doc_id,
            "chunk_id":            chunk_id,
            "embedding_s3_bucket": bucket,
            "embedding_s3_key":    key,
        }),
    )


# ─── notify server ────────────────────────────────────────────────────────────

def _notify(payload: dict) -> None:
    if not NOTIFY_SERVER_URL:
        return
    url = NOTIFY_SERVER_URL.rstrip("/") + "/chunk"
    try:
        http_requests.post(url, json=payload, timeout=NOTIFY_SERVER_TIMEOUT)
    except Exception as exc:
        logger.warning("[EmbeddingWorker] notify failed url=%s error=%s", url, exc)


# ─── core processing ──────────────────────────────────────────────────────────

def _run_message(msg: dict, context: Any = None) -> dict:
    t_total = time.perf_counter()

    doc_id   = msg["document_id"]
    chunk_id = msg["chunk_id"]
    bucket   = msg["enriched_s3_bucket"]
    key      = msg["enriched_s3_key"]

    _notify({"event": "chunk_embedding", "document_id": doc_id, "chunk_id": chunk_id, "status": "EMBEDDING"})

    # ── Idempotency: duplicate SQS delivery → skip GPU, re-queue to index ─────

    if _embedding_exists(doc_id, chunk_id):
        existing_key = _embedding_key(doc_id, chunk_id)
        logger.info("[EmbeddingWorker] idempotency HIT chunk_id=%s — skipping GPU", chunk_id)
        _push_to_index_queue(doc_id, chunk_id, ARTIFACT_BUCKET, existing_key)
        _notify({
            "event":          "chunk_embedded",
            "document_id":    doc_id,
            "chunk_id":       chunk_id,
            "status":         "EMBEDDED",
            "idempotency_hit": True,
            "elapsed_s":      0.0,
        })
        return {"ok": True, "document_id": doc_id, "chunk_id": chunk_id, "skipped": True, "elapsed_s": 0.0}

    # ── Load enriched chunk from S3 ───────────────────────────────────────────

    chunk = _load_enriched_from_s3(bucket, key)

    # ── Collect ordered text slots ────────────────────────────────────────────

    text         = chunk["normalization"]["normalized_text"]
    tokens       = chunk["normalization"]["tokens"]
    heading_text = chunk.get("heading_embedding_text", "")
    objects      = chunk.get("extraction", {}).get("objects", [])

    slot_texts:   list[str]                  = [text[:_MAX_INPUT_CHARS]]
    heading_slot: int | None                 = None
    obj_slots:    list[tuple[int, int]]      = []
    sent_slots:   list[tuple[int, int, int]] = []

    if heading_text:
        heading_slot = len(slot_texts)
        slot_texts.append(heading_text[:_MAX_INPUT_CHARS])

    for obj_idx, obj in enumerate(objects):
        if obj.get("text"):
            slot_texts.append(_object_text(obj)[:_MAX_INPUT_CHARS])
            obj_slots.append((len(slot_texts) - 1, obj_idx))
        for span_idx, span in enumerate(obj.get("display_spans", [])):
            if span.get("type") == "sentence" and span.get("text"):
                slot_texts.append(_sentence_text(obj, span)[:_MAX_INPUT_CHARS])
                sent_slots.append((len(slot_texts) - 1, obj_idx, span_idx))

    n_embed_requests = len(slot_texts)

    # ── GPU embedding API ─────────────────────────────────────────────────────

    t_embed = time.perf_counter()
    all_vecs, gpu_metrics = _embed(
        slot_texts, input_type="search_document", doc_id=doc_id, chunk_id=chunk_id
    )
    embed_ms = int((time.perf_counter() - t_embed) * 1000)

    # ── Distribute vectors ────────────────────────────────────────────────────

    dense_vector         = all_vecs[0]
    sparse_vector        = _sparse(tokens)
    heading_dense_vector = all_vecs[heading_slot] if heading_slot is not None else []

    for slot_idx, obj_idx in obj_slots:
        objects[obj_idx]["embedding"] = all_vecs[slot_idx]
    for slot_idx, obj_idx, span_idx in sent_slots:
        objects[obj_idx]["display_spans"][span_idx]["embedding"] = all_vecs[slot_idx]

    if objects and "extraction" in chunk:
        chunk = {**chunk, "extraction": {**chunk["extraction"], "objects": objects}}

    chunk = {
        **chunk,
        "embedding": {
            "dense_vector":         dense_vector,
            "heading_dense_vector": heading_dense_vector,
            "sparse_vector":        sparse_vector,
            "model":                EMBEDDING_MODEL,
            "dimensions":           len(dense_vector),
        },
    }

    # ── Save embedding artifact to S3 ─────────────────────────────────────────

    embedding_key = _save_embedded_to_s3(doc_id, chunk_id, chunk)

    # ── Push pointer to index queue ───────────────────────────────────────────

    _push_to_index_queue(doc_id, chunk_id, ARTIFACT_BUCKET, embedding_key)

    elapsed = round(time.perf_counter() - t_total, 3)

    logger.info(
        "[EmbeddingWorker] chunk_id=%s embed_requests=%d embed_ms=%d "
        "gpu_batch_size=%d gpu_inference_ms=%d gpu_total_request_ms=%d [EMBEDDED→QUEUED]",
        chunk_id, n_embed_requests, embed_ms,
        gpu_metrics["gpu_batch_size"],
        gpu_metrics["gpu_inference_ms"],
        gpu_metrics["gpu_total_request_ms"],
    )

    _notify({
        "event":                "chunk_embedded",
        "document_id":          doc_id,
        "chunk_id":             chunk_id,
        "status":               "EMBEDDED",
        "embed_requests":       n_embed_requests,
        "embed_ms":             embed_ms,
        "gpu_queue_wait_ms":    gpu_metrics["gpu_queue_wait_ms"],
        "gpu_batch_size":       gpu_metrics["gpu_batch_size"],
        "gpu_inference_ms":     gpu_metrics["gpu_inference_ms"],
        "gpu_total_request_ms": gpu_metrics["gpu_total_request_ms"],
        "elapsed_s":            elapsed,
    })

    return {
        "ok":             True,
        "document_id":    doc_id,
        "chunk_id":       chunk_id,
        "embed_requests": n_embed_requests,
        "elapsed_s":      elapsed,
    }


_MIN_REMAINING_MS = 30_000


def handler(event: dict, context: Any) -> dict:
    # Direct invoke for smoke testing
    if "Records" not in event:
        result = _run_message(event, context)
        logger.info("[EmbeddingWorker] done chunk_id=%s elapsed_s=%s",
                    result.get("chunk_id"), result.get("elapsed_s"))
        return result

    failures: list[dict[str, str]] = []
    processed = 0

    for record in event.get("Records", []):
        message_id = record.get("messageId", "")

        if context is not None and context.get_remaining_time_in_millis() < _MIN_REMAINING_MS:
            logger.warning("[EmbeddingWorker] low time — requeueing message_id=%s", message_id)
            if message_id:
                failures.append({"itemIdentifier": message_id})
            continue

        msg = None
        try:
            body = record.get("body", "{}")
            msg  = json.loads(body) if isinstance(body, str) else body

            sent_ts = record.get("attributes", {}).get("SentTimestamp")
            if sent_ts:
                msg["_queue_wait_ms"] = int(time.time() * 1000) - int(sent_ts)

            result = _run_message(msg, context)
            processed += 1
            logger.info("[EmbeddingWorker] done chunk_id=%s elapsed_s=%s",
                        result.get("chunk_id"), result.get("elapsed_s"))

        except Exception as exc:
            chunk_id = (msg or {}).get("chunk_id", "?")
            doc_id   = (msg or {}).get("document_id", "")
            logger.exception("[EmbeddingWorker] failed message_id=%s chunk_id=%s error=%s",
                             message_id, chunk_id, exc)
            _notify({
                "event":       "chunk_failed",
                "document_id": doc_id,
                "chunk_id":    chunk_id,
                "status":      "FAILED",
                "stage":       "embedding",
                "error":       str(exc),
            })
            if message_id:
                failures.append({"itemIdentifier": message_id})

    return {
        "processed":         processed,
        "failed":            len(failures),
        "batchItemFailures": failures,
    }
