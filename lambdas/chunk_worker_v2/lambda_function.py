"""
Chunk Worker v2 — Stage 1: Enrichment
======================================
New worker; does NOT modify chunk_worker v1.

Full 3-stage decoupled pipeline:

  Stage 1  chunk_worker_v2   normalize → NER → ontology → S3 enriched cache
                                  ↓  SQS pointer
  Stage 2  embedding_worker  load enriched → GPU API → S3 embedding artifact
                                  ↓  SQS pointer
  Stage 3  index_worker      load embedded → OpenSearch → notify_server

This lambda is Stage 1 only.  It enriches the chunk, writes a vector-free
artifact to S3, then drops a lightweight pointer onto the embedding queue.
Lambda concurrency is therefore NOT tied to GPU latency.

S3 enriched artifact
---------------------
  {ARTIFACT_BUCKET}/{doc_id}/enriched/v{ENRICHMENT_VERSION}/{chunk_id}.json
  (no embedding vectors — always model-agnostic)

Embedding queue message out
----------------------------
  {"document_id": "...", "chunk_id": "...",
   "enriched_s3_bucket": "...", "enriched_s3_key": "..."}

Notify event
-------------
  {"event": "chunk_enriched", "status": "ENRICHED", ...}

Env vars
---------
  Required:
    ARTIFACT_BUCKET       S3 bucket for enriched-chunk cache
    EMBEDDING_QUEUE_URL   SQS URL for the embedding stage queue

  Optional:
    ENRICHMENT_VERSION    Cache key version; bump to re-enrich    (default: "1")
    NOTIFY_SERVER_URL     Status update endpoint                  (default: "" = off)
    NOTIFY_SERVER_TIMEOUT Notify POST timeout seconds             (default: 5)
    NER_MODEL             (same as v1, default: gliner)
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import platform
import resource
import sys
import time
import types
from pathlib import Path
from typing import Any

import boto3
import requests as http_requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ─── paths ────────────────────────────────────────────────────────────────────

_task_root_env = os.environ.get("LAMBDA_TASK_ROOT")
if _task_root_env and (Path(_task_root_env) / "lambdas").exists():
    ROOT = Path(_task_root_env)
else:
    ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── env ──────────────────────────────────────────────────────────────────────

ARTIFACT_BUCKET       = os.environ.get("ARTIFACT_BUCKET", "")
ENRICHMENT_VERSION    = os.environ.get("ENRICHMENT_VERSION", "1")
EMBEDDING_QUEUE_URL   = os.environ.get("EMBEDDING_QUEUE_URL", "")
NOTIFY_SERVER_URL     = os.environ.get("NOTIFY_SERVER_URL", "")
NOTIFY_SERVER_TIMEOUT = int(os.environ.get("NOTIFY_SERVER_TIMEOUT", "5"))

# ─── module + AWS client cache ────────────────────────────────────────────────

_loaded: dict[str, types.ModuleType] = {}
_aws: dict[str, Any] = {}


def _get(service: str):
    if service not in _aws:
        _aws[service] = boto3.client(service)
    return _aws[service]


def _load(rel_path: str, alias: str) -> types.ModuleType:
    if alias in _loaded:
        return _loaded[alias]
    lf_path = ROOT / "lambdas" / rel_path / "lambda_function.py"
    spec = importlib.util.spec_from_file_location(alias, lf_path)
    mod = importlib.util.module_from_spec(spec)
    lf_dir = str(lf_path.parent)
    if lf_dir not in sys.path:
        sys.path.insert(0, lf_dir)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    _loaded[alias] = mod
    return mod


# ─── S3 enrichment cache ──────────────────────────────────────────────────────

def _enriched_key(doc_id: str, chunk_id: str) -> str:
    return f"{doc_id}/enriched/v{ENRICHMENT_VERSION}/{chunk_id}.json"


def _load_enriched(doc_id: str, chunk_id: str) -> dict | None:
    """Return cached enriched chunk or None on miss / disabled."""
    if not ARTIFACT_BUCKET:
        return None
    key = _enriched_key(doc_id, chunk_id)
    try:
        obj = _get("s3").get_object(Bucket=ARTIFACT_BUCKET, Key=key)
        return json.loads(obj["Body"].read())
    except Exception as exc:
        code = getattr(getattr(exc, "response", None), "get", lambda *_: {})("Error", {}).get("Code", "")
        if code not in ("NoSuchKey", "404"):
            logger.warning("[ChunkWorkerV2] S3 cache read error %s: %s", key, exc)
        return None


def _strip_embeddings(chunk: dict) -> dict:
    """Remove all vectors so the S3 enriched artifact is model-agnostic."""
    chunk = {k: v for k, v in chunk.items() if k != "embedding"}
    extraction = chunk.get("extraction", {})
    if "objects" not in extraction:
        return chunk
    clean_objects = []
    for obj in extraction["objects"]:
        obj = {k: v for k, v in obj.items() if k != "embedding"}
        spans = obj.get("display_spans")
        if spans:
            obj = {**obj, "display_spans": [
                {k: v for k, v in s.items() if k != "embedding"} for s in spans
            ]}
        clean_objects.append(obj)
    return {**chunk, "extraction": {**extraction, "objects": clean_objects}}


def _save_enriched(doc_id: str, chunk_id: str, chunk: dict) -> None:
    if not ARTIFACT_BUCKET:
        return
    key = _enriched_key(doc_id, chunk_id)
    try:
        _get("s3").put_object(
            Bucket=ARTIFACT_BUCKET,
            Key=key,
            Body=json.dumps(_strip_embeddings(chunk), default=str).encode(),
            ContentType="application/json",
        )
    except Exception as exc:
        logger.warning("[ChunkWorkerV2] S3 cache write error %s: %s", key, exc)


# ─── embedding queue dispatch ─────────────────────────────────────────────────

def _push_to_embedding_queue(doc_id: str, chunk_id: str, bucket: str, key: str) -> None:
    if not EMBEDDING_QUEUE_URL:
        raise RuntimeError("EMBEDDING_QUEUE_URL is not configured")
    _get("sqs").send_message(
        QueueUrl=EMBEDDING_QUEUE_URL,
        MessageBody=json.dumps({
            "document_id":        doc_id,
            "chunk_id":           chunk_id,
            "enriched_s3_bucket": bucket,
            "enriched_s3_key":    key,
        }),
    )


# ─── notify server ────────────────────────────────────────────────────────────

def _notify(payload: dict) -> None:
    """Fire-and-forget POST; failures are logged but never raise."""
    if not NOTIFY_SERVER_URL:
        return
    url = NOTIFY_SERVER_URL.rstrip("/") + "/chunk"
    try:
        http_requests.post(url, json=payload, timeout=NOTIFY_SERVER_TIMEOUT)
    except Exception as exc:
        logger.warning("[ChunkWorkerV2] notify failed url=%s error=%s", url, exc)


# ─── SQS payload decode (identical to v1) ────────────────────────────────────

def _decode_payload(record_or_event: dict) -> dict:
    payload = record_or_event
    if "body" in record_or_event:
        payload = json.loads(record_or_event["body"])

    if isinstance(payload, dict) and isinstance(payload.get("Message"), str):
        try:
            payload = json.loads(payload["Message"])
        except Exception:
            pass

    if isinstance(payload, dict) and isinstance(payload.get("chunk"), dict):
        payload = payload["chunk"]

    s3_ref = payload.get("s3_payload") if isinstance(payload, dict) else None
    if isinstance(s3_ref, dict):
        obj = _get("s3").get_object(Bucket=s3_ref["bucket"], Key=s3_ref["key"])
        payload = json.loads(obj["Body"].read())

    if not isinstance(payload, dict):
        raise ValueError("Message payload must be a JSON object")

    payload.setdefault("source_type", "document")
    return payload


_MIN_REMAINING_MS = 60_000


# ─── core processing ──────────────────────────────────────────────────────────

def _run_chunk(chunk: dict, context: Any = None) -> dict:
    timings: dict[str, float] = {}
    t_total = time.perf_counter()

    doc_id   = chunk.get("document_id", "")
    chunk_id = chunk.get("chunk_id", "")

    def _timed(label: str, fn, *args, **kwargs):
        t = time.perf_counter()
        result = fn(*args, **kwargs)
        timings[label] = round(time.perf_counter() - t, 3)
        return result

    # ── Enrichment: load from S3 cache or run normalize → NER → ontology ─────

    cached = _load_enriched(doc_id, chunk_id)
    if cached is not None:
        chunk = cached
        timings.update(normalize=0.0, ner=0.0, ontology=0.0)
        logger.info("[ChunkWorkerV2] enrichment cache HIT chunk_id=%s", chunk_id)
    else:
        normalize = _load("normalize", "normalize")
        ner       = _load("ner",       "ner")
        ontology  = _load("ontology",  "ontology")
        chunk = _timed("normalize", normalize._process_document, chunk)
        chunk = _timed("ner",       ner._process_document,       chunk)
        chunk = _timed("ontology",  ontology._process_document,  chunk)
        _save_enriched(doc_id, chunk_id, chunk)

    # ── Push pointer to embedding queue (Lambda is now free) ─────────────────

    enriched_key = _enriched_key(doc_id, chunk_id)
    _push_to_embedding_queue(doc_id, chunk_id, ARTIFACT_BUCKET, enriched_key)

    # ── Telemetry ─────────────────────────────────────────────────────────────

    elapsed = round(time.perf_counter() - t_total, 3)
    obj_list    = chunk.get("extraction", {}).get("objects", [])
    n_objects   = len(obj_list)
    n_sentences = sum(
        sum(1 for s in o.get("display_spans", []) if s.get("type") == "sentence")
        for o in obj_list
    )
    pages = chunk.get("page_end", 0) - chunk.get("page_start", 0) + 1

    _rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = round(_rss_raw / 1024 if platform.system() != "Darwin" else _rss_raw / 1_048_576, 1)

    log_fn = logger.warning if elapsed > 300 else logger.info
    log_fn(
        "[ChunkWorker] stats chunk_id=%s pages=%d objects=%d sentences=%d rss_mb=%.1f "
        "normalize=%.3fs ner=%.3fs ontology=%.3fs total=%.3fs [ENRICHED→QUEUED]",
        chunk_id, pages, n_objects, n_sentences, rss_mb,
        timings.get("normalize", 0), timings.get("ner", 0),
        timings.get("ontology", 0), elapsed,
    )

    result = {
        "ok":          True,
        "document_id": doc_id,
        "chunk_id":    chunk_id,
        "pages":       pages,
        "objects":     n_objects,
        "sentences":   n_sentences,
        "elapsed_s":   elapsed,
        "timings":     timings,
    }

    _notify({
        "event":       "chunk_enriched",
        "document_id": doc_id,
        "chunk_id":    chunk_id,
        "status":      "ENRICHED",
        "timings":     timings,
        "objects":     n_objects,
        "sentences":   n_sentences,
    })

    return result


# ─── Lambda handler ───────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    # Direct invoke (smoke test / local run)
    if "Records" not in event:
        result = _run_chunk(_decode_payload(event), context)
        logger.info("[ChunkWorkerV2] done chunk_id=%s elapsed_s=%s",
                    result.get("chunk_id"), result.get("elapsed_s"))
        return result

    failures: list[dict[str, str]] = []
    processed = 0

    for record in event.get("Records", []):
        message_id = record.get("messageId", "")

        if context is not None:
            remaining_ms = context.get_remaining_time_in_millis()
            if remaining_ms < _MIN_REMAINING_MS:
                logger.warning(
                    "[ChunkWorkerV2] only %d ms remaining — requeueing message_id=%s",
                    remaining_ms, message_id,
                )
                if message_id:
                    failures.append({"itemIdentifier": message_id})
                continue

        chunk = None
        try:
            chunk = _decode_payload(record)
            sent_ts = record.get("attributes", {}).get("SentTimestamp")
            if sent_ts:
                chunk["_queue_wait_ms"] = int(time.time() * 1000) - int(sent_ts)
            if context is not None:
                chunk["_memory_limit_mb"] = int(getattr(context, "memory_limit_in_mb", 0) or 0)

            result = _run_chunk(chunk, context)
            processed += 1
            logger.info("[ChunkWorkerV2] done chunk_id=%s elapsed_s=%s",
                        result.get("chunk_id"), result.get("elapsed_s"))

        except Exception as exc:
            chunk_id = (chunk or {}).get("chunk_id", "?")
            doc_id   = (chunk or {}).get("document_id", "")
            logger.exception(
                "[ChunkWorkerV2] failed message_id=%s chunk_id=%s error=%s",
                message_id, chunk_id, exc,
            )
            _notify({
                "event":       "chunk_failed",
                "document_id": doc_id,
                "chunk_id":    chunk_id,
                "status":      "FAILED",
                "error":       str(exc),
            })
            if message_id:
                failures.append({"itemIdentifier": message_id})

    return {
        "processed":         processed,
        "failed":            len(failures),
        "batchItemFailures": failures,
    }
