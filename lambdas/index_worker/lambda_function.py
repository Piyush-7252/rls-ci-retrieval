"""
Index Worker — Stage 3: OpenSearch Indexing
============================================
Reads embedded-chunk pointers from the index queue, loads the full
embedded chunk from S3, indexes it to OpenSearch via the shared index
module, then fires a DONE notification to notify_server.

Message in (from index queue):
  {"document_id": "...", "chunk_id": "...",
   "embedding_s3_bucket": "...", "embedding_s3_key": "..."}

Notify event:
  {"event": "chunk_indexed", "status": "DONE", ...}

Env vars:
  Required:
    OPENSEARCH_ENDPOINT

  Optional:
    OPENSEARCH_INDEX, SEMANTIC_OBJECTS_INDEX, OPENSEARCH_CI_INDEX  (same as v1)
    NOTIFY_SERVER_URL                                              (default: "" = off)
    NOTIFY_SERVER_TIMEOUT                                          (default: 5)
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
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


# ─── notify server ────────────────────────────────────────────────────────────

def _notify(payload: dict) -> None:
    if not NOTIFY_SERVER_URL:
        return
    url = NOTIFY_SERVER_URL.rstrip("/") + "/chunk"
    try:
        http_requests.post(url, json=payload, timeout=NOTIFY_SERVER_TIMEOUT)
    except Exception as exc:
        logger.warning("[IndexWorker] notify failed url=%s error=%s", url, exc)


# ─── core processing ──────────────────────────────────────────────────────────

def _run_message(msg: dict, context: Any = None) -> dict:
    t_total = time.perf_counter()

    doc_id   = msg["document_id"]
    chunk_id = msg["chunk_id"]
    bucket   = msg["embedding_s3_bucket"]
    key      = msg["embedding_s3_key"]

    _notify({"event": "chunk_indexing", "document_id": doc_id, "chunk_id": chunk_id, "status": "INDEXING"})

    # ── Load embedded chunk from S3 ───────────────────────────────────────────

    obj   = _get("s3").get_object(Bucket=bucket, Key=key)
    chunk = json.loads(obj["Body"].read())

    # ── Index to OpenSearch ───────────────────────────────────────────────────
    # Duplicate SQS delivery is safe: the index lambda uses chunk_id / object_id
    # as deterministic OpenSearch _id values, so re-indexing is a no-op upsert.

    idx = _load("index", "idx")
    idx._process_document(chunk)

    elapsed = round(time.perf_counter() - t_total, 3)

    logger.info("[IndexWorker] chunk_id=%s elapsed_s=%.3f [INDEXED→DONE]",
                chunk_id, elapsed)

    _notify({
        "event":       "chunk_indexed",
        "document_id": doc_id,
        "chunk_id":    chunk_id,
        "status":      "DONE",
        "elapsed_s":   elapsed,
    })

    return {"ok": True, "document_id": doc_id, "chunk_id": chunk_id, "elapsed_s": elapsed}


_MIN_REMAINING_MS = 30_000


def handler(event: dict, context: Any) -> dict:
    # Direct invoke for smoke testing
    if "Records" not in event:
        result = _run_message(event, context)
        logger.info("[IndexWorker] done chunk_id=%s elapsed_s=%s",
                    result.get("chunk_id"), result.get("elapsed_s"))
        return result

    failures: list[dict[str, str]] = []
    processed = 0

    for record in event.get("Records", []):
        message_id = record.get("messageId", "")

        if context is not None and context.get_remaining_time_in_millis() < _MIN_REMAINING_MS:
            logger.warning("[IndexWorker] low time — requeueing message_id=%s", message_id)
            if message_id:
                failures.append({"itemIdentifier": message_id})
            continue

        msg = None
        try:
            body = record.get("body", "{}")
            msg  = json.loads(body) if isinstance(body, str) else body

            result = _run_message(msg, context)
            processed += 1
            logger.info("[IndexWorker] done chunk_id=%s elapsed_s=%s",
                        result.get("chunk_id"), result.get("elapsed_s"))

        except Exception as exc:
            chunk_id = (msg or {}).get("chunk_id", "?")
            doc_id   = (msg or {}).get("document_id", "")
            logger.exception("[IndexWorker] failed message_id=%s chunk_id=%s error=%s",
                             message_id, chunk_id, exc)
            _notify({
                "event":       "chunk_failed",
                "document_id": doc_id,
                "chunk_id":    chunk_id,
                "status":      "FAILED",
                "stage":       "index",
                "error":       str(exc),
            })
            if message_id:
                failures.append({"itemIdentifier": message_id})

    return {
        "processed":         processed,
        "failed":            len(failures),
        "batchItemFailures": failures,
    }
