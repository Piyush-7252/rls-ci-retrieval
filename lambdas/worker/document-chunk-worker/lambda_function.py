from __future__ import annotations

import importlib.util
import json
import logging
import os
import resource
import sys
import time
import types
from pathlib import Path
from typing import Any

from shared.server_notify import (
    notify_document_failed_chunk,
    notify_document_indexed_chunk,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# In Lambda containers, code lives under /var/task. Use that root when present.
_task_root_env = os.environ.get("LAMBDA_TASK_ROOT")
if _task_root_env and (Path(_task_root_env) / "lambdas").exists():
    ROOT = Path(_task_root_env)
else:
    # Local fallback when running from repository checkout.
    ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_loaded: dict[str, types.ModuleType] = {}
_aws: dict[str, Any] = {}

# Selects which embedding lambda module to load; default keeps Titan behaviour unchanged.
_EMBEDDING_MODULE: tuple[str, str] = {
    "cohere": ("embedding_cohere", "embedding_cohere"),
}.get(os.environ.get("EMBEDDING_PROVIDER", "titan"), ("embedding", "embedding"))


def _get(service: str):
    if service not in _aws:
        import boto3

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


def _decode_payload(record_or_event: dict) -> dict:
    payload = record_or_event
    if "body" in record_or_event:
        payload = json.loads(record_or_event["body"])

    # SNS-wrapped SQS payload support
    if isinstance(payload, dict) and isinstance(payload.get("Message"), str):
        try:
            payload = json.loads(payload["Message"])
        except Exception:
            pass

    # Optional wrapper format: {"chunk": {...}}
    if isinstance(payload, dict) and isinstance(payload.get("chunk"), dict):
        payload = payload["chunk"]

    # Optional large-payload indirection format
    # {"chunk_id": "...", "s3_payload": {"bucket": "...", "key": "..."}}
    s3_payload = payload.get("s3_payload") if isinstance(payload, dict) else None
    if isinstance(s3_payload, dict):
        bucket = s3_payload["bucket"]
        key = s3_payload["key"]
        obj = _get("s3").get_object(Bucket=bucket, Key=key)
        payload = json.loads(obj["Body"].read())

    if not isinstance(payload, dict):
        raise ValueError("Message payload must be a JSON object")

    payload.setdefault("source_type", "document")
    return payload


# Requeue if less than this many ms remain in the Lambda invocation.
_MIN_REMAINING_MS = 60_000


def _run_chunk(chunk: dict, context: Any = None) -> dict:
    normalize = _load("normalize", "normalize")
    ner = _load("ner", "ner")
    ontology = _load("ontology", "ontology")
    embedding = _load(*_EMBEDDING_MODULE)
    idx = _load("index", "idx")

    timings: dict[str, float] = {}
    t_total = time.perf_counter()

    def _timed(label: str, fn, *args, **kwargs):
        t = time.perf_counter()
        result = fn(*args, **kwargs)
        timings[label] = round(time.perf_counter() - t, 3)
        return result
    chunk = _timed("normalize", normalize._process_document, chunk)
    chunk = _timed("ner",       ner._process_document,       chunk)
    chunk = _timed("ontology",  ontology._process_document,  chunk)
    chunk = _timed("embedding", embedding._process_document, chunk, lambda_ctx=context)
    _timed("index",     idx._process_document,       chunk)

    elapsed = round(time.perf_counter() - t_total, 3)
    extraction = chunk.get("extraction", {})
    obj_list   = extraction.get("objects", [])
    objects    = len(obj_list)
    sentences  = sum(
        sum(1 for s in o.get("display_spans", []) if s.get("type") == "sentence")
        for o in obj_list
    )
    embeddings = sum(1 for o in obj_list if o.get("embedding"))
    embed_api_calls = (
        1  # chunk text
        + (1 if chunk.get("heading_embedding_text") else 0)
        + embeddings  # one per object
        + sum(
            1 for o in obj_list
            for s in o.get("display_spans", [])
            if s.get("type") == "sentence" and s.get("embedding")
        )
    )
    pages = chunk.get("page_end", 0) - chunk.get("page_start", 0) + 1

    # Peak RSS in MB (Linux: ru_maxrss is in KB; macOS: bytes — normalise to MB).
    _rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import platform
    rss_mb = round(_rss_raw / 1024 if platform.system() != "Darwin" else _rss_raw / 1_048_576, 1)

    # Warn on unusually long chunks so pathological sections can be identified.
    _SLOW_CHUNK_THRESHOLD_S = 300
    log_fn = logger.warning if elapsed > _SLOW_CHUNK_THRESHOLD_S else logger.info
    log_fn(
        "[ChunkWorker] stats chunk_id=%s pages=%d objects=%d sentences=%d "
        "embeddings=%d embed_api_calls=%d rss_mb=%.1f "
        "normalize=%.3fs ner=%.3fs ontology=%.3fs embedding=%.3fs index=%.3fs total=%.3fs%s",
        chunk.get("chunk_id"), pages, objects, sentences,
        embeddings, embed_api_calls, rss_mb,
        timings.get("normalize", 0), timings.get("ner", 0),
        timings.get("ontology", 0), timings.get("embedding", 0),
        timings.get("index", 0), elapsed,
        " [SLOW]" if elapsed > _SLOW_CHUNK_THRESHOLD_S else "",
    )

    return {
        "ok": True,
        "document_id": chunk.get("document_id"),
        "chunk_id": chunk.get("chunk_id"),
        "pages": pages,
        "objects": objects,
        "sentences": sentences,
        "embeddings": embeddings,
        "embed_api_calls": embed_api_calls,
        "elapsed_s": elapsed,
        "timings": timings,
    }


def _get_attempt_id(chunk: dict) -> str | None:
    value = (
        chunk.get("attemptId")
        or chunk.get("attempt_id")
        or (chunk.get("metadata") or {}).get("attemptId")
        or (chunk.get("metadata") or {}).get("attempt_id")
    )
    return str(value) if value is not None and str(value).strip() else None


def _is_final_sqs_attempt(record: dict) -> bool:
    """
    Return True only when this delivery is the final retry before SQS moves
    the message to the DLQ.

    Configure MAX_RECEIVE_COUNT to match the SQS redrive policy. If it is
    unavailable, default to 5 and log the decision.
    """
    attributes = record.get("attributes") or {}

    try:
        receive_count = int(attributes.get("ApproximateReceiveCount", "1"))
    except (TypeError, ValueError):
        receive_count = 1

    try:
        max_receive_count = int(os.environ.get("SQS_MAX_RECEIVE_COUNT", "5"))
    except (TypeError, ValueError):
        max_receive_count = 5

    if max_receive_count < 1:
        max_receive_count = 1

    return receive_count >= max_receive_count


def _notify_indexed(chunk: dict, result: dict) -> None:
    attempt_id = _get_attempt_id(chunk)
    tenant_schema = chunk.get("tenant_schema")

    if not attempt_id:
        logger.warning(
            "[ChunkWorker] indexed callback skipped: missing attemptId chunk_id=%s",
            result.get("chunk_id"),
        )
        return

    if not tenant_schema:
        logger.warning(
            "[ChunkWorker] indexed callback skipped: missing tenant_schema attempt_id=%s chunk_id=%s",
            attempt_id,
            result.get("chunk_id"),
        )
        return

    notify_document_indexed_chunk(
        attempt_id=attempt_id,
        tenant_schema=tenant_schema,
    )


def _notify_failed(chunk: dict, exc: Exception) -> None:
    attempt_id = _get_attempt_id(chunk)
    tenant_schema = chunk.get("tenant_schema")
    chunk_id = str(chunk.get("chunk_id") or "")

    if not attempt_id:
        logger.warning(
            "[ChunkWorker] failed callback skipped: missing attemptId chunk_id=%s",
            chunk_id,
        )
        return

    if not tenant_schema:
        logger.warning(
            "[ChunkWorker] failed callback skipped: missing tenant_schema attempt_id=%s chunk_id=%s",
            attempt_id,
            chunk_id,
        )
        return

    notify_document_failed_chunk(
        attempt_id=attempt_id,
        chunk_id=chunk_id,
        error=str(exc),
        tenant_schema=tenant_schema,
    )


def handler(event: dict, context: Any) -> dict:
    # Direct invoke for smoke testing
    if "Records" not in event:
        result = _run_chunk(_decode_payload(event), context)
        logger.info(
            "[ChunkWorker] done chunk_id=%s elapsed_s=%s",
            result.get("chunk_id"),
            result.get("elapsed_s"),
        )
        return result

    # SQS batch invoke
    failures: list[dict[str, str]] = []
    processed = 0

    for record in event.get("Records", []):
        message_id = record.get("messageId", "")

        # Safety: if Lambda is nearly out of time, requeue rather than timeout.
        if context is not None:
            remaining_ms = context.get_remaining_time_in_millis()
            if remaining_ms < _MIN_REMAINING_MS:
                logger.warning(
                    "[ChunkWorker] only %d ms remaining — requeueing message_id=%s",
                    remaining_ms, message_id,
                )
                if message_id:
                    failures.append({"itemIdentifier": message_id})
                continue

        chunk = None
        try:
            chunk = _decode_payload(record)
            # Stamp queue wait time and memory limit for observability
            sent_ts_str = record.get("attributes", {}).get("SentTimestamp")
            if sent_ts_str:
                chunk["_queue_wait_ms"] = int(time.time() * 1000) - int(sent_ts_str)
            if context is not None:
                chunk["_memory_limit_mb"] = int(getattr(context, "memory_limit_in_mb", 0) or 0)
            result = _run_chunk(chunk, context)

            # The chunk is considered indexed only after _run_chunk completes
            # successfully. Callback failures are non-fatal.
            _notify_indexed(chunk, result)

            processed += 1
            logger.info(
                "[ChunkWorker] done chunk_id=%s elapsed_s=%s",
                result.get("chunk_id"),
                result.get("elapsed_s"),
            )
        except Exception as exc:
            chunk_id = chunk.get("chunk_id", "?") if chunk else "?"
            logger.exception(
                "[ChunkWorker] failed message_id=%s chunk_id=%s error=%s",
                message_id, chunk_id, exc,
            )

            # Do not increment failedChunks for transient attempts. The same
            # SQS message can be retried multiple times. Count it only on the
            # final configured receive attempt, immediately before DLQ.
            if chunk is not None and _is_final_sqs_attempt(record):
                _notify_failed(chunk, exc)

            if message_id:
                failures.append({"itemIdentifier": message_id})

    return {
        "processed": processed,
        "failed": len(failures),
        "batchItemFailures": failures,
    }
