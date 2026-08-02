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


def _run_chunk(chunk: dict) -> dict:
    normalize = _load("normalize", "normalize")
    ner = _load("ner", "ner")
    ontology = _load("ontology", "ontology")
    embedding = _load("embedding", "embedding")
    idx = _load("index", "idx")

    t0 = time.perf_counter()

    chunk = normalize._process_document(chunk)
    chunk = ner._process_document(chunk)
    chunk = ontology._process_document(chunk)
    chunk = embedding._process_document(chunk)
    idx._process_document(chunk)

    elapsed = round(time.perf_counter() - t0, 3)
    objects = len(chunk.get("extraction", {}).get("objects", []))

    return {
        "ok": True,
        "document_id": chunk.get("document_id"),
        "chunk_id": chunk.get("chunk_id"),
        "objects": objects,
        "elapsed_s": elapsed,
    }


def handler(event: dict, context: Any) -> dict:
    # Direct invoke for smoke testing
    if "Records" not in event:
        result = _run_chunk(_decode_payload(event))
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
        try:
            chunk = _decode_payload(record)
            result = _run_chunk(chunk)
            processed += 1
            logger.info(
                "[ChunkWorker] done chunk_id=%s elapsed_s=%s",
                result.get("chunk_id"),
                result.get("elapsed_s"),
            )
        except Exception as exc:
            logger.exception("[ChunkWorker] failed message_id=%s error=%s", message_id, exc)
            if message_id:
                failures.append({"itemIdentifier": message_id})

    return {
        "processed": processed,
        "failed": len(failures),
        "batchItemFailures": failures,
    }
