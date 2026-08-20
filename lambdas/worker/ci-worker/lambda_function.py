"""
CI Worker Lambda
================
SQS-triggered Lambda that runs a single raw CI through the full enrichment
pipeline and writes it to the ci-objects OpenSearch index.

Pipeline
--------
    raw CI  →  normalize._process_ci
            →  ner._process_ci
            →  ontology._process_ci
            →  embedding._process_ci
            →  index._process_ci   (writes to ci-objects)

SQS message body (JSON)
-----------------------
    { raw CI dict with "id", "knownCI", ... }

Direct invoke (smoke test)
--------------------------
    event = { raw CI dict }

Env vars
--------
    OPENSEARCH_ENDPOINT   — host only (no https:// prefix)
    OPENSEARCH_CI_INDEX   — default: ci-objects
    NER_MODEL             — default: gliner
    EMBEDDING_MODEL       — default: amazon.titan-embed-text-v2:0
    EMBEDDING_MAX_WORKERS — default: 4
    AWS_REGION            — default: eu-west-1
"""

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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# In Lambda containers, code lives under /var/task. Use that root when present.
_task_root_env = os.environ.get("LAMBDA_TASK_ROOT")
if _task_root_env and (Path(_task_root_env) / "lambdas").exists():
    ROOT = Path(_task_root_env)
else:
    ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_loaded: dict[str, types.ModuleType] = {}

# Selects which embedding lambda module to load; default keeps Titan behaviour unchanged.
_EMBEDDING_MODULE: tuple[str, str] = {
    "cohere": ("embedding_cohere", "embedding_cohere"),
}.get(os.environ.get("EMBEDDING_PROVIDER", "titan"), ("embedding", "embedding"))

# Requeue if less than this many ms remain in the Lambda invocation.
_MIN_REMAINING_MS = 30_000


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
    """Extract the CI dict from an SQS record or a direct-invoke event."""
    payload = record_or_event

    # SQS record wraps the body as a string.
    if "body" in record_or_event:
        payload = json.loads(record_or_event["body"])

    # SNS-wrapped SQS message support.
    if isinstance(payload, dict) and isinstance(payload.get("Message"), str):
        try:
            payload = json.loads(payload["Message"])
        except Exception:
            pass

    if not isinstance(payload, dict):
        raise ValueError("CI payload must be a JSON object")

    payload.setdefault("source_type", "ci")
    return payload


def _run_ci(ci: dict, context: Any = None) -> dict:
    normalize = _load("normalize", "normalize")
    ner       = _load("ner",       "ner")
    ontology  = _load("ontology",  "ontology")
    embedding = _load(*_EMBEDDING_MODULE)
    idx       = _load("index",     "idx")

    timings: dict[str, float] = {}
    t_total = time.perf_counter()

    def _timed(label: str, fn, *args, **kwargs):
        t = time.perf_counter()
        result = fn(*args, **kwargs)
        timings[label] = round(time.perf_counter() - t, 3)
        return result

    ci = _timed("normalize", normalize._process_ci, ci)
    ci = _timed("ner",       ner._process_ci,       ci)
    ci = _timed("ontology",  ontology._process_ci,  ci)
    ci = _timed("embedding", embedding._process_ci, ci)
    _timed("index",          idx._process_ci,       ci)

    elapsed  = round(time.perf_counter() - t_total, 3)
    ci_id    = ci.get("id", "unknown")
    entities = len(ci.get("ner", {}).get("entities", []))
    patterns = len(ci.get("ontology", {}).get("regex_patterns", []))
    embed_ok = bool(ci.get("embedding", {}).get("vector"))

    _rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import platform
    rss_mb = round(
        _rss_raw / 1024 if platform.system() != "Darwin" else _rss_raw / 1_048_576, 1
    )

    logger.info(
        "[CIWorker] done ci_id=%s entities=%d patterns=%d embedding=%s rss_mb=%.1f "
        "normalize=%.3fs ner=%.3fs ontology=%.3fs embedding=%.3fs index=%.3fs total=%.3fs",
        ci_id, entities, patterns, embed_ok, rss_mb,
        timings.get("normalize", 0), timings.get("ner", 0),
        timings.get("ontology", 0), timings.get("embedding", 0),
        timings.get("index", 0), elapsed,
    )

    return {
        "ok":        True,
        "ci_id":     ci_id,
        "entities":  entities,
        "patterns":  patterns,
        "embedding": embed_ok,
        "elapsed_s": elapsed,
        "timings":   timings,
    }


def handler(event: dict, context: Any) -> dict:
    # Direct invoke (no SQS Records) — smoke testing / orchestrator fan-out.
    if "Records" not in event:
        ci = _decode_payload(event)
        ci_id = ci.get("id", "unknown")
        logger.info("[CIWorker] start (direct) ci_id=%s", ci_id)
        result = _run_ci(ci, context)
        return result

    # SQS batch invoke.
    failures: list[dict[str, str]] = []
    processed = 0

    for record in event.get("Records", []):
        message_id = record.get("messageId", "")

        # Requeue if Lambda is nearly out of time.
        if context is not None:
            remaining_ms = context.get_remaining_time_in_millis()
            if remaining_ms < _MIN_REMAINING_MS:
                logger.warning(
                    "[CIWorker] only %d ms remaining — requeueing message_id=%s",
                    remaining_ms, message_id,
                )
                if message_id:
                    failures.append({"itemIdentifier": message_id})
                continue

        try:
            ci = _decode_payload(record)
            ci_id = ci.get("id", "unknown")
            logger.info("[CIWorker] start ci_id=%s", ci_id)
            result = _run_ci(ci, context)
            processed += 1
            logger.info(
                "[CIWorker] done ci_id=%s elapsed_s=%s",
                result.get("ci_id"), result.get("elapsed_s"),
            )
        except Exception as exc:
            logger.exception("[CIWorker] failed message_id=%s ci_id=%s error=%s",
                             message_id, record.get("body", "")[:80], exc)
            if message_id:
                failures.append({"itemIdentifier": message_id})

    return {
        "processed":        processed,
        "failed":           len(failures),
        "batchItemFailures": failures,
    }
