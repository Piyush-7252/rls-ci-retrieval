"""
Search Orchestrator Lambda
===========================
Splits a list of CIs into fixed-size batches, fans out to Search Worker
Lambdas (direct synchronous invoke via ThreadPoolExecutor), collects all
results and returns the merged response.

Input
-----
{
    "search_id":        str (optional, generated if absent),
    "cis":              list[dict],   # raw CIs (looked up from ci-objects index)
    "document_id":      str,
    "document_context": dict (optional, derived from document_assets.json if absent),
    "batch_size":       int  (default: 50),
    "skip_rerank":      bool (default: false),
    "skip_verify":      bool (default: false),
    "max_workers":      int  (default: number of batches)
}

Output
------
{
    "search_id":    str,
    "document_id":  str,
    "n_cis":        int,
    "n_batches":    int,
    "results":      list[dict],     # per-CI result with final_hits + timings
    "stage_wall":   dict[str, float],  # max wall-clock per stage across all batches
    "wall_time":    float
}

Env vars
--------
  WORKER_LAMBDA_ARN       — ARN of the Search Worker Lambda (required)
  OPENSEARCH_ENDPOINT     — host only (no https://)
  OPENSEARCH_CI_INDEX     — default: ci-objects
  EMBEDDING_MODEL         — Bedrock embedding model for CI lookup fallback
  AWS_REGION
  DOCUMENT_ASSETS_PATH    — local path to document_assets.json (optional)
  RESULTS_BUCKET          — S3 bucket to write full result JSON (required for Lambda invoke)
  RESULTS_PREFIX          — S3 key prefix (default: search-results)
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_task_root_env = os.environ.get("LAMBDA_TASK_ROOT")
if _task_root_env and (Path(_task_root_env) / "lambdas").exists():
    ROOT = Path(_task_root_env)
else:
    ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Env config ─────────────────────────────────────────────────────────────────
WORKER_LAMBDA_ARN    = os.environ.get("WORKER_LAMBDA_ARN", "")
OPENSEARCH_ENDPOINT  = os.environ.get("OPENSEARCH_ENDPOINT", "localhost")
OPENSEARCH_CI_INDEX  = os.environ.get("OPENSEARCH_CI_INDEX", "ci-objects")
AWS_REGION           = os.environ.get("AWS_REGION", "us-east-1")
EMBEDDING_MODEL      = os.environ.get("EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
DOCUMENT_ASSETS_PATH = os.environ.get(
    "DOCUMENT_ASSETS_PATH",
    str(ROOT / "localfiles" / "assets" / "document_assets.json"),
)
RESULTS_BUCKET  = os.environ.get("RESULTS_BUCKET", "")
RESULTS_PREFIX  = os.environ.get("RESULTS_PREFIX", "search-results")
# Max CIs processed in parallel within a single worker Lambda.
# Keep low on small OpenSearch clusters (2 search threads) to avoid 429s.
SEARCH_CI_WORKERS = int(os.environ.get("SEARCH_CI_WORKERS", "5"))

_DEFAULT_BATCH_SIZE = 50

# ── Lazy singletons ────────────────────────────────────────────────────────────
_aws: dict = {}
_os_client = None


def _get(service: str, region: str | None = None):
    key = f"{service}:{region or ''}"
    if key not in _aws:
        import boto3
        _aws[key] = boto3.client(service, region_name=region) if region else boto3.client(service)
    return _aws[key]


def _get_os():
    global _os_client
    if _os_client is None:
        import boto3
        from opensearchpy import OpenSearch, RequestsHttpConnection
        from requests_aws4auth import AWS4Auth
        frozen  = boto3.Session().get_credentials().get_frozen_credentials()
        awsauth = AWS4Auth(frozen.access_key, frozen.secret_key, AWS_REGION, "es",
                          session_token=frozen.token)
        _os_client = OpenSearch(
            hosts=[{"host": OPENSEARCH_ENDPOINT, "port": 443}],
            http_auth=awsauth, use_ssl=True, verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30,
            max_retries=2,
            retry_on_timeout=True,
        )
    return _os_client


# ── Document context ───────────────────────────────────────────────────────────

def _load_document_context(document_id: str) -> dict:
    path = Path(DOCUMENT_ASSETS_PATH)
    if not path.exists():
        return {}
    try:
        with path.open() as fh:
            return json.load(fh).get(document_id, {})
    except Exception as exc:
        logger.warning("[Orchestrator] document_context load failed: %s", exc)
        return {}


# ── CI enrichment lookup ───────────────────────────────────────────────────────

def _lookup_ci(raw_ci: dict) -> dict | None:
    """Fetch the enriched CI from the ci-objects OpenSearch index."""
    ci_id = raw_ci.get("id")
    if ci_id is None:
        return None
    try:
        from shared.opensearch_enrichment import ENRICHMENT_DEFAULTS
        resp = _get_os().get(index=OPENSEARCH_CI_INDEX, id=str(ci_id), ignore=[404])
        if not resp.get("found"):
            logger.warning("[Orchestrator] CI %s not found in ci-objects", ci_id)
            return None
        doc = resp["_source"]
        enrichment_fields = {k: doc.get(k, default)
                             for k, default in ENRICHMENT_DEFAULTS.items()}
        return {
            **raw_ci,
            **enrichment_fields,
            "entities":  doc.get("entities", []),
            "knownCI":   doc.get("known_ci", raw_ci.get("knownCI", "")),
            "normalization": {
                "normalized_text":     doc.get("normalized_text", ""),
                "tokens":              doc.get("tokens", []),
                "abbreviations_found": {},
            },
            "ner": {
                "entities": doc.get("entities", []),
                "model":    doc.get("ner_model", "gliner"),
            },
            "ontology": {
                "expansions":     doc.get("ontology_expansions", []),
                "synonyms":       doc.get("ontology_synonyms", {}),
                "regex_patterns": doc.get("regex_patterns", []),
            },
            "embedding": {
                "dense_vector":  doc.get("dense_vector", []),
                "sparse_vector": doc.get("sparse_vector", {}),
                "model":         doc.get("embedding_model", EMBEDDING_MODEL),
                "dimensions":    len(doc.get("dense_vector", [])),
            },
        }
    except Exception as exc:
        logger.warning("[Orchestrator] lookup failed ci_id=%s: %s", ci_id, exc)
        return None


def _load_cis_parallel(raw_cis: list[dict], n_workers: int) -> list[dict]:
    """Bulk-fetch enriched CIs from ci-objects in parallel."""
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_lookup_ci, raw_cis))
    enriched = [r for r in results if r is not None]
    logger.info("[Orchestrator] enriched %d/%d CIs", len(enriched), len(raw_cis))
    return enriched


# ── Worker invocation ──────────────────────────────────────────────────────────

def _invoke_worker(batch_payload: dict) -> dict:
    """Synchronously invoke the Search Worker Lambda and return its response."""
    if not WORKER_LAMBDA_ARN:
        raise RuntimeError("WORKER_LAMBDA_ARN env var is not set")
    resp = _get("lambda").invoke(
        FunctionName   = WORKER_LAMBDA_ARN,
        InvocationType = "RequestResponse",
        Payload        = json.dumps(batch_payload).encode(),
    )
    raw      = resp["Payload"].read()
    result   = json.loads(raw)
    # Lambda wraps unhandled errors in {"errorMessage": ..., "errorType": ...}
    if "errorMessage" in result:
        raise RuntimeError(
            f"Worker batch_idx={batch_payload['batch_idx']} failed: "
            f"{result.get('errorType')}: {result.get('errorMessage')}"
        )
    return result


# ── Result merge ───────────────────────────────────────────────────────────────

def _merge_stage_walls(walls: list[dict[str, float]]) -> dict[str, float]:
    """Return max wall-clock per stage across all batches (worst-case latency)."""
    merged: dict[str, float] = {}
    for w in walls:
        for k, v in w.items():
            merged[k] = max(merged.get(k, 0.0), v)
    return merged


# ── Handler ────────────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    search_id   = event.get("search_id") or str(uuid.uuid4())
    raw_cis     = event.get("cis", [])
    document_id = event.get("document_id", "")
    batch_size  = int(event.get("batch_size",  _DEFAULT_BATCH_SIZE))
    skip_rerank = bool(event.get("skip_rerank", False))
    skip_verify = bool(event.get("skip_verify",  False))
    ci_workers  = int(event.get("ci_workers", SEARCH_CI_WORKERS))

    logger.info("[Orchestrator] start search_id=%s doc=%s cis=%d batch_size=%d",
                search_id, document_id, len(raw_cis), batch_size)

    # ── Document context ────────────────────────────────────────────────────
    doc_context = event.get("document_context") or _load_document_context(document_id)

    # ── Load enriched CIs ───────────────────────────────────────────────────
    t0         = time.perf_counter()
    n_lookup_workers = min(max(len(raw_cis), 1), 50)
    enriched   = _load_cis_parallel(raw_cis, n_lookup_workers)
    if not enriched:
        logger.warning("[Orchestrator] no enriched CIs found — returning empty result")
        return {
            "search_id": search_id, "document_id": document_id,
            "n_cis": 0, "n_batches": 0, "results": [],
            "stage_wall": {}, "wall_time": round(time.perf_counter() - t0, 3),
        }

    # ── Split into batches ──────────────────────────────────────────────────
    batches = [enriched[i:i + batch_size] for i in range(0, len(enriched), batch_size)]
    n_batches = len(batches)
    logger.info("[Orchestrator] %d CIs → %d batches of ≤%d",
                len(enriched), n_batches, batch_size)

    # ── Fan-out: invoke all workers concurrently ────────────────────────────
    payloads = [
        {
            "search_id":        search_id,
            "batch_idx":        idx,
            "cis":              batch,
            "document_id":      document_id,
            "document_context": doc_context,
            "skip_rerank":      skip_rerank,
            "skip_verify":      skip_verify,
            "workers":          min(len(batch), ci_workers),
        }
        for idx, batch in enumerate(batches)
    ]

    n_invoke_workers = int(event.get("max_workers", n_batches))
    all_results:  list[dict]             = [{}] * n_batches
    all_walls:    list[dict[str, float]] = []
    errors:       list[str]              = []

    with ThreadPoolExecutor(max_workers=n_invoke_workers) as pool:
        futures = {pool.submit(_invoke_worker, p): p["batch_idx"] for p in payloads}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                batch_response = future.result()
                all_results[idx] = batch_response          # keyed by batch_idx
                all_walls.append(batch_response.get("stage_wall", {}))
                logger.info("[Orchestrator] batch %d/%d done (%.1fs)",
                            idx + 1, n_batches,
                            batch_response.get("wall_time", 0.0))
            except Exception as exc:
                logger.error("[Orchestrator] batch %d failed: %s", idx, exc)
                errors.append(str(exc))

    # Flatten per-CI results in original order
    flat_results: list[dict] = []
    for batch_resp in all_results:
        flat_results.extend(batch_resp.get("results", []))

    wall_time = round(time.perf_counter() - t0, 3)
    stage_wall = _merge_stage_walls(all_walls)

    total_hits = sum(len(r.get("final_hits", [])) for r in flat_results)
    logger.info("[Orchestrator] done search_id=%s wall=%.1fs cis=%d hits=%d errors=%d",
                search_id, wall_time, len(flat_results), total_hits, len(errors))

    response = {
        "search_id":   search_id,
        "document_id": document_id,
        "n_cis":       len(enriched),
        "n_batches":   n_batches,
        "results":     flat_results,
        "stage_wall":  stage_wall,
        "wall_time":   wall_time,
        **({"errors": errors} if errors else {}),
    }

    # ── Write full results to S3 (payload is too large for Lambda response) ──
    if RESULTS_BUCKET:
        import datetime
        ts      = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        s3_key  = f"{RESULTS_PREFIX}/{ts}_{search_id}_{document_id}.json"
        body    = json.dumps(response, default=str).encode()
        _get("s3").put_object(
            Bucket      = RESULTS_BUCKET,
            Key         = s3_key,
            Body        = body,
            ContentType = "application/json",
        )
        logger.info("[Orchestrator] results written s3://%s/%s (%d bytes)",
                    RESULTS_BUCKET, s3_key, len(body))
        # Return lightweight summary + pointer instead of 13MB payload
        return {
            "search_id":   search_id,
            "document_id": document_id,
            "n_cis":       len(enriched),
            "total_hits":  total_hits,
            "wall_time":   wall_time,
            "s3_bucket":   RESULTS_BUCKET,
            "s3_key":      s3_key,
            **({"errors": errors} if errors else {}),
        }

    return response
