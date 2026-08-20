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
}

Output
------
{
    "search_id":      str,
    "document_id":    str,
    "status":         str (COMPLETED | PARTIAL | FAILED),
    "expected_cis":   int,
    "completed_cis":  int,
    "failed_cis":     int,
    "batch_summary":  list[dict],      # per-batch status with CI failures
    "results":        list[dict],      # per-CI result with final_hits
    "stage_wall":     dict[str, float],  # max wall-clock per stage
    "wall_time":      float
}

Env vars
--------
  WORKER_LAMBDA_ARN       — ARN of the Search Worker Lambda (required)
                             e.g., arn:aws:lambda:eu-west-1:064051750322:function:rls-ci-retrieval-search-worker
  OPENSEARCH_ENDPOINT     — host only, no https:// (required)
                             e.g., search-rls-dev-rhitzxwnctmuyq2l4kny5kwelu.eu-west-1.es.amazonaws.com
  OPENSEARCH_CI_INDEX     — CI objects index (default: ci-objects)
  OPENSEARCH_MAXSIZE      — Connection pool size (default: 128)
  EMBEDDING_MODEL         — Bedrock embedding model for CI lookup fallback
  AWS_REGION              — AWS region (required)
  CI_LOOKUP_WORKERS       — Concurrent threads for CI enrichment (default: 10)
  MAX_WORKERS             — Concurrent Worker Lambda invocations (default: 3, prevents OpenSearch 429s)
  DOCUMENT_ASSETS_PATH    — Local path to document_assets.json (optional)
  RESULTS_BUCKET          — S3 bucket for full result JSON (required)
                             e.g., rls-file-bucket-eu
  RESULTS_PREFIX          — S3 key prefix (default: rls-ci-retrieval-search-results)
  
  Connection pool: OPENSEARCH_MAXSIZE (default 128) provides safety margin for concurrent
  CI lookups (up to CI_LOOKUP_WORKERS × 2 threads per invocation)
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
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from shared.utility import getTenantFromEvent

# ── Logging Context Variables (thread-safe for concurrent batch execution) ───────
_ctx_tenant = ContextVar("tenant", default="-")
_ctx_document_id = ContextVar("document_id", default="-")
_ctx_search_id = ContextVar("search_id", default="-")
_ctx_batch_idx = ContextVar("batch_idx", default="-")


class SearchContextFilter(logging.Filter):
    """Logging filter that injects search context into all log records.
    
    Automatically adds [tenant=...] [document=...] [search=...] [batch=...]
    prefix to every log from any module, without requiring manual propagation.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        tenant = _ctx_tenant.get()
        document_id = _ctx_document_id.get()
        search_id = _ctx_search_id.get()
        batch_idx = _ctx_batch_idx.get()
        
        prefix = f"[tenant={tenant}] [document={document_id}] [search={search_id}]"
        if batch_idx != "-":
            prefix += f" [batch={batch_idx}]"
        
        record.msg = f"{prefix} {record.msg}"
        return True


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configure root logger so all dynamically loaded modules inherit the context filter
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Add context filter to all existing handlers (preserve Lambda's CloudWatch handler)
context_filter = SearchContextFilter()
for handler in root_logger.handlers:
    handler.addFilter(context_filter)

# If no handlers exist, create one (e.g., local testing)
if not root_logger.handlers:
    import sys
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '[%(levelname)s]\t%(asctime)s\t%(name)s\t%(message)s'
    )
    handler.setFormatter(formatter)
    handler.addFilter(context_filter)
    root_logger.addHandler(handler)

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
OPENSEARCH_TIMEOUT   = int(os.environ.get("OPENSEARCH_TIMEOUT", "30"))
OPENSEARCH_MAXSIZE   = int(os.environ.get("OPENSEARCH_MAXSIZE", "256"))  # Connection pool size
AWS_REGION           = os.environ.get("AWS_REGION", "eu-west-1")
EMBEDDING_MODEL      = os.environ.get("EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
DOCUMENT_ASSETS_PATH = os.environ.get(
    "DOCUMENT_ASSETS_PATH",
    str(ROOT / "localfiles" / "assets" / "document_assets.json"),
)
RESULTS_BUCKET  = os.environ.get("RESULTS_BUCKET", "")
RESULTS_PREFIX  = os.environ.get("RESULTS_PREFIX", "search-results")
# Max CIs processed in parallel within a single worker Lambda.
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "3"))
CI_LOOKUP_WORKERS = int(os.environ.get("CI_LOOKUP_WORKERS", "10"))

_DEFAULT_BATCH_SIZE = 50

# ── Lazy singletons ────────────────────────────────────────────────────────────
_aws: dict = {}
_os_client = None


def _get(service: str, region: str | None = None):
    key = f"{service}:{region or ''}"
    if key not in _aws:
        import boto3
        from botocore.config import Config
        
        # For Lambda invocations, set longer read timeout (workers can take 100+ seconds)
        config = Config(
            read_timeout=180,  # 3 minutes for worker Lambda responses
            retries={'max_attempts': 1}  # Don't retry on timeout
        ) if service == "lambda" else None
        
        _aws[key] = boto3.client(service, region_name=region, config=config)
    return _aws[key]


def _get_os():
    global _os_client
    if _os_client is None:
        import boto3
        from opensearchpy import OpenSearch, RequestsHttpConnection
        from requests_aws4auth import AWS4Auth
        
        # Custom connection class that configures HTTPAdapter pool sizes
        class PooledRequestsHttpConnection(RequestsHttpConnection):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                from requests.adapters import HTTPAdapter
                adapter = HTTPAdapter(
                    pool_connections=OPENSEARCH_MAXSIZE,
                    pool_maxsize=OPENSEARCH_MAXSIZE,
                )
                self.session.mount("https://", adapter)
                self.session.mount("http://", adapter)
        
        frozen  = boto3.Session().get_credentials().get_frozen_credentials()
        awsauth = AWS4Auth(frozen.access_key, frozen.secret_key, AWS_REGION, "es",
                          session_token=frozen.token)
        _os_client = OpenSearch(
            hosts=[{"host": OPENSEARCH_ENDPOINT, "port": 443}],
            http_auth=awsauth, use_ssl=True, verify_certs=True,
            connection_class=PooledRequestsHttpConnection,
            timeout=OPENSEARCH_TIMEOUT,
            max_retries=2,
            retry_on_timeout=True,
            maxsize=OPENSEARCH_MAXSIZE,  # Connection pool size
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
        results = list(pool.map(lambda ci: _lookup_ci(ci), raw_cis))
    enriched = [r for r in results if r is not None]
    logger.info("[Orchestrator] enriched %d/%d CIs", len(enriched), len(raw_cis))
    return enriched


# ── Worker invocation ──────────────────────────────────────────────────────────

def _invoke_worker(batch_payload: dict) -> dict:
    """Synchronously invoke the Search Worker Lambda and return its response."""
    if not WORKER_LAMBDA_ARN:
        raise RuntimeError("WORKER_LAMBDA_ARN env var is not set")
    # Extract function name from ARN (format: arn:aws:lambda:region:account:function:name)
    function_name = WORKER_LAMBDA_ARN.split(":")[-1]
    resp = _get("lambda").invoke(
        FunctionName   = function_name,
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
    tenant     = getTenantFromEvent(event)
    batch_size  = int(event.get("batch_size",  _DEFAULT_BATCH_SIZE))
    skip_rerank = bool(event.get("skip_rerank", False))
    skip_verify = bool(event.get("skip_verify",  False))

    # ── Set logging context via ContextVar (applies to all logs automatically) ──────────────────
    _ctx_tenant.set(tenant["name"])
    _ctx_document_id.set(document_id)
    _ctx_search_id.set(search_id)

    logger.info("[Orchestrator] start cis=%d batch_size=%d", len(raw_cis), batch_size)

    # ── Document context ────────────────────────────────────────────────────
    doc_context = event.get("document_context") or _load_document_context(document_id)

    # ── Load enriched CIs ───────────────────────────────────────────────────
    t0         = time.perf_counter()
    # Create mapping of raw CIs by ID for later retrieval
    raw_ci_map = {ci.get("id"): ci for ci in raw_cis}
    n_lookup_workers = min(CI_LOOKUP_WORKERS, len(raw_cis)) if raw_cis else 1
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
    logger.info("[Orchestrator] %d CIs → %d batches of ≤%d", len(enriched), n_batches, batch_size)

    # ── Fan-out: invoke all workers concurrently ────────────────────────────
    payloads = [
        {
            "search_id":        search_id,
            "batch_idx":        idx,
            "cis":              batch,
            "document_id":      document_id,
            "tenant_name":      tenant["name"],  # Pass tenant for worker logging context
            "document_context": doc_context,
            "skip_rerank":      skip_rerank,
            "skip_verify":      skip_verify,
        }
        for idx, batch in enumerate(batches)
    ]

    # ── Concurrency: default to conservative 3 batches max (prevent OpenSearch 429s)
    n_invoke_workers = int(min(MAX_WORKERS, n_batches))
    logger.info("[Orchestrator] parallelism: %d batches max", n_invoke_workers)

    all_results:  list[dict]             = [{}] * n_batches
    batch_times:  dict[int, float]       = {}     # Track execution time per batch
    failed_batches: set[int]             = set()  # Track which batches failed
    errors:       list[dict]             = []     # Track batch_idx + error

    with ThreadPoolExecutor(max_workers=n_invoke_workers) as pool:
        futures = {pool.submit(_invoke_worker, p): p["batch_idx"] for p in payloads}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                batch_response = future.result()
                all_results[idx] = batch_response
                # Store batch execution time
                batch_times[idx] = batch_response.get("wall_time", 0.0)
                logger.info("[Orchestrator] batch %d/%d done (%.1fs)",
                         idx + 1, n_batches,
                         batch_response.get("wall_time", 0.0))
            except Exception as exc:
                logger.error("[Orchestrator] batch %d failed: %s", idx, exc)
                failed_batches.add(idx)
                batch_times[idx] = 0.0  # Mark failed batch with 0 time
                errors.append({
                    "batch_idx": idx,
                    "error": str(exc),
                })

    # Flatten per-CI results in original order, enriching with raw CI data
    flat_results: list[dict] = []
    total_worker_completed = 0
    total_worker_failed = 0
    batch_summary: list[dict] = []
    
    for batch_idx, batch_resp in enumerate(all_results):
        batch_results = batch_resp.get("results", [])
        
        # Attach raw CI data from the original raw_cis (not enriched)
        for result in batch_results:
            ci_id = result.get("ci_id") or result.get("id")
            if ci_id and ci_id in raw_ci_map:
                result["raw_ci"] = raw_ci_map[ci_id]
            flat_results.append(result)
        
        batch_completed = batch_resp.get("completed_cis", 0)
        batch_failed = batch_resp.get("failed_cis", 0)
        total_worker_completed += batch_completed
        total_worker_failed += batch_failed
        
        # For expected_cis: use actual count from original batches
        expected_batch_cis = len(batches[batch_idx]) if batch_idx < len(batches) else 0
        
        # Build per-batch summary with execution time
        batch_status = "FAILED" if batch_idx in failed_batches else (
            "PARTIAL" if batch_failed > 0 else "COMPLETED"
        )
        batch_summary.append({
            "batch_idx": batch_idx,
            "status": batch_status,
            "expected_cis": expected_batch_cis,
            "completed_cis": batch_completed,
            "failed_cis": batch_failed,
            "execution_time_ms": round(batch_times.get(batch_idx, 0.0) * 1000, 1),
            "ci_failures": batch_resp.get("ci_failures", []),  # Detailed CI failures from Worker
        })

    wall_time = round(time.perf_counter() - t0, 3)

    total_hits = sum(len(r.get("final_hits", [])) for r in flat_results)
    
    # ── Determine completion status ─────────────────────────────────────────────
    # Track both orchestrator-level (batch failures) and worker-level (CI failures)
    total_cis_at_orchestrator = len(enriched)
    
    # Status logic:
    # - FAILED: all batches failed (complete failure)
    # - PARTIAL: some batches failed OR some CIs failed within batches
    # - COMPLETED: all batches returned and all CIs completed
    if failed_batches and len(failed_batches) == n_batches:
        status = "FAILED"  # Complete failure
    elif failed_batches or total_worker_failed > 0:
        status = "PARTIAL"  # Some batches or CIs failed
    else:
        status = "COMPLETED"  # Everything succeeded
    
    logger.info(
        "[Orchestrator] done wall=%.1fs status=%s "
        "cis_expected=%d cis_completed=%d cis_failed_in_worker=%d failed_batches=%d",
        wall_time, status,
        total_cis_at_orchestrator, total_worker_completed, total_worker_failed, len(failed_batches),
    )

    # Build failed_batches list with execution times
    failed_batches_with_times = []
    for batch_idx in sorted(failed_batches):
        failed_batches_with_times.append({
            "batch_idx": batch_idx,
            "execution_time_ms": round(batch_times.get(batch_idx, 0.0) * 1000, 1),
            "error": next((e.get("error") for e in errors if e.get("batch_idx") == batch_idx), None),
        })
    
    response = {
        "search_id":       search_id,
        "document_id":     document_id,
        "status":          status,
        "expected_cis":    total_cis_at_orchestrator,
        "completed_cis":   total_worker_completed,
        "failed_cis":      total_worker_failed,
        "n_cis":           total_cis_at_orchestrator,
        "n_batches":       n_batches,
        "failed_batches":  failed_batches_with_times if failed_batches else [],
        "batch_summary":   batch_summary,
        "results":         flat_results,
        "wall_time":       wall_time,
    }
    if errors:
        response["errors"] = errors

    # ── Write full results to S3 (payload is too large for Lambda response) ──
    if RESULTS_BUCKET:
        s3_key = f"{RESULTS_PREFIX}/{tenant['name']}/{search_id}/{document_id}.json"

        body    = json.dumps(response, default=str).encode()
        _get("s3").put_object(
            Bucket      = RESULTS_BUCKET,
            Key         = s3_key,
            Body        = body,
            ContentType = "application/json",
        )
        logger.info("[Orchestrator] results written s3://%s/%s (%d bytes)",
                 RESULTS_BUCKET, s3_key, len(body))
        # Return lightweight summary + pointer
        summary = {
            "search_id":      search_id,
            "document_id":    document_id,
            "status":         status,
            "expected_cis":   total_cis_at_orchestrator,
            "completed_cis":  total_worker_completed,
            "failed_cis":     total_worker_failed,
            "total_hits":     total_hits,
            "wall_time":      wall_time,
            "batch_summary":  batch_summary,
            "failed_batches": failed_batches_with_times if failed_batches else [],
            "s3_bucket":      RESULTS_BUCKET,
            "s3_key":         s3_key,
        }
        if errors:
            summary["errors"] = errors
        return summary

    return response
