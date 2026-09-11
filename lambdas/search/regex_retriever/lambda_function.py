"""
Search Pipeline — Stage 2e: Regex Retriever
=============================================
Applies the CI's compiled regex patterns against raw_text.
Best for:  IDENTIFIER (NCT numbers, protocol IDs, patient IDs, dates).

Strategy: fetch candidate chunks via BM25 (broad), then apply Python regex
for precision.  OpenSearch regexp is limited; Python re gives full control.

Input:  classified search request  (ci must have "ontology.regex_patterns")
Output: { "retriever": "regex", "hits": list[Hit] }
"""

from __future__ import annotations

import logging
import json
import os
import time
import re
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OPENSEARCH_ENDPOINT = os.environ.get("OPENSEARCH_ENDPOINT", "localhost")
OPENSEARCH_INDEX    = os.environ.get("OPENSEARCH_INDEX", "document-chunks")
OPENSEARCH_MAXSIZE  = int(os.environ.get("OPENSEARCH_MAXSIZE", "256"))  # Connection pool size
AWS_REGION          = os.environ.get("AWS_REGION", "us-east-1")
TOP_K               = int(os.environ.get("RETRIEVER_TOP_K", "10"))
FETCH_SIZE          = int(os.environ.get("REGEX_FETCH_SIZE", "200"))

from shared.opensearch_client import get_opensearch_client

def _get_os():
    return get_opensearch_client()


# ─────────────────────────────────────────────────────────────────────────────


# ── Diagnostic OpenSearch logging ─────────────────────────────────────────────
VECTOR_DEBUG_LOG_BODY = os.environ.get("VECTOR_DEBUG_LOG_BODY", "true").lower() == "true"

def _debug_redact(value: Any) -> Any:
    """Redact embedding vectors while preserving the exact query structure."""
    if isinstance(value, dict):
        return {
            k: ("<VECTOR REDACTED>" if k == "vector" else _debug_redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_debug_redact(v) for v in value]
    return value

def _debug_request(label: str, operation: str, index: str | None, body: Any = None) -> None:
    if not VECTOR_DEBUG_LOG_BODY:
        return
    try:
        logger.info(
            "[OS DEBUG] REQUEST op=%s label=%s index=%s body=%s",
            operation, label, index,
            json.dumps(_debug_redact(body), separators=(",", ":"), default=str)
        )
    except Exception as exc:
        logger.warning("[OS DEBUG] REQUEST_LOG_FAILED label=%s error=%s", label, exc)

def _debug_search(label: str, index: str, body: dict, **kwargs):
    t0 = time.perf_counter()
    _debug_request(label, "search", index, body)
    try:
        resp = _get_os().search(index=index, body=body, **kwargs)
        logger.info(
            "[OS DEBUG] DONE op=search label=%s index=%s elapsed_ms=%.1f "
            "status=%s took_ms=%s hits=%s shard_total=%s shard_failed=%s",
            label, index, (time.perf_counter()-t0)*1000,
            getattr(getattr(resp, "meta", None), "status", None),
            resp.get("took"),
            len(resp.get("hits", {}).get("hits", [])),
            resp.get("_shards", {}).get("total"),
            resp.get("_shards", {}).get("failed"),
        )
        return resp
    except Exception as exc:
        logger.error(
            "[OS DEBUG] ERROR op=search label=%s index=%s elapsed_ms=%.1f "
            "error_type=%s error=%s",
            label, index, (time.perf_counter()-t0)*1000,
            type(exc).__name__, exc, exc_info=True
        )
        raise

def _debug_msearch(label: str, index: str, body: Any, **kwargs):
    t0 = time.perf_counter()
    _debug_request(label, "msearch", index, body)
    try:
        resp = _debug_msearch("_debug_msearch", index=index, **kwargs)
        responses = resp.get("responses", [])
        logger.info(
            "[OS DEBUG] DONE op=msearch label=%s index=%s elapsed_ms=%.1f "
            "responses=%s errors=%s",
            label, index, (time.perf_counter()-t0)*1000,
            len(responses),
            sum(1 for r in responses if r.get("error")),
        )
        return resp
    except Exception as exc:
        logger.error(
            "[OS DEBUG] ERROR op=msearch label=%s index=%s elapsed_ms=%.1f "
            "error_type=%s error=%s",
            label, index, (time.perf_counter()-t0)*1000,
            type(exc).__name__, exc, exc_info=True
        )
        raise

def _debug_mget(label: str, index: str, body: Any, **kwargs):
    t0 = time.perf_counter()
    _debug_request(label, "mget", index, body)
    try:
        resp = _debug_mget("_regex_search", index=index, body=body, **kwargs)
        logger.info(
            "[OS DEBUG] DONE op=mget label=%s index=%s elapsed_ms=%.1f docs=%s",
            label, index, (time.perf_counter()-t0)*1000,
            len(resp.get("docs", [])),
        )
        return resp
    except Exception as exc:
        logger.error(
            "[OS DEBUG] ERROR op=mget label=%s index=%s elapsed_ms=%.1f "
            "error_type=%s error=%s",
            label, index, (time.perf_counter()-t0)*1000,
            type(exc).__name__, exc, exc_info=True
        )
        raise

def handler(event: dict, context: Any) -> dict:
    logger.info(
        "[Retriever DEBUG] START retriever=%s search_id=%s ci_id=%s ci_type=%s "
        "strategies=%s document_id=%s",
        __name__,
        event.get("search_id"),
        (event.get("ci") or {}).get("id"),
        ((event.get("classification") or {}).get("ci_type")
         if isinstance(event.get("classification"), dict) else None),
        (event.get("classification") or {}).get("strategies", [])
        if isinstance(event.get("classification"), dict) else [],
        event.get("document_id"),
    )
    search_id = event.get("search_id", "unknown")
    logger.info("[Regex Retriever] start search_id=%s", search_id)
    try:
        result = _process(event)
    except Exception as exc:
        logger.error("[Regex Retriever] failed search_id=%s error=%s", search_id, exc)
        raise
    logger.info("[Regex Retriever] done search_id=%s hits=%d", search_id, len(result["hits"]))
    return result


def _process(req: dict) -> dict:
    patterns    = req["ci"].get("ontology", {}).get("regex_patterns", [])
    ci_text     = req["ci"].get("knownCI", "")
    document_id = req.get("document_id")
    tenant = req.get("tenant")
    project_id = req.get("project_id")
    tenant_id = tenant.get("tenant_id")

    if not patterns:
        # Fallback: build patterns from raw CI text
        patterns = [re.escape(ci_text)]

    # Compile once — skip invalid patterns
    compiled: list[re.Pattern] = []
    for pat in patterns:
        try:
            compiled.append(re.compile(pat, re.IGNORECASE))
        except re.error:
            logger.warning("[Regex Retriever] invalid pattern skipped: %s", pat)

    if not compiled:
        return {"retriever": "regex", "hits": []}

    hits = _regex_search(compiled, document_id, tenant_id=tenant_id, project_id=project_id)

    return {
        "retriever": "regex",
        "hits":      hits,
    }


def _regex_search(
    patterns: list[re.Pattern],
    document_id: str | None,
    tenant_id: str | None = None,
    project_id: str | None = None
) -> list[dict]:
    """Fetch all chunks and apply Python regex to raw_text."""
    filter_clause = [{"term": {"document_id": document_id}}] if document_id else []
    if tenant_id:
        filter_clause.append({"term": {"tenant_id": tenant_id}})
    if project_id:
        filter_clause.append({"term": {"project_id": project_id}})

    body = {
        "size": FETCH_SIZE,
        "query": {
            "bool": {
                "filter": filter_clause,
                "must":   [{"match_all": {}}],
            }
        },
        "_source": ["chunk_id", "document_id", "page_start", "page_end", "raw_text"],
    }

    resp = _debug_search("_regex_search", OPENSEARCH_INDEX, body)
    hits: list[dict] = []

    for h in resp.get("hits", {}).get("hits", []):
        src      = h.get("_source", {})
        raw_text = src.get("raw_text", "")

        # Count how many patterns match; use match count as score proxy
        match_count = 0
        first_snippet = ""
        for pat in patterns:
            m = pat.search(raw_text)
            if m:
                match_count += 1
                if not first_snippet:
                    start   = max(0, m.start() - 80)
                    end     = min(len(raw_text), m.end() + 80)
                    first_snippet = raw_text[start:end]

        if match_count > 0:
            hits.append({
                "chunk_id":   src.get("chunk_id", h["_id"]),
                "score":      float(match_count),
                "page_start": src.get("page_start", 0),
                "page_end":   src.get("page_end",   0),
                "snippet":    first_snippet[:200],
            })

    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits[:TOP_K]
