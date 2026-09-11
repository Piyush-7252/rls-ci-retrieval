"""
Search Pipeline — Stage 2f: NER Retriever
===========================================
Matches by overlapping named entities between the CI and document chunks.
Best for:  PERSON names, ORGANIZATION names, clinical identifiers.

Input:  classified search request  (ci must have "ner.entities")
Output: { "retriever": "ner", "hits": list[Hit] }
"""

from __future__ import annotations

import logging
import json
import os
import time
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OPENSEARCH_ENDPOINT = os.environ.get("OPENSEARCH_ENDPOINT", "localhost")
OPENSEARCH_INDEX    = os.environ.get("OPENSEARCH_INDEX", "document-chunks")
AWS_REGION          = os.environ.get("AWS_REGION", "us-east-1")
TOP_K               = int(os.environ.get("RETRIEVER_TOP_K", "10"))


def _adaptive_k(page_count: int, base_k: int = 10) -> int:
    if page_count <= 0:    return base_k
    if page_count < 500:   return base_k
    if page_count < 3_000: return max(base_k, 25)
    if page_count < 10_000: return max(base_k, 50)
    return max(base_k, 75)

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
        resp = _debug_mget("_parse_hits", index=index, body=body, **kwargs)
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
    logger.info("[NER Retriever] start search_id=%s", search_id)
    try:
        result = _process(event)
    except Exception as exc:
        logger.error("[NER Retriever] failed search_id=%s error=%s", search_id, exc)
        raise
    logger.info("[NER Retriever] done search_id=%s hits=%d", search_id, len(result["hits"]))
    return result


def _process(req: dict) -> dict:
    entities    = req["ci"].get("ner", {}).get("entities", [])
    document_id = req.get("document_id")
    tenant = req.get("tenant")
    project_id = req.get("project_id")
    tenant_id = tenant.get("tenant_id")

    # Extract unique entity texts
    entity_texts = list({e.get("text", "").lower() for e in entities if e.get("text")})

    if not entity_texts:
        return {"retriever": "ner", "hits": []}

    page_count = int(req.get("document_page_count", 0))
    k          = _adaptive_k(page_count, TOP_K)
    hits = _ner_search(entity_texts, document_id, tenant_id=tenant_id, project_id=project_id, k=k)

    return {
        "retriever": "ner",
        "hits":      hits,
    }


def _ner_search(entity_texts: list[str], document_id: str | None, tenant_id: str | None = None, project_id: str | None = None, k: int = TOP_K) -> list[dict]:
    filter_clause = [{"term": {"document_id": document_id}}] if document_id else []
    if tenant_id:
        filter_clause.append({"term": {"tenant_id": tenant_id}})
    if project_id:
        filter_clause.append({"term": {"project_id": project_id}})

    # Search for chunks whose entity list overlaps with CI entity texts
    should_clauses = [
        {
            "match": {
                "normalized_text": {
                    "query": text,
                    "boost": 2.0,
                }
            }
        }
        for text in entity_texts
    ]

    body = {
        "size": k,
        "query": {
            "bool": {
                "filter": filter_clause,
                "should": should_clauses,
                "minimum_should_match": 1,
            }
        },
        "_source": ["chunk_id", "document_id", "page_start", "page_end", "raw_text",
                    "entities"],
    }

    resp = _debug_search("_ner_search", OPENSEARCH_INDEX, body)
    return _parse_hits(resp, entity_texts)


def _parse_hits(resp: dict, entity_texts: list[str]) -> list[dict]:
    hits = []
    for h in resp.get("hits", {}).get("hits", []):
        src = h.get("_source", {})
        hits.append({
            "chunk_id":   src.get("chunk_id", h["_id"]),
            "score":      round(h.get("_score", 0.0), 4),
            "page_start": src.get("page_start", 0),
            "page_end":   src.get("page_end",   0),
            "snippet":    src.get("raw_text", "")[:200],
        })
    return hits
