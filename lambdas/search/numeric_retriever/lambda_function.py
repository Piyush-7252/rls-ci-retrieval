"""
Search Pipeline — Stage 2f: Numeric Retriever
===============================================
Structured numeric/statistical retrieval for CIs where the number IS the secret.

Activated for CI types: CONFIDENCE_INTERVAL, P_VALUE, HAZARD_RATIO, ODDS_RATIO,
NUMERIC_PERCENTAGE, NUMERIC_SAMPLE_SIZE, MEDIAN (and legacy NUMERIC / STATISTICAL).

Two-tier retrieval strategy
-----------------------------
  Tier 1 — Structured filter (primary, near-zero false positives):
    Build an OpenSearch filter query on statistical_identity.type + value fields.
    Only documents where the enrichment pipeline extracted the SAME statistical
    type and matching values are returned.

    Example for "95% CI: 27-48 days":
      filter: statistical_identity.type = "confidence_interval"
              statistical_identity.lower_ci = 27
              statistical_identity.upper_ci = 48

    This eliminates "Treatment started on Day 27. Assessment on Day 48." because
    that object has no statistical_identity.type = "confidence_interval".

  Tier 2 — Token must query (fallback, only when tier 1 returns 0 hits):
    Require the key numeric tokens to appear in the document text.
    Used when the document was indexed BEFORE statistical_identity was added to
    the enrichment schema (i.e., a stale index that hasn't been reprocessed yet).

Why not vector search?
-----------------------
  "n = 8" has no semantic neighbourhood — "n = 8" and "n = 80" are equidistant
  in vector space.  "dose level 8" is semantically similar to "n = 8".

  Typed filter queries resolve both problems:
    filter: statistical_identity.type = "sample_size"
            statistical_identity.sample_size = 8
  -> only documents where a sample size of 8 was explicitly extracted pass.

Input:  classified search request  (ci must have statistical_identity set)
Output: { "retriever": "numeric", "hits": list[Hit] }
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OPENSEARCH_ENDPOINT    = os.environ.get("OPENSEARCH_ENDPOINT", "localhost")
SEMANTIC_OBJECTS_INDEX = os.environ.get("SEMANTIC_OBJECTS_INDEX", "semantic-objects")
AWS_REGION             = os.environ.get("AWS_REGION", "us-east-1")
TOP_K                  = int(os.environ.get("RETRIEVER_TOP_K", "10"))


def _adaptive_k(page_count: int, base_k: int = 10) -> int:
    if page_count <= 0:    return base_k
    if page_count < 500:   return base_k
    if page_count < 3_000: return max(base_k, 25)
    if page_count < 10_000: return max(base_k, 50)
    return max(base_k, 75)

_os_client = None


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


# Source fields — full ENRICHMENT_DEFAULTS parity
_SOURCE_FIELDS: list[str] = [
    "object_id", "parent_chunk_id", "document_id",
    "position", "global_position", "document_position",
    "type", "text", "page", "bbox",
    "display_spans",
    "section_category", "heading_path", "semantic_path", "section_confidence",
    "prev_sentence_text", "next_sentence_text", "paragraph_text",
    "entities",
    "facts", "own_facts", "effective_facts", "inherited_slots", "slot_provenance",
    "study_context", "statement_type", "object_subtype", "modality",
    "clinical_relations",
    "clinical_identity", "treatment_identity", "endpoint_identity",
    "population_identity", "temporal_context",
    "statistical_identity",
    "study_hierarchy", "negated_slots", "clinical_signature",
]


# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    search_id = event.get("search_id", "unknown")
    logger.info("[Numeric Retriever] start search_id=%s", search_id)
    try:
        result = _process(event)
    except Exception as exc:
        logger.error("[Numeric Retriever] failed search_id=%s error=%s", search_id, exc)
        raise
    logger.info("[Numeric Retriever] done search_id=%s hits=%d", search_id, len(result["hits"]))
    return result


def _process(req: dict) -> dict:
    ci          = req.get("ci", {})
    ci_text     = ci.get("knownCI", "")
    si          = ci.get("statistical_identity") or {}
    document_id = req.get("document_id")

    page_count = int(req.get("document_page_count", 0))
    k          = _adaptive_k(page_count, TOP_K)

    # Tier 1: structured filter query (near-zero false positives)
    body = _build_structured_query(si, document_id)
    if body is not None:
        body["size"] = k
        try:
            resp = _get_os().search(index=SEMANTIC_OBJECTS_INDEX, body=body)
            hits = _parse_hits(resp)
            if hits:
                logger.info("[Numeric Retriever] structured filter: %d hits", len(hits))
                return {"retriever": "numeric", "hits": hits}
            logger.debug(
                "[Numeric Retriever] structured filter: 0 hits "
                "(document may predate statistical_identity indexing) — falling back"
            )
        except Exception as exc:
            logger.warning("[Numeric Retriever] structured filter failed: %s — falling back", exc)

    # Tier 2: token must query (fallback for un-reindexed documents)
    body = _build_token_query(si, ci_text, document_id)
    if body is None:
        logger.debug("[Numeric Retriever] no numeric tokens found — returning empty")
        return {"retriever": "numeric", "hits": []}
    body["size"] = k

    try:
        resp = _get_os().search(index=SEMANTIC_OBJECTS_INDEX, body=body)
        hits = _parse_hits(resp)
        logger.info("[Numeric Retriever] token fallback: %d hits", len(hits))
        return {"retriever": "numeric", "hits": hits}
    except Exception as exc:
        logger.warning("[Numeric Retriever] token fallback failed: %s", exc)
        return {"retriever": "numeric", "hits": []}


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — Structured filter query
# ─────────────────────────────────────────────────────────────────────────────

def _build_structured_query(si: dict, document_id: str | None) -> dict | None:
    """
    Filter on statistical_identity.type + value fields.

    Returns None when si has no type discriminant.
    All clauses are filters (no scoring) — the aggregator re-scores candidates.
    """
    si_type = si.get("type")
    if not si_type:
        return None

    filter_clauses: list[dict] = []
    if document_id:
        filter_clauses.append({"term": {"document_id": document_id}})

    # Type gate: eliminates every object of the wrong statistical kind.
    # "dose level 8" has no statistical_identity.type="sample_size" -> filtered out.
    filter_clauses.append({"term": {"statistical_identity.type.keyword": si_type}})

    # Value filters: exact match on the numeric value(s).
    if si_type == "confidence_interval":
        lower = si.get("lower_ci")
        upper = si.get("upper_ci")
        if lower is not None:
            filter_clauses.append({"term": {"statistical_identity.lower_ci": lower}})
        if upper is not None:
            filter_clauses.append({"term": {"statistical_identity.upper_ci": upper}})

    elif si_type == "sample_size":
        val = si.get("sample_size")
        if val is not None:
            filter_clauses.append({"term": {"statistical_identity.sample_size": val}})

    elif si_type == "p_value":
        val = si.get("p_value")
        if val is not None:
            filter_clauses.append({"term": {"statistical_identity.p_value": val}})

    elif si_type == "hazard_ratio":
        val = si.get("hazard_ratio")
        if val is not None:
            filter_clauses.append({"term": {"statistical_identity.hazard_ratio": val}})

    elif si_type == "odds_ratio":
        val = si.get("odds_ratio")
        if val is not None:
            filter_clauses.append({"term": {"statistical_identity.odds_ratio": val}})

    elif si_type == "median":
        val = si.get("median")
        if val is not None:
            filter_clauses.append({"term": {"statistical_identity.median": val}})

    elif si_type == "percentage":
        val = si.get("percentage")
        if val is not None:
            filter_clauses.append({"term": {"statistical_identity.percentage": val}})

    return {
        "size":    TOP_K,
        "query":   {"bool": {"filter": filter_clauses}},
        "_source": _SOURCE_FIELDS,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — Token must query (fallback for un-reindexed documents)
# ─────────────────────────────────────────────────────────────────────────────

_NUMERIC_TOKEN_RE = re.compile(r'\b\d+(?:\.\d+)?\b')


def _num_tok(val: float | int) -> str:
    """Return the canonical string token for a numeric value (no trailing .0)."""
    if isinstance(val, float) and val == int(val):
        return str(int(val))
    return str(val)


def _extract_numeric_tokens(text: str) -> list[str]:
    """Extract distinct numeric strings from raw CI text."""
    seen: dict[str, None] = {}
    for m in _NUMERIC_TOKEN_RE.finditer(text):
        seen[m.group()] = None
    return list(seen.keys())


def _build_token_query(si: dict, ci_text: str, document_id: str | None) -> dict | None:
    """
    Token-based must query: require the key numeric tokens to appear in text.
    Fallback for documents indexed before statistical_identity was added.
    """
    filter_clauses: list[dict] = [{"term": {"document_id": document_id}}] if document_id else []
    must: list[dict] = []

    lower = si.get("lower_ci")
    upper = si.get("upper_ci")

    if lower is not None and upper is not None:
        must.append({"match": {"text": {"query": _num_tok(lower)}}})
        must.append({"match": {"text": {"query": _num_tok(upper)}}})
    elif si.get("sample_size") is not None:
        must.append({"match": {"text": {"query": str(si["sample_size"])}}})
    elif si.get("p_value") is not None:
        must.append({"match": {"text": {"query": str(si["p_value"])}}})
    elif si.get("hazard_ratio") is not None:
        must.append({"match": {"text": {"query": _num_tok(si["hazard_ratio"])}}})
    elif si.get("odds_ratio") is not None:
        must.append({"match": {"text": {"query": _num_tok(si["odds_ratio"])}}})
    elif si.get("median") is not None:
        must.append({"match": {"text": {"query": _num_tok(si["median"])}}})
    elif si.get("percentage") is not None:
        must.append({"match": {"text": {"query": _num_tok(si["percentage"])}}})
    else:
        tokens = _extract_numeric_tokens(ci_text)
        if not tokens:
            return None
        for tok in tokens[:3]:
            must.append({"match": {"text": {"query": tok}}})

    return {
        "size":    TOP_K,
        "query":   {"bool": {"filter": filter_clauses, "must": must}},
        "_source": _SOURCE_FIELDS,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Hit parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_hits(resp: dict) -> list[dict]:
    hits = []
    seen_chunks: set[str] = set()   # deduplicate: keep only the first (best) object per chunk
    for h in resp.get("hits", {}).get("hits", []):
        src        = h.get("_source", {})
        chunk_id   = src.get("parent_chunk_id", "")
        if chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk_id)
        hits.append({
            "chunk_id":   chunk_id,
            "score":      round(h.get("_score", 0.0), 4),
            "page_start": src.get("page", 0),
            "page_end":   src.get("page", 0),
            "snippet":    src.get("text", "")[:200],
            "matched_object": {
                "object_id":            src.get("object_id"),
                "parent_chunk_id":      src.get("parent_chunk_id"),
                "document_id":          src.get("document_id"),
                "position":             src.get("position"),
                "global_position":      src.get("global_position"),
                "document_position":    src.get("document_position"),
                "type":                 src.get("type"),
                "text":                 src.get("text"),
                "page":                 src.get("page"),
                "bbox":                 src.get("bbox", []),
                "display_spans":        src.get("display_spans", []),
                "section_category":     src.get("section_category"),
                "heading_path":         src.get("heading_path"),
                "semantic_path":        src.get("semantic_path"),
                "section_confidence":   src.get("section_confidence"),
                "prev_sentence_text":   src.get("prev_sentence_text"),
                "next_sentence_text":   src.get("next_sentence_text"),
                "paragraph_text":       src.get("paragraph_text"),
                "entities":             src.get("entities", []),
                "facts":                src.get("facts", {}),
                "own_facts":            src.get("own_facts", {}),
                "effective_facts":      src.get("effective_facts", {}),
                "inherited_slots":      src.get("inherited_slots", []),
                "slot_provenance":      src.get("slot_provenance", {}),
                "clinical_identity":    src.get("clinical_identity", {}),
                "treatment_identity":   src.get("treatment_identity", {}),
                "endpoint_identity":    src.get("endpoint_identity", {}),
                "population_identity":  src.get("population_identity", {}),
                "temporal_context":     src.get("temporal_context", {}),
                "statistical_identity": src.get("statistical_identity", {}),
                "modality":             src.get("modality", "GENERAL"),
                "object_subtype":       src.get("object_subtype", "GENERAL"),
                "clinical_relations":   src.get("clinical_relations", []),
                "statement_type":       src.get("statement_type"),
                "study_context":        src.get("study_context", "GENERAL"),
                "study_hierarchy":      src.get("study_hierarchy", {}),
                "negated_slots":        src.get("negated_slots", []),
                "clinical_signature":   src.get("clinical_signature", {}),
            },
        })
    return hits
