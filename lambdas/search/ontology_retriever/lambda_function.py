"""
Search Pipeline — Stage 2d: Ontology Retriever
================================================
Searches using the CI's curated synonyms and abbreviation expansions.
Best for:  CLINICAL_ROLE ("PI" → "Principal Investigator" → "Lead Investigator").

Input:  classified search request  (ci must have "ontology")
Output: { "retriever": "ontology", "hits": list[Hit] }
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OPENSEARCH_ENDPOINT = os.environ.get("OPENSEARCH_ENDPOINT", "localhost")
OPENSEARCH_INDEX    = os.environ.get("OPENSEARCH_INDEX", "document-chunks")
AWS_REGION          = os.environ.get("AWS_REGION", "us-east-1")
TOP_K               = int(os.environ.get("RETRIEVER_TOP_K", "10"))
TIE_BUFFER          = int(os.environ.get("RETRIEVER_TIE_BUFFER", "15"))


def _adaptive_k(page_count: int, base_k: int = 10) -> int:
    if page_count <= 0:    return base_k
    if page_count < 500:   return base_k
    if page_count < 3_000: return max(base_k, 25)
    if page_count < 10_000: return max(base_k, 50)
    return max(base_k, 75)


def _with_ties(sorted_hits: list[dict], k: int) -> list[dict]:
    if len(sorted_hits) <= k:
        return sorted_hits
    cutoff = sorted_hits[k - 1]["score"]
    result = sorted_hits[:k]
    for h in sorted_hits[k:]:
        if h["score"] == cutoff:
            result.append(h)
        else:
            break
    return result

from shared.opensearch_client import get_opensearch_client

def _get_os():
    return get_opensearch_client()


# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    search_id = event.get("search_id", "unknown")
    logger.info("[Ontology Retriever] start search_id=%s", search_id)
    try:
        result = _process(event)
    except Exception as exc:
        logger.error("[Ontology Retriever] failed search_id=%s error=%s", search_id, exc)
        raise
    logger.info("[Ontology Retriever] done search_id=%s hits=%d", search_id, len(result["hits"]))
    return result


def _process(req: dict) -> dict:
    ontology    = req["ci"].get("ontology", {})
    document_id = req.get("document_id")

    # Use canonical identity fields — not raw ontology expansions.
    # Canonical names are already resolved; searching for them in normalized_text
    # gives high-precision matches without the noise of raw abbreviations.
    ci_obj      = req.get("ci", {})
    search_terms: list[str] = []

    def _collect(identity: dict) -> None:
        for v in identity.values():
            if isinstance(v, list):
                search_terms.extend(str(x).lower() for x in v if x)
            elif isinstance(v, str) and v:
                search_terms.append(v.lower())

    _collect(ci_obj.get("clinical_identity",   {}))
    _collect(ci_obj.get("treatment_identity",  {}))
    _collect(ci_obj.get("endpoint_identity",   {}))
    _collect(ci_obj.get("population_identity", {}))

    # Deduplicate preserving order
    seen: set[str] = set()
    search_terms = [t for t in search_terms if t not in seen and not seen.add(t)]

    if not search_terms:
        return {"retriever": "ontology", "hits": []}

    page_count = int(req.get("document_page_count", 0))
    k          = _adaptive_k(page_count, TOP_K)
    hits = _ontology_search(search_terms, document_id, k)

    return {
        "retriever": "ontology",
        "hits":      _with_ties(hits, k),
    }


def _ontology_search(terms: list[str], document_id: str | None, k: int = TOP_K) -> list[dict]:
    filter_clause = [{"term": {"document_id": document_id}}] if document_id else []

    # One match clause per term — any match counts
    should_clauses = [
        {"match": {"normalized_text": {"query": term, "boost": 1.0}}}
        for term in terms
    ]

    body = {
        "size": k + TIE_BUFFER,
        "query": {
            "bool": {
                "filter": filter_clause,
                "should": should_clauses,
                "minimum_should_match": 1,
            }
        },
        "_source": ["chunk_id", "document_id", "page_start", "page_end", "raw_text"],
    }

    resp = _get_os().search(index=OPENSEARCH_INDEX, body=body)
    return _parse_hits(resp)


def _parse_hits(resp: dict) -> list[dict]:
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
