"""
Search Pipeline — Stage 2a: Literal Retriever
===============================================
Finds chunks via exact phrase match and fuzzy match on ``raw_text``.
Best for:  PERSON names, verbatim CI text.

Input:  classified search request  (must have "classification")
Output: { "retriever": "literal", "hits": list[Hit] }

Hit schema
----------
{ "chunk_id": str, "score": float, "page_start": int, "page_end": int, "snippet": str }
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OPENSEARCH_ENDPOINT = os.environ.get("OPENSEARCH_ENDPOINT", "localhost")
OPENSEARCH_INDEX    = os.environ.get("OPENSEARCH_INDEX", "document-chunks")
AWS_REGION          = os.environ.get("AWS_REGION", "us-east-1")
TOP_K               = int(os.environ.get("RETRIEVER_TOP_K", "10"))
# Literal retriever returns ALL exact matches (up to LITERAL_MAX).
# Exact/phrase hits are always evidence — never cap on a hard k.
LITERAL_MAX         = int(os.environ.get("LITERAL_MAX", "200"))

from shared.opensearch_client import get_opensearch_client

def _get_os():
    return get_opensearch_client()


# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    search_id = event.get("search_id", "unknown")
    logger.info("[Literal Retriever] start search_id=%s", search_id)
    try:
        result = _process(event)
    except Exception as exc:
        logger.error("[Literal Retriever] failed search_id=%s error=%s", search_id, exc)
        raise
    logger.info("[Literal Retriever] done search_id=%s hits=%d", search_id, len(result["hits"]))
    return result


def _process(req: dict) -> dict:
    ci_text     = req["ci"].get("knownCI", "")
    norm_text   = req["ci"].get("normalization", {}).get("normalized_text", ci_text)
    document_id = req.get("document_id")
    tenant = req.get("tenant")
    project_id = req.get("project_id")
    tenant_id = tenant.get("tenant_id")

    hits = _literal_search(ci_text, norm_text, document_id, tenant_id=tenant_id, project_id=project_id)

    return {
        "retriever": "literal",
        "hits":      hits,
    }


def _extract_literal_matches(ci_text: str, raw_text: str) -> list[dict]:
    """
    Find where ci_text (or its significant sub-phrases) appear in raw_text.

    Returns [{"text": str, "start": int, "end": int}, ...] sorted by position.
    Passed forward so Stage 6.5 can surface the exact matched term instead of
    re-discovering it through the scorer registry.

    Strategy 1 — whole phrase: "adverse events, serious AEs" as one substring.
    Strategy 2 — sub-phrases: split on commas/semicolons/newlines, search each
                  phrase ≥ 5 chars independently.
    """
    matches: list[dict] = []
    raw_lower = raw_text.lower()

    # Strategy 1 — whole phrase
    ci_s = ci_text.strip()
    idx  = raw_lower.find(ci_s.lower())
    if idx >= 0:
        return [{"text": raw_text[idx: idx + len(ci_s)], "start": idx, "end": idx + len(ci_s)}]

    # Strategy 2 — significant sub-phrases
    sub_phrases = [p.strip() for p in re.split(r'[,;\n]+', ci_s) if len(p.strip()) >= 5]
    for phrase in sub_phrases:
        idx = raw_lower.find(phrase.lower())
        if idx >= 0:
            matches.append({"text": raw_text[idx: idx + len(phrase)],
                            "start": idx, "end": idx + len(phrase)})

    # Deduplicate overlapping spans, keep leftmost
    seen: set[int] = set()
    unique: list[dict] = []
    for m in sorted(matches, key=lambda x: x["start"]):
        if m["start"] not in seen:
            seen.add(m["start"])
            unique.append(m)
    return unique


def _literal_search(ci_text: str, norm_text: str, document_id: str | None, tenant_id: str | None = None, project_id: str | None = None) -> list[dict]:
    filter_clause = [{"term": {"document_id": document_id}}] if document_id else []
    if tenant_id:
        filter_clause.append({"term": {"tenant_id": tenant_id}})
    if project_id:
        filter_clause.append({"term": {"project_id": project_id}})

    body = {
        "size": LITERAL_MAX,
        "query": {
            "bool": {
                "filter": filter_clause,
                "should": [
                    # Exact phrase match on raw text (highest weight)
                    {
                        "match_phrase": {
                            "raw_text": {
                                "query": ci_text,
                                "boost": 3.0,
                            }
                        }
                    },
                    # Phrase on normalized text
                    {
                        "match_phrase": {
                            "normalized_text": {
                                "query": norm_text,
                                "boost": 2.0,
                            }
                        }
                    },
                    # Fuzzy match for OCR errors / typos (short CI texts only —
                    # long texts expand to thousands of term variations and hit
                    # OpenSearch's 1024 maxClauseCount limit).
                    *([{
                        "match": {
                            "raw_text": {
                                "query":     ci_text,
                                "fuzziness": "AUTO",
                                "boost":     1.0,
                            }
                        }
                    }] if len(ci_text) <= 50 else []),
                ],
                "minimum_should_match": 1,
            }
        },
        "_source": ["chunk_id", "document_id", "page_start", "page_end", "raw_text"],
    }

    resp = _get_os().search(index=OPENSEARCH_INDEX, body=body)
    return _parse_hits(resp, ci_text)


def _parse_hits(resp: dict, ci_text: str = "") -> list[dict]:
    hits = []
    for h in resp.get("hits", {}).get("hits", []):
        src = h.get("_source", {})
        raw = src.get("raw_text", "")
        hits.append({
            "chunk_id":        src.get("chunk_id", h["_id"]),
            "score":           round(h.get("_score", 0.0), 4),
            "page_start":      src.get("page_start", 0),
            "page_end":        src.get("page_end",   0),
            "snippet":         raw[:200],
            "literal_matches": _extract_literal_matches(ci_text, raw) if ci_text else [],
        })
    return hits
