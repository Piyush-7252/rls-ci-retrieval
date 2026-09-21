"""
Search Pipeline — Stage 2b: BM25 Retriever
============================================
Multi-field BM25 keyword search across raw text, tokens, and entities.
Best for:  PHRASE, IDENTIFIER, any text-based CI.

Input:  classified search request
Output: { "retriever": "bm25", "hits": list[Hit] }
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OPENSEARCH_ENDPOINT    = os.environ.get("OPENSEARCH_ENDPOINT", "localhost")
OPENSEARCH_INDEX       = os.environ.get("OPENSEARCH_INDEX", "document-chunks")
SEMANTIC_OBJECTS_INDEX = os.environ.get("SEMANTIC_OBJECTS_INDEX", "semantic-objects")
AWS_REGION             = os.environ.get("AWS_REGION", "us-east-1")
TOP_K                  = int(os.environ.get("RETRIEVER_TOP_K", "10"))
TIE_BUFFER             = int(os.environ.get("RETRIEVER_TIE_BUFFER", "15"))
# Score-decay threshold (relative to top score): keep all hits >= top * ratio.
# 0.0 = disabled → falls back to adaptive-k + tie-inclusion.
# 0.80 is a sensible starting point: drops candidates that score < 80% of the best hit.
BM25_SCORE_RATIO       = float(os.environ.get("BM25_SCORE_RATIO", "0.0"))
BM25_MAX_HITS          = int(os.environ.get("BM25_MAX_HITS", "150"))

from shared.opensearch_client import get_opensearch_client

def _get_os():
    return get_opensearch_client()


# ─── Adaptive retrieval depth ─────────────────────────────────────────────────

def _adaptive_k(page_count: int, base_k: int = 10) -> int:
    """Scale retrieval depth with document size.

    Small documents (<500 pages) keep the base TOP_K to avoid unnecessary cost.
    Large clinical documents (CSRs, dossiers) scale up so evidence beyond rank-10
    is reachable.  Set document_page_count=0 (or omit) to fall back to base_k.
    """
    if page_count <= 0:
        return base_k
    if page_count < 500:
        return base_k
    if page_count < 3_000:
        return max(base_k, 25)
    if page_count < 10_000:
        return max(base_k, 50)
    return max(base_k, 75)


def _with_ties(sorted_hits: list[dict], k: int) -> list[dict]:
    """Return top-k hits plus any additional hits that tie the score at rank k.

    Prevents cutting through identical-scored candidates at the boundary, which
    is common when many document sections mention the same clinical term.
    """
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

def _score_decay_filter(sorted_hits: list[dict], ratio: float, max_hits: int) -> list[dict]:
    """Keep all hits whose score is within `ratio` of the top score.

    Example: ratio=0.80, top_score=95 → keep all hits with score >= 76.
    Dense distributions (many hits at similar scores) produce many results;
    sparse distributions (big score drop after rank-1) produce few.
    Always bounded by max_hits.
    """
    if not sorted_hits:
        return sorted_hits
    threshold = sorted_hits[0]["score"] * ratio
    return [h for h in sorted_hits if h["score"] >= threshold][:max_hits]

# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    search_id = event.get("search_id", "unknown")
    logger.info("[BM25 Retriever] start search_id=%s", search_id)
    try:
        result = _process(event)
    except Exception as exc:
        logger.error("[BM25 Retriever] failed search_id=%s error=%s", search_id, exc)
        raise
    logger.info("[BM25 Retriever] done search_id=%s hits=%d", search_id, len(result["hits"]))
    return result


def _process(req: dict) -> dict:
    norm_text   = req["ci"].get("normalization", {}).get("normalized_text", "")
    tokens      = req["ci"].get("normalization", {}).get("tokens", [])
    document_id = req.get("document_id")
    tenant = req.get("tenant")
    project_id = req.get("project_id")
    tenant_id = tenant.get("tenant_id")

    page_count  = int(req.get("document_page_count", 0))
    k           = _adaptive_k(page_count, TOP_K)

    # Choose fetch size: when score-decay is active we need to see a wider pool
    # so the threshold operates on enough candidates; otherwise use adaptive k.
    ratio      = BM25_SCORE_RATIO
    fetch_size = BM25_MAX_HITS if ratio > 0.0 else k + TIE_BUFFER

    # Search semantic-objects for precision; chunk index for recall
    obj_hits   = _bm25_search_objects(norm_text, tokens, document_id, tenant_id=tenant_id, project_id=project_id, fetch_size=fetch_size)
    chunk_hits = _bm25_search_chunks(norm_text, tokens, document_id, tenant_id=tenant_id, project_id=project_id, fetch_size=fetch_size)

    # Object hits take priority; fill remaining slots with chunk hits not already covered
    seen_chunks: set[str] = set()
    hits: list[dict] = []
    for h in obj_hits:
        hits.append(h)
        seen_chunks.add(h["chunk_id"])
    for h in chunk_hits:
        if h["chunk_id"] not in seen_chunks:
            hits.append(h)
            seen_chunks.add(h["chunk_id"])

    hits.sort(key=lambda x: x["score"], reverse=True)

    # Score-decay mode: drop candidates below ratio * top_score
    # Fallback: adaptive-k with tie inclusion (no threshold configured)
    if ratio > 0.0 and hits:
        hits = _score_decay_filter(hits, ratio, BM25_MAX_HITS)
    else:
        hits = _with_ties(hits, k)

    return {
        "retriever": "bm25",
        "hits":      hits,
    }


def _bm25_search_objects(norm_text: str, tokens: list[str], document_id: str | None, tenant_id: str | None = None, project_id: str | None = None, fetch_size: int = TOP_K) -> list[dict]:
    """BM25 search against semantic-objects index."""
    filter_clause = [{"term": {"document_id": document_id}}] if document_id else []
    if tenant_id:
        filter_clause.append({"term": {"tenant_id": tenant_id}})
    if project_id:
        filter_clause.append({"term": {"project_id": project_id}})
    query_text    = " ".join(tokens[:50]) if tokens else norm_text

    body = {
        "size": fetch_size,
        "query": {
            "bool": {
                "filter": filter_clause,
                "should": [
                    {
                        "multi_match": {
                            "query":  query_text,
                            "fields": ["text^2"],
                            "type":   "best_fields",
                        }
                    },
                    {
                        "multi_match": {
                            "query":  norm_text,
                            "fields": ["text"],
                            "type":   "cross_fields",
                            "boost":  1.5,
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        },
        "_source": [
            "object_id", "parent_chunk_id", "document_id",
            "position", "global_position", "document_position", "type", "text", "page", "bbox", "geometry",
    "list_id", "list_level", "list_label", "list_number_format",
    "table_id", "row_index", "row_start", "col_start", "row_span", "col_span",
            "list_id", "list_level", "list_label", "list_number_format",
            "table_id", "row_index", "row_start", "col_start", "row_span", "col_span",
            "entities",
            "section_category", "heading_path", "semantic_path", "section_confidence",
            "prev_sentence_text", "next_sentence_text", "paragraph_text",
            # raw + propagated fact slots
            "facts", "own_facts", "effective_facts", "inherited_slots", "slot_provenance",
            # classification
            "study_context", "statement_type", "object_subtype", "modality",
            # relations
            "clinical_relations",
            # identity layer
            "clinical_identity", "treatment_identity", "endpoint_identity",
            "population_identity", "temporal_context",
            # structural / provenance
            "study_hierarchy", "negated_slots", "clinical_signature",
            # numeric / statistical
            "statistical_identity",
        ],
    }

    try:
        resp = _get_os().search(index=SEMANTIC_OBJECTS_INDEX, body=body)
    except Exception as exc:
        logger.warning("[BM25 Retriever] semantic-objects search failed: %s", exc)
        return []

    return _parse_object_hits(resp)


def _bm25_search_chunks(norm_text: str, tokens: list[str], document_id: str | None, tenant_id: str | None = None, project_id: str | None = None, fetch_size: int = TOP_K) -> list[dict]:
    """BM25 fallback against document-chunks for broad recall."""
    filter_clause = [{"term": {"document_id": document_id}}] if document_id else []
    if tenant_id:
        filter_clause.append({"term": {"tenant_id": tenant_id}})
    if project_id:
        filter_clause.append({"term": {"project_id": project_id}})
    query_text    = " ".join(tokens[:50]) if tokens else norm_text

    body = {
        "size": fetch_size,
        "query": {
            "bool": {
                "filter": filter_clause,
                "should": [
                    {
                        "multi_match": {
                            "query":  query_text,
                            "fields": ["normalized_text^2", "raw_text"],
                            "type":   "best_fields",
                        }
                    },
                    {
                        "multi_match": {
                            "query":  norm_text,
                            "fields": ["normalized_text"],
                            "type":   "cross_fields",
                            "boost":  1.5,
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        },
        "_source": ["chunk_id", "document_id", "page_start", "page_end", "raw_text"],
    }

    try:
        resp = _get_os().search(index=OPENSEARCH_INDEX, body=body)
    except Exception as exc:
        logger.warning("[BM25 Retriever] document-chunks search failed: %s", exc)
        return []

    return _parse_chunk_hits(resp)


def _parse_object_hits(resp: dict) -> list[dict]:
    hits = []
    for h in resp.get("hits", {}).get("hits", []):
        src = h.get("_source", {})
        hits.append({
            "chunk_id":       src.get("parent_chunk_id", ""),
            "score":          round(h.get("_score", 0.0), 4),
            "page_start":     src.get("page", 0),
            "page_end":       src.get("page", 0),
            "snippet":        src.get("text", "")[:200],
            "matched_object": {
                "object_id":          src["object_id"],
                "parent_chunk_id":    src["parent_chunk_id"],
                "document_id":        src["document_id"],
                "position":           src.get("position"),
                "global_position":    src.get("global_position"),
                "document_position":  src.get("document_position"),
                "type":               src["type"],
                "list_id":            src.get("list_id"),
                "list_level":         src.get("list_level"),
                "list_label":         src.get("list_label"),
                "list_number_format": src.get("list_number_format"),
                "table_id":           src.get("table_id", src.get("table_key")),
                "row_index":          src.get("row_index", src.get("row_start")),
                "row_start":          src.get("row_start"),
                "col_start":          src.get("col_start"),
                "row_span":           src.get("row_span"),
                "col_span":           src.get("col_span"),
                "text":               src["text"],
                "page":               src.get("page"),
                "bbox":               src.get("bbox", []),
                "geometry":           src.get("geometry") or {},
                "entities":           src.get("entities", []),
                "section_category":   src.get("section_category"),
                "heading_path":       src.get("heading_path"),
                "semantic_path":      src.get("semantic_path"),
                "section_confidence": src.get("section_confidence"),
                "prev_sentence_text": src.get("prev_sentence_text"),
                "next_sentence_text": src.get("next_sentence_text"),
                "paragraph_text":     src.get("paragraph_text"),
                "facts":               src.get("facts", {}),
                "own_facts":           src.get("own_facts", {}),
                "study_context":       src.get("study_context", "GENERAL"),
                "statement_type":      src.get("statement_type"),
                "clinical_relations":  src.get("clinical_relations", []),
                # Semantic enrichment layer — full ENRICHMENT_DEFAULTS parity with fact_retriever
                "effective_facts":     src.get("effective_facts", {}),
                "inherited_slots":     src.get("inherited_slots", []),
                "slot_provenance":     src.get("slot_provenance", {}),
                "clinical_identity":   src.get("clinical_identity", {}),
                "treatment_identity":  src.get("treatment_identity", {}),
                "endpoint_identity":   src.get("endpoint_identity", {}),
                "population_identity": src.get("population_identity", {}),
                "temporal_context":    src.get("temporal_context", {}),
                "modality":            src.get("modality", "GENERAL"),
                "object_subtype":      src.get("object_subtype", "GENERAL"),
                "study_hierarchy":     src.get("study_hierarchy", {}),
                "negated_slots":       src.get("negated_slots", []),
                "clinical_signature":  src.get("clinical_signature", {}),
                "statistical_identity": src.get("statistical_identity", {}),
            },
        })
    return hits


def _parse_chunk_hits(resp: dict) -> list[dict]:
    hits = []
    for h in resp.get("hits", {}).get("hits", []):
        src = h.get("_source", {})
        hits.append({
            "chunk_id":   src.get("chunk_id", h["_id"]),
            "score":      round(h.get("_score", 0.0), 4),
            "page_start": src.get("page_start", 0),
            "page_end":   src.get("page_end",   0),
            "snippet":    src.get("raw_text", "")[:200],
            # no matched_object — context_expander fetches context_objects by position
        })
    return hits
