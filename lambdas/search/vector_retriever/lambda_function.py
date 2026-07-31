"""
Search Pipeline — Stage 2c: Vector Retriever
=============================================
Cosine similarity search using the CI's dense embedding.
Best for:  CLINICAL_ROLE, PHRASE, semantic variants.

Strategy: fetch all document chunks, compute cosine similarity in Python.
In production: use knn_vector mapping + OpenSearch k-NN plugin for efficiency.

Input:  classified search request  (ci must have "embedding.dense_vector")
Output: { "retriever": "vector", "hits": list[Hit] }
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OPENSEARCH_ENDPOINT    = os.environ.get("OPENSEARCH_ENDPOINT", "localhost")
OPENSEARCH_INDEX       = os.environ.get("OPENSEARCH_INDEX", "document-chunks")
SEMANTIC_OBJECTS_INDEX = os.environ.get("SEMANTIC_OBJECTS_INDEX", "semantic-objects")
AWS_REGION             = os.environ.get("AWS_REGION", "us-east-1")
TOP_K                  = int(os.environ.get("RETRIEVER_TOP_K", "10"))
FETCH_SIZE             = int(os.environ.get("VECTOR_FETCH_SIZE", "100"))

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
        )
    return _os_client


# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    search_id = event.get("search_id", "unknown")
    logger.info("[Vector Retriever] start search_id=%s", search_id)
    try:
        result = _process(event)
    except Exception as exc:
        logger.error("[Vector Retriever] failed search_id=%s error=%s", search_id, exc)
        raise
    logger.info("[Vector Retriever] done search_id=%s hits=%d", search_id, len(result["hits"]))
    return result


def _process(req: dict) -> dict:
    ci_embedding = req["ci"].get("embedding", {}).get("dense_vector", [])
    document_id  = req.get("document_id")

    if not ci_embedding:
        logger.warning("[Vector Retriever] no embedding on CI — returning empty")
        return {"retriever": "vector", "hits": []}

    # Lane 1: body-vector search (object text vs CI embedding)
    obj_hits   = _vector_search_objects(ci_embedding, document_id)
    # Lane 2: heading-vector search (heading text vs CI embedding)
    head_hits  = _vector_search_objects_heading(ci_embedding, document_id)
    # Lane 3: chunk-level fallback for broad recall
    chunk_hits = _vector_search_chunks(ci_embedding, document_id)

    # Merge: body hits first (most precise), then heading hits, then chunk fallback
    seen_chunks: set[str] = set()
    hits: list[dict] = []
    for h in obj_hits:
        hits.append(h)
        seen_chunks.add(h["chunk_id"])
    for h in head_hits:
        if h["chunk_id"] not in seen_chunks:
            hits.append(h)
            seen_chunks.add(h["chunk_id"])
    for h in chunk_hits:
        if h["chunk_id"] not in seen_chunks:
            hits.append(h)
            seen_chunks.add(h["chunk_id"])

    hits.sort(key=lambda x: x["score"], reverse=True)
    return {
        "retriever": "vector",
        "hits":      hits[:TOP_K],
    }


def _vector_search_objects(ci_embedding: list[float], document_id: str | None) -> list[dict]:
    """
    Search semantic-objects index by body dense_vector.
    Returns hits with full matched_object metadata.
    """
    filter_clause = [{"term": {"document_id": document_id}}] if document_id else []

    body = {
        "size": FETCH_SIZE,
        "query": {
            "bool": {
                "filter": filter_clause,
                "must":   [{"exists": {"field": "dense_vector"}}],
            }
        },
        "_source": [
            "object_id", "parent_chunk_id", "document_id",
            "position", "global_position", "type", "text", "page", "bbox",
            "display_spans", "dense_vector",
            "section_category", "heading_path", "semantic_path",
            "section_confidence", "document_position",
            "chunk_idx", "parent_chunk_idx", "prev_chunk_idx", "next_chunk_idx",
            # Semantic layer: canonical identity fields for reranker
            "effective_facts", "clinical_identity",
            "treatment_identity", "endpoint_identity", "population_identity",
            "modality", "study_context", "statement_type",
        ],
    }

    try:
        resp = _get_os().search(index=SEMANTIC_OBJECTS_INDEX, body=body)
    except Exception as exc:
        logger.warning("[Vector Retriever] semantic-objects search failed: %s", exc)
        return []

    raw_hits = resp.get("hits", {}).get("hits", [])
    scored: list[tuple[float, dict]] = []
    for h in raw_hits:
        src = h.get("_source", {})
        vec = src.get("dense_vector", [])
        if not vec:
            continue
        score = _cosine(ci_embedding, vec)
        scored.append((score, src))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        {
            "chunk_id":   s["parent_chunk_id"],
            "score":      round(score, 4),
            "page_start": s.get("page", 0),
            "page_end":   s.get("page", 0),
            "snippet":    s.get("text", "")[:200],
            "matched_object": _build_matched_object(s),
        }
        for score, s in scored[:TOP_K]
    ]


def _build_matched_object(s: dict) -> dict:
    """Build the full matched_object dict from a semantic-objects _source."""
    return {
        "object_id":          s["object_id"],
        "parent_chunk_id":    s["parent_chunk_id"],
        "document_id":        s["document_id"],
        "position":           s.get("position"),
        "global_position":    s.get("global_position"),
        "type":               s["type"],
        "text":               s["text"],
        "page":               s.get("page"),
        "bbox":               s.get("bbox", []),
        "display_spans":      s.get("display_spans", []),
        "entities":           s.get("entities", []),
        # Section / semantic metadata
        "section_category":   s.get("section_category"),
        "heading_path":       s.get("heading_path"),
        "semantic_path":      s.get("semantic_path"),
        "section_confidence": s.get("section_confidence"),
        "document_position":  s.get("document_position"),
        # Sentence neighbour context
        "prev_sentence_text": s.get("prev_sentence_text"),
        "next_sentence_text": s.get("next_sentence_text"),
        "paragraph_text":     s.get("paragraph_text"),
        # Chunk adjacency
        "chunk_idx":          s.get("chunk_idx"),
        "parent_chunk_idx":   s.get("parent_chunk_idx"),
        "prev_chunk_idx":     s.get("prev_chunk_idx"),
        "next_chunk_idx":     s.get("next_chunk_idx"),
        # Semantic layer: canonical identity fields (no raw entities / neighbor text)
        "effective_facts":     s.get("effective_facts", {}),
        "clinical_identity":   s.get("clinical_identity", {}),
        "treatment_identity":  s.get("treatment_identity", {}),
        "endpoint_identity":   s.get("endpoint_identity", {}),
        "population_identity": s.get("population_identity", {}),
        "modality":            s.get("modality", "GENERAL"),
        "study_context":       s.get("study_context", "GENERAL"),
        "statement_type":      s.get("statement_type"),
    }


def _vector_search_objects_heading(ci_embedding: list[float], document_id: str | None) -> list[dict]:
    """
    Search semantic-objects index by heading_dense_vector.

    Retrieves objects whose *heading* is semantically close to the CI query,
    even when the object body text doesn’t directly mention the topic.
    Scores are slightly penalised (0.90×) so body-vector hits take priority
    when both lanes return the same chunk.
    """
    if not ci_embedding:
        return []
    filter_clause = [{"term": {"document_id": document_id}}] if document_id else []
    body = {
        "size": FETCH_SIZE,
        "query": {"bool": {
            "filter": filter_clause,
            "must":   [{"exists": {"field": "heading_dense_vector"}}],
        }},
        "_source": [
            "object_id", "parent_chunk_id", "document_id",
            "position", "global_position", "type", "text", "page", "bbox",
            "display_spans", "heading_dense_vector",
            "section_category", "heading_path", "semantic_path",
            "section_confidence", "document_position",
            "chunk_idx", "parent_chunk_idx", "prev_chunk_idx", "next_chunk_idx",
            "effective_facts", "clinical_identity",
            "treatment_identity", "endpoint_identity", "population_identity",
            "modality", "study_context", "statement_type",
        ],
    }
    try:
        resp = _get_os().search(index=SEMANTIC_OBJECTS_INDEX, body=body)
    except Exception as exc:
        logger.warning("[Vector Retriever] heading vector search failed: %s", exc)
        return []

    raw_hits = resp.get("hits", {}).get("hits", [])
    scored: list[tuple[float, dict]] = []
    for h in raw_hits:
        src = h.get("_source", {})
        vec = src.get("heading_dense_vector", [])
        if not vec:
            continue
        score = _cosine(ci_embedding, vec) * 0.90   # slight penalty vs body search
        scored.append((score, src))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "chunk_id":   s["parent_chunk_id"],
            "score":      round(score, 4),
            "page_start": s.get("page", 0),
            "page_end":   s.get("page", 0),
            "snippet":    s.get("text", "")[:200],
            "matched_object": _build_matched_object(s),
        }
        for score, s in scored[:TOP_K]
    ]


def _vector_search_chunks(ci_embedding: list[float], document_id: str | None) -> list[dict]:
    """
    Fallback: search document-chunks for broad recall.
    Returns hits WITHOUT matched_object (context_expander handles those).
    """
    filter_clause = [{"term": {"document_id": document_id}}] if document_id else []

    body = {
        "size": FETCH_SIZE,
        "query": {
            "bool": {
                "filter": filter_clause,
                "must":   [{"exists": {"field": "dense_vector"}}],
            }
        },
        "_source": ["chunk_id", "document_id", "page_start", "page_end", "raw_text",
                    "dense_vector"],
    }

    try:
        resp = _get_os().search(index=OPENSEARCH_INDEX, body=body)
    except Exception as exc:
        logger.warning("[Vector Retriever] document-chunks search failed: %s", exc)
        return []

    raw_hits = resp.get("hits", {}).get("hits", [])
    scored: list[tuple[float, dict]] = []
    for h in raw_hits:
        src = h.get("_source", {})
        vec = src.get("dense_vector", [])
        if not vec:
            continue
        score = _cosine(ci_embedding, vec)
        scored.append((score, src))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        {
            "chunk_id":   s["chunk_id"],
            "score":      round(score, 4),
            "page_start": s.get("page_start", 0),
            "page_end":   s.get("page_end",   0),
            "snippet":    s.get("raw_text", "")[:200],
            # no matched_object — context_expander will fetch context_objects by position
        }
        for score, s in scored[:TOP_K]
    ]


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a))
    nb   = math.sqrt(sum(y * y for y in b))
    denom = na * nb
    return dot / denom if denom else 0.0
