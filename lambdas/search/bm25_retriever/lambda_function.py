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

    # Search semantic-objects for precision; chunk index for recall
    obj_hits   = _bm25_search_objects(norm_text, tokens, document_id)
    chunk_hits = _bm25_search_chunks(norm_text, tokens, document_id)

    # Object hits take priority; fill remaining TOP_K with chunk hits not already covered
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

    return {
        "retriever": "bm25",
        "hits":      hits[:TOP_K],
    }


def _bm25_search_objects(norm_text: str, tokens: list[str], document_id: str | None) -> list[dict]:
    """BM25 search against semantic-objects index."""
    filter_clause = [{"term": {"document_id": document_id}}] if document_id else []
    query_text    = " ".join(tokens[:50]) if tokens else norm_text

    body = {
        "size": TOP_K,
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
            "position", "global_position", "document_position", "type", "text", "page", "bbox",
            "display_spans", "entities",
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


def _bm25_search_chunks(norm_text: str, tokens: list[str], document_id: str | None) -> list[dict]:
    """BM25 fallback against document-chunks for broad recall."""
    filter_clause = [{"term": {"document_id": document_id}}] if document_id else []
    query_text    = " ".join(tokens[:50]) if tokens else norm_text

    body = {
        "size": TOP_K,
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
                "text":               src["text"],
                "page":               src.get("page"),
                "bbox":               src.get("bbox", []),
                "display_spans":      src.get("display_spans", []),
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
