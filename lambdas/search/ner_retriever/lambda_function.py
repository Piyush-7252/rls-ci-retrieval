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
import os
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

    # Extract unique entity texts
    entity_texts = list({e.get("text", "").lower() for e in entities if e.get("text")})

    if not entity_texts:
        return {"retriever": "ner", "hits": []}

    page_count = int(req.get("document_page_count", 0))
    k          = _adaptive_k(page_count, TOP_K)
    hits = _ner_search(entity_texts, document_id, k)

    return {
        "retriever": "ner",
        "hits":      hits,
    }


def _ner_search(entity_texts: list[str], document_id: str | None, k: int = TOP_K) -> list[dict]:
    filter_clause = [{"term": {"document_id": document_id}}] if document_id else []

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

    resp = _get_os().search(index=OPENSEARCH_INDEX, body=body)
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
