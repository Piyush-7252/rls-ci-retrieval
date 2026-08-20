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
import os
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
            maxsize=OPENSEARCH_MAXSIZE,  # Connection pool size
        )
    return _os_client


# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
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

    hits = _regex_search(compiled, document_id)

    return {
        "retriever": "regex",
        "hits":      hits,
    }


def _regex_search(
    patterns: list[re.Pattern],
    document_id: str | None,
) -> list[dict]:
    """Fetch all chunks and apply Python regex to raw_text."""
    filter_clause = [{"term": {"document_id": document_id}}] if document_id else []

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

    resp = _get_os().search(index=OPENSEARCH_INDEX, body=body)
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
