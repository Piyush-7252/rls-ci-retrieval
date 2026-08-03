"""
Search Pipeline — Stage 4: Context Expander
=============================================
For each candidate chunk, fetches the chunk's text from OpenSearch and
retrieves neighbouring chunks (±1 page range) to provide surrounding context.

Input:  aggregated search request  (must have "candidates")
Appends: "expanded_candidates": list[ExpandedCandidate]

ExpandedCandidate schema
-------------------------
{
    "chunk_id":      str,
    "page_start":    int,
    "page_end":      int,
    "sources":       list[str],
    "agg_score":     float,
    "context": {
        "prev_text":    str,   # last 500 chars of preceding chunk (if any)
        "current_text": str,   # full raw_text of this chunk
        "next_text":    str,   # first 500 chars of following chunk (if any)
    }
}
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OPENSEARCH_ENDPOINT    = os.environ.get("OPENSEARCH_ENDPOINT", "localhost")
OPENSEARCH_INDEX       = os.environ.get("OPENSEARCH_INDEX", "document-chunks")
SEMANTIC_OBJECTS_INDEX = os.environ.get("SEMANTIC_OBJECTS_INDEX", "semantic-objects")
AWS_REGION             = os.environ.get("AWS_REGION", "us-east-1")
RERANKER_LAMBDA_ARN    = os.environ.get("RERANKER_LAMBDA_ARN", "")
CONTEXT_CHARS          = int(os.environ.get("CONTEXT_CHARS", "500"))
CONTEXT_WINDOW         = int(os.environ.get("CONTEXT_WINDOW", "3"))   # objects before+after match

_aws: dict = {}
_os_client = None

def _get(service: str):
    if service not in _aws:
        import boto3
        _aws[service] = boto3.client(service)
    return _aws[service]

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
    logger.info("[Context Expander] start search_id=%s", search_id)
    try:
        result = _process(event)
    except Exception as exc:
        logger.error("[Context Expander] failed search_id=%s error=%s", search_id, exc)
        raise
    logger.info("[Context Expander] done search_id=%s expanded=%d",
                search_id, len(result["expanded_candidates"]))
    if RERANKER_LAMBDA_ARN:
        _get("lambda").invoke(
            FunctionName   = RERANKER_LAMBDA_ARN,
            InvocationType = "Event",
            Payload        = json.dumps(result).encode(),
        )
    return result


def _process(req: dict) -> dict:
    candidates  = req.get("candidates", [])
    document_id = req.get("document_id")

    expanded = [_expand(cand, document_id) for cand in candidates]

    return {
        **req,
        "expanded_candidates": expanded,
    }


def _expand(candidate: dict, document_id: str | None) -> dict:
    chunk_id   = candidate["chunk_id"]
    page_start = candidate.get("page_start", 0)
    page_end   = candidate.get("page_end",   0)

    # Fetch the chunk's raw text (for broad LLM context)
    chunk_doc    = _fetch_chunk(chunk_id)
    current_text = chunk_doc.get("raw_text", "")

    # If candidate came from semantic-objects index it already has the matched object
    matched_obj = candidate.get("matched_object")   # set by retriever for object-level hits

    # For sentence-level hits, replace current_text with the tight 3-sentence window
    # stored inline (prev_sentence_text + sentence + next_sentence_text).
    # This gives the verifier precise, focused context rather than the full paragraph.
    # The chunk-level prev/next text still provides broader surrounding context.
    if matched_obj and matched_obj.get("type") == "sentence":
        parts = [
            matched_obj.get("prev_sentence_text") or "",
            matched_obj.get("text") or "",
            matched_obj.get("next_sentence_text") or "",
        ]
        sentence_window = " ".join(p for p in parts if p).strip()
        if sentence_window:
            current_text = sentence_window

    # Extract adjacency indices set by the section chunker
    obj_meta         = matched_obj or {}
    chunk_idx        = obj_meta.get("chunk_idx")
    prev_chunk_idx   = obj_meta.get("prev_chunk_idx")
    next_chunk_idx   = obj_meta.get("next_chunk_idx")
    parent_chunk_idx = obj_meta.get("parent_chunk_idx")

    # Prefer exact adjacency-index lookup over page-range heuristic
    if prev_chunk_idx is not None:
        prev_text = _fetch_chunk_by_idx(document_id, prev_chunk_idx)
    else:
        prev_text = _fetch_neighbor(document_id, page_end=page_start - 1)

    if next_chunk_idx is not None:
        next_text = _fetch_chunk_by_idx(document_id, next_chunk_idx)
    else:
        next_text = _fetch_neighbor(document_id, page_start=page_end + 1)

    # Parent chunk carries the heading context for child sections.
    # E.g. parent "6.9 Dose Modifications" gives BGE context for
    # the child "6.9.2 Pomalidomide Dose Modifications".
    parent_text = ""
    if parent_chunk_idx is not None:
        parent_text = _fetch_chunk_by_idx(document_id, parent_chunk_idx)

    # Fetch neighboring objects — use global_position for cross-chunk context
    context_objects = _fetch_context_objects(
        document_id  = document_id or candidate.get("document_id", ""),
        chunk_id     = chunk_id,
        center_pos   = obj_meta.get("global_position"),
    )

    # For chunk-level hits (no matched semantic object), promote the best object
    # found within the chunk so that statement_type, facts, entities, and
    # clinical_relations are available to the reranker and scoring stages.
    # _fetch_context_objects already queries by parent_chunk_id when center_pos
    # is None, so this reuses results that were fetched anyway — no extra query.
    if matched_obj is None and context_objects:
        _OBJECT_TYPE_PRIORITY = {"sentence": 3, "paragraph": 2, "heading": 1, "table_row": 1}
        matched_obj = max(
            context_objects,
            key=lambda o: _OBJECT_TYPE_PRIORITY.get(o.get("type", ""), 0),
        )

    return {
        # Carry forward ALL aggregator-computed fields (agg_score, score_breakdown,
        # entity_overlap, contradiction, study_context, etc.) so the reranker and
        # downstream diagnostics can see them.  Context-specific keys override below.
        **candidate,
        "matched_object":  matched_obj,       # the exact retrieved semantic object
        "context_objects": context_objects,   # neighbouring objects for display
        "context": {
            "parent_text":  parent_text[:CONTEXT_CHARS]  if parent_text else "",
            "prev_text":    prev_text[-CONTEXT_CHARS:]   if prev_text  else "",
            "current_text": current_text,
            "next_text":    next_text[:CONTEXT_CHARS]    if next_text  else "",
        },
    }


def _fetch_chunk_by_idx(document_id: str | None, chunk_idx: int) -> str:
    """
    Exact lookup: fetch raw_text for the chunk at ``chunk_idx`` within
    the given document.  More reliable than page-range heuristics because
    chunk_idx is a monotone integer assigned by the section chunker.
    """
    if not document_id:
        return ""
    body = {
        "size": 1,
        "query": {"bool": {"filter": [
            {"term": {"document_id.keyword": document_id}},
            {"term": {"chunk_idx":   chunk_idx}},
        ]}},
        "_source": ["raw_text"],
    }
    try:
        resp = _get_os().search(index=OPENSEARCH_INDEX, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        return hits[0]["_source"].get("raw_text", "") if hits else ""
    except Exception as exc:
        logger.warning("[Context Expander] chunk_idx=%s lookup failed: %s", chunk_idx, exc)
        return ""


def _fetch_context_objects(
    document_id: str,
    chunk_id:    str,
    center_pos:  int | None,
    window:      int = CONTEXT_WINDOW,
) -> list[dict]:
    """
    Fetch the window of semantic objects surrounding a matched position.

    Uses global_position for the query so context expansion works seamlessly
    across chunk boundaries (e.g. Object 198 in chunk 1, Object 199 in chunk 2).
    Falls back to parent_chunk_id filter when global_position is unavailable.
    """
    if not document_id:
        return []

    if center_pos is not None:
        # Primary: global_position range (document-wide, crosses chunk boundaries)
        filter_clauses: list[dict] = [
            {"term":  {"document_id.keyword": document_id}},
            {"range": {"global_position": {"gte": center_pos - window,
                                            "lte": center_pos + window}}},
        ]
        fetch_size = window * 2 + 1
    else:
        # Fallback for chunk-level hits with no matched object
        filter_clauses = [{"term": {"parent_chunk_id": chunk_id}}]
        fetch_size = 100

    body = {
        "size": fetch_size,
        "query": {"bool": {"filter": filter_clauses}},
        "sort":  [{"global_position": "asc"}],
    }

    try:
        resp = _get_os().search(index=SEMANTIC_OBJECTS_INDEX, body=body)
        return [h["_source"] for h in resp.get("hits", {}).get("hits", [])]
    except Exception as exc:
        logger.warning("[Context Expander] context objects fetch failed chunk=%s: %s", chunk_id, exc)
        return []


def _fetch_chunk(chunk_id: str) -> dict:
    """Fetch raw_text for a chunk (broad display context only)."""
    try:
        resp = _get_os().get(index=OPENSEARCH_INDEX, id=chunk_id)
        return resp.get("_source", {})
    except Exception as exc:
        logger.warning("[Context Expander] could not fetch chunk %s: %s", chunk_id, exc)
        return {}


def _fetch_chunk_text(chunk_id: str) -> str:
    return _fetch_chunk(chunk_id).get("raw_text", "")


def _fetch_neighbor(
    document_id: str | None,
    page_start:  int | None = None,
    page_end:    int | None = None,
) -> str:
    """Fetch the chunk whose page range is adjacent to the given boundary."""
    if not document_id:
        return ""

    filter_clauses: list[dict] = [{"term": {"document_id.keyword": document_id}}]
    if page_start is not None:
        filter_clauses.append({"term": {"page_start": page_start}})
    if page_end is not None:
        filter_clauses.append({"term": {"page_end": page_end}})

    body = {
        "size": 1,
        "query": {"bool": {"filter": filter_clauses}},
        "_source": ["raw_text"],
    }

    try:
        resp = _get_os().search(index=OPENSEARCH_INDEX, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        if hits:
            return hits[0].get("_source", {}).get("raw_text", "")
    except Exception as exc:
        logger.warning("[Context Expander] neighbor fetch failed: %s", exc)
    return ""
