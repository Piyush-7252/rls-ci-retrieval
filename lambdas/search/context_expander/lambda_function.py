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
import re
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
            timeout=30,
            max_retries=2,
            retry_on_timeout=True,
        )
        from requests.adapters import HTTPAdapter as _HA
        for _conn in _os_client.transport.connection_pool.connections:
            if hasattr(_conn, "session"):
                _conn.session.mount("https://", _HA(pool_maxsize=64, pool_connections=16))
                _conn.session.mount("http://",  _HA(pool_maxsize=64, pool_connections=16))
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
    document_id = req.get("document_id") or ""

    if not candidates:
        return {**req, "expanded_candidates": []}

    # ── Phase 1: collect all lookup keys ──────────────────────────────────────
    primary_ids:  list[str]                          = []
    idx_needed:   set[int]                           = set()
    ctx_keys:     list[tuple[str, int | None]]       = []
    _ctx_key_set: set[tuple[str, int | None]]        = set()

    for c in candidates:
        cid      = c.get("chunk_id", "")
        obj_meta = c.get("matched_object") or {}
        primary_ids.append(cid)
        for key in ("prev_chunk_idx", "next_chunk_idx", "parent_chunk_idx"):
            idx = obj_meta.get(key)
            if idx is not None:
                idx_needed.add(idx)
        ctx_key = (cid, obj_meta.get("global_position"))
        if ctx_key not in _ctx_key_set:
            ctx_keys.append(ctx_key)
            _ctx_key_set.add(ctx_key)

    # ── Phase 2: run all 3 fetches concurrently (they're independent) ───────────
    from concurrent.futures import ThreadPoolExecutor as _TPE
    with _TPE(max_workers=3) as _pool:
        _f_chunk = _pool.submit(_mget_chunks, list(dict.fromkeys(primary_ids)))
        _f_idx   = _pool.submit(_msearch_by_idx, document_id, list(idx_needed))
        _f_ctx   = _pool.submit(_fetch_context_objects_merged, document_id, ctx_keys)
        chunk_cache: dict[str, dict]         = _f_chunk.result()
        idx_cache:   dict[int, str]          = _f_idx.result()
        ctx_cache:   dict[tuple, list[dict]] = _f_ctx.result()

    logger.info(
        "[Context Expander] search_id=%s  candidates=%d  idx_lookups=%d"
        "  ctx_queries=%d  chunk_cache_hits=%d/%d",
        req.get("search_id"), len(candidates), len(idx_needed),
        len(ctx_keys), len(chunk_cache), len(primary_ids),
    )

    # ── Phase 3: expand each candidate from caches ────────────────────────────
    expanded = [_expand(c, document_id, chunk_cache, idx_cache, ctx_cache) for c in candidates]

    if expanded:
        avg_chars = sum(e.get("current_text_chars", 0) for e in expanded) / len(expanded)
        logger.info(
            "[Context Expander] search_id=%s  avg_context_chars=%.0f  max_context_chars=%d",
            req.get("search_id"), avg_chars,
            max(e.get("current_text_chars", 0) for e in expanded),
        )

    return {**req, "expanded_candidates": expanded}


_OBJECT_TYPE_PRIORITY: dict[str, int] = {
    "sentence": 3, "paragraph": 2, "heading": 1, "table_row": 1,
}

# ── Heading normalizer ────────────────────────────────────────────────────────

_SECTION_NUM_RE = re.compile(r'^[\d\.]+\s+')

def _normalize_heading(heading: str) -> str:
    """Strip leading section numbers from each breadcrumb segment.

    Example: "5.1 Primary Objective > 5.1.2 ORR" → "Primary Objective > ORR"
    Passes through headings that have no numeric prefix unchanged.
    """
    parts   = heading.split(" > ")
    cleaned = [_SECTION_NUM_RE.sub("", seg.strip()) for seg in parts if seg.strip()]
    return " > ".join(cleaned)


# ── Context object sorter ─────────────────────────────────────────────────────

_CTX_TYPE_ORDER: dict[str, int] = {
    "heading":   1,
    "paragraph": 2,
    "sentence":  3,
    "table_row": 4,
    "list":      4,
}

def _sort_context_objects(
    objs:       list[dict],
    matched_id: str | None,
    center_pos: int | None,
) -> list[dict]:
    """Sort context objects for verifier relevance.

    Order:
      1. The matched object itself (always first)
      2. Heading objects — nearest first
      3. Paragraph objects — nearest first
      4. Sentence objects — nearest first
      5. Table / list objects — nearest first
      6. Other types — nearest first
    """
    cp = center_pos or 0
    def _key(o: dict) -> tuple:
        is_match = 0 if o.get("object_id") == matched_id else 1
        tord     = _CTX_TYPE_ORDER.get(o.get("type", ""), 5)
        dist     = abs((o.get("global_position") or 0) - cp)
        return (is_match, tord, dist)
    return sorted(objs, key=_key)


def _expand(
    candidate:   dict,
    document_id: str | None,
    chunk_cache: dict[str, dict],
    idx_cache:   dict[int, str],
    ctx_cache:   dict[tuple, list[dict]],
) -> dict:
    chunk_id   = candidate["chunk_id"]
    page_start = candidate.get("page_start", 0)
    page_end   = candidate.get("page_end",   0)

    # Fetch the chunk's raw text — use mget cache, fall back to single GET
    chunk_doc    = chunk_cache.get(chunk_id) or _fetch_chunk(chunk_id)
    current_text = chunk_doc.get("raw_text", "")

    # If candidate came from semantic-objects index it already has the matched object
    matched_obj = candidate.get("matched_object")   # set by retriever for object-level hits
    # Capture origin before context_expander potentially assigns matched_obj from chunk context
    origin_is_direct = matched_obj is not None

    # context_strategy describes what actually ended up in current_text.
    # Set here to "chunk_fallback"; updated once we know the matched object type.
    context_strategy = "chunk_fallback"

    # For sentence-level hits, build hierarchical context:
    #   normalized heading → parent paragraph (only when multi-sentence) → 3-sentence window.
    # Preserves the semantic hierarchy (drug / endpoint / disease inherited from heading)
    # while keeping the verifier's context tight and precise.
    if matched_obj and matched_obj.get("type") == "sentence":
        raw_heading = (matched_obj.get("heading_path") or matched_obj.get("semantic_path") or "").strip()
        heading     = _normalize_heading(raw_heading) if raw_heading else ""
        para_text   = (matched_obj.get("paragraph_text") or "").strip()
        sent_text   = (matched_obj.get("text") or "").strip()
        prev_s      = (matched_obj.get("prev_sentence_text") or "").strip()
        next_s      = (matched_obj.get("next_sentence_text") or "").strip()
        sent_window = " ".join(p for p in [prev_s, sent_text, next_s] if p)

        ctx_parts: list[str] = []
        if heading:
            ctx_parts.append(heading)
        # Include parent paragraph only when it contains more than this one sentence
        # (para_text == sent_text means single-sentence paragraph — would be redundant)
        if para_text and para_text != sent_text:
            ctx_parts.append(para_text)
        if sent_window:
            ctx_parts.append(sent_window)

        if ctx_parts:
            current_text     = "\n\n".join(ctx_parts)
            context_strategy = "sentence_hierarchical"

    # Extract adjacency indices set by the section chunker
    obj_meta         = matched_obj or {}
    chunk_idx        = obj_meta.get("chunk_idx")
    prev_chunk_idx   = obj_meta.get("prev_chunk_idx")
    next_chunk_idx   = obj_meta.get("next_chunk_idx")
    parent_chunk_idx = obj_meta.get("parent_chunk_idx")

    # Prefer exact adjacency-index lookup (from msearch cache) over page-range heuristic
    if prev_chunk_idx is not None:
        prev_text = idx_cache.get(prev_chunk_idx, "")
    else:
        prev_text = _fetch_neighbor(document_id, page_end=page_start - 1)

    if next_chunk_idx is not None:
        next_text = idx_cache.get(next_chunk_idx, "")
    else:
        next_text = _fetch_neighbor(document_id, page_start=page_end + 1)

    parent_text = idx_cache.get(parent_chunk_idx, "") if parent_chunk_idx is not None else ""

    # Context objects from msearch cache (deduped by chunk_id + center_pos)
    center_pos      = obj_meta.get("global_position")
    context_objects = ctx_cache.get((chunk_id, center_pos), [])

    # Track why this object was selected — useful for debugging retrieval decisions:
    #   retriever_direct  — retriever set matched_object directly from semantic-objects
    #   literal_match     — via_chunk: chosen because text contained a literal CI match
    #   highest_priority  — via_chunk: chosen by type-priority (sentence > paragraph > …)
    #   chunk_only        — via_chunk: no context objects found at all
    selection_reason    = "retriever_direct" if origin_is_direct else None
    literal_match_count = 0      # set below when selection_reason == "literal_match"

    if matched_obj is None and context_objects:
        lit_texts = [lm["text"].lower()
                     for lm in candidate.get("literal_matches", [])
                     if lm.get("text")]
        if lit_texts:
            # Prefer the object whose text contains the literal match.
            # Among objects that contain the match, prefer the most specific
            # type (sentence > paragraph) so the UI highlights the tightest span.
            # Fall back to type-priority ordering when no object contains the match.
            def _lit_key(o: dict) -> tuple:
                obj_lower = (o.get("text") or "").lower()
                return (int(any(lt in obj_lower for lt in lit_texts)),
                        _OBJECT_TYPE_PRIORITY.get(o.get("type", ""), 0))
            matched_obj         = max(context_objects, key=_lit_key)
            selection_reason    = "literal_match"
            # How many context objects contained the literal — low count = high confidence
            literal_match_count = sum(
                1 for o in context_objects
                if any(lt in (o.get("text") or "").lower() for lt in lit_texts)
            )
        else:
            matched_obj      = max(
                context_objects,
                key=lambda o: _OBJECT_TYPE_PRIORITY.get(o.get("type", ""), 0),
            )
            selection_reason = "highest_priority"
    elif matched_obj is None:
        selection_reason = "chunk_only"   # no context objects available

    # Finalize context_strategy now that matched_obj is settled.
    # "sentence_hierarchical" is already set above; all other types take the object type name.
    if context_strategy == "chunk_fallback" and matched_obj is not None:
        context_strategy = matched_obj.get("type", "unknown")

    # Sort context_objects by verifier relevance:
    # matched object first, then headings, paragraphs, sentences, tables — each nearest first.
    matched_id      = (matched_obj or {}).get("object_id")
    context_objects = _sort_context_objects(context_objects, matched_id, center_pos)

    # Distance from the retrieval center to the matched object's global_position.
    # distance=0 → exact indexed position; distance>0 → pulled from surrounding window.
    matched_pos      = (matched_obj or {}).get("global_position")
    matched_distance = (
        abs(matched_pos - center_pos)
        if matched_pos is not None and center_pos is not None
        else None
    )

    # Character length of current_text sent to the verifier.
    # Tracks whether hierarchical context (heading + para + window) is growing too large.
    current_text_chars = len(current_text)

    # distance_ratio = matched_distance / CONTEXT_WINDOW
    # Normalises distance against the configured window size so analytics remain
    # comparable if CONTEXT_WINDOW changes (e.g. 3 → 5).
    #   0.0 = exact hit   1.0 = edge of window   >1.0 = outside window (shouldn't happen)
    distance_ratio = (
        round(matched_distance / CONTEXT_WINDOW, 3)
        if matched_distance is not None and CONTEXT_WINDOW > 0
        else None
    )

    context_quality = {
        "parent":    bool(parent_text),
        "prev":      bool(prev_text),
        "next":      bool(next_text),
        "n_objects": len(context_objects),
    }
    # Granular origin: "{direct|via_chunk}_{object_type}"
    #   direct    — retriever found this object in semantic-objects index directly
    #   via_chunk — BM25 chunk fallback; context_expander assigned best matching object
    obj_type = (matched_obj or {}).get("type") or "unknown"
    prefix   = "direct" if origin_is_direct else "via_chunk"
    retrieval_origin = f"{prefix}_{obj_type}"

    return {
        **candidate,
        "matched_object":     matched_obj,
        "retrieval_origin":   retrieval_origin,
        "selection_reason":   selection_reason,
        "literal_match_count": literal_match_count,
        "context_strategy":   context_strategy,
        "matched_distance":   matched_distance,
        "distance_ratio":     distance_ratio,
        "current_text_chars": current_text_chars,
        "context_objects":    context_objects,
        "context_quality":    context_quality,
        "context": {
            "parent_text":  parent_text[:CONTEXT_CHARS]  if parent_text else "",
            "prev_text":    prev_text[-CONTEXT_CHARS:]   if prev_text  else "",
            "current_text": current_text,
            "next_text":    next_text[:CONTEXT_CHARS]    if next_text  else "",
        },
    }


def _mget_chunks(chunk_ids: list[str]) -> dict[str, dict]:
    """Batch-fetch primary chunk docs via mget (one round-trip)."""
    if not chunk_ids:
        return {}
    try:
        resp = _get_os().mget(index=OPENSEARCH_INDEX, body={"ids": chunk_ids})
        return {
            doc["_id"]: doc["_source"]
            for doc in resp.get("docs", [])
            if doc.get("found")
        }
    except Exception as exc:
        logger.warning("[Context Expander] mget failed, will fall back per-doc: %s", exc)
        return {}


def _msearch_by_idx(document_id: str, idx_list: list[int]) -> dict[int, str]:
    """Batch-fetch raw_text for prev/next/parent chunks by chunk_idx via msearch."""
    if not idx_list or not document_id:
        return {}
    body: list[dict] = []
    for idx in idx_list:
        body.append({})
        body.append({
            "size": 1,
            "query": {"bool": {"filter": [
                {"term": {"document_id": document_id}},
                {"term": {"chunk_idx": idx}},
            ]}},
            "_source": ["raw_text"],
        })
    try:
        resp = _get_os().msearch(body=body, index=OPENSEARCH_INDEX)
        result: dict[int, str] = {}
        for i, r in enumerate(resp.get("responses", [])):
            hits = r.get("hits", {}).get("hits", [])
            if hits:
                result[idx_list[i]] = hits[0]["_source"].get("raw_text", "")
        return result
    except Exception as exc:
        logger.warning("[Context Expander] msearch by idx failed: %s", exc)
        return {}


def _fetch_context_objects_merged(
    document_id: str,
    ctx_keys:    list[tuple[str, int | None]],
) -> dict[tuple, list[dict]]:
    """Fetch context objects for all candidates in ONE OpenSearch terms query.

    Instead of N range sub-queries, expand every center_pos by ±CONTEXT_WINDOW,
    deduplicate all resulting positions, then issue a single terms(global_position)
    query.  For 190 dispersed candidates × 7 positions each = ~1,288 unique
    positions vs 190 individual searches.
    """
    if not ctx_keys:
        return {}

    result: dict[tuple, list[dict]] = {}
    with_pos    = [(key, key[1]) for key in ctx_keys if key[1] is not None]
    without_pos = [key for key in ctx_keys if key[1] is None]

    # ── Chunk-only fallbacks (no global_position) — still need per-chunk queries ──
    if without_pos:
        body: list[dict] = []
        for key in without_pos:
            body.append({})
            body.append({
                "size": 100,
                "query": {"bool": {"filter": [{"term": {"parent_chunk_id": key[0]}}]}},
                "sort": [{"global_position": "asc"}],
            })
        try:
            resp = _get_os().msearch(body=body, index=SEMANTIC_OBJECTS_INDEX)
            for i, r in enumerate(resp.get("responses", [])):
                result[without_pos[i]] = [h["_source"] for h in r.get("hits", {}).get("hits", [])]
        except Exception as exc:
            logger.warning("[Context Expander] chunk-only msearch failed: %s", exc)
            for key in without_pos:
                result[key] = []

    if not with_pos:
        return result

    # ── Expand all center positions to full ±CONTEXT_WINDOW neighbourhoods ────
    needed: set[int] = set()
    for _, center_pos in with_pos:
        for offset in range(-CONTEXT_WINDOW, CONTEXT_WINDOW + 1):
            needed.add(center_pos + offset)

    logger.info(
        "[Context Expander] ctx_keys=%d  needed_positions=%d  os_queries=1",
        len(with_pos), len(needed),
    )

    # ── ONE terms query — all needed positions in a single request ────────────
    pos_to_objs: dict[int, list[dict]] = {}
    try:
        fetch_size = min(len(needed) * 6, 10000)
        resp = _get_os().search(
            index=SEMANTIC_OBJECTS_INDEX,
            body={
                "size": fetch_size,
                "query": {"bool": {"filter": [
                    {"term":  {"document_id": document_id}},
                    {"terms": {"global_position": sorted(needed)}},
                ]}},
                "sort": [{"global_position": "asc"}],
            },
        )
        hits = resp.get("hits", {}).get("hits", [])
        logger.info(
            "[Context Expander] needed_positions=%d  returned_objects=%d  fetch_size=%d",
            len(needed), len(hits), fetch_size,
        )
        for h in hits:
            src  = h["_source"]
            gpos = src.get("global_position")
            if gpos is not None:
                pos_to_objs.setdefault(gpos, []).append(src)
    except Exception as exc:
        logger.warning("[Context Expander] terms context fetch failed: %s", exc)

    # ── Resolve per-key neighborhood from the local position dict ─────────────
    for key, center_pos in with_pos:
        neighborhood = []
        for offset in range(-CONTEXT_WINDOW, CONTEXT_WINDOW + 1):
            neighborhood.extend(pos_to_objs.get(center_pos + offset, []))
        result[key] = sorted(neighborhood, key=lambda o: o.get("global_position", 0))

    return result


def _msearch_context_objects(
    document_id: str,
    ctx_keys:    list[tuple[str, int | None]],
) -> dict[tuple, list[dict]]:
    """
    Batch-fetch context objects for all (chunk_id, center_pos) pairs via msearch.
    Deduplication already done in _process so each key appears once.
    """
    if not ctx_keys:
        return {}
    body: list[dict] = []
    for chunk_id, center_pos in ctx_keys:
        body.append({})
        if center_pos is not None:
            body.append({
                "size": CONTEXT_WINDOW * 2 + 1,
                "query": {"bool": {"filter": [
                    {"term":  {"document_id": document_id}},
                    {"range": {"global_position": {
                        "gte": center_pos - CONTEXT_WINDOW,
                        "lte": center_pos + CONTEXT_WINDOW,
                    }}},
                ]}},
                "sort": [{"global_position": "asc"}],
            })
        else:
            body.append({
                "size": 100,
                "query": {"bool": {"filter": [
                    {"term": {"parent_chunk_id": chunk_id}},
                ]}},
                "sort": [{"global_position": "asc"}],
            })
    try:
        resp = _get_os().msearch(body=body, index=SEMANTIC_OBJECTS_INDEX)
        return {
            ctx_keys[i]: [h["_source"] for h in r.get("hits", {}).get("hits", [])]
            for i, r in enumerate(resp.get("responses", []))
        }
    except Exception as exc:
        logger.warning("[Context Expander] msearch context objects failed: %s", exc)
        return {}


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
            {"term": {"document_id": document_id}},
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
            {"term":  {"document_id": document_id}},
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

    filter_clauses: list[dict] = [{"term": {"document_id": document_id}}]
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
