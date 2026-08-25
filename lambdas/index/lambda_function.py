"""
Index Lambda  (Unified — CI and Document)
==========================================
Routes on ``event["source_type"]``:
  "ci"       → write CI to the ``ci-objects`` OpenSearch index
  "document" → write chunk + objects + sentences to ``document-chunks``
                and ``semantic-objects`` indices

Both paths call ``build_enrichment_fields(obj)`` from
``shared.opensearch_enrichment`` to copy the 17 ClinicalObject enrichment
fields.  Adding a new Knowledge Layer field requires a change in ONE place
(``shared/opensearch_enrichment.py``) and both indices are updated
automatically.

Env vars
--------
  OPENSEARCH_ENDPOINT     — host name only (no https:// prefix)
  OPENSEARCH_CI_INDEX     — default: ci-objects
  OPENSEARCH_INDEX        — default: document-chunks
  SEMANTIC_OBJECTS_INDEX  — default: semantic-objects
  AWS_REGION              — default: us-east-1
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ── ensure shared/ is importable when running as a loaded module ──────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from shared.opensearch_enrichment import build_enrichment_fields
except ImportError:
    logger.warning("shared.opensearch_enrichment not found — enrichment fields will be empty")
    def build_enrichment_fields(obj: dict) -> dict:  # type: ignore[misc]
        return {}

# ── env config ────────────────────────────────────────────────────────────────
OPENSEARCH_ENDPOINT    = os.environ.get("OPENSEARCH_ENDPOINT", "localhost")
OPENSEARCH_CI_INDEX    = os.environ.get("OPENSEARCH_CI_INDEX", "ci-objects")
OPENSEARCH_INDEX       = os.environ.get("OPENSEARCH_INDEX", "document-chunks")
SEMANTIC_OBJECTS_INDEX = os.environ.get("SEMANTIC_OBJECTS_INDEX", "semantic-objects")
AWS_REGION             = os.environ.get("AWS_REGION", "us-east-1")

# ─── lazy OpenSearch client ───────────────────────────────────────────────────
_os_client = None

def _get_os():
    global _os_client
    if _os_client is None:
        import boto3
        from opensearchpy import OpenSearch, RequestsHttpConnection
        from requests_aws4auth import AWS4Auth

        frozen  = boto3.Session().get_credentials().get_frozen_credentials()
        awsauth = AWS4Auth(
            frozen.access_key,
            frozen.secret_key,
            AWS_REGION,
            "es",
            session_token=frozen.token,
        )
        _os_client = OpenSearch(
            hosts              = [{"host": OPENSEARCH_ENDPOINT, "port": 443}],
            http_auth          = awsauth,
            use_ssl            = True,
            verify_certs       = True,
            connection_class   = RequestsHttpConnection,
            timeout            = 60,    # raise from 10s default; single chunk can be 17+ writes
            max_retries        = 3,     # retry transient failures (ConnectionTimeout, 429, 503)
            retry_on_timeout   = True,  # auto-retry on ConnectionTimeout without SQS requeue
        )
    return _os_client


# ─────────────────────────────────────────────────────────────────────────────
# Bulk helper with HTTP-level retry + full-jitter backoff
# ─────────────────────────────────────────────────────────────────────────────

_BULK_MAX_RETRIES  = 3
_BULK_BACKOFF_BASE = 2.0
_BULK_BACKOFF_CAP  = 30.0
_OS_RETRY_STATUSES = frozenset({429, 503, 502, 504})


def _bulk_with_retry(bulk_body: list, chunk_id: str) -> tuple[dict, int]:
    """
    Execute a bulk request with full-jitter exponential backoff on HTTP errors.

    Returns (response, http_retries) — http_retries is the number of extra
    attempts made due to HTTP-level errors (0 = succeeded on first try).
    """
    import opensearchpy.exceptions as _osx
    last_exc: Exception | None = None
    for attempt in range(_BULK_MAX_RETRIES):
        try:
            return _get_os().bulk(body=bulk_body), attempt
        except (_osx.ConnectionTimeout, _osx.ConnectionError) as exc:
            last_exc = exc
        except _osx.TransportError as exc:
            if exc.status_code not in _OS_RETRY_STATUSES:
                raise
            last_exc = exc
        window = min(_BULK_BACKOFF_CAP, _BULK_BACKOFF_BASE * (2 ** attempt))
        delay  = random.uniform(0, window)
        logger.warning(
            "[Index] bulk HTTP error chunk_id=%s attempt=%d/%d error=%s retrying_in=%.1fs",
            chunk_id, attempt + 1, _BULK_MAX_RETRIES, last_exc, delay,
        )
        time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def _index_with_retry(index: str, doc_id: str, body: dict, chunk_id: str) -> int:
    """
    Single-document index with the same full-jitter backoff used for bulk.

    Returns http_retries (0 = succeeded on first try).
    """
    import opensearchpy.exceptions as _osx
    last_exc: Exception | None = None
    for attempt in range(_BULK_MAX_RETRIES):
        try:
            _get_os().index(index=index, id=doc_id, body=body)
            return attempt
        except (_osx.ConnectionTimeout, _osx.ConnectionError) as exc:
            last_exc = exc
        except _osx.TransportError as exc:
            if exc.status_code not in _OS_RETRY_STATUSES:
                raise
            last_exc = exc
        window = min(_BULK_BACKOFF_CAP, _BULK_BACKOFF_BASE * (2 ** attempt))
        delay  = random.uniform(0, window)
        logger.warning(
            "[Index] single-doc HTTP error chunk_id=%s attempt=%d/%d error=%s retrying_in=%.1fs",
            chunk_id, attempt + 1, _BULK_MAX_RETRIES, last_exc, delay,
        )
        time.sleep(delay)
    raise last_exc  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    source_type = event.get("source_type", "document")

    if source_type == "ci":
        ci_id = event.get("id", "unknown")
        logger.info("[Index] start source=ci ci_id=%s", ci_id)
        try:
            result = _process_ci(event)
        except Exception as exc:
            logger.error("[Index] failed source=ci ci_id=%s error=%s", ci_id, exc)
            raise
        logger.info("[Index] done source=ci ci_id=%s", ci_id)
        return result

    else:
        chunk_id = event.get("chunk_id", "unknown")
        logger.info("[Index] start source=document chunk_id=%s", chunk_id)
        try:
            result = _process_document(event)
        except Exception as exc:
            logger.error("[Index] failed source=document chunk_id=%s error=%s", chunk_id, exc)
            raise
        logger.info("[Index] done source=document chunk_id=%s", chunk_id)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# CI path
# ─────────────────────────────────────────────────────────────────────────────

def _process_ci(ci: dict) -> dict:
    doc        = _build_ci_doc(ci)
    ci_id_str  = str(ci["id"])
    http_retries = _index_with_retry(OPENSEARCH_CI_INDEX, ci_id_str, doc, chunk_id=f"ci-{ci_id_str}")
    if http_retries:
        logger.warning("[Index] CI indexed after %d retries ci_id=%s", http_retries, ci_id_str)
    return {"stored": True, "ci_id": ci["id"]}


def _build_ci_doc(ci: dict) -> dict:
    """
    Flatten an enriched CI into a single OpenSearch document.

    Schema is intentionally symmetric with the document semantic-objects index
    so that the reranker compares two ClinicalObjects with identical fields.
    CI-specific fields come first; the shared enrichment fields are appended
    via build_enrichment_fields().
    """
    return {
        # ── CI identity ───────────────────────────────────────────────────────
        "ci_id":    ci["id"],
        "known_ci": ci.get("knownCI", ""),
        "category": ci.get("category", {}).get("name", ""),
        "status":   ci.get("status"),
        "assets":   ci.get("assets", []),
        # ── NLP pipeline outputs ──────────────────────────────────────────────
        "normalized_text":     ci["normalization"]["normalized_text"],
        "tokens":              ci["normalization"]["tokens"],
        "entities":            ci["ner"]["entities"],
        "ner_model":           ci["ner"].get("model", "gliner"),
        "ontology_expansions": ci["ontology"]["expansions"],
        "ontology_synonyms":   ci["ontology"]["synonyms"],
        "regex_patterns":      ci["ontology"]["regex_patterns"],
        # ── Embeddings ───────────────────────────────────────────────────────
        "dense_vector":      ci["embedding"]["dense_vector"],
        "sparse_vector":     ci["embedding"]["sparse_vector"],
        "embedding_model":   ci["embedding"]["model"],
        # ── ClinicalObject enrichment (symmetric with semantic-objects) ───────
        # build_enrichment_fields() is the SINGLE source of truth — both CI
        # and document objects get the exact same enrichment schema.
        **build_enrichment_fields(ci),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Document path
# ─────────────────────────────────────────────────────────────────────────────

def _process_document(chunk: dict) -> dict:
    chunk_id    = chunk["chunk_id"]
    document_id = chunk["document_id"]

    chunk_doc = _build_chunk_doc(chunk)
    obj_docs  = _build_object_docs(chunk)
    sent_docs = _build_sentence_docs(chunk)

    # ── object size stats ─────────────────────────────────────────────────────
    obj_chars = [len(d.get("text", "")) for d in obj_docs]
    avg_obj_chars = int(sum(obj_chars) / max(len(obj_chars), 1))
    max_obj_chars = max(obj_chars) if obj_chars else 0

    # Build one bulk body: chunk → objects → sentences (single HTTPS round-trip).
    # Pre-serialize every item exactly once: the same bytes are used for both
    # size accounting AND the wire payload.  opensearch-py passes str items
    # through verbatim (no re-serialisation), so this guarantees exactly one
    # json.dumps call per document — no double serialisation anywhere.
    bulk_body: list[str]  = []
    bulk_ids:  list[str]  = []   # _id per (action, doc) pair — for retry reconstruction
    bulk_size_bytes: int  = 0

    def _append(action: dict, doc: dict) -> None:
        nonlocal bulk_size_bytes
        action_s = json.dumps(action)
        doc_s    = json.dumps(doc)
        bulk_body.append(action_s)
        bulk_body.append(doc_s)
        bulk_ids.append(action["index"]["_id"])
        bulk_size_bytes += len(action_s.encode()) + len(doc_s.encode())

    _append({"index": {"_index": OPENSEARCH_INDEX, "_id": chunk_id}}, chunk_doc)
    for doc in obj_docs:
        _append({"index": {"_index": SEMANTIC_OBJECTS_INDEX, "_id": doc["object_id"]}}, doc)
    for doc in sent_docs:
        _append({"index": {"_index": SEMANTIC_OBJECTS_INDEX, "_id": doc["object_id"]}}, doc)

    bulk_docs    = 1 + len(obj_docs) + len(sent_docs)
    bulk_size_mb = bulk_size_bytes / (1024 * 1024)

    t0                    = time.monotonic()
    resp, http_retries    = _bulk_with_retry(bulk_body, chunk_id)
    bulk_latency          = time.monotonic() - t0
    partial_doc_retries   = 0
    docs_per_sec  = bulk_docs        / max(bulk_latency, 0.001)
    obj_per_sec   = len(obj_docs)    / max(bulk_latency, 0.001)
    sent_per_sec  = len(sent_docs)   / max(bulk_latency, 0.001)

    failed_items = [
        item for item in resp.get("items", [])
        if item.get("index", {}).get("error")
    ] if resp.get("errors") else []

    # ── retry failed docs only (don't re-send successful ones) ───────────────
    if failed_items:
        failed_ids = {
            item["index"]["_id"]
            for item in failed_items
        }
        # bulk_body contains pre-serialised strings; use parallel bulk_ids to
        # identify which (action, doc) pairs belong to failed documents without
        # parsing the strings back.
        retry_body: list[str] = []
        for idx, doc_id in enumerate(bulk_ids):
            if doc_id in failed_ids:
                retry_body.append(bulk_body[idx * 2])
                retry_body.append(bulk_body[idx * 2 + 1])

        if retry_body:
            logger.warning(
                "[Index] retrying failed docs chunk_id=%s failed=%d",
                chunk_id, len(failed_items),
            )
            partial_doc_retries = len(failed_items)
            t1                   = time.monotonic()
            retry_resp, _hr      = _bulk_with_retry(retry_body, chunk_id)
            http_retries        += _hr
            retry_latency        = time.monotonic() - t1
            still_failed = [
                item for item in retry_resp.get("items", [])
                if item.get("index", {}).get("error")
            ] if retry_resp.get("errors") else []

            if still_failed:
                logger.error(
                    "[Index] bulk retry still failing chunk_id=%s failed=%d first_error=%s",
                    chunk_id, len(still_failed),
                    still_failed[0]["index"]["error"],
                )
                raise RuntimeError(
                    f"Bulk index failed for {len(still_failed)} docs in {chunk_id}"
                )
            failed_items = []   # retry succeeded

    logger.info(
        "[IndexSummary] chunk=%s doc=%s bulk_docs=%d chunk_docs=1 objects=%d sentences=%d "
        "bulk_size_mb=%.2f bulk_latency=%.3fs docs_per_sec=%.1f "
        "obj_per_sec=%.1f sent_per_sec=%.1f "
        "avg_obj_chars=%d max_obj_chars=%d "
        "http_retries=%d partial_doc_retries=%d failed=%d",
        chunk_id, document_id, bulk_docs, len(obj_docs), len(sent_docs),
        bulk_size_mb, bulk_latency, docs_per_sec,
        obj_per_sec, sent_per_sec,
        avg_obj_chars, max_obj_chars,
        http_retries, partial_doc_retries, len(failed_items),
    )
    return {
        "indexed":           True,
        "document_id":       document_id,
        "chunk_id":          chunk["chunk_id"],
        "objects_indexed":   len(obj_docs),
        "sentences_indexed": len(sent_docs),
    }


def _build_chunk_doc(chunk: dict) -> dict:
    """Chunk-level doc for document-chunks index (high-recall retrieval)."""
    return {
        "document_id":          chunk["document_id"],
        "chunk_id":             chunk["chunk_id"],
        "page_start":           chunk["page_start"],
        "page_end":             chunk["page_end"],
        "raw_text":             chunk["extraction"]["raw_text"],
        "normalized_text":      chunk["normalization"]["normalized_text"],
        "tokens":               chunk["normalization"]["tokens"],
        "entities":             chunk.get("ner", {}).get("entities", []),
        "ontology_expansions":  chunk.get("ontology", {}).get("expansions", []),
        "ontology_synonyms":    json.dumps(chunk.get("ontology", {}).get("synonyms", {})),
        "dense_vector":         chunk.get("embedding", {}).get("dense_vector", []),
        "heading_dense_vector": chunk.get("embedding", {}).get("heading_dense_vector", []),
        "sparse_vector_json":   json.dumps(chunk.get("embedding", {}).get("sparse_vector", {})),
        "embedding_model":      chunk.get("embedding", {}).get("model", ""),
        "chunk_idx":            chunk.get("chunk_idx"),
        "parent_chunk_idx":     chunk.get("parent_chunk_idx"),
        "prev_chunk_idx":       chunk.get("prev_chunk_idx"),
        "next_chunk_idx":       chunk.get("next_chunk_idx"),
        "geometry":              chunk.get("geometry") or {},
    }



def _build_object_docs(chunk: dict) -> list[dict]:
    """
    One OpenSearch doc per semantic object for the semantic-objects index.

    Fields are split into two sections:
    RETRIEVAL — text, vectors, entities, section, enrichment.  Reasoned over.
    DISPLAY   — page, bbox, display_spans.  UI only; never embedded.
    """
    objects     = chunk.get("extraction", {}).get("objects", [])
    document_id = chunk["document_id"]
    chunk_id    = chunk["chunk_id"]
    page_start  = chunk.get("page_start", 0)
    docs        = []

    for obj in objects:
        if not obj.get("searchable", True):
            continue
        if not obj.get("indexable", True):
            continue

        docs.append({
            # ── Identity ──────────────────────────────────────────────────────
            "object_id":        obj["object_id"],
            "document_id":      document_id,
            "parent_chunk_id":  chunk_id,
            "position":         obj["position"],
            "global_position":  obj.get("global_position", obj["position"]),
            "type":             obj["type"],
            "list_id":         obj.get("list_id"),
            "list_level":      obj.get("list_level"),
            "list_label":      obj.get("list_label"),
            "list_number_format": obj.get("list_number_format"),
            "table_id":        obj.get("table_id", obj.get("table_key")),
            "cell_id":         obj.get("cell_id"),
            "table_role":      obj.get("table_role"),
            "row_index":       obj.get("row_index", obj.get("row_start")),
            "row_start":       obj.get("row_start"),
            "col_start":       obj.get("col_start"),
            "row_span":        obj.get("row_span"),
            "col_span":        obj.get("col_span"),
            # ── Retrieval ─────────────────────────────────────────────────────
            "text":             obj["text"],
            "normalized_text":  obj.get("normalized_text", ""),
            "section":          obj.get("section"),
            "section_number":   obj.get("section_number"),
            "section_depth":    obj.get("section_depth"),
            "section_level":    obj.get("section_level"),
            "section_category":    obj.get("section_category"),
            "heading_path":        obj.get("heading_path"),
            "semantic_path":       obj.get("semantic_path"),
            "section_confidence":  obj.get("section_confidence"),
            "document_position":   obj.get("document_position"),
            "chunk_idx":           obj.get("chunk_idx"),
            "parent_chunk_idx":    obj.get("parent_chunk_idx"),
            "prev_chunk_idx":      obj.get("prev_chunk_idx"),
            "next_chunk_idx":      obj.get("next_chunk_idx"),
            "category":         obj.get("category", "clinical"),
            "boost_weight":     obj.get("boost_weight", 1.0),
            "indexable":        obj.get("indexable", True),
            "parent_heading":   obj.get("parent_heading"),
            "prev_object_pos":  obj.get("prev_object_pos"),
            "next_object_pos":  obj.get("next_object_pos"),
            "dense_vector":     obj.get("embedding", []),
            # knn_vector rejects empty arrays; omit the field entirely when absent
            **({"heading_dense_vector": _hdv} if (_hdv := chunk.get("embedding", {}).get("heading_dense_vector")) else {}),
            "entities":         obj.get("entities", []),
            # ── ClinicalObject enrichment (single source of truth) ────────────
            **build_enrichment_fields(obj),
            # ── Display (never embedded) ──────────────────────────────────────
            "page":          obj.get("page", page_start),
            "bbox":          [float(v) for v in obj.get("bbox", [])],
            "geometry": obj.get("geometry") or {},
            "display_spans": obj.get("display_spans", []),
        })

    return docs


def _build_sentence_docs(chunk: dict) -> list[dict]:
    """Build sentence documents from self-contained display spans.

    ``display_spans`` are the sentence transport units.  Each sentence span
    carries its canonical upstream ``geometry`` directly, so indexing never
    performs a geometry lookup, reconstruction, or text/index matching.
    """
    objects     = chunk.get("extraction", {}).get("objects", [])
    document_id = chunk["document_id"]
    chunk_id    = chunk["chunk_id"]
    docs = []

    for obj in objects:
        if not obj.get("searchable", True) or not obj.get("indexable", True):
            continue

        object_id = obj["object_id"]
        spans = [
            (idx, span)
            for idx, span in enumerate(obj.get("display_spans", []))
            if span.get("type") == "sentence"
            and span.get("text")
        ]

        for list_pos, (idx, span) in enumerate(spans):
            sentence_id = f"{object_id}_s{idx}"
            geometry = span.get("geometry") or {}

            prev_id = (
                f"{object_id}_s{spans[list_pos - 1][0]}"
                if list_pos > 0 else None
            )
            next_id = (
                f"{object_id}_s{spans[list_pos + 1][0]}"
                if list_pos < len(spans) - 1 else None
            )
            prev_text = (
                spans[list_pos - 1][1].get("text")
                if list_pos > 0 else None
            )
            next_text = (
                spans[list_pos + 1][1].get("text")
                if list_pos < len(spans) - 1 else None
            )

            docs.append({
                "object_id": sentence_id,
                "parent_object_id": object_id,
                "parent_chunk_id": chunk_id,
                "document_id": document_id,
                "type": "sentence",
                "text": span["text"],
                "normalized_text": span["text"],
                **({"dense_vector": span["embedding"]} if span.get("embedding") else {}),
                "entities": obj.get("entities", []),
                "paragraph_text": obj.get("text", ""),
                "prev_sentence_text": prev_text,
                "next_sentence_text": next_text,
                **build_enrichment_fields(obj),
                "section": obj.get("section"),
                "section_number": obj.get("section_number"),
                "section_depth": obj.get("section_depth"),
                "section_level": obj.get("section_level"),
                "section_category": obj.get("section_category"),
                "heading_path": obj.get("heading_path"),
                "semantic_path": obj.get("semantic_path"),
                "section_confidence": obj.get("section_confidence"),
                "parent_heading": obj.get("parent_heading"),
                "document_position": obj.get("document_position"),
                "global_position": obj.get("global_position", obj.get("position")),
                "category": obj.get("category", "clinical"),
                "boost_weight": obj.get("boost_weight", 1.0),
                "list_id": obj.get("list_id"),
                "list_level": obj.get("list_level"),
                "list_label": obj.get("list_label"),
                "list_number_format": obj.get("list_number_format"),
                "table_id": obj.get("table_id", obj.get("table_key")),
                "cell_id": obj.get("cell_id"),
                "table_role": obj.get("table_role"),
                "row_index": obj.get("row_index", obj.get("row_start")),
                "prev_sentence_id": prev_id,
                "next_sentence_id": next_id,
                "page": geometry.get("page", obj.get("page", 0)),
                "bbox": [float(v) for v in (geometry.get("bbox") or [])],
                "geometry": geometry,
            })

    return docs
