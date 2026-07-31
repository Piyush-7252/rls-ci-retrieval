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
import sys
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
            hosts            = [{"host": OPENSEARCH_ENDPOINT, "port": 443}],
            http_auth        = awsauth,
            use_ssl          = True,
            verify_certs     = True,
            connection_class = RequestsHttpConnection,
        )
    return _os_client


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
    doc = _build_ci_doc(ci)
    _get_os().index(index=OPENSEARCH_CI_INDEX, id=str(ci["id"]), body=doc)
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
    chunk_doc = _build_chunk_doc(chunk)
    obj_docs  = _build_object_docs(chunk)
    sent_docs = _build_sentence_docs(chunk)

    _get_os().index(index=OPENSEARCH_INDEX, id=chunk["chunk_id"], body=chunk_doc)

    for doc in obj_docs:
        _get_os().index(index=SEMANTIC_OBJECTS_INDEX, id=doc["object_id"], body=doc)

    for doc in sent_docs:
        _get_os().index(index=SEMANTIC_OBJECTS_INDEX, id=doc["object_id"], body=doc)

    logger.info(
        "[Index document] chunk_id=%s chunk=1 objects=%d sentences=%d",
        chunk["chunk_id"], len(obj_docs), len(sent_docs),
    )
    return {
        "indexed":           True,
        "document_id":       chunk["document_id"],
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
            "heading_dense_vector": chunk.get("embedding", {}).get("heading_dense_vector", []),
            "entities":         obj.get("entities", []),
            # ── ClinicalObject enrichment (single source of truth) ────────────
            **build_enrichment_fields(obj),
            # ── Display (never embedded) ──────────────────────────────────────
            "page":          obj.get("page", page_start),
            "bbox":          obj.get("bbox", []),
            "display_spans": obj.get("display_spans", []),
        })

    return docs


def _build_sentence_docs(chunk: dict) -> list[dict]:
    """
    One OpenSearch doc per sentence for fine-grained retrieval.

    Sentences are display_spans of type="sentence" that received an embedding.
    Each sentence doc inherits all enrichment fields from its parent object
    via build_enrichment_fields(obj) — so the same enrichment schema applies
    at chunk, object, and sentence granularity.
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

        object_id    = obj["object_id"]
        obj_entities = obj.get("entities", [])
        spans = [
            (idx, span)
            for idx, span in enumerate(obj.get("display_spans", []))
            if span.get("type") == "sentence" and span.get("text") and span.get("embedding")
        ]

        for list_pos, (idx, span) in enumerate(spans):
            sentence_id = f"{object_id}_s{idx}"
            span_start  = span.get("start", 0)
            span_end    = span.get("end", len(span.get("text", "")))

            # Entities overlapping this sentence's char range
            sent_entities = [
                e for e in obj_entities
                if e.get("object_start", 0) < span_end
                and e.get("object_end",   0) > span_start
            ]

            prev_id   = f"{object_id}_s{spans[list_pos - 1][0]}" if list_pos > 0             else None
            next_id   = f"{object_id}_s{spans[list_pos + 1][0]}" if list_pos < len(spans) - 1 else None
            prev_text = spans[list_pos - 1][1].get("text") if list_pos > 0             else None
            next_text = spans[list_pos + 1][1].get("text") if list_pos < len(spans) - 1 else None

            docs.append({
                # ── Identity ──────────────────────────────────────────────────
                "object_id":         sentence_id,
                "parent_object_id":  object_id,
                "parent_chunk_id":   chunk_id,
                "document_id":       document_id,
                "type":              "sentence",
                "char_start":        span_start,
                "char_end":          span_end,
                # ── Retrieval ─────────────────────────────────────────────────
                "text":              span["text"],
                "normalized_text":   span["text"],
                "dense_vector":      span["embedding"],
                "entities":          sent_entities,
                # ── Context (never embedded — available without extra lookup) ──
                "paragraph_text":        obj.get("text", ""),
                "prev_sentence_text":    prev_text,
                "next_sentence_text":    next_text,
                # ── ClinicalObject enrichment (inherited from parent object) ──
                # Sentences do not have their own enrichment run — they inherit
                # everything from their parent object including effective_facts,
                # slot_provenance, endpoint_identity, population_identity, etc.
                **build_enrichment_fields(obj),
                # ── Section context (inherited from parent object) ─────────────
                "section":           obj.get("section"),
                "section_number":    obj.get("section_number"),
                "section_depth":     obj.get("section_depth"),
                "section_level":     obj.get("section_level"),
                "section_category":  obj.get("section_category"),
                "heading_path":      obj.get("heading_path"),
                "semantic_path":     obj.get("semantic_path"),
                "section_confidence": obj.get("section_confidence"),
                "parent_heading":    obj.get("parent_heading"),
                "document_position": obj.get("document_position"),
                "global_position":   obj.get("global_position", obj.get("position")),
                "category":          obj.get("category", "clinical"),
                "boost_weight":      obj.get("boost_weight", 1.0),
                # ── Sentence adjacency ────────────────────────────────────────
                "prev_sentence_id":  prev_id,
                "next_sentence_id":  next_id,
                # ── Display ───────────────────────────────────────────────────
                "page":              obj.get("page", page_start),
                "bbox":              span.get("bbox", obj.get("bbox", [])),
            })

    return docs
