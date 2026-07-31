"""
patch_semantic_objects.py
══════════════════════════════════════════════════════════════════════════════
Backfill enrichment fields on already-indexed semantic-objects documents.

Documents indexed before the full enrichment pipeline was deployed have
``facts`` / ``statement_type`` / ``clinical_relations`` but are missing:

    effective_facts      own_facts          inherited_slots    slot_provenance
    study_hierarchy      clinical_identity  treatment_identity endpoint_identity
    population_identity  temporal_context   clinical_signature
    modality             object_subtype     negated_slots

This script:

1.  Scrolls through every doc in ``semantic-objects`` that lacks
    ``effective_facts``.
2.  Groups docs by ``parent_chunk_id`` so context flows correctly
    heading → paragraph → table → sentence (same logic as the NER Lambda).
3.  For non-sentence objects calls ``enrich_object(text, entities)``
    then ``propagate_effective_facts([...])`` on the full sorted chunk.
4.  For sentence objects (type=sentence) inherits enrichment from the
    parent paragraph object identified by stripping ``_s{n}`` from
    the sentence ``object_id``; falls back to self-contained enrichment
    if the parent is not in the same scroll batch.
5.  Bulk-updates OpenSearch with only the enrichment diff — existing
    retrieval fields (vectors, tokens, etc.) are never touched.

Usage
─────
    python tests/patch_semantic_objects.py
    python tests/patch_semantic_objects.py --dry-run
    python tests/patch_semantic_objects.py --limit 500
    python tests/patch_semantic_objects.py --chunk-id 10993-co-jnj-64407564_chunk_0110
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OPENSEARCH_ENDPOINT   = os.environ.get(
    "OPENSEARCH_ENDPOINT",
    "search-rls-dev-rhitzxwnctmuyq2l4kny5kwelu.eu-west-1.es.amazonaws.com",
)
AWS_REGION            = os.environ.get("AWS_REGION", "eu-west-1")
SEMANTIC_OBJECTS_INDEX = "semantic-objects"
SCROLL_SIZE           = 500   # docs per scroll page
BULK_SIZE             = 200   # docs per bulk update request

os.environ.update({
    "AWS_DEFAULT_REGION":  AWS_REGION,
    "AWS_REGION":          AWS_REGION,
    "OPENSEARCH_ENDPOINT": OPENSEARCH_ENDPOINT,
})

logging.basicConfig(
    level   = logging.WARNING,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("patch_semantic_objects")

# ── enrichment field names (mirrors shared/opensearch_enrichment.ENRICHMENT_DEFAULTS) ──
_ENRICH_FIELDS = (
    "study_context", "statement_type", "object_subtype", "modality",
    "facts", "own_facts", "effective_facts", "inherited_slots", "slot_provenance",
    "study_hierarchy", "clinical_identity", "clinical_relations",
    "negated_slots", "treatment_identity", "endpoint_identity",
    "population_identity", "temporal_context", "clinical_signature",
)

# Regex that matches the sentence suffix added by the index lambda:
#   "{parent_object_id}_s{n}"  →  captures "{parent_object_id}"
_SENTENCE_ID_RE = re.compile(r'^(.+)_s\d+$')


# ─────────────────────────────────────────────────────────────────────────────
# OpenSearch client
# ─────────────────────────────────────────────────────────────────────────────

_os_client = None

def _get_os():
    global _os_client
    if _os_client is not None:
        return _os_client
    import boto3
    from opensearchpy import OpenSearch, RequestsHttpConnection
    from requests_aws4auth import AWS4Auth

    frozen  = boto3.Session().get_credentials().get_frozen_credentials()
    awsauth = AWS4Auth(
        frozen.access_key, frozen.secret_key, AWS_REGION, "es",
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
# Enrichment helpers
# ─────────────────────────────────────────────────────────────────────────────

def _enrich_chunk_objects(objects: list[dict]) -> dict[str, dict]:
    """
    Re-derive all enrichment fields for every object in a chunk.

    Pipeline:
        enrich_object()            — per-object facts + identity fields
        propagate_effective_facts  — context inheritance + clinical_identity

    Returns {object_id: {field: value, ...}} with only the enrichment diff
    (fields that changed or were previously absent).
    """
    from shared.clinical_fact_extractor import enrich_object
    from shared.clinical_enrichment_pipeline import enrich_document_objects

    # Separate sentences from structural objects (sentences must not drive context)
    sentences: list[dict]   = []
    structural: list[dict]  = []
    for obj in objects:
        if (obj.get("type") or "").lower() == "sentence":
            sentences.append(obj)
        else:
            structural.append(obj)

    # Sort structural objects by position for correct context flow
    structural.sort(key=lambda o: float(o.get("position") or 0))

    # ── Pass 1: per-object enrichment ────────────────────────────────────────
    for obj in structural:
        text     = (obj.get("normalized_text") or obj.get("text") or "").strip()
        entities = obj.get("entities") or []
        # Adapt entity offsets — old docs stored start/end; enrich_object
        # needs object_start/object_end.
        adapted  = [
            {**e, "object_start": e.get("object_start", e.get("start", 0)),
                  "object_end":   e.get("object_end",   e.get("end",   0))}
            for e in entities
        ]
        if not text:
            continue
        try:
            enrichment = enrich_object(
                text             = text,
                entities         = adapted,
                section_category = obj.get("section_category", ""),
                heading_path     = obj.get("heading_path") or [],
            )
            obj.update(enrichment)
        except Exception as exc:
            logger.debug("enrich_object failed for %s: %s", obj.get("object_id"), exc)

    # ── Pass 2: chunk-level propagation (effective_facts + clinical_identity) ─
    if structural:
        try:
            enrich_document_objects(structural)
        except Exception as exc:
            logger.debug("propagate_effective_facts failed for chunk: %s", exc)

    # Build parent map for sentence inheritance
    parent_map: dict[str, dict] = {o["object_id"]: o for o in structural}

    # ── Pass 3: sentences inherit parent enrichment ───────────────────────────
    for sent in sentences:
        sid = sent.get("object_id", "")
        m   = _SENTENCE_ID_RE.match(sid)
        parent_id = m.group(1) if m else None
        parent    = parent_map.get(parent_id) if parent_id else None

        if parent is not None:
            # Inherit all enrichment fields from the re-derived parent object
            for field in _ENRICH_FIELDS:
                if field in parent:
                    sent[field] = parent[field]
        else:
            # Parent not in this scroll batch — treat sentence as self-contained
            from shared.clinical_enrichment_pipeline import enrich_ci
            text     = (sent.get("normalized_text") or sent.get("text") or "").strip()
            entities = sent.get("entities") or []
            adapted  = [
                {**e, "object_start": e.get("object_start", e.get("start", 0)),
                      "object_end":   e.get("object_end",   e.get("end",   0))}
                for e in entities
            ]
            if text:
                try:
                    enrichment = enrich_ci(text, adapted)
                    for field in _ENRICH_FIELDS:
                        if field in enrichment:
                            sent[field] = enrichment[field]
                except Exception as exc:
                    logger.debug("self-contained enrichment failed for %s: %s", sid, exc)

    # Collect diff — only fields with non-empty new values
    result: dict[str, dict] = {}
    for obj in structural + sentences:
        oid  = obj.get("object_id")
        if not oid:
            continue
        diff = {f: obj[f] for f in _ENRICH_FIELDS if f in obj}
        if diff:
            result[oid] = diff
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Bulk update
# ─────────────────────────────────────────────────────────────────────────────

def _bulk_update(updates: dict[str, dict], dry_run: bool = False) -> tuple[int, int]:
    """Send partial-update (_update) requests for each object_id → diff.

    Returns (succeeded, failed).
    """
    if not updates:
        return 0, 0
    if dry_run:
        print(f"    [dry-run] would update {len(updates)} objects")
        return len(updates), 0

    from opensearchpy import helpers

    actions = [
        {
            "_op_type": "update",
            "_index":   SEMANTIC_OBJECTS_INDEX,
            "_id":      oid,
            "doc":      diff,
        }
        for oid, diff in updates.items()
    ]

    succeeded = 0
    failed    = 0
    for ok, info in helpers.streaming_bulk(
        _get_os(),
        actions,
        chunk_size     = BULK_SIZE,
        raise_on_error = False,
    ):
        if ok:
            succeeded += 1
        else:
            failed += 1
            logger.warning("Bulk update failed: %s", info)

    return succeeded, failed


# ─────────────────────────────────────────────────────────────────────────────
# Scroll helpers
# ─────────────────────────────────────────────────────────────────────────────

def _scroll_stale(chunk_id: str | None = None):
    """Yield all stale docs from semantic-objects (missing effective_facts).

    If chunk_id is given, only that chunk is processed.
    """
    query: dict = {"bool": {"must_not": [{"exists": {"field": "effective_facts"}}]}}
    if chunk_id:
        query["bool"]["must"] = [{"term": {"parent_chunk_id": chunk_id}}]

    resp = _get_os().search(
        index  = SEMANTIC_OBJECTS_INDEX,
        scroll = "5m",
        size   = SCROLL_SIZE,
        body   = {
            "query":   query,
            "_source": list(_ENRICH_FIELDS) + [
                "object_id", "parent_chunk_id", "type",
                "text", "normalized_text", "entities",
                "section_category", "heading_path", "position",
            ],
        },
    )
    scroll_id = resp["_scroll_id"]
    hits = resp["hits"]["hits"]

    while hits:
        yield from hits
        resp  = _get_os().scroll(scroll_id=scroll_id, scroll="5m")
        scroll_id = resp["_scroll_id"]
        hits  = resp["hits"]["hits"]

    try:
        _get_os().clear_scroll(scroll_id=scroll_id)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill enrichment on semantic-objects")
    p.add_argument("--dry-run",   action="store_true",
                   help="Print what would be updated without writing to OpenSearch")
    p.add_argument("--limit",     type=int, default=None,
                   help="Stop after processing this many stale docs")
    p.add_argument("--chunk-id",  default=None,
                   help="Only process objects from this parent_chunk_id")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"\nSemantic Objects Enrichment Backfill")
    print(f"  Index    : {SEMANTIC_OBJECTS_INDEX} @ {OPENSEARCH_ENDPOINT[:45]}…")
    print(f"  Dry-run  : {args.dry_run}")
    if args.chunk_id:
        print(f"  Chunk    : {args.chunk_id}")
    if args.limit:
        print(f"  Limit    : {args.limit} stale docs")
    print()

    # ── Group stale docs by parent_chunk_id ───────────────────────────────────
    print("Scanning semantic-objects for stale enrichment…")
    chunk_buckets: dict[str, list[dict]] = defaultdict(list)
    n_stale = 0

    for hit in _scroll_stale(chunk_id=args.chunk_id):
        src = hit["_source"]
        src["object_id"] = hit["_id"]   # ensure object_id is always present
        cid = src.get("parent_chunk_id", "__no_chunk__")
        chunk_buckets[cid].append(src)
        n_stale += 1
        if args.limit and n_stale >= args.limit:
            break

    print(f"  Found {n_stale} stale objects across {len(chunk_buckets)} chunks")
    if n_stale == 0:
        print("  Nothing to do — all objects already enriched.")
        return

    # ── Process each chunk ────────────────────────────────────────────────────
    total_ok  = 0
    total_err = 0

    for chunk_idx, (cid, objects) in enumerate(chunk_buckets.items(), 1):
        print(f"\n[{chunk_idx}/{len(chunk_buckets)}] chunk={cid}  objects={len(objects)}")

        try:
            updates = _enrich_chunk_objects(objects)
            ok, err = _bulk_update(updates, dry_run=args.dry_run)
            total_ok  += ok
            total_err += err
            print(f"    updated={ok}  failed={err}  "
                  f"enriched={sum(1 for d in updates.values() if d.get('effective_facts'))}")
        except Exception as exc:
            logger.exception("Failed to process chunk %s", cid)
            print(f"    ✗ ERROR: {exc}")

    print(f"\n{'═' * 55}")
    print(f"  Done — objects_updated={total_ok}  errors={total_err}")
    print()


if __name__ == "__main__":
    main()
