#!/usr/bin/env python3
"""
Migrate ALREADY-INDEXED geometry (document-chunks + semantic-objects) to S3.

One-time, deterministic export — indexing is currently stopped, so this is a
static source dataset, not a live migration. A local manifest.json is the
single control plane / ledger for the whole run:

  PENDING -> PROCESSING -> SUCCESS
                        -> FAILED (from either PENDING or PROCESSING)

For every distinct (tenant_id, project_id, document_id) triple (discovered
once via composite aggregations against BOTH indices — each bucket's
doc_count becomes the EXPECTED row count for that index, an independent
number to validate against later). document_id alone is NOT a safe identity
key here: it's the raw per-file value the live pipeline was invoked with,
never tenant-prefixed at indexing time, so the same document_id can occur
under different tenants/projects — the manifest's "id" field is therefore
the composite "{tenant_id}__{project_id}__{document_id}" global id (same
shape as shared.id_resolver.get_global_document_id()), with the raw
document_id kept separately as "documentId" for OpenSearch scans / S3 keys:
  1. Fetch every chunk doc AND every object/sentence doc for that one
     (tenant_id, project_id, document_id) (the two index streams run
     concurrently, bounded by a global OpenSearch-concurrency semaphore
     shared across all document workers, and every scan filters on
     document_id AND tenant_id AND project_id together). A missing id, a
     duplicate id, or a scroll exception in EITHER stream is a hard failure
     and cancels the sibling stream immediately.
  2. Build one flat {chunk_id|object_id|sentence_id: geometry} map for it.
  3. Validate fetchedChunks == expectedChunks AND fetchedObjects == expectedObjects
     AND len(geometry) == fetchedChunks + fetchedObjects — no silent partial data.
  4. Upload plain JSON directly to the deterministic key
     s3://{bucket}/{extraction_prefix}/geometry/geometry.json, where
     extraction_prefix = get_rls_file_s3_extraction_prefix(tenant_name, project_id, document_id).
     S3 durability is trusted — no read-back verification — but a raised
     exception from the put_object call itself still hard-fails the document.
  5. A document is SUCCESS only if every one of the above holds — any failure
     marks the WHOLE document FAILED, so a retry can simply overwrite the
     same S3 key.

This script never mutates OpenSearch — the manifest and S3 are the only
outputs. Dropping the "geometry" field from OpenSearch is a deliberately
separate, later step, gated on this manifest showing FAILED=0.

Usage
-----
  # Dry run (no S3 writes), first 5 documents only
  python3 tools/migrate_geometry_to_s3.py --bucket my-bucket --limit 5 --dry-run

  # Real run, all documents, resumable via manifest.json
  python3 tools/migrate_geometry_to_s3.py --bucket my-bucket --manifest-file geometry_migration_manifest.json

  # Re-run only documents the manifest marked FAILED
  python3 tools/migrate_geometry_to_s3.py --bucket my-bucket --manifest-file geometry_migration_manifest.json --retry-failed
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import resource
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3

# app.py sits directly at /app/app.py in the container alongside /app/shared —
# python already puts the script's own directory on sys.path, so no manual
# sys.path insert is needed here (unlike the old tools/ location).

from shared.opensearch_client import get_opensearch_client
from shared.id_resolver import get_rls_file_s3_extraction_prefix

logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)s [GeometryMigration] %(message)s",
)
logger = logging.getLogger("GeometryMigration")

# opensearchpy logs one line per HTTP call (every scroll page) at INFO, which
# drowns out the per-document progress lines below with no document identity
# attached to any of them — quiet it down to WARNING so our own logs are the
# signal, not the noise.
logging.getLogger("opensearch").setLevel(logging.WARNING)
logging.getLogger("opensearchpy").setLevel(logging.WARNING)

OPENSEARCH_INDEX       = "document-chunks"
SEMANTIC_OBJECTS_INDEX = "semantic-objects"
SCROLL_SIZE            = 2000
SCROLL_TTL             = "5m"
# Bounds every individual scroll HTTP call (see _scan_document_docs) — without
# this, a slow/stuck shard can hang a document's scan indefinitely with no
# further progress and no way to move on to the next document.
SCAN_REQUEST_TIMEOUT_SECONDS = 60


class GeometryValidationError(Exception):
    """Raised for anything that must hard-fail a document: missing id, duplicate
    id, expected/fetched count mismatch, or a scroll/S3 failure. Carries the
    partial fetched_count so a failure still records real progress in the
    manifest instead of collapsing to 0. is_cancellation marks the "sibling
    stream stopped early because the other one failed" case, so the caller
    can prefer reporting the actual root cause over this secondary message."""

    def __init__(self, message: str, fetched_count: int = 0, is_cancellation: bool = False):
        super().__init__(message)
        self.fetched_count = fetched_count
        self.is_cancellation = is_cancellation


def _peak_rss_mb() -> float:
    """Process-level peak resident set size, in MB. Note this is a PROCESS peak,
    not memory attributable to any one document — useful as a coarse signal,
    not a per-document measurement. ru_maxrss is bytes on macOS, KB on Linux."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return rss / divisor


# Substrings (lowercased) that mean the AWS session/temporary credentials used
# for this run have expired or been invalidated — e.g. temporary SSO/assume-role
# creds with a 2-3h lifetime dying mid-run. Once this happens EVERY subsequent
# OpenSearch/S3 call will fail the exact same way, so retrying other documents
# is pointless: the whole run should abort immediately, leaving not-yet-started
# documents PENDING for a rerun with fresh credentials.
CREDENTIALS_EXPIRED_SIGNATURES = (
    "security token included in the request is invalid",
    "security token included in the request is expired",
    "expiredtoken",
    "expiredtokenexception",
    "unrecognizedclientexception",
    "invalidclienttokenid",
    "requestexpired",
)


def _is_credentials_expired_error(error: object) -> bool:
    text = str(error).lower()
    return any(sig in text for sig in CREDENTIALS_EXPIRED_SIGNATURES)


# ─────────────────────────────────────────────────────────────────────────────
# Document discovery
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_tenant_names(client, index: str) -> dict[str, str]:
    """One cheap terms aggregation on tenant_id -> tenant_name, bucketed per
    DISTINCT TENANT (a handful of buckets), not per document. A per-document
    top_hits sub-agg on the main composite aggregation was measured to force
    a real document fetch for every one of ~13K composite buckets per page,
    taking 100+ seconds per page and intermittently hitting hard read
    timeouts — this does the same lookup at a tiny fraction of the cost."""
    resp = client.search(
        index=index,
        body={
            "size": 0,
            "aggs": {
                "tenants": {
                    "terms": {"field": "tenant_id", "size": 10000},
                    "aggs": {
                        "sample": {"top_hits": {"size": 1, "_source": ["tenant_name"]}}
                    },
                }
            },
        },
    )
    names: dict[str, str] = {}
    for bucket in resp["aggregations"]["tenants"]["buckets"]:
        hits = bucket["sample"]["hits"]["hits"]
        if hits:
            names[bucket["key"]] = hits[0]["_source"].get("tenant_name") or ""
    return names


def _discover_documents(client) -> dict[str, dict[str, Any]]:
    """List every distinct (tenant_id, project_id, document_id) triple with its
    tenant_name AND its expected row count per index, via composite
    aggregations against BOTH indices (union, not just document-chunks) — a
    document that only ever landed in semantic-objects (e.g. chunk indexing
    failed/was skipped) must still be discovered so coverage stays 100%.

    document_id alone is NOT used as the identity key: it's just the raw
    per-file value the ECS task was invoked with (see app.py's --document-id),
    never tenant-prefixed at indexing time, so the same document_id can occur
    under different tenants/projects. The composite (tenant_id, project_id,
    document_id) triple is the true unique document identity for this
    migration — same shape as shared.id_resolver.get_global_document_id().

    One bucket per document per index, not one hit per row, so this stays
    fast even with millions of indexed chunks. The composite bucket's own
    doc_count IS the expected row count for that index — this becomes the
    independent "expected" side of the fetched-vs-expected validation done
    later during export. No top_hits sub-agg here (see _fetch_tenant_names) —
    tenant_name is resolved separately, once per distinct tenant."""
    documents: dict[str, dict[str, Any]] = {}
    tenant_names: dict[str, str] = {}

    for index in (OPENSEARCH_INDEX, SEMANTIC_OBJECTS_INDEX):
        for tenant_id, name in _fetch_tenant_names(client, index).items():
            tenant_names.setdefault(tenant_id, name)

        after_key = None
        pages = 0

        while True:
            composite: dict[str, Any] = {
                "size": 1000,
                "sources": [
                    {"tenant_id":   {"terms": {"field": "tenant_id"}}},
                    {"project_id":  {"terms": {"field": "project_id"}}},
                    {"document_id": {"terms": {"field": "document_id"}}},
                ],
            }
            if after_key:
                composite["after"] = after_key

            resp = client.search(
                index=index,
                body={"size": 0, "aggs": {"documents": {"composite": composite}}},
                request_timeout=120,
            )
            agg = resp["aggregations"]["documents"]
            buckets = agg.get("buckets", [])
            pages += 1

            for bucket in buckets:
                tenant_id   = bucket["key"]["tenant_id"]
                project_id  = bucket["key"]["project_id"]
                document_id = bucket["key"]["document_id"]
                global_id = f"{tenant_id}__{project_id}__{document_id}"
                doc_count = bucket.get("doc_count", 0)
                existing = documents.get(global_id, {})
                entry = {
                    "document_id": document_id,
                    "tenant_id":   tenant_id,
                    "project_id":  project_id,
                    "tenant_name": tenant_names.get(tenant_id, ""),
                    "expected_chunks":  existing.get("expected_chunks", 0),
                    "expected_objects": existing.get("expected_objects", 0),
                }
                if index == OPENSEARCH_INDEX:
                    entry["expected_chunks"] = doc_count
                else:
                    entry["expected_objects"] = doc_count
                documents[global_id] = entry

            after_key = agg.get("after_key")
            if not after_key or not buckets:
                break

        logger.info("discovery pass complete index=%s pages=%d running_distinct_documents=%d", index, pages, len(documents))

    logger.info("discovery complete distinct_documents=%d", len(documents))
    return documents


# ─────────────────────────────────────────────────────────────────────────────
# Per-document geometry collection
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_SOURCE_FIELDS  = ["chunk_id", "page_start", "page_end"]
# NOTE: object/sentence docs never persist page_start/page_end as their own
# top-level fields in OpenSearch (only chunk docs do, via _build_chunk_doc).
# For objects, page_start/page_end only exist nested inside the "geometry"
# field (see _build_object_docs in lambdas/index/lambda_function.py), so we
# must read them from there, not from the top level of the object doc.
OBJECT_SOURCE_FIELDS = ["object_id", "type", "page", "bbox", "geometry"]


def _scan_document_docs(client, index: str, document_id: str, tenant_id: str, project_id: str, source_fields: list[str]):
    """Streaming generator over one document's rows in one index — never
    materializes the full result set, and only pulls the handful of fields
    actually needed (excludes embeddings/vectors and other large fields).

    Filters on document_id AND tenant_id AND project_id together: document_id
    alone is not guaranteed unique across tenants/projects (see _discover_documents),
    so a document_id-only term query could pull in rows from an unrelated
    tenant that happens to share the same raw document_id.

    request_timeout bounds EVERY individual HTTP call (the initial search AND
    every scroll continuation, via scroll_kwargs — helpers.scan only applies
    request_timeout to the first call otherwise) so a stuck/slow shard can
    never hang this generator indefinitely: it raises after SCAN_REQUEST_TIMEOUT_SECONDS
    instead, which the caller turns into a hard failure for just this one
    document — no more requests are sent for it, and (since this isn't a
    credentials-expired error) the rest of the run is unaffected."""
    from opensearchpy import helpers

    hits = helpers.scan(
        client,
        index=index,
        query={"query": {"bool": {"filter": [
            {"term": {"document_id": document_id}},
            {"term": {"tenant_id": tenant_id}},
            {"term": {"project_id": project_id}},
        ]}}},
        _source=source_fields,
        size=SCROLL_SIZE,
        scroll=SCROLL_TTL,
        preserve_order=False,
        request_timeout=SCAN_REQUEST_TIMEOUT_SECONDS,
        scroll_kwargs={"request_timeout": SCAN_REQUEST_TIMEOUT_SECONDS},
    )
    for hit in hits:
        yield hit["_source"]


ABORTED_MESSAGE = "aborted: another document hit an unrecoverable AWS credentials error, stopping all in-flight scans"


def _scan_with_semaphore(client, index: str, document_id: str, tenant_id: str, project_id: str, source_fields: list[str], semaphore: threading.Semaphore, abort_event: threading.Event):
    """Same as _scan_document_docs, but holds one slot of the global
    OpenSearch-concurrency semaphore for the whole scan — bounds total
    concurrent scroll streams across ALL document workers, regardless of how
    many documents are being processed in parallel. Checks the global
    abort_event both before AND after acquiring the semaphore: checking only
    before leaves a race where a thread already blocked on a full semaphore
    could acquire it and start a brand-new scan just after another thread sets
    abort_event while it waits — the post-acquire check closes that window."""
    if abort_event.is_set():
        raise GeometryValidationError(ABORTED_MESSAGE)
    semaphore.acquire()
    if abort_event.is_set():
        semaphore.release()
        raise GeometryValidationError(ABORTED_MESSAGE)
    try:
        yield from _scan_document_docs(client, index, document_id, tenant_id, project_id, source_fields)
    finally:
        semaphore.release()


def _collect_chunk_geometry(
    client, document_id: str, tenant_id: str, project_id: str, semaphore: threading.Semaphore, cancel_event: threading.Event, abort_event: threading.Event,
) -> tuple[dict[str, dict], int]:
    geometry: dict[str, dict] = {}
    count = 0
    try:
        for doc in _scan_with_semaphore(client, OPENSEARCH_INDEX, document_id, tenant_id, project_id, CHUNK_SOURCE_FIELDS, semaphore, abort_event):
            if abort_event.is_set():
                raise GeometryValidationError(ABORTED_MESSAGE, fetched_count=count)
            if cancel_event.is_set():
                raise GeometryValidationError("chunk scan cancelled: sibling object scan failed", fetched_count=count, is_cancellation=True)
            count += 1
            chunk_id = doc.get("chunk_id")
            if not chunk_id:
                raise GeometryValidationError(f"chunk doc missing chunk_id (record #{count})", fetched_count=count)
            if chunk_id in geometry:
                raise GeometryValidationError(f"duplicate chunk_id={chunk_id}", fetched_count=count)
            geometry[chunk_id] = {
                "page_start": doc.get("page_start"),
                "page_end":   doc.get("page_end"),
            }
    except GeometryValidationError:
        cancel_event.set()
        raise
    except Exception as exc:
        if _is_credentials_expired_error(exc):
            abort_event.set()
        cancel_event.set()
        raise GeometryValidationError(f"chunk scroll failed after {count} records: {exc!r}", fetched_count=count) from exc
    return geometry, count


def _collect_object_geometry(
    client, document_id: str, tenant_id: str, project_id: str, semaphore: threading.Semaphore, cancel_event: threading.Event, abort_event: threading.Event,
) -> tuple[dict[str, dict], int]:
    geometry: dict[str, dict] = {}
    count = 0
    try:
        for doc in _scan_with_semaphore(client, SEMANTIC_OBJECTS_INDEX, document_id, tenant_id, project_id, OBJECT_SOURCE_FIELDS, semaphore, abort_event):
            if abort_event.is_set():
                raise GeometryValidationError(ABORTED_MESSAGE, fetched_count=count)
            if cancel_event.is_set():
                raise GeometryValidationError("object scan cancelled: sibling chunk scan failed", fetched_count=count, is_cancellation=True)
            count += 1
            object_id = doc.get("object_id")
            if not object_id:
                raise GeometryValidationError(f"object/sentence doc missing object_id (record #{count})", fetched_count=count)
            if object_id in geometry:
                raise GeometryValidationError(f"duplicate object_id={object_id}", fetched_count=count)
            doc_geometry = doc.get("geometry") or {}
            if doc.get("type") == "sentence":
                geometry[object_id] = {
                    "page": doc.get("page"),
                    "bbox": doc.get("bbox", []),
                    "geometry": doc_geometry,
                }
            else:
                geometry[object_id] = {
                    "page":       doc.get("page"),
                    # page_start/page_end for objects live only inside geometry (see note above).
                    "page_start": doc_geometry.get("page_start", doc.get("page")),
                    "page_end":   doc_geometry.get("page_end", doc.get("page")),
                    "bbox":       doc.get("bbox", []),
                    "geometry":   doc_geometry,
                }
    except GeometryValidationError:
        cancel_event.set()
        raise
    except Exception as exc:
        if _is_credentials_expired_error(exc):
            abort_event.set()
        cancel_event.set()
        raise GeometryValidationError(f"object scroll failed after {count} records: {exc!r}", fetched_count=count) from exc
    return geometry, count


def _build_geometry_map_for_document(
    client, document_id: str, tenant_id: str, project_id: str, semaphore: threading.Semaphore, abort_event: threading.Event,
) -> tuple[dict[str, dict], int, int, Exception | None]:
    """Returns (geometry_map, chunk_count, object_and_sentence_count, error).
    The two index streams run concurrently (each still bound by the shared
    global semaphore) instead of sequentially. A shared cancel_event means
    that if either stream hard-fails (missing/duplicate id, scroll exception),
    the sibling stream stops at its next record instead of scanning to
    completion for no reason. abort_event is the PROCESS-WIDE credentials-expiry
    signal: a collector sets it itself the instant it detects a credentials-expired
    exception (not only after _process_document returns), and every stream of
    every document checks it on every record, so an in-flight scroll stops within
    one record of the failure being detected anywhere in the process. Both
    futures are always awaited so neither leaks a scroll context. Failures are
    returned (not raised) so the caller still gets the real partial counts
    instead of them collapsing to 0."""
    cancel_event = threading.Event()
    with ThreadPoolExecutor(max_workers=2) as pool:
        chunk_future  = pool.submit(_collect_chunk_geometry, client, document_id, tenant_id, project_id, semaphore, cancel_event, abort_event)
        object_future = pool.submit(_collect_object_geometry, client, document_id, tenant_id, project_id, semaphore, cancel_event, abort_event)

        chunk_geometry, chunk_count, chunk_error = {}, 0, None
        try:
            chunk_geometry, chunk_count = chunk_future.result()
        except Exception as exc:
            chunk_error = exc
            chunk_count = getattr(exc, "fetched_count", 0)

        object_geometry, object_count, object_error = {}, 0, None
        try:
            object_geometry, object_count = object_future.result()
        except Exception as exc:
            object_error = exc
            object_count = getattr(exc, "fetched_count", 0)

    geometry = {**chunk_geometry, **object_geometry}
    root_error = None
    for err in (chunk_error, object_error):
        if err is not None and not getattr(err, "is_cancellation", False):
            root_error = err
            break
    else:
        root_error = chunk_error or object_error
    return geometry, chunk_count, object_count, root_error


# ─────────────────────────────────────────────────────────────────────────────
# S3 upload + verification
# ─────────────────────────────────────────────────────────────────────────────

def _upload_to_s3(s3, bucket: str, key: str, geometry: dict, dry_run: bool) -> tuple[int, float, float]:
    """Plain JSON, single put_object to the final deterministic key — S3 PUT is
    atomic (no partial object from one PUT call), and put_object raises on any
    failure, which is enough to hard-fail the document. No read-back/HEAD
    verification: S3 durability is trusted, not re-checked.
    Returns (byte_size, serialize_seconds, s3_put_seconds)."""
    t0 = time.perf_counter()
    body = json.dumps(geometry, separators=(",", ":")).encode("utf-8")
    serialize_seconds = time.perf_counter() - t0

    if dry_run:
        logger.info("[DRY RUN] would upload bucket=%s key=%s ids=%d bytes=%d", bucket, key, len(geometry), len(body))
        return len(body), serialize_seconds, 0.0

    t1 = time.perf_counter()
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    put_seconds = time.perf_counter() - t1
    return len(body), serialize_seconds, put_seconds


# ─────────────────────────────────────────────────────────────────────────────
# Manifest (the migration's control plane / ledger — no OpenSearch mutation)
# ─────────────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _init_manifest(documents: dict[str, dict[str, Any]]) -> dict:
    document_data = [
        {
            "id": global_id,
            "documentId": meta.get("document_id") or "",
            "tenantName": meta.get("tenant_name") or "",
            "tenantId": meta.get("tenant_id") or "",
            "projectId": meta.get("project_id") or "",
            "expectedChunks": meta.get("expected_chunks", 0),
            "expectedObjects": meta.get("expected_objects", 0),
            "expectedTotal": meta.get("expected_chunks", 0) + meta.get("expected_objects", 0),
            "status": "PENDING",
            "fetchedChunks": 0,
            "fetchedObjects": 0,
            "fetchedTotal": 0,
            "s3Key": None,
            "error": None,
            "geometryBytes": 0,
            "scanSeconds": 0.0,
            "serializeSeconds": 0.0,
            "s3PutSeconds": 0.0,
            "elapsedSeconds": 0.0,
            "peakRssMb": 0.0,
        }
        for global_id, meta in documents.items()
    ]
    return {"totalDocuments": len(document_data), "createdAt": _now_iso(), "documentData": document_data}


VALID_STATUSES = {"PENDING", "PROCESSING", "SUCCESS", "FAILED"}
REQUIRED_MANIFEST_FIELDS = (
    "id", "documentId", "tenantName", "tenantId", "projectId",
    "expectedChunks", "expectedObjects", "expectedTotal", "status",
    "fetchedChunks", "fetchedObjects", "fetchedTotal", "s3Key", "error",
)
NUMERIC_MANIFEST_FIELDS = ("expectedChunks", "expectedObjects", "expectedTotal", "fetchedChunks", "fetchedObjects", "fetchedTotal")


def _is_valid_count(value: object) -> bool:
    # bool is a subclass of int in Python — explicitly reject True/False so they
    # don't silently pass as 0/1.
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_manifest(manifest: dict) -> list[str]:
    """Structural sanity checks on the manifest before any processing starts —
    catches a corrupted/hand-edited manifest early instead of discovering it
    mid-run. Returns a list of problem descriptions (empty = valid)."""
    problems: list[str] = []
    document_data = manifest.get("documentData")
    if not isinstance(document_data, list):
        return ["documentData is missing or not a list"]

    declared_total = manifest.get("totalDocuments")
    if declared_total != len(document_data):
        problems.append(f"totalDocuments={declared_total} does not match len(documentData)={len(document_data)}")

    seen_ids: set[str] = set()
    for idx, entry in enumerate(document_data):
        missing = [f for f in REQUIRED_MANIFEST_FIELDS if f not in entry]
        if missing:
            problems.append(f"entry[{idx}] missing fields: {missing}")
            continue

        entry_id = entry["id"]
        if entry_id in seen_ids:
            problems.append(f"duplicate manifest id: {entry_id}")
        seen_ids.add(entry_id)

        if entry["status"] not in VALID_STATUSES:
            problems.append(f"entry[{idx}] id={entry_id} has invalid status={entry['status']!r}")

        # A corrupted manifest could hold a string/float/bool/None where an int is
        # expected — validate the type BEFORE any arithmetic/comparison below, or a
        # hand-edited manifest crashes this function with a TypeError instead of
        # producing a clean "manifest validation failed" report.
        bad_types = [f for f in NUMERIC_MANIFEST_FIELDS if not _is_valid_count(entry.get(f))]
        if bad_types:
            problems.append(f"entry[{idx}] id={entry_id} has non-integer value(s) for: {bad_types}")
            continue

        expected_chunks = entry["expectedChunks"]
        expected_objects = entry["expectedObjects"]
        expected_total = entry["expectedTotal"]
        if expected_chunks < 0 or expected_objects < 0:
            problems.append(f"entry[{idx}] id={entry_id} has negative expected counts")
        if expected_total != expected_chunks + expected_objects:
            problems.append(
                f"entry[{idx}] id={entry_id} expectedTotal={expected_total} "
                f"!= expectedChunks+expectedObjects={expected_chunks + expected_objects}"
            )

        fetched_chunks = entry["fetchedChunks"]
        fetched_objects = entry["fetchedObjects"]
        fetched_total = entry["fetchedTotal"]
        if fetched_chunks < 0 or fetched_objects < 0:
            problems.append(f"entry[{idx}] id={entry_id} has negative fetched counts")
        if fetched_total != fetched_chunks + fetched_objects:
            problems.append(
                f"entry[{idx}] id={entry_id} fetchedTotal={fetched_total} "
                f"!= fetchedChunks+fetchedObjects={fetched_chunks + fetched_objects}"
            )

    return problems


def _load_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    # A PROCESSING entry means the previous run crashed/was killed mid-document —
    # safe to retry from scratch since a document is only ever SUCCESS as a whole.
    reset = 0
    for entry in manifest.get("documentData", []):
        if entry.get("status") == "PROCESSING":
            entry["status"] = "PENDING"
            reset += 1
    if reset:
        logger.info("reset %d PROCESSING entries to PENDING (crash recovery)", reset)
    return manifest


def _save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(path)  # atomic — only the main thread ever writes the manifest


def _download_manifest_from_s3(s3, bucket: str, key: str, path: Path) -> bool:
    """Pulls the manifest down to `path` before anything else runs, so a task
    resumes from wherever the last run (possibly on a different container/host)
    left off. Returns True if an object was found and downloaded, False if the
    key doesn't exist yet (first-ever run for this bucket/key — caller falls
    back to discovery or a baked-in seed)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(path))
        logger.info("resumed manifest from s3://%s/%s -> %s", bucket, key, path)
        return True
    except Exception as exc:
        if "404" in str(exc) or "Not Found" in str(exc) or "NoSuchKey" in str(exc):
            logger.info("no manifest at s3://%s/%s yet — starting fresh", bucket, key)
            return False
        raise


def _upload_manifest_to_s3(s3, bucket: str, key: str, path: Path) -> None:
    """Pushes the manifest up after every local save — the manifest is the only
    durable progress record for this run, and the container's disk itself is
    not: a crash, an OOM kill, or the task simply being stopped must still
    leave the latest state in S3 so a rerun resumes from it, not from scratch."""
    s3.upload_file(str(path), bucket, key)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _process_document(
    client, s3, document_id: str, tenant_id: str, project_id: str, entry: dict, bucket: str, dry_run: bool, semaphore: threading.Semaphore, abort_event: threading.Event,
) -> dict:
    """Does all the work for one (tenant_id, project_id, document_id). SUCCESS
    requires ALL of:
      - both scans complete without exception (no scroll failure)
      - every record had an id, and no id was seen twice
      - fetchedChunks == expectedChunks AND fetchedObjects == expectedObjects
        (expected counts come from discovery's composite-bucket doc_count —
        an independent count, not just "however many the scan happened to yield")
      - the S3 upload completed without raising
      - abort_event was never observed set at any checkpoint below
    Any failure marks the WHOLE document FAILED — never a partial artifact —
    but still records the real partial counts reached before the failure.

    Simple, deliberate rule for credential expiry: the instant abort_event is
    set — by THIS document's own scan/upload, or by ANY other document running
    concurrently — every in-flight document is marked FAILED, full stop. This
    is checked before the scan starts, on every scroll record (inside the
    collectors), immediately before the S3 upload, and immediately after it
    succeeds. There's deliberately no special case for "the upload already
    finished with no error" — a same-run FAILED verdict for a document whose
    S3 write happened to already land is harmless, since the S3 key is
    deterministic and a retry simply overwrites it."""
    tenant_name = entry.get("tenantName")
    expected_chunks  = entry.get("expectedChunks", 0)
    expected_objects = entry.get("expectedObjects", 0)

    logger.info(
        "start global_id=%s document_id=%s tenant=%s project_id=%s expected_chunks=%d expected_objects=%d expected_total=%d",
        entry.get("id"), document_id, tenant_name, project_id, expected_chunks, expected_objects, expected_chunks + expected_objects,
    )

    t_start = time.perf_counter()

    def _failed(
        error: str, chunk_count: int = 0, object_count: int = 0, key: str | None = None, scan_seconds: float = 0.0,
    ) -> dict:
        if _is_credentials_expired_error(error):
            abort_event.set()
        return {
            "status": "FAILED", "s3Key": key, "error": error,
            "fetchedChunks": chunk_count, "fetchedObjects": object_count, "fetchedTotal": chunk_count + object_count,
            "geometryBytes": 0, "scanSeconds": scan_seconds, "serializeSeconds": 0.0, "s3PutSeconds": 0.0,
            "elapsedSeconds": time.perf_counter() - t_start, "peakRssMb": _peak_rss_mb(),
            "fatal": _is_credentials_expired_error(error),
        }

    if abort_event.is_set():
        return _failed(ABORTED_MESSAGE)
    if not tenant_name or not project_id:
        return _failed("missing tenantName/projectId")

    prefix = get_rls_file_s3_extraction_prefix(tenant_name, project_id, document_id)
    key = f"{prefix}/geometry/geometry.json"

    chunk_count = object_count = 0
    try:
        t_scan = time.perf_counter()
        geometry, chunk_count, object_count, scan_error = _build_geometry_map_for_document(client, document_id, tenant_id, project_id, semaphore, abort_event)
        scan_seconds = time.perf_counter() - t_scan

        if scan_error is not None:
            return _failed(str(scan_error), chunk_count, object_count, scan_seconds=scan_seconds)
        if chunk_count != expected_chunks:
            return _failed(f"chunk count mismatch: expected={expected_chunks} fetched={chunk_count}", chunk_count, object_count, scan_seconds=scan_seconds)
        if object_count != expected_objects:
            return _failed(f"object count mismatch: expected={expected_objects} fetched={object_count}", chunk_count, object_count, scan_seconds=scan_seconds)
        if len(geometry) != chunk_count + object_count:
            return _failed(f"geometry id count mismatch: ids={len(geometry)} fetched_total={chunk_count + object_count}", chunk_count, object_count, scan_seconds=scan_seconds)
        if not geometry:
            return _failed("empty_geometry", chunk_count, object_count, scan_seconds=scan_seconds)
        if abort_event.is_set():
            return _failed(ABORTED_MESSAGE, chunk_count, object_count, scan_seconds=scan_seconds)

        geometry_bytes, serialize_seconds, s3_put_seconds = _upload_to_s3(s3, bucket, key, geometry, dry_run)
        elapsed_seconds = time.perf_counter() - t_start
        peak_rss_mb = _peak_rss_mb()

        if abort_event.is_set():
            # Upload itself succeeded, but another document's credentials-expiry
            # abort was observed before/while it ran — mark FAILED anyway per the
            # simple "any in-flight document = FAILED" rule; harmless to overwrite
            # this same deterministic S3 key on the next run.
            return _failed(ABORTED_MESSAGE, chunk_count, object_count, key=key, scan_seconds=scan_seconds)

        logger.info(
            "done global_id=%s document_id=%s key=%s chunks=%d objects_and_sentences=%d ids=%d "
            "geometry_bytes=%d scan_s=%.2f serialize_s=%.3f s3_put_s=%.2f elapsed_s=%.2f peak_rss_mb=%.1f",
            entry.get("id"), document_id, key, chunk_count, object_count, len(geometry),
            geometry_bytes, scan_seconds, serialize_seconds, s3_put_seconds, elapsed_seconds, peak_rss_mb,
        )
        return {
            "status": "SUCCESS", "s3Key": key, "error": None,
            "fetchedChunks": chunk_count, "fetchedObjects": object_count, "fetchedTotal": chunk_count + object_count,
            "geometryBytes": geometry_bytes, "scanSeconds": scan_seconds, "serializeSeconds": serialize_seconds,
            "s3PutSeconds": s3_put_seconds, "elapsedSeconds": elapsed_seconds, "peakRssMb": peak_rss_mb,
            "fatal": False,
        }
    except Exception as exc:
        logger.exception("failed document_id=%s", document_id)
        return _failed(repr(exc)[:500], chunk_count, object_count, key=key)


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def main() -> int:
    p = argparse.ArgumentParser(description="One-time migration of already-indexed geometry to S3, driven by a manifest.json ledger")
    # Every default falls back to an env var so an ECS task definition's plain
    # environment block can drive this with zero command-line overrides —
    # same convention as ecs/nlp-sentence-builder/app.py.
    p.add_argument("--bucket", default=os.environ.get("BUCKET", ""), help="S3 bucket the extraction outputs live in")
    p.add_argument("--manifest-file", default=os.environ.get("MANIFEST_FILE", "geometry_migration_manifest.json"), help="Path to the manifest.json control plane / ledger")
    p.add_argument("--limit", type=int, default=int(os.environ.get("LIMIT", "0")), help="Only process the first N documents (0 = no limit)")
    p.add_argument("--only-document-id", default=os.environ.get("ONLY_DOCUMENT_ID", ""), help="Process a single manifest entry by its composite id (tenantId__projectId__documentId; debugging)")
    p.add_argument("--dry-run", action="store_true", default=_bool_env("DRY_RUN"), help="Don't write to S3, just log what would happen")
    p.add_argument("--overwrite", action="store_true", default=_bool_env("OVERWRITE"), help="Reprocess documents even if the manifest already marks them SUCCESS")
    p.add_argument("--retry-failed", action="store_true", default=_bool_env("RETRY_FAILED"), help="Only (re)process documents the manifest marks FAILED")
    p.add_argument("--retry-error-contains", default=os.environ.get("RETRY_ERROR_CONTAINS", ""), help="With --retry-failed, further narrow to FAILED documents whose manifest error message contains this substring (case-insensitive) — e.g. 'scroll failed' to retry only transient network/scroll errors, leaving other failures (like count mismatches) untouched for investigation")
    p.add_argument(
        "--skip-tenant", action="append",
        default=[t for t in os.environ.get("SKIP_TENANT", "").split(",") if t.strip()],
        help="Exclude documents belonging to this tenantName from the processing plan (case-insensitive; repeatable, or comma-separated) — e.g. --skip-tenant 'QA AUTH TEST'",
    )
    p.add_argument("--document-workers", type=int, default=int(os.environ.get("DOCUMENT_WORKERS", "8")), help="Number of documents to process concurrently")
    p.add_argument("--opensearch-concurrency", type=int, default=int(os.environ.get("OPENSEARCH_CONCURRENCY", "16")), help="Global cap on concurrent OpenSearch scan streams across ALL document workers combined")
    p.add_argument("--aws-profile", default=os.environ.get("AWS_PROFILE", ""), help="Named AWS profile (~/.aws/credentials / ~/.aws/config) to use for both OpenSearch and S3 calls, e.g. 'qa' — leave unset in ECS, where the task role already provides credentials")
    p.add_argument("--manifest-s3-bucket", default=os.environ.get("MANIFEST_S3_BUCKET", ""), help="If set (with --manifest-s3-key), the manifest ledger is pulled from this bucket before the run and pushed back after every save — required for ECS, where local disk doesn't survive across task runs")
    p.add_argument("--manifest-s3-key", default=os.environ.get("MANIFEST_S3_KEY", ""), help="S3 key for the manifest ledger within --manifest-s3-bucket")
    args = p.parse_args()

    if not args.bucket:
        p.error("--bucket (or BUCKET env var) is required")
    if bool(args.manifest_s3_bucket) != bool(args.manifest_s3_key):
        p.error("--manifest-s3-bucket and --manifest-s3-key must be set together")
    if args.document_workers < 1:
        p.error("--document-workers must be >= 1")
    if args.opensearch_concurrency < 1:
        p.error("--opensearch-concurrency must be >= 1")
    if args.limit < 0:
        p.error("--limit must be >= 0")
    if args.retry_failed and args.overwrite:
        p.error("--retry-failed and --overwrite are mutually exclusive (ambiguous which documents to process)")
    if args.retry_error_contains and not args.retry_failed:
        p.error("--retry-error-contains requires --retry-failed")

    if args.aws_profile:
        # Setting the env var (rather than boto3.setup_default_session, which
        # only affects boto3.client()/resource() shortcuts) is what actually
        # reaches shared.opensearch_client's explicit boto3.Session() call too —
        # both that and every thread's boto3.client("s3") honor AWS_PROFILE.
        os.environ["AWS_PROFILE"] = args.aws_profile
        logger.info("using AWS profile=%s", args.aws_profile)

    client = get_opensearch_client()
    manifest_path = Path(args.manifest_file)
    manifest_s3 = boto3.client("s3") if args.manifest_s3_bucket else None

    def _persist_manifest() -> None:
        _save_manifest(manifest_path, manifest)
        if manifest_s3:
            _upload_manifest_to_s3(manifest_s3, args.manifest_s3_bucket, args.manifest_s3_key, manifest_path)

    resumed_from_s3 = False
    if manifest_s3:
        resumed_from_s3 = _download_manifest_from_s3(manifest_s3, args.manifest_s3_bucket, args.manifest_s3_key, manifest_path)

    manifest = _load_manifest(manifest_path)
    if manifest is None:
        logger.info("no existing manifest at %s — discovering documents", manifest_path)
        documents = _discover_documents(client)
        manifest = _init_manifest(documents)
        _persist_manifest()
    else:
        logger.info(
            "loaded existing manifest %s total_documents=%d resumed_from_s3=%s",
            manifest_path, len(manifest["documentData"]), resumed_from_s3,
        )

    problems = _validate_manifest(manifest)
    if problems:
        logger.error("manifest failed validation with %d problem(s):", len(problems))
        for problem in problems:
            logger.error("  - %s", problem)
        return 1

    entries_by_id = {e["id"]: e for e in manifest["documentData"]}

    if args.only_document_id:
        if args.only_document_id not in entries_by_id:
            logger.error("document_id=%s not found in manifest", args.only_document_id)
            return 1
        todo = [entries_by_id[args.only_document_id]]
    elif args.retry_failed:
        todo = [e for e in manifest["documentData"] if e["status"] == "FAILED"]
        if args.retry_error_contains:
            needle = args.retry_error_contains.lower()
            todo = [e for e in todo if needle in (e.get("error") or "").lower()]
    elif args.overwrite:
        todo = list(manifest["documentData"])
    else:
        todo = [e for e in manifest["documentData"] if e["status"] != "SUCCESS"]

    skip_tenants = {
        name.strip().lower()
        for entry in args.skip_tenant
        for name in entry.split(",")
        if name.strip()
    }
    if skip_tenants:
        before = len(todo)
        todo = [e for e in todo if (e.get("tenantName") or "").strip().lower() not in skip_tenants]
        logger.info("skip_tenant filter excluded %d document(s) for tenant(s): %s", before - len(todo), sorted(skip_tenants))

    if args.limit:
        todo = todo[: args.limit]

    logger.info(
        "processing plan documents=%d dry_run=%s document_workers=%d opensearch_concurrency=%d retry_failed=%s retry_error_contains=%r",
        len(todo), args.dry_run, args.document_workers, args.opensearch_concurrency, args.retry_failed, args.retry_error_contains,
    )

    semaphore = threading.Semaphore(args.opensearch_concurrency)
    lock = threading.Lock()
    completed = 0
    skipped_pending = 0
    totals: dict[str, int] = {"SUCCESS": 0, "FAILED": 0}
    total_geometry_bytes = 0
    total_elapsed_seconds = 0.0
    max_peak_rss_mb = 0.0
    # Set the instant ANY document, at ANY stage (OpenSearch scroll or S3 put),
    # hits an expired/invalid AWS session token — every subsequent OpenSearch/S3
    # call would fail identically. Once this trips:
    #   - no NEW document is started (stays PENDING)
    #   - every OTHER in-flight document is marked FAILED immediately, at whatever
    #     checkpoint it's currently at (mid-scroll, about to upload, or just
    #     finished uploading) — deliberately simple, no exception for "the upload
    #     already technically succeeded": the S3 key is deterministic, so
    #     reprocessing that document on the next run just overwrites it.
    abort_event = threading.Event()

    def _run(entry: dict) -> tuple[str, dict | None]:
        # Each thread gets its own S3 client (boto3 clients aren't guaranteed thread-safe);
        # the OpenSearch client is shared — its underlying requests session pool is sized
        # for concurrent use (see OPENSEARCH_MAXSIZE in shared/opensearch_client.py), and
        # the global semaphore caps how many scans actually run at once regardless.
        global_id = entry["id"]
        if abort_event.is_set():
            return global_id, None  # never started — caller reverts it to PENDING
        with lock:
            entry["status"] = "PROCESSING"
        document_id = entry["documentId"]
        thread_s3 = boto3.client("s3")
        result = _process_document(
            client, thread_s3, document_id, entry["tenantId"], entry["projectId"], entry, args.bucket, args.dry_run, semaphore, abort_event,
        )
        if result.get("fatal"):
            abort_event.set()
        return global_id, result

    with ThreadPoolExecutor(max_workers=args.document_workers) as pool:
        futures = [pool.submit(_run, entry) for entry in todo]
        for future in as_completed(futures):
            global_id, result = future.result()
            with lock:
                entry = entries_by_id[global_id]
                if result is None:
                    entry["status"] = "PENDING"
                    skipped_pending += 1
                    continue
                completed += 1
                entry["status"] = result["status"]
                entry["fetchedChunks"] = result["fetchedChunks"]
                entry["fetchedObjects"] = result["fetchedObjects"]
                entry["fetchedTotal"] = result["fetchedTotal"]
                entry["s3Key"] = result["s3Key"]
                entry["error"] = result["error"]
                entry["geometryBytes"] = result["geometryBytes"]
                entry["scanSeconds"] = result["scanSeconds"]
                entry["serializeSeconds"] = result["serializeSeconds"]
                entry["s3PutSeconds"] = result["s3PutSeconds"]
                entry["elapsedSeconds"] = result["elapsedSeconds"]
                entry["peakRssMb"] = result["peakRssMb"]
                totals[result["status"]] = totals.get(result["status"], 0) + 1
                if result["status"] == "SUCCESS":
                    total_geometry_bytes += result["geometryBytes"]
                total_elapsed_seconds += result["elapsedSeconds"]
                max_peak_rss_mb = max(max_peak_rss_mb, result["peakRssMb"])
                if result.get("fatal"):
                    logger.error(
                        "FATAL: AWS credentials expired/invalid on global_id=%s (%s) — aborting dispatch of new documents; "
                        "not-yet-started documents remain PENDING for a rerun with fresh credentials",
                        global_id, result["error"],
                    )
                # Saved after EVERY document (not batched) — with atomic replace this
                # is cheap at ~13K documents and makes the manifest a genuine durable
                # ledger: a crash never loses a completed result. Pushed to S3 in the
                # same breath (when configured) so a killed/restarted task resumes
                # from here too, not just from this container's local disk.
                _persist_manifest()
                remaining = len(todo) - completed - skipped_pending
                avg_so_far = total_elapsed_seconds / completed if completed else 0.0
                eta_seconds = avg_so_far * remaining / args.document_workers if avg_so_far else 0.0
                logger.info(
                    "progress %d/%d remaining=%d (skipped_pending=%d success=%d failed=%d) last=%s document_id=%s tenant=%s "
                    "fetched=%d/%d eta_min=%.1f",
                    completed, len(todo), remaining, skipped_pending, totals.get("SUCCESS", 0), totals.get("FAILED", 0),
                    global_id, entry.get("documentId"), entry.get("tenantName"),
                    result["fetchedTotal"], entry.get("expectedTotal", 0), eta_seconds / 60,
                )

    _persist_manifest()

    success_count = totals.get("SUCCESS", 0)
    avg_elapsed_seconds = total_elapsed_seconds / completed if completed else 0.0
    logger.info(
        "MIGRATION SUMMARY total_documents=%d attempted=%d skipped_pending=%d success=%d failed=%d "
        "total_geometry_gb=%.3f avg_doc_elapsed_s=%.2f max_peak_rss_mb=%.1f aborted=%s",
        len(todo), completed, skipped_pending, success_count, totals.get("FAILED", 0),
        total_geometry_bytes / (1024 ** 3), avg_elapsed_seconds, max_peak_rss_mb, abort_event.is_set(),
    )
    if abort_event.is_set():
        return 1
    return 0 if totals.get("FAILED", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
