from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import boto3

from shared.id_resolver import get_global_document_id
from shared.section_chunker import build_section_chunks
from shared.apryse_parser import parse_pages
from shared.sentence_builder import _build_objects
from shared.server_notify import notify_document_indexing_dispatch_status

logger = logging.getLogger("NLP_SENTENCE_BUILDER")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [NLP_SENTENCE_BUILDER] %(message)s",
)

SQS_MAX_ENTRIES = 10
SQS_MAX_BATCH_BYTES = int(os.environ.get("SQS_BATCH_MAX_BYTES", "900000"))
SQS_MAX_MESSAGE_BYTES = 256 * 1024
S3_OFFLOAD_THRESHOLD = int(os.environ.get("SQS_S3_OFFLOAD_THRESHOLD", "240000"))

s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "eu-west-1"))
sqs = boto3.client("sqs", region_name=os.environ.get("AWS_REGION", "eu-west-1"))


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build NLP sentence/semantic objects and dispatch document chunks to SQS")
    p.add_argument("--document-id", default=os.getenv("DOCUMENT_ID", ""))
    p.add_argument("--full-tables-key", default=os.getenv("FULL_TABLES_KEY", ""))
    p.add_argument("--input-bucket", default=os.getenv("INPUT_BUCKET", ""))
    p.add_argument("--source-s3-key", default=os.getenv("SOURCE_S3_KEY", ""))
    p.add_argument("--queue-url", default=os.getenv("QUEUE_URL", ""))
    p.add_argument("--payload-bucket", default=os.getenv("PAYLOAD_BUCKET", ""))
    p.add_argument("--payload-prefix", default=os.getenv("PAYLOAD_PREFIX", "nlp-sentence-builder-payloads"))
    p.add_argument("--tenant-id", default=os.getenv("TENANT_ID", ""))
    p.add_argument("--tenant-name", default=os.getenv("TENANT_NAME", ""))
    p.add_argument("--tenant-schema", default=os.getenv("TENANT_SCHEMA", ""))
    p.add_argument("--project-id", default=os.getenv("PROJECT_ID", ""))
    p.add_argument("--file-id", default=os.getenv("FILE_ID", ""))
    p.add_argument("--callback-url", default=os.getenv("CALLBACK_URL", ""))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--attempt_id", default=os.getenv("ATTEMPT_ID", ""))
    return p.parse_args()


def _require(args: argparse.Namespace) -> None:
    required = {
        "document-id": args.document_id,
        "full-tables-key": args.full_tables_key,
        "input-bucket": args.input_bucket,
        "queue-url": args.queue_url,
        "tenant-id": args.tenant_id,
        "tenant-name": args.tenant_name,
        "tenant-schema": args.tenant_schema,
        "project-id": args.project_id,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(f"Missing required configuration: {', '.join(missing)}")


def _tenant(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "tenant_id": str(args.tenant_id),
        "tenant_name": args.tenant_name,
        "tenant_schema": args.tenant_schema
    }


def _download_full_tables(bucket: str, key: str) -> tuple[dict, Path, float]:
    target = Path("/tmp/full_tables.json")
    t0 = time.perf_counter()
    logger.info("input download start bucket=%s key=%s", bucket, key)
    s3.download_file(bucket, key, str(target))
    elapsed = time.perf_counter() - t0
    size_mb = target.stat().st_size / 1024 / 1024
    logger.info("input download complete bytes=%d size_mb=%.2f elapsed_s=%.3f", target.stat().st_size, size_mb, elapsed)
    with target.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError("full_tables.json must contain a JSON object")
    return raw, target, elapsed


def _merge_doc_structure(raw: dict) -> dict:
    pages: list[dict] = []
    props: dict = {}
    for chunk in sorted(raw.get("chunks", []), key=lambda c: c.get("chunkIndex", 0)):
        ds = chunk.get("docStructure", {}) or {}
        if not props:
            props = ds.get("properties", {}) or {}
        offset = int(chunk.get("startPage", 1) or 1) - 1
        for page in ds.get("pages", []) or []:
            if offset:
                page = {
                    **page,
                    "properties": {
                        **page.get("properties", {}),
                        "pageNumber": int(page.get("properties", {}).get("pageNumber", 0) or 0) + offset,
                    },
                }
            pages.append(page)
    return {"properties": props, "pages": pages}


def _build_chunk_payloads(doc_structure: dict, args: argparse.Namespace, tenant: dict[str, Any]) -> list[dict]:
    total_pages = len(doc_structure.get("pages", []))
    pages = parse_pages(doc_structure, 1, total_pages)
    logger.info("parsed pages=%d", len(pages))
    if not pages:
        return []

    t0 = time.perf_counter()
    sections = build_section_chunks(pages, total_pages=total_pages)
    logger.info("section chunking complete sections=%d elapsed_s=%.3f", len(sections), time.perf_counter() - t0)

    global_document_id = get_global_document_id(
        str(args.document_id),
        tenant_id=str(args.tenant_id),
        project_id=str(args.project_id),
    )
    attempt_id = args.attempt_id

    payloads: list[dict] = []
    global_obj_counter = 0
    for idx, sec in enumerate(sections):
        chunk_id = f"{global_document_id}_chunk_{idx:04d}"
        t_chunk = time.perf_counter()
        objects = _build_objects(chunk_id, sec.virtual_pages, global_offset=global_obj_counter)
        for obj in objects:
            obj["section_category"] = sec.section_category
            obj["heading_path"] = " > ".join(sec.heading_path)
            obj["semantic_path"] = " > ".join(sec.semantic_path)
            obj["section_confidence"] = sec.section_confidence
            obj["document_position"] = sec.document_position
            obj["chunk_idx"] = sec.chunk_idx
            obj["parent_chunk_idx"] = sec.parent_chunk_idx
            obj["prev_chunk_idx"] = sec.prev_chunk_idx
            obj["next_chunk_idx"] = sec.next_chunk_idx
        global_obj_counter += len(objects)

        payloads.append({
            "source_type": "document",
            "document_id": str(args.document_id),
            "tenant_id": str(args.tenant_id),
            "attempt_id": attempt_id,
            "tenant_name": args.tenant_name,
            "tenant_schema": args.tenant_schema,
            "project_id": str(args.project_id),
            "file_id": str(args.file_id) if args.file_id else None,
            "chunk_id": chunk_id,
            "s3_bucket": args.input_bucket,
            "s3_key": args.source_s3_key,
            "page_start": sec.page_start,
            "page_end": sec.page_end,
            "section": sec.section,
            "subsection": sec.subsection,
            "section_category": sec.section_category,
            "heading_path": " > ".join(sec.heading_path),
            "heading_level": sec.heading_level,
            "word_count": sec.word_count,
            "chunk_idx": sec.chunk_idx,
            "parent_chunk_idx": sec.parent_chunk_idx,
            "prev_chunk_idx": sec.prev_chunk_idx,
            "next_chunk_idx": sec.next_chunk_idx,
            "section_confidence": sec.section_confidence,
            "document_position": sec.document_position,
            "semantic_path": " > ".join(sec.semantic_path),
            "heading_embedding_text": sec.heading_embedding_text,
            "extraction": {
                "raw_text": sec.text,
                "pages": sec.virtual_pages,
                "objects": objects,
            },
        })
        logger.info(
            "chunk built index=%d/%d chunk_id=%s pages=%d-%d objects=%d elapsed_s=%.3f",
            idx + 1, len(sections), chunk_id, sec.page_start, sec.page_end, len(objects), time.perf_counter() - t_chunk,
        )
    return payloads


def _send_batch(
    entries: list[dict],
    batch_number: int,
    total_batches_hint: int | None = None,
) -> tuple[int, int]:
    """Send one SQS batch.

    Individual SQS entry failures are NOT raised here. They are counted so the
    caller can report DISPATCH_PARTIAL. Only an actual SQS API/transport
    exception escapes and becomes DISPATCH_FAILED.
    """
    t0 = time.perf_counter()

    try:
        resp = sqs.send_message_batch(
            QueueUrl=_CURRENT_QUEUE_URL,
            Entries=entries,
        )
    except Exception:
        elapsed = time.perf_counter() - t0
        logger.exception(
            "sqs batch request failed batch=%d%s entries=%d elapsed_s=%.3f",
            batch_number,
            f"/{total_batches_hint}" if total_batches_hint else "",
            len(entries),
            elapsed,
        )
        raise

    failed = resp.get("Failed", []) or []
    successful = resp.get("Successful", []) or []
    elapsed = time.perf_counter() - t0

    logger.info(
        "sqs batch complete batch=%d%s entries=%d successful=%d failed=%d "
        "bytes=%d elapsed_s=%.3f",
        batch_number,
        f"/{total_batches_hint}" if total_batches_hint else "",
        len(entries),
        len(successful),
        len(failed),
        sum(len(e["MessageBody"].encode("utf-8")) for e in entries),
        elapsed,
    )

    for failure in failed:
        logger.error(
            "sqs entry failed batch=%d id=%s code=%s message=%s",
            batch_number,
            failure.get("Id"),
            failure.get("Code"),
            failure.get("Message"),
        )

    return len(successful), len(failed)


_CURRENT_QUEUE_URL = ""


def _dispatch(payloads: list[dict], args: argparse.Namespace) -> tuple[int, int, int]:
    global _CURRENT_QUEUE_URL
    _CURRENT_QUEUE_URL = args.queue_url
    payload_bucket = args.payload_bucket or args.input_bucket
    global_document_id = get_global_document_id(str(args.document_id), tenant_id=str(args.tenant_id), project_id=str(args.project_id))
    tenant_name = args.tenant_name 
    project_id = args.project_id

    batches: list[list[dict]] = []
    current: list[dict] = []
    current_bytes = 0
    offloaded = 0

    for payload in payloads:
        body_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        if len(body_bytes) > SQS_MAX_MESSAGE_BYTES:
            key = f"{args.payload_prefix.rstrip('/')}/{tenant_name}/{project_id}/{global_document_id}/{payload['chunk_id']}.json"
            t0 = time.perf_counter()
            s3.put_object(Bucket=payload_bucket, Key=key, Body=body_bytes, ContentType="application/json")
            elapsed = time.perf_counter() - t0
            logger.info("s3 offload chunk_id=%s bytes=%d elapsed_s=%.3f bucket=%s key=%s", payload["chunk_id"], len(body_bytes), elapsed, payload_bucket, key)
            offloaded += 1
            body = json.dumps({
                "source_type": "document",
                "document_id": payload["document_id"],
                "tenant_id": payload["tenant_id"],
                "tenant_name": payload["tenant_name"],
                "tenant_schema": payload["tenant_schema"],
                "project_id": payload["project_id"],
                "file_id": payload.get("file_id"),
                "attemptId": payload.get("attemptId"),
                "chunk_id": payload["chunk_id"],
                "s3_payload": {"bucket": payload_bucket, "key": key},
            })
        else:
            body = body_bytes.decode("utf-8")

        entry_bytes = len(body.encode("utf-8")) + 512
        if current and (len(current) >= SQS_MAX_ENTRIES or current_bytes + entry_bytes > SQS_MAX_BATCH_BYTES):
            batches.append(current)
            current = []
            current_bytes = 0
        current.append({"Id": f"m{len(batches):04d}_{len(current):02d}", "MessageBody": body})
        current_bytes += entry_bytes
    if current:
        batches.append(current)

    logger.info("dispatch prepared chunks=%d sqs_batches=%d offloaded=%d", len(payloads), len(batches), offloaded)
    sent = 0
    failed_dispatch = 0

    for number, entries in enumerate(batches, 1):
        batch_sent, batch_failed = _send_batch(
            entries,
            number,
            len(batches),
        )
        sent += batch_sent
        failed_dispatch += batch_failed

        logger.info(
            "dispatch batch accounted batch=%d/%d sent=%d failed=%d "
            "running_sent=%d running_failed=%d",
            number,
            len(batches),
            batch_sent,
            batch_failed,
            sent,
            failed_dispatch,
        )

    logger.info(
        "dispatch totals expected=%d dispatched=%d failed_dispatch=%d",
        len(payloads),
        sent,
        failed_dispatch,
    )

    return sent, offloaded, failed_dispatch


def main() -> int:
    args = _args()
    tenant_schema = args.tenant_schema
    attempt_id = os.getenv("ATTEMPT_ID", "")

    logger.info(
        "START document_id=%s file_id=%s tenant_schema=%s project_id=%s attempt_id=%s",
        args.document_id,
        args.file_id,
        args.tenant_schema,
        args.project_id,
        attempt_id,
    )

    try:
        # Validate before starting any work. If configuration is invalid,
        # report DISPATCH_FAILED and let the process exit with code 1.
        _require(args)

        tenant = _tenant(args)

        # Callback failures are deliberately non-fatal; the backend DB is the
        # state system of record.
        if args.file_id:
            notify_document_indexing_dispatch_status(
                job_id=args.file_id,
                tenant_schema=tenant_schema,
                status="PROCESSING",
                attempt_id=attempt_id,
                callback_url=args.callback_url,
            )

        raw, temp_path, _ = _download_full_tables(args.input_bucket, args.full_tables_key)
        t0 = time.perf_counter()
        doc_structure = _merge_doc_structure(raw)
        del raw
        logger.info("document structure merged pages=%d elapsed_s=%.3f", len(doc_structure.get("pages", [])), time.perf_counter() - t0)

        payloads = _build_chunk_payloads(doc_structure, args, tenant)
        expected = len(payloads)
        logger.info("object build complete document_id=%s expected_chunks=%d", args.document_id, expected)

        if args.file_id:
            notify_document_indexing_dispatch_status(
                job_id=args.file_id,
                tenant_schema=tenant_schema,
                status="DISPATCH_PREPARED",
                attempt_id=attempt_id,
                expected_chunks=expected,
                callback_url=args.callback_url,
            )

        if args.dry_run:
            logger.info("DRY RUN complete expected_chunks=%d", expected)
            return 0

        sent, offloaded, failed_dispatch = _dispatch(payloads, args)

        status = "DISPATCHED" if failed_dispatch == 0 else "DISPATCH_PARTIAL"

        logger.info(
            "DISPATCH COMPLETE document_id=%s expected_chunks=%d "
            "dispatched_chunks=%d failed_dispatch_chunks=%d offloaded=%d status=%s",
            args.document_id,
            expected,
            sent,
            failed_dispatch,
            offloaded,
            status,
        )

        if args.file_id:
            notify_document_indexing_dispatch_status(
                job_id=args.file_id,
                tenant_schema=tenant_schema,
                status=status,
                attempt_id=attempt_id,
                expected_chunks=expected,
                dispatched_chunks=sent,
                failed_dispatch_chunks=failed_dispatch,
                callback_url=args.callback_url,
            )

        # Partial entry failures are an expected/recoverable dispatch state.
        # Do NOT raise and do NOT turn the whole ECS task into DISPATCH_FAILED.
        return 0
    except Exception as exc:
        logger.exception("FAILED document_id=%s error=%s", args.document_id, exc)
        if args.file_id:
            notify_document_indexing_dispatch_status(
                job_id=args.file_id,
                tenant_schema=tenant_schema,
                status="DISPATCH_FAILED",
                attempt_id=attempt_id,
                error=str(exc),
                callback_url=args.callback_url,
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
