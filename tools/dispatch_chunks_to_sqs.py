from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_loaded: dict[str, types.ModuleType] = {}


def _load(rel_path: str, alias: str) -> types.ModuleType:
    if alias in _loaded:
        return _loaded[alias]

    lf_path = ROOT / "lambdas" / rel_path / "lambda_function.py"
    spec = importlib.util.spec_from_file_location(alias, lf_path)
    mod = importlib.util.module_from_spec(spec)
    lf_dir = str(lf_path.parent)
    if lf_dir not in sys.path:
        sys.path.insert(0, lf_dir)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    _loaded[alias] = mod
    return mod


def _merge_doc_structure(raw: dict) -> dict:
    all_pages: list[dict] = []
    merged_props: dict = {}
    for chunk in sorted(raw.get("chunks", []), key=lambda c: c.get("chunkIndex", 0)):
        ds = chunk.get("docStructure", {})
        if not merged_props:
            merged_props = ds.get("properties", {})
        # Apryse numbers pages locally (1–N per extraction chunk).
        # Offset by chunk.startPage - 1 to get global document page numbers.
        offset = chunk.get("startPage", 1) - 1
        for page in ds.get("pages", []):
            if offset:
                local_num = page.get("properties", {}).get("pageNumber", 0)
                page = {**page, "properties": {**page.get("properties", {}), "pageNumber": local_num + offset}}
            all_pages.append(page)
    return {"properties": merged_props, "pages": all_pages}


def _load_doc_structure(s3, bucket: str, key: str, cache_path: Path | None) -> dict:
    if cache_path and cache_path.exists():
        raw = json.loads(cache_path.read_bytes())
        return _merge_doc_structure(raw)

    obj = s3.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
    raw = json.loads(data)
    return _merge_doc_structure(raw)


def _build_chunks(
    doc_structure: dict,
    document_id: str,
    s3_bucket: str,
    s3_key: str,
    page_start: int,
    page_end: int,
    max_chunks: int = 0,
) -> list[dict]:
    from shared.apryse_parser import parse_pages
    from shared.section_chunker import build_section_chunks

    extraction = _load("extraction", "extraction")

    all_pages = parse_pages(doc_structure, page_start, page_end)
    if not all_pages:
        return []

    sections = build_section_chunks(all_pages, total_pages=len(doc_structure.get("pages", [])))
    if not sections:
        return []

    chunks: list[dict] = []
    global_obj_counter = 0

    for sec in sections:
        if max_chunks > 0 and len(chunks) >= max_chunks:
            break
        chunk_id = f"{document_id}_chunk_{len(chunks):04d}"

        objects = extraction._build_objects(
            chunk_id,
            sec.virtual_pages,
            global_offset=global_obj_counter,
        )
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

        chunks.append(
            {
                "source_type": "document",
                "document_id": document_id,
                "chunk_id": chunk_id,
                "s3_bucket": s3_bucket,
                "s3_key": s3_key,
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
            }
        )

    return chunks


def _send_chunk_messages(
    sqs,
    s3,
    queue_url: str,
    payload_bucket: str,
    payload_prefix: str,
    document_id: str,
    chunks: list[dict],
) -> tuple[int, int]:
    is_fifo = queue_url.endswith(".fifo")
    sent = 0
    offloaded = 0

    entries: list[dict] = []
    for i, chunk in enumerate(chunks):
        body_bytes = json.dumps(chunk).encode()

        if len(body_bytes) > 240_000:
            offloaded += 1
            payload_key = f"{payload_prefix}/{chunk['chunk_id']}.json"
            s3.put_object(
                Bucket=payload_bucket,
                Key=payload_key,
                Body=body_bytes,
                ContentType="application/json",
            )
            body = json.dumps(
                {
                    "document_id": document_id,
                    "chunk_id": chunk["chunk_id"],
                    "s3_payload": {
                        "bucket": payload_bucket,
                        "key": payload_key,
                    },
                }
            )
        else:
            body = body_bytes.decode()

        entry = {
            "Id": f"m{sent + i}"[-80:],
            "MessageBody": body,
        }
        if is_fifo:
            entry["MessageGroupId"] = document_id
            entry["MessageDeduplicationId"] = chunk["chunk_id"]

        entries.append(entry)

        if len(entries) == 10:
            resp = sqs.send_message_batch(QueueUrl=queue_url, Entries=entries)
            failed = resp.get("Failed", [])
            if failed:
                raise RuntimeError(f"SQS batch send failed: {failed}")
            sent += len(entries)
            entries = []

    if entries:
        resp = sqs.send_message_batch(QueueUrl=queue_url, Entries=entries)
        failed = resp.get("Failed", [])
        if failed:
            raise RuntimeError(f"SQS batch send failed: {failed}")
        sent += len(entries)

    return sent, offloaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Build section-aware chunks and enqueue to SQS")
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--s3-bucket", required=True)
    parser.add_argument("--s3-key", required=True, help="Source PDF key for metadata")
    parser.add_argument("--full-tables-key", required=True)
    parser.add_argument("--queue-url", default="")
    parser.add_argument("--region", default="eu-west-1")
    parser.add_argument("--cache-path", default="")
    parser.add_argument("--payload-bucket", default="")
    parser.add_argument("--payload-prefix", default="fanout-payloads")
    parser.add_argument("--page-start", type=int, default=1)
    parser.add_argument("--page-end", type=int, default=0, help="0 means full document")
    parser.add_argument("--page-batch-size", type=int, default=0,
                        help="stream dispatch in batches of N pages (0 = all at once)")
    parser.add_argument("--limit", type=int, default=0, help="optional chunk limit for smoke tests")
    parser.add_argument("--dry-run", action="store_true", help="build chunks only; do not send to SQS")
    args = parser.parse_args()

    if not args.dry_run and not args.queue_url:
        raise SystemExit("--queue-url is required unless --dry-run is set")

    session = boto3.Session(region_name=args.region)
    s3 = session.client("s3")
    sqs = session.client("sqs")

    cache_path = Path(args.cache_path) if args.cache_path else None
    doc_structure = _load_doc_structure(s3, args.s3_bucket, args.full_tables_key, cache_path)

    total_pages = len(doc_structure.get("pages", []))
    page_end = args.page_end if args.page_end > 0 else total_pages

    payload_bucket = args.payload_bucket or args.s3_bucket
    payload_prefix = f"{args.payload_prefix.rstrip('/')}/{args.document_id}"

    batch_size = args.page_batch_size
    if batch_size <= 0:
        # Original behaviour: build all chunks then send
        ranges = [(args.page_start, page_end)]
    else:
        ranges = [
            (s, min(s + batch_size - 1, page_end))
            for s in range(args.page_start, page_end + 1, batch_size)
        ]

    total_chunks_built = 0
    total_sent = 0
    total_offloaded = 0
    chunks_remaining = args.limit  # 0 means unlimited

    for batch_start, batch_end in ranges:
        max_for_batch = chunks_remaining if chunks_remaining > 0 else 0
        chunks = _build_chunks(
            doc_structure=doc_structure,
            document_id=args.document_id,
            s3_bucket=args.s3_bucket,
            s3_key=args.s3_key,
            page_start=batch_start,
            page_end=batch_end,
            max_chunks=max_for_batch,
        )
        total_chunks_built += len(chunks)

        if args.dry_run:
            continue

        sent, offloaded = _send_chunk_messages(
            sqs=sqs,
            s3=s3,
            queue_url=args.queue_url,
            payload_bucket=payload_bucket,
            payload_prefix=payload_prefix,
            document_id=args.document_id,
            chunks=chunks,
        )
        total_sent += sent
        total_offloaded += offloaded

        if batch_size > 0:
            print(
                f"batch pages={batch_start}-{batch_end} "
                f"chunks_built={len(chunks)} queued={sent}",
                flush=True,
            )

        if chunks_remaining > 0:
            chunks_remaining -= len(chunks)
            if chunks_remaining <= 0:
                break

    if args.dry_run:
        print(f"pages={total_pages} chunks_built={total_chunks_built} (dry-run)")
        return

    print(f"pages={total_pages} chunks_built={total_chunks_built} queued={total_sent} offloaded_to_s3={total_offloaded}")


if __name__ == "__main__":
    main()
