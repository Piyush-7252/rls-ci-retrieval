"""
dispatch_cis_to_sqs.py
======================
Reads a CI JSON file and pushes each CI as an SQS message to the
rls-ci-retrieval-ci-chunk-worker-queue (or any queue you point at).  Mirrors the pattern
of dispatch_chunks_to_sqs.py.

Usage
-----
    python tools/dispatch_cis_to_sqs.py \\
        --ci-file  localfiles/ci/ahmedCis.json \\
        --queue-url https://sqs.eu-west-1.amazonaws.com/064051750322/rls-ci-retrieval-ci-chunk-worker-queue \\
        --region eu-west-1

    # dry-run (print messages, do not send)
    python tools/dispatch_cis_to_sqs.py \\
        --ci-file localfiles/ci/ahmedCis.json \\
        --dry-run

    # only the first 5 CIs
    python tools/dispatch_cis_to_sqs.py \\
        --ci-file localfiles/ci/ahmedCis.json \\
        --max-cis 5 \\
        --queue-url ...

    # specific CI IDs
    python tools/dispatch_cis_to_sqs.py \\
        --ci-file localfiles/ci/ahmedCis.json \\
        --ci-ids 32 35 36 \\
        --queue-url ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parent.parent
_SQS_MAX_BATCH = 10          # SQS send_message_batch limit
_SQS_MAX_BODY  = 256 * 1024  # 256 KB SQS message size limit
TENANT = {"tenant_name": "RLS Test Script", "tenant_id": "1", "tenant_schema": "rls-test-script"}
PROJECT_ID="123"

def _send_batch(sqs, queue_url: str, entries: list[dict]) -> int:
    """Send a batch of SQS entries; return the count of successes."""
    resp   = sqs.send_message_batch(QueueUrl=queue_url, Entries=entries)
    failed = resp.get("Failed", [])
    if failed:
        for f in failed:
            print(f"  WARN: SQS send failed Id={f['Id']} Code={f['Code']} Msg={f['Message']}")
    return len(entries) - len(failed)


def dispatch(
    ci_file: Path,
    queue_url: str | None,
    region: str,
    max_cis: int | None,
    ci_ids: list[int] | None,
    dry_run: bool,
    tenant: dict,
    project_id: str
) -> None:
    with ci_file.open() as fh:
        raw_cis = json.load(fh)
    if isinstance(raw_cis, dict):
        raw_cis = list(raw_cis.values())
    if not isinstance(raw_cis, list):
        raw_cis = [raw_cis]

    # Filter
    if ci_ids:
        raw_cis = [c for c in raw_cis if c.get("id") in ci_ids]
    elif max_cis:
        raw_cis = raw_cis[:max_cis]

    print(f"\nCI dispatch")
    print(f"  File      : {ci_file.name}")
    print(f"  CIs       : {len(raw_cis)}")
    print(f"  Queue     : {queue_url or '(dry-run)'}")
    print(f"  Dry-run   : {dry_run}")
    print()

    sqs        = boto3.client("sqs", region_name=region) if not dry_run and queue_url else None
    batch:    list[dict] = []
    sent      = 0
    skipped   = 0
    t_start   = time.perf_counter()

    for raw_ci in raw_cis:
        ci_id    = raw_ci.get("id", "?")
        body_str = json.dumps({**raw_ci, "source_type": "ci",
                               "tenant_id": tenant["tenant_id"],
                               "tenant_name": tenant["tenant_name"],
                               "tenant_schema": tenant["tenant_schema"],
                               "project_id": project_id})

        if len(body_str.encode()) > _SQS_MAX_BODY:
            print(f"  WARN: CI {ci_id} body exceeds 256 KB — skipping")
            skipped += 1
            continue

        if dry_run:
            ci_text = (raw_ci.get("knownCI") or "")[:60].replace("\n", " ")
            print(f"  [DRY] ci_id={ci_id}  \"{ci_text}\"")
            sent += 1
            continue

        batch.append({
            "Id":          str(uuid.uuid4()),
            "MessageBody": body_str,
        })

        if len(batch) == _SQS_MAX_BATCH:
            sent += _send_batch(sqs, queue_url, batch)
            batch = []

    # Flush remaining
    if batch and not dry_run:
        sent += _send_batch(sqs, queue_url, batch)

    elapsed = time.perf_counter() - t_start
    print(f"\n  sent={sent}  skipped={skipped}  elapsed={elapsed:.1f}s")
    if sent and not dry_run:
        rate = sent / elapsed
        print(f"  rate ≈ {rate:.1f} CIs/s")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Dispatch CIs to SQS for rls-ci-worker")
    p.add_argument("--ci-file",   required=True, help="Path to CI JSON file")
    p.add_argument("--queue-url", default=None,  help="SQS queue URL")
    p.add_argument("--region",    default="eu-west-1")
    p.add_argument("--max-cis",   type=int, default=None, help="Limit number of CIs")
    p.add_argument("--ci-ids",    nargs="+", type=int, default=None,
                   help="Only dispatch these CI IDs")
    p.add_argument("--dry-run",   action="store_true",
                   help="Print messages without sending to SQS")
    args = p.parse_args()

    if not args.dry_run and not args.queue_url:
        print("ERROR: --queue-url is required (or use --dry-run)")
        sys.exit(1)

    ci_path = Path(args.ci_file)
    tenant = TENANT
    project_id = PROJECT_ID
    if not ci_path.is_absolute():
        ci_path = ROOT / ci_path
    if not ci_path.exists():
        print(f"ERROR: CI file not found: {ci_path}")
        sys.exit(1)

    dispatch(
        ci_file   = ci_path,
        queue_url = args.queue_url,
        region    = args.region,
        max_cis   = args.max_cis,
        ci_ids    = args.ci_ids,
        dry_run   = args.dry_run,
        tenant     = tenant,
        project_id = project_id
    )


if __name__ == "__main__":
    main()
