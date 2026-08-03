"""
Batch dispatch all CI files under localfiles/ci/ to rls-ci-worker-queue.

Usage
-----
    python tools/batch_dispatch_cis.py
    python tools/batch_dispatch_cis.py --dry-run
    python tools/batch_dispatch_cis.py --files ahmedCis.json christineCIs.json
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

ROOT       = Path(__file__).resolve().parent.parent
CI_DIR     = ROOT / "localfiles" / "ci"
QUEUE_URL  = "https://sqs.eu-west-1.amazonaws.com/064051750322/rls-ci-worker-queue"
REGION     = "eu-west-1"

# All CI files in dispatch order (smallest → largest to warm up queue fast)
CI_FILES = [
    "rlsTestScriptTenantCis.json",    #  5
    "random.json",                     # 11
    "ahmedFalseNumaricCis.json",       # 13
    "Anonymize_fixture.json",          # 20
    "ProtocoI_301.json",               # 20
    "RxPharmaProtocolv1.json",         # 20
    "numericCis.json",                 # 22
    "ahmedCis.json",                   # 34
    "christineCIs.json",               # 61
]


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def dispatch_file(ci_file: Path, dry_run: bool, index: int, total: int) -> bool:
    print(f"\n{'='*70}")
    print(f"[{index}/{total}] {now_utc()}")
    print(f"  file : {ci_file.name}")
    print(f"{'='*70}", flush=True)

    cmd = [
        sys.executable, "tools/dispatch_cis_to_sqs.py",
        "--ci-file", str(ci_file),
        "--region",  REGION,
    ]
    if dry_run:
        cmd.append("--dry-run")
    else:
        cmd += ["--queue-url", QUEUE_URL]

    t0     = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"  ✓ done in {elapsed:.1f}s", flush=True)
        return True
    else:
        print(f"  ✗ FAILED (exit {result.returncode}) after {elapsed:.1f}s", flush=True)
        return False


def main() -> None:
    p = argparse.ArgumentParser(description="Batch dispatch all CI files to rls-ci-worker-queue")
    p.add_argument("--dry-run", action="store_true", help="Print without sending to SQS")
    p.add_argument("--files",   nargs="+", default=None,
                   help="Only dispatch these filenames (e.g. ahmedCis.json christineCIs.json)")
    args = p.parse_args()

    files_to_run = [CI_DIR / f for f in (args.files or CI_FILES)]

    # Validate all files exist before starting
    missing = [f for f in files_to_run if not f.exists()]
    if missing:
        for f in missing:
            print(f"ERROR: not found: {f}")
        sys.exit(1)

    total  = len(files_to_run)
    failed = []

    print(f"=== BATCH CI DISPATCH — {now_utc()} ===")
    print(f"  {total} files  |  queue: {QUEUE_URL if not args.dry_run else '(dry-run)'}")

    for i, ci_file in enumerate(files_to_run, start=1):
        ok = dispatch_file(ci_file, args.dry_run, i, total)
        if not ok:
            failed.append(ci_file.name)

    print(f"\n{'='*70}")
    print(f"=== BATCH DONE — {now_utc()} ===")
    print(f"  {total - len(failed)}/{total} succeeded")
    if failed:
        print(f"  FAILED ({len(failed)}):")
        for f in failed:
            print(f"    {f}")


if __name__ == "__main__":
    main()
