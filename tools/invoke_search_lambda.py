"""
invoke_search_lambda.py
=======================
Fire a search run via the rls-search-orchestrator Lambda, wait for it to
finish, download the full results JSON from S3, and save it locally.

The orchestrator writes full results to:
  s3://<RESULTS_BUCKET>/search-results/<timestamp>_<search_id>_<document_id>.json
and returns a lightweight summary + s3_key in its response.

Usage
-----
    # Search all CIs in ahmedCis.json against the default document
    python tools/invoke_search_lambda.py \\
        --ci-file localfiles/ci/ahmedCis.json

    # Specify document explicitly
    python tools/invoke_search_lambda.py \\
        --ci-file localfiles/ci/ahmedCis.json \\
        --document-id Combined_REDACTED_CSR-Full-co-jnj-64407564

    # Only first 5 CIs, skip rerank
    python tools/invoke_search_lambda.py \\
        --ci-file localfiles/ci/ahmedCis.json \\
        --max-cis 5 \\
        --skip-rerank

    # Point at a different orchestrator or region
    python tools/invoke_search_lambda.py \\
        --ci-file localfiles/ci/ahmedCis.json \\
        --function rls-search-orchestrator \\
        --region eu-west-1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "localfiles" / "search_results"

DEFAULT_FUNCTION   = "rls-ci-retrieval-search-orchestrator"
DEFAULT_REGION     = "eu-west-1"
DEFAULT_DOCUMENT   = "20260726062234599_4xs0l7p_10993_REDACTED_Protocol-Amendment-1-FD-64407564MMY3002-218114_1245209"
DEFAULT_RESULTS_BUCKET = "rls-file-bucket-eu"
DEFAULT_RESULTS_PREFIX = "search-results"


def load_cis(ci_file: Path, max_cis: int | None) -> list[dict]:
    with ci_file.open() as fh:
        data = json.load(fh)
    cis = data if isinstance(data, list) else list(data.values())
    if max_cis:
        cis = cis[:max_cis]
    return cis


def invoke_orchestrator(
    function: str,
    region: str,
    cis: list[dict],
    document_id: str,
    skip_rerank: bool,
    skip_verify: bool,
    batch_size: int,
    ci_workers: int = 5,
    max_workers: int | None = None,
) -> dict:
    """Synchronously invoke the orchestrator Lambda and return its JSON response."""
    client = boto3.client(
        "lambda", region_name=region,
        config=Config(read_timeout=900, connect_timeout=10, retries={"max_attempts": 0}),
    )

    # Pass raw CIs — orchestrator looks them up from ci-objects by id
    payload = {
        "document_id":  document_id,
        "cis":          cis,
        "skip_rerank":  skip_rerank,
        "skip_verify":  skip_verify,
        "batch_size":   batch_size,
        "ci_workers":   ci_workers,
        **(({"max_workers": max_workers}) if max_workers else {}),
    }

    log.info("Invoking %s with %d CIs (document=%s)...", function, len(cis), document_id)
    t0 = time.perf_counter()
    resp = client.invoke(
        FunctionName   = function,
        InvocationType = "RequestResponse",
        Payload        = json.dumps(payload).encode(),
    )
    elapsed = time.perf_counter() - t0

    raw    = resp["Payload"].read()
    result = json.loads(raw)

    if "errorMessage" in result:
        raise RuntimeError(
            f"Lambda error ({result.get('errorType')}): {result['errorMessage']}"
        )

    log.info("Lambda returned in %.1fs — n_cis=%s total_hits=%s",
             elapsed, result.get("n_cis"), result.get("total_hits"))
    return result


def download_from_s3(
    bucket: str,
    s3_key: str,
    region: str,
    out_path: Path,
) -> None:
    log.info("Downloading s3://%s/%s ...", bucket, s3_key)
    s3 = boto3.client("s3", region_name=region)
    s3.download_file(bucket, s3_key, str(out_path))
    log.info("Saved → %s  (%.1fMB)", out_path, out_path.stat().st_size / 1024 / 1024)


def print_summary(summary_response: dict, full_results: dict | None = None) -> None:
    print()
    print("=" * 60)
    print("  SEARCH COMPLETE")
    print("=" * 60)
    print(f"  search_id   : {summary_response.get('search_id')}")
    print(f"  document_id : {summary_response.get('document_id')}")
    print(f"  CIs searched: {summary_response.get('n_cis')}")
    print(f"  Total hits  : {summary_response.get('total_hits')}")
    print(f"  Wall time   : {summary_response.get('wall_time', 0):.1f}s")
    if "s3_key" in summary_response:
        print(f"  S3 key      : s3://{summary_response['s3_bucket']}/{summary_response['s3_key']}")
    if "errors" in summary_response:
        print(f"  Errors      : {summary_response['errors']}")
    if full_results:
        s = full_results.get("summary", {})
        if s:
            print()
            print(f"  DIRECT hits : {s.get('direct_hits', '?')}")
            print(f"  RELATED hits: {s.get('related_hits', '?')}")
            print(f"  Rejected    : {s.get('total_rejected', '?')}")
    print("=" * 60)
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Invoke rls-search-orchestrator Lambda")
    ap.add_argument("--ci-file",    type=Path, default=ROOT / "localfiles" / "ci" / "ahmedCis.json",
                    help="Path to CI JSON file")
    ap.add_argument("--document-id", default=DEFAULT_DOCUMENT)
    ap.add_argument("--function",   default=DEFAULT_FUNCTION)
    ap.add_argument("--region",     default=DEFAULT_REGION)
    ap.add_argument("--max-cis",    type=int, default=35)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--skip-rerank",  action="store_true")
    ap.add_argument("--skip-verify",  action="store_true")
    ap.add_argument("--ci-workers",   type=int, default=5,
                    help="Max CIs processed in parallel per worker Lambda (throttles OpenSearch)")
    ap.add_argument("--max-workers",  type=int, default=None,
                    help="Max concurrent worker Lambda invocations (default: all batches)")
    ap.add_argument("--results-bucket", default=DEFAULT_RESULTS_BUCKET)
    ap.add_argument("--results-prefix", default=DEFAULT_RESULTS_PREFIX)
    ap.add_argument("--out-dir",    type=Path, default=RESULTS_DIR)
    args = ap.parse_args()

    ci_file: Path = args.ci_file
    if not ci_file.is_absolute():
        ci_file = ROOT / ci_file
    if not ci_file.exists():
        log.error("CI file not found: %s", ci_file)
        sys.exit(1)

    cis = load_cis(ci_file, args.max_cis)
    log.info("Loaded %d CIs from %s", len(cis), ci_file.name)

    summary = invoke_orchestrator(
        function    = args.function,
        region      = args.region,
        cis         = cis,
        document_id = args.document_id,
        skip_rerank = args.skip_rerank,
        skip_verify = args.skip_verify,
        batch_size  = args.batch_size,
        ci_workers  = args.ci_workers,
        max_workers = args.max_workers,
    )

    # If orchestrator wrote to S3, download it
    full_results = None
    out_path = None
    if "s3_key" in summary:
        ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        doc_id   = args.document_id.replace("/", "_")[:60]
        ci_stem  = ci_file.stem
        out_name = f"{ts}_lambda_{ci_stem}_{doc_id}.json"
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.out_dir / out_name

        download_from_s3(
            bucket   = summary["s3_bucket"],
            s3_key   = summary["s3_key"],
            region   = args.region,
            out_path = out_path,
        )
        with out_path.open() as fh:
            full_results = json.load(fh)
    else:
        # Orchestrator returned inline (no RESULTS_BUCKET set) — save directly
        ts      = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        doc_id  = args.document_id.replace("/", "_")[:60]
        ci_stem = ci_file.stem
        out_name = f"{ts}_lambda_{ci_stem}_{doc_id}.json"
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.out_dir / out_name
        with out_path.open("w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        log.info("Results saved (inline) → %s", out_path)
        full_results = summary

    print_summary(summary, full_results)
    if out_path:
        print(f"Full results: {out_path}")


if __name__ == "__main__":
    main()
