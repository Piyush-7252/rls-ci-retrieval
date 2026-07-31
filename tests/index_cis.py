"""
CI Indexer — Enrich and store CIs to ci-objects
================================================
Runs each CI through the full enrichment pipeline
(normalize → NER → ontology → embedding → store)
and writes the result to the ci-objects OpenSearch index.

Usage
-----
    python tests/index_cis.py --ci-file localfiles/ci/ahmedCis.json
    python tests/index_cis.py --ci-file localfiles/ci/ahmedCis.json --max-cis 3
    python tests/index_cis.py --ci-file localfiles/ci/ahmedCis.json --force   # delete first
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OPENSEARCH_ENDPOINT = os.environ.get(
    "OPENSEARCH_ENDPOINT",
    "search-rls-dev-rhitzxwnctmuyq2l4kny5kwelu.eu-west-1.es.amazonaws.com",
)
AWS_REGION           = os.environ.get("AWS_REGION", "eu-west-1")
EMBEDDING_MODEL      = "amazon.titan-embed-text-v2:0"
DEFAULT_CI_FILE      = str(ROOT / "localfiles" / "ci" / "ahmedCis.json")
# Allow override via env so batch_run.py can use per-CI-file indexes.
OPENSEARCH_CI_INDEX  = os.environ.get("OPENSEARCH_CI_INDEX", "ci-objects")

os.environ.update(
    {
        "AWS_DEFAULT_REGION":  AWS_REGION,
        "AWS_REGION":          AWS_REGION,
        "OPENSEARCH_ENDPOINT": OPENSEARCH_ENDPOINT,
        "OPENSEARCH_CI_INDEX": OPENSEARCH_CI_INDEX,
        "NER_MODEL":           "gliner",
        "EMBEDDING_MODEL":     EMBEDDING_MODEL,
        # fan-out ARNs unused in local mode
        "NER_LAMBDA_ARN":      "",
        "ONTOLOGY_LAMBDA_ARN": "",
        "EMBEDDING_LAMBDA_ARN":"",
        "INDEX_LAMBDA_ARN":    "",
    }
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("index_cis")

# ── module loader ──────────────────────────────────────────────────────────────
_loaded: dict[str, types.ModuleType] = {}

def _load(rel_path: str, alias: str) -> types.ModuleType:
    if alias in _loaded:
        return _loaded[alias]
    lf_path = ROOT / "lambdas" / rel_path / "lambda_function.py"
    spec    = importlib.util.spec_from_file_location(alias, lf_path)
    mod     = importlib.util.module_from_spec(spec)
    lf_dir  = str(lf_path.parent)
    if lf_dir not in sys.path:
        sys.path.insert(0, lf_dir)
    spec.loader.exec_module(mod)
    _loaded[alias] = mod
    return mod

# ── OpenSearch client ──────────────────────────────────────────────────────────
_os_client = None

def _build_os_client():
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

# ── CI enrichment pipeline ─────────────────────────────────────────────────────

def enrich_ci(raw_ci: dict, ci_id: int) -> dict:
    normalize = _load("normalize", "normalize")
    ner       = _load("ner",       "ner")
    ontology  = _load("ontology",  "ontology")
    embedding = _load("embedding", "embedding")

    obj = {**raw_ci, "id": ci_id, "source_type": "ci"}
    _pt: dict[str, float] = {}
    _t_ci = time.perf_counter()

    print(f"    normalize …", end=" ", flush=True)
    _t0 = time.perf_counter(); obj = normalize._process_ci(obj); _pt["normalize"] = round(time.perf_counter() - _t0, 3)
    print(f"NER …", end=" ", flush=True)
    _t0 = time.perf_counter(); obj = ner._process_ci(obj); _pt["ner"] = round(time.perf_counter() - _t0, 3)
    print(f"ontology …", end=" ", flush=True)
    _t0 = time.perf_counter(); obj = ontology._process_ci(obj); _pt["ontology"] = round(time.perf_counter() - _t0, 3)
    print(f"embedding …", end=" ", flush=True)
    _t0 = time.perf_counter(); obj = embedding._process_ci(obj); _pt["embedding"] = round(time.perf_counter() - _t0, 3)
    _pt["total"] = round(time.perf_counter() - _t_ci, 3)
    print(f"done  ({_pt['total']:.1f}s)")
    obj["_pipeline_timings"] = _pt
    return obj


def store_ci(enriched_ci: dict) -> bool:
    idx = _load("index", "idx")                   # unified Index lambda (ci + document)
    # Inject the shared OS client
    if hasattr(idx, "_os_client"):
        idx._os_client = _build_os_client()
    try:
        idx._process_ci(enriched_ci)              # unified path: _process_ci()
        return True
    except Exception as exc:
        logger.error("[store_ci] failed ci_id=%s: %s", enriched_ci.get("id"), exc)
        return False


def delete_ci(ci_id: int) -> None:
    client = _build_os_client()
    try:
        resp   = client.delete(index=OPENSEARCH_CI_INDEX, id=str(ci_id), ignore=[404])
        result = resp.get("result", "not_found")
        print(f"    deleted existing entry (result={result})")
    except Exception as exc:
        logger.warning("[delete_ci] ci_id=%s: %s", ci_id, exc)

# ── main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrich and index CIs to ci-objects")
    p.add_argument("--ci-file",  default=DEFAULT_CI_FILE,
                   help=f"Path to CI JSON file (default: {DEFAULT_CI_FILE})")
    p.add_argument("--max-cis",  type=int, default=None,
                   help="Maximum number of CIs to process (default: all)")
    p.add_argument("--ci-ids",   nargs="+", type=int, default=None,
                   help="Process only these CI IDs  e.g. --ci-ids 32 35 36")
    p.add_argument("--force",    action="store_true",
                   help="Delete existing ci-objects entries before re-indexing")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ci_path = Path(args.ci_file)
    if not ci_path.exists():
        print(f"ERROR: CI file not found: {ci_path}")
        sys.exit(1)

    with ci_path.open() as fh:
        raw_cis = json.load(fh)
    if isinstance(raw_cis, dict):
        raw_cis = list(raw_cis.values())
    if not isinstance(raw_cis, list):
        raw_cis = [raw_cis]

    # Filter by explicit IDs if requested
    if args.ci_ids:
        raw_cis = [c for c in raw_cis if c.get("id") in args.ci_ids]
    elif args.max_cis:
        raw_cis = raw_cis[: args.max_cis]

    print(f"\nCI Indexer")
    print(f"  File   : {ci_path.name}")
    print(f"  CIs    : {len(raw_cis)}")
    print(f"  Index  : {OPENSEARCH_CI_INDEX} @ {OPENSEARCH_ENDPOINT[:40]}…")
    print(f"  Force  : {args.force}")

    print("\nConnecting to OpenSearch …")
    client = _build_os_client()
    info   = client.info()
    print(f"  Cluster: {info.get('cluster_name')}  version: {info['version']['number']}")

    stored = 0
    failed = 0
    all_pt: list[dict] = []
    _t_wall_start = time.perf_counter()

    for raw_ci in raw_cis:
        ci_id   = raw_ci.get("id", "?")
        ci_text = (raw_ci.get("knownCI") or "")[:80].replace("\n", " ")
        print(f"\n  CI {ci_id}: \"{ci_text}\"")

        if args.force:
            delete_ci(ci_id)

        try:
            enriched = enrich_ci(raw_ci, ci_id)
            ok       = store_ci(enriched)
            if ok:
                entities  = len(enriched.get("ner", {}).get("entities", []))
                patterns  = len(enriched.get("ontology", {}).get("regex_patterns", []))
                dim       = enriched.get("embedding", {}).get("dimensions", 0)
                stmt_type = enriched.get("statement_type", "—")
                ctx       = enriched.get("study_context", "—")
                n_rel     = len(enriched.get("clinical_relations", []))
                pt        = enriched.get("_pipeline_timings", {})
                print(f"    ✓ stored  entities={entities}  patterns={patterns}  "
                      f"embedding_dim={dim}  stmt={stmt_type}  ctx={ctx}  relations={n_rel}")
                all_pt.append(pt)
                stored += 1
            else:
                print(f"    ✗ store failed")
                failed += 1
        except Exception as exc:
            print(f"    ✗ ERROR: {exc}")
            logger.exception("Enrichment failed for CI %s", ci_id)
            failed += 1

    print(f"\n{'═'*50}")
    print(f"  Done — stored={stored}  failed={failed}")

    # ── Timing summary ────────────────────────────────────────────────
    if all_pt:
        _wall_time = time.perf_counter() - _t_wall_start
        n_ci       = len(all_pt)
        stages     = ["normalize", "ner", "ontology", "embedding", "total"]

        def _tot(k: str) -> float:
            return sum(p.get(k, 0.0) for p in all_pt)

        print(f"\n{'═'*58}")
        print(f"  TIMING BREAKDOWN  ({n_ci} CI{'s' if n_ci != 1 else ''},  "
              f"wall-clock {_wall_time:.1f}s)")
        print(f"{'═'*58}")
        print(f"  {'Stage':<14}  {'Total':>9}  {'Avg/CI':>9}  {'Share':>7}")
        print(f"  {'─'*14}  {'─'*9}  {'─'*9}  {'─'*7}")
        denom = _tot("total") or 1.0
        for stage in stages[:-1]:
            tot = _tot(stage)
            print(f"  {stage:<14}  {tot:>8.2f}s  {tot/n_ci:>8.2f}s  "
                  f"{100*tot/denom:>6.1f}%")
        print(f"  {'─'*14}  {'─'*9}  {'─'*9}  {'─'*7}")
        tot_all = _tot("total")
        print(f"  {'TOTAL':<14}  {tot_all:>8.2f}s  {tot_all/n_ci:>8.2f}s  100.0%")
        print(f"{'═'*58}")
    print()


if __name__ == "__main__":
    main()
