"""
batch_run.py
------------
Reads full_tables.json and runs the full pipeline for each document:

  Step 1 - Index CIs        (skipped if ci_index already marked indexed=true)
  Step 2 - search_test.py   (per document)
  Step 3 - export_results_csv.py
  Step 4 - compare_results.py  (old ECS vs new, skipped if no old-ecs file)

OPENSEARCH_CI_INDEX is passed as an env var to child processes so each
CI file gets its own isolated OpenSearch index.

Usage:
  python tools/batch_run.py
  python tools/batch_run.py --config localfiles/full_tables.json
  python tools/batch_run.py --run-id run18-10993     # single run
  python tools/batch_run.py --index-only             # only index CIs
  python tools/batch_run.py --skip-index             # skip indexing step
  python tools/batch_run.py --skip-export            # skip CSV export
  python tools/batch_run.py --skip-compare           # skip comparison report
  python tools/batch_run.py --force-reindex          # re-index even if indexed=true
  python tools/batch_run.py --dry-run                # print commands, do not execute
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run(cmd: list[str], extra_env: dict[str, str] | None = None,
         dry_run: bool = False) -> int:
    env = {**os.environ, **(extra_env or {})}
    print(f"\n  $ {' '.join(cmd)}")
    if dry_run:
        print("  [DRY RUN - not executed]")
        return 0
    result = subprocess.run(cmd, env=env, cwd=str(ROOT))
    return result.returncode


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _save_config(path: Path, cfg: dict) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_index(ci_index_name: str, idx_cfg: dict,
               force_reindex: bool, dry_run: bool) -> bool:
    """Index a CI file into its own named OpenSearch index. Returns True on success."""
    ci_file = str(ROOT / idx_cfg["ci_file"])
    max_cis = idx_cfg.get("max_cis")

    cmd = [PYTHON, "tests/index_cis.py", "--ci-file", ci_file]
    if max_cis:
        cmd += ["--max-cis", str(max_cis)]
    if force_reindex:
        cmd += ["--force"]

    rc = _run(cmd, extra_env={"OPENSEARCH_CI_INDEX": ci_index_name}, dry_run=dry_run)
    return rc == 0


def step_index_document(run: dict, dry_run: bool) -> bool:
    """Run s3_pipeline_test.py to index one document into OpenSearch. Returns True on success."""
    doc_id    = run["document_id"]
    s3_folder = run.get("s3_folder", "")
    workers   = run.get("workers", 4)

    if not s3_folder:
        print(f"  [SKIP doc-index] no s3_folder defined for {doc_id}")
        return False

    cmd = [
        PYTHON, "-u", "tests/s3_pipeline_test.py",
        "--document-id", doc_id,
        "--s3-folder",   s3_folder,
        "--pages",       "1-9999",
        "--workers",     str(workers),
        "--force",
    ]
    rc = _run(cmd, extra_env={
        "PIPELINE_DOCUMENT_ID": doc_id,
        "PIPELINE_S3_FOLDER":   s3_folder,
    }, dry_run=dry_run)
    return rc == 0


def step_search(run: dict, idx_cfg: dict, dry_run: bool) -> bool:
    """Run search_test.py for one document. Returns True on success."""
    ci_index_name = run["ci_index"]
    ci_file       = str(ROOT / idx_cfg["ci_file"])
    max_cis       = idx_cfg.get("max_cis", 34)
    output_json   = str(ROOT / run["output_json"])

    # Ensure output directory exists
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON, "tests/search_test.py",
        "--ci-file",     ci_file,
        "--max-cis",     str(max_cis),
        "--document-id", run["document_id"],
        "--workers",     str(run.get("workers", 10)),
        "--output",      output_json,
        "-u",
    ]
    # Remove -u flag if it causes issues (it's a python flag, not script flag)
    cmd = [PYTHON, "-u", "tests/search_test.py",
           "--ci-file",     ci_file,
           "--max-cis",     str(max_cis),
           "--document-id", run["document_id"],
           "--workers",     str(run.get("workers", 10)),
           "--output",      output_json]

    rc = _run(cmd, extra_env={"OPENSEARCH_CI_INDEX": ci_index_name}, dry_run=dry_run)
    return rc == 0


def step_export(run: dict, dry_run: bool) -> bool:
    """Run export_results_csv.py on the output JSON. Returns True on success."""
    output_json = str(ROOT / run["output_json"])
    output_csv  = str(ROOT / run["output_csv"])

    if not dry_run and not Path(output_json).exists():
        print(f"  [SKIP export] JSON not found: {output_json}")
        return False

    cmd = [PYTHON, "tools/export_results_csv.py",
           output_json, "--out", output_csv]

    rc = _run(cmd, dry_run=dry_run)
    return rc == 0


def _find_old_ecs(doc_num: str) -> str | None:
    """Locate localfiles/old-ecs/{doc_num}_.../current.json if it exists."""
    old_ecs = ROOT / "localfiles" / "old-ecs"
    if not old_ecs.exists():
        return None
    for folder in old_ecs.iterdir():
        if folder.name.startswith(doc_num + "_") or folder.name.startswith(doc_num + "-"):
            candidate = folder / "current.json"
            if candidate.exists():
                return str(candidate)
    return None


def step_compare(run: dict, dry_run: bool) -> bool:
    """Run compare_results.py (old ECS vs new). Returns True on success or skip."""
    doc_num  = run["document_id"].split("-")[0]
    ci_stem  = run["ci_index"].replace("ci-objects-", "")
    old_path = _find_old_ecs(doc_num)

    if not old_path:
        print(f"  [SKIP compare] no old-ecs/current.json for doc {doc_num}")
        return True  # not an error — old data simply doesn't exist

    new_json    = str(ROOT / run["output_json"])
    compare_csv = str(ROOT / "localfiles" / "comparison" / f"{doc_num}-{ci_stem}.csv")

    if not dry_run and not Path(new_json).exists():
        print(f"  [SKIP compare] new JSON not found: {new_json}")
        return False

    cmd = [PYTHON, "tools/compare_results.py",
           "--old", old_path,
           "--new", new_json,
           "--out", compare_csv]

    rc = _run(cmd, dry_run=dry_run)
    if rc == 0:
        print(f"  Comparison -> {compare_csv}")
    return rc == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Batch run search pipeline from full_tables.json")
    p.add_argument("--config",        default="localfiles/full_tables.json",
                   help="Path to config JSON (default: localfiles/full_tables.json)")
    p.add_argument("--run-id",        default=None,
                   help="Run only this specific run id (e.g. run18-10993)")
    p.add_argument("--index-only",    action="store_true",
                   help="Only run the CI indexing step, skip search and export")
    p.add_argument("--skip-index",    action="store_true",
                   help="Skip CI indexing (use already-indexed CIs)")
    p.add_argument("--skip-export",   action="store_true",
                   help="Skip CSV export step")
    p.add_argument("--skip-compare",  action="store_true",
                   help="Skip comparison report step")
    p.add_argument("--force-reindex", action="store_true",
                   help="Re-index CIs even if indexed=true in config")
    p.add_argument("--reindex-docs",  action="store_true",
                   help="Also re-index documents into document-chunks/semantic-objects (s3_pipeline_test.py --force)")
    p.add_argument("--skip-doc-index", action="store_true",
                   help="Skip document indexing even when --reindex-docs is set")
    p.add_argument("--dry-run",       action="store_true",
                   help="Print commands without executing them")
    args = p.parse_args()

    config_path = ROOT / args.config
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        sys.exit(1)

    cfg = _load_config(config_path)
    ci_indexes = cfg.get("ci_indexes", {})
    runs       = cfg.get("runs", [])

    # Filter runs
    if args.run_id:
        runs = [r for r in runs if r["id"] == args.run_id]
        if not runs:
            print(f"ERROR: run_id '{args.run_id}' not found in config")
            sys.exit(1)
    else:
        runs = [r for r in runs if r.get("enabled", True)]

    print(f"\nBatch Run")
    print(f"  Config : {config_path}")
    print(f"  Runs   : {len(runs)}")
    steps = []
    if not args.skip_index:              steps.append("ci-index")
    if getattr(args, 'reindex_docs', False) and not getattr(args, 'skip_doc_index', False):
        steps.append("doc-index")
    if not args.index_only:              steps += ["search", "export", "compare"]
    if args.skip_export:                 steps = [s for s in steps if s != "export"]
    if getattr(args, 'skip_compare', False): steps = [s for s in steps if s != "compare"]
    print(f"  Steps  : {' -> '.join(steps)}")

    # ---- Step 1: Index CIs (once per unique ci_index) --------------------
    if not args.skip_index:
        needed_indexes = {r["ci_index"] for r in runs}
        for ci_index_name in sorted(needed_indexes):
            idx_cfg = ci_indexes.get(ci_index_name)
            if not idx_cfg:
                print(f"\nERROR: ci_index '{ci_index_name}' not defined in ci_indexes section")
                sys.exit(1)

            already_indexed = idx_cfg.get("indexed", False) and not args.force_reindex
            if already_indexed:
                print(f"\n[INDEX] {ci_index_name} — already indexed, skipping."
                      f"  (use --force-reindex to redo)")
                continue

            print(f"\n[INDEX] {ci_index_name}  ({idx_cfg['ci_file']})")
            ok = step_index(ci_index_name, idx_cfg, args.force_reindex, args.dry_run)
            if ok:
                cfg["ci_indexes"][ci_index_name]["indexed"]    = True
                cfg["ci_indexes"][ci_index_name]["indexed_at"] = _now()
                _save_config(config_path, cfg)
                print(f"  -> marked indexed=true in config")
            else:
                print(f"  ERROR: indexing failed for {ci_index_name}")
                sys.exit(1)

    if args.index_only and not getattr(args, 'reindex_docs', False):
        print("\nDone (index-only mode).")
        return

    # ---- Step 1b: Index Documents (once per unique document_id) ----------
    if getattr(args, 'reindex_docs', False) and not getattr(args, 'skip_doc_index', False):
        seen_docs: set[str] = set()
        for run in runs:
            doc_key = (run["document_id"], run.get("s3_folder", ""))
            if doc_key in seen_docs:
                continue
            seen_docs.add(doc_key)
            print(f"\n[DOC-INDEX] {run['document_id']}  ({run.get('s3_folder', '?')})")
            ok = step_index_document(run, args.dry_run)
            if not ok:
                print(f"  ERROR: document indexing failed for {run['document_id']}")
                sys.exit(1)

    if args.index_only:
        print("\nDone (index-only mode).")
        return

    # ---- Step 2 + 3: Search + Export per run  ----------------------------
    summary: list[dict] = []
    for run in runs:
        run_id    = run["id"]
        doc_id    = run["document_id"]
        doc_name  = run["document_name"]
        ci_index  = run["ci_index"]
        idx_cfg   = ci_indexes.get(ci_index, {})

        print(f"\n{'='*60}")
        print(f"[RUN] {run_id}  |  {doc_id}")
        print(f"      {doc_name}")

        # Search
        print(f"\n  [SEARCH] document={doc_id}  ci_index={ci_index}")
        ok_search = step_search(run, idx_cfg, args.dry_run)
        if not ok_search:
            print(f"  ERROR: search failed for {run_id}")
            summary.append({"id": run_id, "status": "SEARCH_FAILED"})
            run["status"] = "search_failed"
            run["completed_at"] = _now()
            _save_config(config_path, cfg)
            continue

        # Export
        if not args.skip_export:
            print(f"\n  [EXPORT] -> {run['output_csv']}")
            ok_export = step_export(run, args.dry_run)
            if not ok_export:
                print(f"  WARNING: export failed for {run_id}")
                summary.append({"id": run_id, "status": "EXPORT_FAILED"})
                run["status"] = "export_failed"
                run["completed_at"] = _now()
                _save_config(config_path, cfg)
                continue

        # Compare
        if not getattr(args, 'skip_compare', False):
            doc_num = doc_id.split("-")[0]
            ci_stem = ci_index.replace("ci-objects-", "")
            print(f"\n  [COMPARE] old-ecs/{doc_num} vs new  -> localfiles/comparison/{doc_num}-{ci_stem}.csv")
            step_compare(run, args.dry_run)  # never fatal — old data may not exist

        summary.append({"id": run_id, "status": "OK"})
        run["status"] = "done"
        run["completed_at"] = _now()
        _save_config(config_path, cfg)

    # ---- Summary ---------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Summary  ({len(summary)} runs)")
    for s in summary:
        icon = "OK" if s["status"].startswith("OK") else "FAIL"
        print(f"  [{icon}] {s['id']:40s}  {s['status']}")
    ok_count   = sum(1 for s in summary if s["status"].startswith("OK"))
    fail_count = len(summary) - ok_count
    print(f"\n  {ok_count} succeeded  {fail_count} failed")


if __name__ == "__main__":
    main()
