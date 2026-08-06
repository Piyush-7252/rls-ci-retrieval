"""
compare_results.py
------------------
Compare old ECS result JSON with new search_test.py result JSON.

Produces a CSV with one row per (CI, page) showing old vs new side-by-side.
Rows present only in old get status=OLD ONLY, only in new=NEW ONLY, both=BOTH.

OLD format : localfiles/old-ecs/{doc_folder}/current.json
             combined_semantic_results[]
             fields: ci_id, ci_reference, type, strategy, text, confidence, page_num

NEW format : localfiles/search_results/run18/{doc_id}.json  (search_test.py output)
             results[].final_hits + rejected_hits + skipped_hits
             ALL hit types are included (not just final)

Verdict mapping (new -> compare column):
  YES / DIRECT / SUPPORTING / RELATED* -> TRUE_POSITIVE
  MAYBE                                 -> UNCERTAIN
  NO                                    -> FALSE_POSITIVE
  SKIP                                  -> SKIPPED

Usage (single pair):
  python tools/compare_results.py --old localfiles/old-ecs/10990_.../current.json \
                                   --new localfiles/search_results/run18/10990.json

Usage (batch via full_tables.json):
  python tools/compare_results.py --batch --config localfiles/full_tables.json \
                                   --ci-index ci-objects-ahmed
  python tools/compare_results.py --batch   # uses defaults

Output CSVs go to localfiles/comparison/ (or --out-dir).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT      = Path(__file__).resolve().parent.parent
OLD_ECS   = ROOT / "localfiles" / "old-ecs"
OUT_DIR   = ROOT / "localfiles" / "comparison"
CONFIG    = ROOT / "localfiles" / "full_tables.json"


# ---------------------------------------------------------------------------
# Verdict mapping
# ---------------------------------------------------------------------------

_VERDICT_MAP = {
    # LLM verifier verdicts
    "YES":       "TRUE_POSITIVE",
    "MAYBE":     "UNCERTAIN",
    "NO":        "FALSE_POSITIVE",
    "SKIP":      "SKIPPED",
    # evidence_type values that can appear in verdict field
    "DIRECT":          "TRUE_POSITIVE",
    "SUPPORTING":      "TRUE_POSITIVE",
    "RELATED":         "TRUE_POSITIVE",
    "RELATED_PROTOCOL":"TRUE_POSITIVE",
    "INDIRECT":        "TRUE_POSITIVE",
}


# ---------------------------------------------------------------------------
# Old loader
# ---------------------------------------------------------------------------

def _load_old(path: Path) -> list[dict]:
    """Load combined_semantic_results[] from current.json."""
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    rows = []
    for m in data.get("combined_semantic_results", []):
        rows.append({
            "ci_id":        str(m.get("ci_id", "")),
            "ci_reference": (m.get("ci_reference") or "").replace("\n", " "),
            "ci_type":      m.get("type", ""),
            "page_num":     str(m.get("page_num", "")),
            "confidence":   m.get("confidence", ""),
            "strategy":     m.get("strategy", ""),
            "text":         (m.get("text") or "")[:300].replace("\n", " "),
        })
    return rows


# ---------------------------------------------------------------------------
# New loader  (search_test.py output)
# ---------------------------------------------------------------------------

def _hit_page(hit: dict) -> str:
    """Best page number from a hit."""
    p = hit.get("match_page") or hit.get("page_start")
    return str(p) if p is not None else ""


def _hit_strategy(hit: dict) -> str:
    """Summarise retrieval sources."""
    sources = hit.get("sources") or []
    et = hit.get("evidence_type", "")
    parts = list(sources) + ([et] if et else [])
    return " | ".join(parts) if parts else hit.get("retrieval_object_type", "")


def _hit_text(hit: dict) -> str:
    """Best text snippet."""
    return (hit.get("text") or "").replace("\n", " ")


def _load_new(path: Path) -> list[dict]:
    """Load all hits (final + rejected + skipped) from search_test.py output."""
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    rows = []
    for result in data.get("results", []):
        ci      = result.get("ci") or {}
        ci_id   = str(ci.get("id", ""))
        ci_text = (ci.get("text") or "").replace("\n", " ").strip()

        all_hits = (
            [(h, "final")    for h in (result.get("final_hits")    or [])] +
            [(h, "rejected") for h in (result.get("rejected_hits") or [])] +
            [(h, "skipped")  for h in (result.get("skipped_hits")  or [])]
        )

        for hit, hit_cat in all_hits:
            verdict_raw = hit.get("verdict", "")
            verdict_mapped = _VERDICT_MAP.get(verdict_raw, verdict_raw)

            reason = (hit.get("evidence_reason") or
                      hit.get("verifier_reason") or
                      hit.get("reason") or "")

            rows.append({
                "ci_id":        ci_id,
                "ci_reference": ci_text,
                "ci_type":      (ci.get("ci_type") or ""),
                "page_num":     _hit_page(hit),
                "confidence":   hit.get("confidence", ""),
                "strategy":     _hit_strategy(hit),
                "text":         _hit_text(hit),
                "verdict":          verdict_mapped,
                "reason":           reason[:300].replace("\n", " "),
                "hit_category":     hit_cat,          # final/rejected/skipped
                "evidence_type":    hit.get("evidence_type", ""),
                "match_span":       (hit.get("match_span") or "").replace("\n", " "),
                "context_sentence": (hit.get("context_sentence") or "").replace("\n", " "),
            })
    return rows


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def _ci_text_key(r: dict) -> str:
    """Normalised CI text — used as the stable cross-system match key.

    Old ECS and new search can have different ci_id values for the same CI,
    so we match on lowercased, whitespace-collapsed CI reference text instead.
    """
    return " ".join((r.get("ci_reference") or "").lower().split())


def _group(rows: list[dict]) -> dict:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (_ci_text_key(r), r["page_num"])
        grouped[key].append(r)
    for key in grouped:
        grouped[key].sort(
            key=lambda x: float(x["confidence"]) if x["confidence"] not in ("", None) else 0,
            reverse=True,
        )
    return dict(grouped)


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

def _compare(old_rows: list[dict], new_rows: list[dict]) -> list[dict]:
    # Split new hits: only final hits drive BOTH / NEW ONLY status.
    # rejected + skipped are used only to enrich OLD ONLY rows with the
    # Claude verdict/reason that explains why the hit was not promoted.
    final_rows     = [r for r in new_rows if r.get("hit_category") == "final"]
    non_final_rows = [r for r in new_rows if r.get("hit_category") != "final"]

    old_grp       = _group(old_rows)
    new_final_grp = _group(final_rows)
    new_dropped_grp = _group(non_final_rows)   # rejected + skipped

    # Universe: old hits + new final hits only
    all_keys = sorted(
        set(old_grp) | set(new_final_grp),
        key=lambda k: (k[0], str(k[1]).zfill(6)),
    )

    out = []
    for key in all_keys:
        ci_text_key, page_num = key
        old_matches   = old_grp.get(key, [])
        final_matches = new_final_grp.get(key, [])

        if old_matches and final_matches:
            status = "BOTH"
        elif old_matches:
            status = "OLD ONLY"
        else:
            status = "NEW ONLY"

        # For OLD ONLY rows: try to find a dropped (rejected/skipped) hit to
        # show the Claude reason why this CI×page didn't make it to final.
        dropped_matches = new_dropped_grp.get(key, []) if status == "OLD ONLY" else []

        new_matches = final_matches if final_matches else dropped_matches
        max_len = max(len(old_matches), len(new_matches), 1)

        for i in range(max_len):
            o = old_matches[i] if i < len(old_matches) else {}
            n = new_matches[i] if i < len(new_matches) else {}

            # Per-row status: for BOTH groups, extra old rows that have no
            # corresponding new final row should be OLD ONLY (not BOTH with
            # empty new columns).  OLD ONLY / NEW ONLY groups keep their
            # group status regardless — n may be a dropped hit used only for
            # enrichment display, not an independent match.
            if status == "BOTH":
                if o and n:
                    row_status = "BOTH"
                elif o:
                    row_status = "OLD ONLY"
                else:
                    row_status = "NEW ONLY"
            else:
                row_status = status

            ci_ref  = (o or n).get("ci_reference", "")
            ci_type = (o or n).get("ci_type", "")

            out.append({
                "ci_id":              o.get("ci_id") or n.get("ci_id", ""),

                "ci_reference":       ci_ref,
                "ci_type":            ci_type,
                "page_num":           page_num,
                "status":             row_status,
                # Old
                "old_confidence":     o.get("confidence", ""),
                "old_strategy":       o.get("strategy", ""),
                "old_text":           o.get("text", ""),
                # New — populated for final hits (BOTH/NEW ONLY) and for
                # dropped hits that match an OLD ONLY row
                "new_confidence":     n.get("confidence", ""),
                "new_strategy":       n.get("strategy", ""),
                "new_hit_category":   n.get("hit_category", ""),   # final/rejected/skipped
                "new_evidence_type":  n.get("evidence_type", ""),
                "new_claude_verdict": n.get("verdict", ""),
                "new_claude_reason":  n.get("reason", ""),
                "new_text":           n.get("text", ""),
                "new_match_span":     n.get("match_span", ""),
                "context_sentence":   n.get("context_sentence", ""),
            })
    return out


# ---------------------------------------------------------------------------
# CSV fieldnames
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "ci_id", "ci_reference", "ci_type", "page_num", "status",
    "old_confidence", "old_strategy", "old_text",
    "new_confidence", "new_strategy",
    "new_hit_category", "new_evidence_type",
    "new_claude_verdict", "new_claude_reason",
    "new_text", "new_match_span", "context_sentence",
]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _summary(old_rows, new_rows, output_rows, doc_label=""):
    final_rows = [r for r in new_rows if r.get("hit_category") == "final"]

    old_grp       = _group(old_rows)
    new_final_grp = _group(final_rows)
    old_keys   = set(old_grp)
    final_keys = set(new_final_grp)

    only_old = old_keys - final_keys
    only_new = final_keys - old_keys
    both     = old_keys & final_keys

    tp = sum(1 for r in final_rows if r.get("verdict") == "TRUE_POSITIVE")
    fp = sum(1 for r in final_rows if r.get("verdict") == "FALSE_POSITIVE")
    uc = sum(1 for r in final_rows if r.get("verdict") == "UNCERTAIN")

    label = f"  [{doc_label}]  " if doc_label else ""
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY{label}")
    print(f"{'='*60}")
    print(f"  Old matches        : {len(old_rows):>5}  ({len(old_keys)} unique CI×page)")
    print(f"  New final hits     : {len(final_rows):>5}  ({len(final_keys)} unique CI×page)")
    print(f"  In BOTH            : {len(both):>5}")
    print(f"  OLD ONLY (missed)  : {len(only_old):>5}")
    print(f"  NEW ONLY (added)   : {len(only_new):>5}")
    print(f"  New final verdicts : TP={tp}  FP={fp}  UC={uc}")

    by_cat = defaultdict(int)
    for r in new_rows:
        by_cat[r.get("hit_category", "?")] += 1
    cats = "  ".join(f"{k}={v}" for k, v in sorted(by_cat.items()))
    print(f"  All new hit types  : {cats}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Single-pair run
# ---------------------------------------------------------------------------

def _run_pair(old_path: Path, new_path: Path, out_path: Path,
              doc_label: str = "") -> None:
    print(f"\nOld  : {old_path}")
    old_rows = _load_old(old_path)
    print(f"       {len(old_rows)} matches")

    print(f"New  : {new_path}")
    new_rows = _load_new(new_path)
    print(f"       {len(new_rows)} hits  (final + rejected + skipped)")

    output_rows = _compare(old_rows, new_rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES,
                                extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"CSV  : {out_path}  ({len(output_rows)} rows)")
    _summary(old_rows, new_rows, output_rows, doc_label)


# ---------------------------------------------------------------------------
# Batch run via full_tables.json
# ---------------------------------------------------------------------------

def _find_old_ecs(doc_num: str) -> Path | None:
    """Find the current.json for a document number under old-ecs/."""
    if not OLD_ECS.exists():
        return None
    for folder in OLD_ECS.iterdir():
        if folder.name.startswith(doc_num + "_") or folder.name.startswith(doc_num + "-"):
            candidate = folder / "current.json"
            if candidate.exists():
                return candidate
    return None


def _run_batch(config_path: Path, ci_index_filter: str | None,
               out_dir: Path) -> None:
    cfg  = json.loads(config_path.read_text(encoding="utf-8"))
    runs = [r for r in cfg.get("runs", []) if r.get("enabled", True)]

    if ci_index_filter:
        runs = [r for r in runs if r.get("ci_index") == ci_index_filter]

    # Deduplicate by (document_id, ci_index) — one compare per pair
    seen: set[tuple] = set()
    unique_runs = []
    for r in runs:
        key = (r["document_id"], r["ci_index"])
        if key not in seen:
            seen.add(key)
            unique_runs.append(r)

    print(f"Batch compare: {len(unique_runs)} document+CI combinations")

    for run in unique_runs:
        doc_id  = run["document_id"]
        doc_num = doc_id.split("-")[0]
        ci_idx  = run.get("ci_index", "unknown")
        ci_stem = ci_idx.replace("ci-objects-", "")
        label   = f"{doc_num}-{ci_stem}"

        # Locate old file
        old_path = _find_old_ecs(doc_num)
        if not old_path:
            print(f"\n[SKIP] {label}: no old-ecs/current.json for doc {doc_num}")
            continue

        # Locate new file
        new_json = run.get("output_json")
        if not new_json:
            print(f"\n[SKIP] {label}: no output_json in config")
            continue
        new_path = ROOT / new_json
        if not new_path.exists():
            print(f"\n[SKIP] {label}: new result not found: {new_path}")
            continue

        out_path = out_dir / f"{label}.csv"
        _run_pair(old_path, new_path, out_path, doc_label=label)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare old ECS result vs new search_test.py result."
    )
    p.add_argument("--old",         default=None,
                   help="Path to old current.json")
    p.add_argument("--new",         default=None,
                   help="Path to new search_test.py output JSON")
    p.add_argument("--out",         default=None,
                   help="Output CSV path (single-pair mode)")
    p.add_argument("--batch",       action="store_true",
                   help="Batch mode: read full_tables.json and compare all documents")
    p.add_argument("--config",      default=str(CONFIG),
                   help=f"Config JSON for batch mode (default: {CONFIG})")
    p.add_argument("--ci-index",    default=None,
                   help="Filter batch runs to this ci_index name (e.g. ci-objects-ahmed)")
    p.add_argument("--out-dir",     default=str(OUT_DIR),
                   help=f"Output directory for batch CSVs (default: {OUT_DIR})")
    args = p.parse_args()

    if args.batch:
        _run_batch(
            config_path=Path(args.config),
            ci_index_filter=args.ci_index,
            out_dir=Path(args.out_dir),
        )
        return

    # Single-pair mode
    if not args.old or not args.new:
        p.error("Provide --old and --new, or use --batch mode.")

    old_path = Path(args.old)
    new_path = Path(args.new)
    if not old_path.exists():
        print(f"ERROR: --old not found: {old_path}", file=sys.stderr)
        sys.exit(1)
    if not new_path.exists():
        print(f"ERROR: --new not found: {new_path}", file=sys.stderr)
        sys.exit(1)

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = OUT_DIR / f"comparison_{old_path.parent.name}_vs_{new_path.stem}.csv"

    _run_pair(old_path, new_path, out_path)


if __name__ == "__main__":
    main()
