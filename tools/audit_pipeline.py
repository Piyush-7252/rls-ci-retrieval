#!/usr/bin/env python3
"""
Pipeline Failure Audit Tool
============================
Reads a search-results JSON produced by search_test.py and generates
a per-candidate failure analysis answering the 8-stage review framework:

  Stage              What is inspected
  -----------------  -------------------------------------------------------
  CI                 Enrichment completeness (entities, facts, identities)
  Candidate          Enrichment completeness of the retrieved object
  Candidate context  Signals from neighboring objects (prev/next sentence)
  Retrieval          Which retriever(s) surfaced it and by what margin
  Aggregator         Score component breakdown – what drove the composite
  Cross-encoder      CE score vs semantic reality (lexical-bias detector)
  Verifier           Verdict, reason, and whether it was the last defence
  Root cause         Labelled taxonomy of failure mode

Outputs
-------
  <stem>_audit.csv     — one row per candidate (final + rejected + skipped)
  <stem>_audit_summary.txt — distribution of root-cause labels + top issues

Usage
-----
  python tools/audit_pipeline.py localfiles/search_results/run18/10990.json
  python tools/audit_pipeline.py localfiles/search_results/run18/10990.json \
      --only-fp               # only false-positive final hits
      --min-ce 2.0            # only candidates with CE score >= 2.0
      --ci-id 36              # only a specific CI
      --output localfiles/audit/10990-audit.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ──────────────────────────────────────────────────────────────────────────────
# Root-cause taxonomy
# ──────────────────────────────────────────────────────────────────────────────

ROOT_CAUSES = {
    # CI-side
    "CI_NO_ENTITIES":      "CI has no extracted entities – retrieval net was too broad",
    "CI_NO_FACTS":         "CI has no extracted facts – fact-slot comparisons blind",
    "CI_SPARSE_IDENTITY":  "CI treatment/endpoint/population identity slots are empty",
    "CI_ONTOLOGY_EMPTY":   "CI has no ontology expansions – ontology retriever was silent",

    # Candidate-side
    "CAND_NO_ENTITIES":    "Candidate object has no extracted entities",
    "CAND_NO_FACTS":       "Candidate object has no extracted facts",
    "CAND_SPARSE_ENRICH":  "Candidate enrichment_status reports missing slots",

    # Retrieval
    "LEXICAL_ONLY":        "Only BM25 surfaced the candidate – pure lexical overlap",
    "ONTOLOGY_BROAD":      "Only ontology surfaced it – expansion may be too broad",
    "FACT_ONLY":           "Only fact-retrieval surfaced it – fact match without semantic signal",
    "SINGLE_RETRIEVER":    "Only one retriever surfaced it – weak multi-signal evidence",

    # Aggregator
    "WRONG_DRUG":          "Drug identity mismatch (overlap < 0.5) – different compound",
    "NO_IDENTITY":         "No slot overlapped at all – zero_id_pen fired",
    "LOW_ENTITY_OVERLAP":  "Entity term overlap < 0.2 – objects share no clinical terms",
    "LOW_FACT_OVERLAP":    "Fact slot overlap < 0.2 – different clinical facts",
    "LOW_GRANULARITY":     "Granularity factor < 0.05 – text too coarse / fragment",
    "SECTION_DRIFT":       "Section-drift penalty applied – wrong section type",
    "STRUCTURAL_PENALTY":  "Structural penalty > 0.4 – non-assertive / comparison context",
    "ZERO_ENRICH_PEN":     "Zero-enrichment penalty applied – enrichment completely absent",

    # Cross-encoder / reranker
    "CE_LEXICAL_BIAS":     "High CE score but low fact+entity overlap – lexical token match",
    "CE_WEAK":             "Low CE score (< 1.0) – reranker was already sceptical",

    # Verifier
    "VERIFIER_DRUG_MISMATCH": "Verifier rejected for drug / study mismatch",
    "VERIFIER_STMT_MISMATCH": "Verifier rejected for statement-type mismatch",
    "VERIFIER_CORRECT":    "Verifier correctly rejected with clear clinical reasoning",
    "VERIFIER_MARGINAL":   "Verifier verdict MAYBE – ambiguous evidence",
    "VERIFIER_FP_PASSED":  "Verifier passed it but may be a false positive",

    # Chunk / context
    "CHUNK_BOUNDARY":      "Evidence likely split across chunk boundaries",
    "CONTEXT_ONLY":        "Match is in context_chunk, not in the object itself",
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _safe(d: dict, *keys, default=None):
    """Safe nested dict access."""
    v = d
    for k in keys:
        if not isinstance(v, dict):
            return default
        v = v.get(k, default)
    return v


def _sources(candidate: dict) -> list[str]:
    """Return list of retriever source names from the sources array or agg_score_breakdown."""
    srcs: list[str] = []
    ab = candidate.get("agg_score_breakdown", {})
    for name in ("bm25", "vector", "literal", "ontology", "fact_ret", "numeric_ret"):
        val = ab.get(name, 0)
        if isinstance(val, (int, float)) and val > 0:
            srcs.append(name.replace("_ret", ""))
    # also check candidate.sources list (strings like "bm25", "vector", ...)
    for s in candidate.get("sources", []):
        label = s if isinstance(s, str) else s.get("retriever", "")
        if label and label not in srcs:
            srcs.append(label)
    return srcs


def _enrichment_gaps(status: dict, side: str) -> list[str]:
    """Return list of missing enrichment slot names for 'ci' or 'candidate'."""
    slots = status.get(side, {})
    return [k for k, v in slots.items() if v is False]


def _identity_drug_overlap(ab: dict) -> float:
    io = ab.get("identity_overlap", {})
    if isinstance(io, dict):
        return io.get("drug", {}).get("overlap", 1.0)
    return 1.0


def _identity_all_zero(ab: dict) -> bool:
    io = ab.get("identity_overlap", {})
    if not isinstance(io, dict):
        return False
    for slot, v in io.items():
        if isinstance(v, dict) and v.get("overlap", 0.0) > 0:
            return False
        if isinstance(v, float) and v > 0:
            return False
    return True


def _infer_root_causes(
    ci: dict,
    candidate: dict,
    candidate_role: str,  # "final" | "rejected" | "skipped"
) -> list[str]:
    """Infer root-cause labels for a single candidate."""
    causes: list[str] = []

    sb  = candidate.get("score_breakdown", {})
    ab  = candidate.get("agg_score_breakdown", {})
    io  = candidate.get("indexed_object", {})
    enrich_status = sb.get("enrichment_status", {})

    srcs = _sources(candidate)
    ce   = candidate.get("cross_encoder_score", 0.0) or 0.0
    agg  = candidate.get("agg_score", 0.0) or 0.0

    # ── CI side ───────────────────────────────────────────────────────────────
    ci_enrich_gaps = _enrichment_gaps(enrich_status, "ci")
    if "entities" in ci_enrich_gaps or not ci.get("entities"):
        causes.append("CI_NO_ENTITIES")
    if "facts" in ci_enrich_gaps or not ci.get("effective_facts"):
        causes.append("CI_NO_FACTS")
    if not ci_enrich_gaps:
        ti = ci.get("treatment_identity", {})
        if not ti or (not ti.get("drug") and not ti.get("compound_id")):
            causes.append("CI_SPARSE_IDENTITY")
    if not ci.get("ontology_expansions"):
        causes.append("CI_ONTOLOGY_EMPTY")

    # ── Candidate side ────────────────────────────────────────────────────────
    cand_enrich_gaps = _enrichment_gaps(enrich_status, "candidate")
    if "entities" in cand_enrich_gaps or not io.get("entities"):
        causes.append("CAND_NO_ENTITIES")
    if "facts" in cand_enrich_gaps or not io.get("facts"):
        causes.append("CAND_NO_FACTS")
    if len(cand_enrich_gaps) > 2:
        causes.append("CAND_SPARSE_ENRICH")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    if srcs:
        if len(srcs) == 1:
            causes.append("SINGLE_RETRIEVER")
            if srcs[0] == "bm25":
                causes.append("LEXICAL_ONLY")
            elif srcs[0] in ("ontology",):
                causes.append("ONTOLOGY_BROAD")
            elif srcs[0] in ("fact", "fact_ret"):
                causes.append("FACT_ONLY")

    # ── Aggregator ────────────────────────────────────────────────────────────
    drug_overlap = _identity_drug_overlap(ab)
    if drug_overlap < 0.5 and ci.get("treatment_identity", {}).get("drug"):
        causes.append("WRONG_DRUG")

    if ab.get("zero_id_pen", 0.0) < -0.01:
        causes.append("NO_IDENTITY")
    elif _identity_all_zero(ab):
        causes.append("NO_IDENTITY")

    entity_olap = ab.get("entity_olap", 1.0) or 0.0
    if entity_olap < 0.2:
        causes.append("LOW_ENTITY_OVERLAP")

    fact_olap = ab.get("fact_olap", 1.0) or 0.0
    if fact_olap < 0.2:
        causes.append("LOW_FACT_OVERLAP")

    gran = sb.get("gran_factor", 1.0) or 0.0
    if gran < 0.05:
        causes.append("LOW_GRANULARITY")

    if (ab.get("sect_drift_pen", 0.0) or 0.0) < -0.05:
        causes.append("SECTION_DRIFT")

    struct_pen = abs(sb.get("struct_penalty", 0.0) or 0.0)
    if struct_pen > 0.35:
        causes.append("STRUCTURAL_PENALTY")

    if (ab.get("zero_enrich_pen", 0.0) or 0.0) < -0.05:
        causes.append("ZERO_ENRICH_PEN")

    # ── Cross-encoder ─────────────────────────────────────────────────────────
    if ce < 1.0:
        causes.append("CE_WEAK")
    elif ce >= 2.5 and entity_olap < 0.2 and fact_olap < 0.2:
        causes.append("CE_LEXICAL_BIAS")

    # ── Verifier ──────────────────────────────────────────────────────────────
    verdict  = candidate.get("verdict", candidate.get("verifier_verdict", ""))
    v_reason = (candidate.get("verifier_reason") or candidate.get("reason") or "").lower()

    if verdict == "MAYBE":
        causes.append("VERIFIER_MARGINAL")
    elif verdict == "NO":
        if any(w in v_reason for w in ("drug", "compound", "talquetamab", "teclistamab", "wrong drug")):
            causes.append("VERIFIER_DRUG_MISMATCH")
        elif any(w in v_reason for w in ("statement", "not an assertion", "comparison")):
            causes.append("VERIFIER_STMT_MISMATCH")
        else:
            causes.append("VERIFIER_CORRECT")
    elif verdict in ("YES", "") and candidate_role == "final":
        causes.append("VERIFIER_FP_PASSED")

    # ── Chunk boundary heuristic ──────────────────────────────────────────────
    ctx = io.get("context_chunk_text", "")
    text = io.get("text", "")
    match_span = candidate.get("match_span", "")
    if match_span and text and match_span not in text and ctx and match_span in ctx:
        causes.append("CONTEXT_ONLY")

    # Deduplicate preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for c in causes:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


# ──────────────────────────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────────────────────────

def _analyse_result(result: dict) -> list[dict]:
    """Return a list of analysis rows for all candidates in one CI result."""
    rows: list[dict] = []
    ci     = result.get("ci", {})
    ci_id  = result.get("ci_id")
    ci_txt = result.get("ci_text", "")[:120].replace("\n", " ")

    # Enrich status summary for the CI
    ci_entities_count    = len(ci.get("entities", []) or [])
    ci_facts_count       = sum(len(v) for v in (ci.get("effective_facts") or {}).values()
                               if isinstance(v, list))
    ci_ontology_count    = len(ci.get("ontology_expansions") or [])
    ci_drug              = ", ".join((ci.get("treatment_identity") or {}).get("drug") or [])
    ci_phase             = (ci.get("study_hierarchy") or {}).get("phase") or ""
    ci_section           = (ci.get("clinical_identity") or {}).get("section") or ""
    ci_strategies        = ", ".join(result.get("strategies") or [])

    def _build_row(cand: dict, role: str) -> dict:
        ab  = cand.get("agg_score_breakdown", {})
        sb  = cand.get("score_breakdown", {})
        io  = cand.get("indexed_object", {})
        enrich_status = sb.get("enrichment_status", {})
        enrich_diag   = sb.get("enrichment_diagnostics", {})

        srcs        = _sources(cand)
        root_causes = _infer_root_causes(ci, cand, role)

        io_entities_count = len(io.get("entities") or [])
        io_facts_count    = sum(len(v) for v in (io.get("facts") or {}).values()
                                if isinstance(v, list))

        # Identity overlap summary
        io_id = ab.get("identity_overlap", {})
        drug_overlap    = io_id.get("drug",     {}).get("overlap", "") if isinstance(io_id, dict) else ""
        phase_overlap   = io_id.get("phase",    {}).get("overlap", "") if isinstance(io_id, dict) else ""
        endpoint_overlap= io_id.get("endpoint", {}).get("overlap", "") if isinstance(io_id, dict) else ""

        cand_drug       = ", ".join(io_id.get("drug", {}).get("candidate", []) if isinstance(io_id, dict) else [])

        verdict         = cand.get("verdict", "")
        v_reason        = (cand.get("verifier_reason") or cand.get("reason") or "")[:200]

        ci_enrich_gaps  = _enrichment_gaps(enrich_status, "ci")
        cand_enrich_gaps= _enrichment_gaps(enrich_status, "candidate")

        return {
            # Identity
            "ci_id":                 ci_id,
            "ci_text_short":         ci_txt,
            "ci_type":               result.get("ci_type", ""),
            "candidate_role":        role,
            "doc_page":              f"{cand.get('page_start','')}–{cand.get('page_end','')}",
            "object_type":           cand.get("retrieval_object_type", ""),
            "section":               cand.get("retrieval_section", ""),
            "heading_path":          cand.get("retrieval_heading_path", "")[:80],
            "match_span":            (cand.get("match_span") or "")[:100],

            # CI enrichment
            "ci_strategies":         ci_strategies,
            "ci_drug":               ci_drug,
            "ci_phase":              ci_phase,
            "ci_section":            ci_section,
            "ci_entities_n":         ci_entities_count,
            "ci_facts_n":            ci_facts_count,
            "ci_ontology_n":         ci_ontology_count,
            "ci_enrich_gaps":        ", ".join(ci_enrich_gaps),

            # Candidate enrichment
            "cand_drug":             cand_drug,
            "cand_entities_n":       io_entities_count,
            "cand_facts_n":          io_facts_count,
            "cand_enrich_gaps":      ", ".join(cand_enrich_gaps),
            "cand_enrich_status":    (enrich_diag.get("candidate") or {}).get("status", ""),
            "cand_missing_slots":    ", ".join((enrich_diag.get("candidate") or {}).get("missing_slots", [])),

            # Retrieval
            "retrievers":            ", ".join(srcs),
            "n_retrievers":          len(srcs),

            # Aggregator scores
            "agg_score":             round(float(cand.get("agg_score", 0) or 0), 4),
            "ret_component":         round(float(ab.get("ret_component", 0) or 0), 4),
            "entity_olap":           round(float(ab.get("entity_olap", 0) or 0), 4),
            "fact_olap":             round(float(ab.get("fact_olap", 0) or 0), 4),
            "identity_bonus":        round(float(ab.get("identity_bonus", 0) or 0), 4),
            "zero_id_pen":           round(float(ab.get("zero_id_pen", 0) or 0), 4),
            "zero_enrich_pen":       round(float(ab.get("zero_enrich_pen", 0) or 0), 4),
            "sect_drift_pen":        round(float(ab.get("sect_drift_pen", 0) or 0), 4),
            "bm25":                  round(float(ab.get("bm25", 0) or 0), 4),
            "vector":                round(float(ab.get("vector", 0) or 0), 4),
            "ontology":              round(float(ab.get("ontology", 0) or 0), 4),
            "fact_ret":              round(float(ab.get("fact_ret", 0) or 0), 4),
            "contradiction":         round(float(ab.get("contradiction", 0) or 0), 4),

            # Composite scoring
            "drug_identity_score":   round(float(sb.get("drug_identity", 0) or 0), 4),
            "fact_slot_score":       round(float(sb.get("fact_slot", 0) or 0), 4),
            "intent_align":          round(float(sb.get("intent_align", 0) or 0), 4),
            "gran_factor":           round(float(sb.get("gran_factor", 0) or 0), 4),
            "struct_penalty":        round(float(sb.get("struct_penalty", 0) or 0), 4),
            "composite":             round(float(sb.get("composite", 0) or 0), 4),

            # Cross-encoder
            "cross_encoder_score":   round(float(cand.get("cross_encoder_score", 0) or 0), 4),
            "highlight_score":       round(float(cand.get("highlight_score", 0) or 0), 4),

            # Identity overlap
            "drug_overlap":          drug_overlap,
            "phase_overlap":         phase_overlap,
            "endpoint_overlap":      endpoint_overlap,

            # Verifier
            "verdict":               verdict,
            "confidence":            round(float(cand.get("confidence", 0) or 0), 4),
            "verifier_reason":       v_reason,

            # Root cause
            "root_causes":           " | ".join(root_causes),
            "n_root_causes":         len(root_causes),

            # ci_facts vs cand_facts (what did the comparator actually see)
            "ci_facts_str":          str(sb.get("ci_facts", ""))[:200],
            "cand_facts_str":        str(sb.get("cand_facts", ""))[:200],
            "clinical_reasoning":    str(sb.get("clinical_reasoning", ""))[:300],

            # Context text for manual review
            "object_text":           (io.get("text") or "")[:300].replace("\n", " "),
            "context_sentence":      (cand.get("context_sentence") or io.get("prev_sentence_text") or "")[:200].replace("\n", " "),
        }

    for cand in result.get("candidates", []):
        rows.append(_build_row(cand, "pre_verifier"))
    for cand in result.get("final_hits", []):
        rows.append(_build_row(cand, "final"))
    for cand in result.get("rejected_hits", []):
        rows.append(_build_row(cand, "rejected"))
    for cand in result.get("skipped_hits", []):
        rows.append(_build_row(cand, "skipped"))

    return rows


def _print_summary(rows: list[dict], out_csv: str) -> None:
    """Print a rich terminal summary report."""
    total      = len(rows)
    final_n    = sum(1 for r in rows if r["candidate_role"] == "final")
    rejected_n = sum(1 for r in rows if r["candidate_role"] == "rejected")
    skipped_n  = sum(1 for r in rows if r["candidate_role"] == "skipped")
    pre_n      = sum(1 for r in rows if r["candidate_role"] == "pre_verifier")

    # Root cause distribution
    cause_counter: Counter = Counter()
    for r in rows:
        for c in r["root_causes"].split(" | "):
            if c.strip():
                cause_counter[c.strip()] += 1

    # Top false-positive CIs (final hits with most causes)
    fp_rows = [r for r in rows if r["candidate_role"] == "final"]
    ci_fp_causes: dict[int, list[str]] = defaultdict(list)
    for r in fp_rows:
        ci_fp_causes[r["ci_id"]].extend(r["root_causes"].split(" | "))

    print()
    print("=" * 70)
    print("  PIPELINE AUDIT SUMMARY")
    print("=" * 70)
    print(f"  Output CSV       : {out_csv}")
    print(f"  Total candidates : {total}")
    print(f"    pre-verifier   : {pre_n}")
    print(f"    final (passed) : {final_n}")
    print(f"    rejected       : {rejected_n}")
    print(f"    skipped        : {skipped_n}")
    print()

    print("  ROOT CAUSE DISTRIBUTION (all candidates)")
    print("  " + "-" * 60)
    for cause, count in cause_counter.most_common(20):
        pct = 100 * count / total if total else 0
        bar = "█" * int(pct / 2)
        print(f"  {cause:<35} {count:>4}  {pct:5.1f}%  {bar}")
    print()

    print("  TOP CIs WITH MOST ROOT CAUSE LABELS ON FINAL HITS")
    print("  " + "-" * 60)
    for ci_id, all_causes in sorted(ci_fp_causes.items(),
                                    key=lambda x: len(set(x[1])), reverse=True)[:10]:
        unique = sorted(set(c for c in all_causes if c))
        ci_text = next((r["ci_text_short"] for r in fp_rows if r["ci_id"] == ci_id), "")
        print(f"  CI {ci_id:<4}  ({len(fp_rows)}/{total} final)  {ci_text[:60]}")
        for c in unique:
            desc = ROOT_CAUSES.get(c, "")[:70]
            print(f"         ├─ {c}: {desc}")
        print()

    print("  STAGE GAPS — CI enrichment problems")
    print("  " + "-" * 60)
    ci_enrich_problems: Counter = Counter()
    for r in rows:
        if r["ci_enrich_gaps"]:
            for gap in r["ci_enrich_gaps"].split(", "):
                ci_enrich_problems[gap.strip()] += 1
    if ci_enrich_problems:
        for gap, count in ci_enrich_problems.most_common():
            print(f"    {gap}: {count} candidates affected")
    else:
        print("    No CI enrichment gaps detected")
    print()

    print("  STAGE GAPS — Candidate enrichment problems")
    print("  " + "-" * 60)
    cand_enrich_problems: Counter = Counter()
    for r in rows:
        if r["cand_enrich_gaps"]:
            for gap in r["cand_enrich_gaps"].split(", "):
                cand_enrich_problems[gap.strip()] += 1
    if cand_enrich_problems:
        for gap, count in cand_enrich_problems.most_common():
            print(f"    {gap}: {count} candidates affected")
    else:
        print("    No candidate enrichment gaps detected")
    print()

    print("  RETRIEVER DISTRIBUTION")
    print("  " + "-" * 60)
    ret_counter: Counter = Counter()
    for r in rows:
        for ret in r["retrievers"].split(", "):
            if ret.strip():
                ret_counter[ret.strip()] += 1
    for ret, count in ret_counter.most_common():
        print(f"    {ret}: {count}")
    print()

    # CE score distribution by verdict
    by_verdict: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_verdict[r["candidate_role"]].append(r["cross_encoder_score"])
    print("  CROSS-ENCODER SCORE — mean by stage")
    print("  " + "-" * 60)
    for role, scores in by_verdict.items():
        if scores:
            print(f"    {role:<15} mean={sum(scores)/len(scores):.3f}  "
                  f"max={max(scores):.3f}  min={min(scores):.3f}  n={len(scores)}")
    print()
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline failure audit tool")
    ap.add_argument("results_json", help="Path to search_test.py output JSON")
    ap.add_argument("--output",    "-o", help="Output CSV path (default: auto)")
    ap.add_argument("--only-fp",   action="store_true",
                    help="Only include final-hit rows (potential false positives)")
    ap.add_argument("--only-rejected", action="store_true",
                    help="Only include rejected-hit rows")
    ap.add_argument("--min-ce",    type=float, default=0.0,
                    help="Only include candidates with CE score >= this value")
    ap.add_argument("--ci-id",     type=int, default=None,
                    help="Only audit a specific CI id")
    ap.add_argument("--no-summary", action="store_true",
                    help="Skip terminal summary, just write CSV")
    args = ap.parse_args()

    results_path = Path(args.results_json)
    if not results_path.exists():
        sys.exit(f"ERROR: {results_path} not found")

    with results_path.open() as f:
        data = json.load(f)

    doc_id   = data.get("run", {}).get("document_id", results_path.stem)
    out_csv  = args.output or str(results_path.parent / f"{results_path.stem}_audit.csv")

    all_rows: list[dict] = []
    for result in data.get("results", []):
        if args.ci_id is not None and result.get("ci_id") != args.ci_id:
            continue
        rows = _analyse_result(result)
        all_rows.extend(rows)

    # Filter
    if args.only_fp:
        all_rows = [r for r in all_rows if r["candidate_role"] == "final"]
    elif args.only_rejected:
        all_rows = [r for r in all_rows if r["candidate_role"] == "rejected"]

    if args.min_ce > 0:
        all_rows = [r for r in all_rows if r["cross_encoder_score"] >= args.min_ce]

    if not all_rows:
        print("No rows matched the filters.")
        return

    # Write CSV
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows → {out_csv}")

    if not args.no_summary:
        _print_summary(all_rows, out_csv)


if __name__ == "__main__":
    main()
