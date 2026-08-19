#!/usr/bin/env python3
"""
Compare a reranked (RR) run with a no-rerank (NR) run.

Passage identity:
    (ci_id, page_start, first_80_chars_of_underlying_passage_text)

This intentionally does not use chunk_id, retrieval_origin, score, or LLM verdict
as part of passage identity.
"""
import argparse
import json
from pathlib import Path
import pandas as pd

OUTPUT_COLUMNS = [
    "ci_id","ci_type","ci_text","page","section","highlight_text",
    "final_hit_in",
    "rr_candidates_after_retrieve","rr_candidates_passed_to_llm","rr_rerank_time_s",
    "rr_agg_score","rr_ce_score","rr_composite_score","rr_result_category",
    "rr_llm_verdict","rr_llm_confidence","rr_llm_reason","rr_highlight_score",
    "rr_llm_verify_time_s","rr_evidence_type","rr_evidence_confidence","rr_evidence_reason",
    "nr_candidates_after_retrieve","nr_candidates_passed_to_llm","nr_rerank_time_s",
    "nr_agg_score","nr_ce_score","nr_composite_score","nr_result_category",
    "nr_llm_verdict","nr_llm_confidence","nr_llm_reason","nr_highlight_score",
    "nr_llm_verify_time_s","nr_evidence_type","nr_evidence_confidence","nr_evidence_reason",
]

CATEGORY_PRIORITY = {
    "FINAL_HIT": 4,
    "LLM_REJECTED": 3,
    "BLOCKED_PRE_LLM": 2,
    "SKIPPED": 1,
}

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def indexed_text(entry):
    obj = entry.get("indexed_object") or {}
    return obj.get("text") or ""

def underlying_text(entry):
    # indexed_object.text is the actual passage. The top-level `text` in
    # final/rejected hits may additionally contain the heading path.
    return indexed_text(entry) or entry.get("text") or ""

def passage_key(ci_id, entry):
    page = entry.get("page_start", entry.get("match_page"))
    text = underlying_text(entry)
    return (ci_id, page, text[:80])

def display_text(entry):
    # Preserve the report-style display text where available.
    if entry.get("text"):
        return entry["text"]
    text = indexed_text(entry)
    heading = entry.get("retrieval_heading_path") or ""
    return f"{heading}\n\n{text}".strip()

def score_values(entry):
    sb = entry.get("score_breakdown") or {}
    return (
        entry.get("agg_score"),
        sb.get("ce", entry.get("cross_encoder_score")),
        sb.get("composite"),
    )

def build_side(data):
    side = {}
    for result in data.get("results", []):
        ci_id = result.get("ci_id")
        meta = {
            "ci_type": result.get("ci_type"),
            "ci_text": result.get("ci_text"),
            "candidates_after_retrieve": result.get("candidates_found"),
            "candidates_passed_to_llm": (result.get("timings") or {}).get("n_candidates_to_verifier"),
            # The supplied report calls this rerank_time_s, but the value is
            # the complete per-CI pipeline time.
            "rerank_time_s": (result.get("timings") or {}).get("total"),
            "llm_verify_time_s": (result.get("timings") or {}).get("llm_verifier"),
        }

        # Outcome-bearing records are preferred over candidate-only records.
        sources = [
            ("candidates", result.get("candidates", [])),
            ("skipped", result.get("skipped_hits", [])),
            ("rejected", result.get("rejected_hits", [])),
            ("final", result.get("final_hits", [])),
        ]

        for source_name, entries in sources:
            for entry in entries:
                key = passage_key(ci_id, entry)
                if key[1] is None or not key[2]:
                    continue

                category = {
                    "final": "FINAL_HIT",
                    "rejected": "LLM_REJECTED",
                    "skipped": "SKIPPED",
                    "candidates": "BLOCKED_PRE_LLM",
                }[source_name]

                old = side.get(key)
                if old and CATEGORY_PRIORITY[old["result_category"]] >= CATEGORY_PRIORITY[category]:
                    continue

                agg, ce, composite = score_values(entry)
                side[key] = {
                    **meta,
                    "page": key[1],
                    "section": (
                        entry.get("retrieval_section")
                        or (entry.get("indexed_object") or {}).get("section_category")
                    ),
                    "highlight_text": display_text(entry),
                    "result_category": category,
                    "llm_verdict": entry.get("verdict") if category != "BLOCKED_PRE_LLM" else "SKIP",
                    "llm_confidence": entry.get("confidence"),
                    "llm_reason": (
                        entry.get("reason")
                        if category != "BLOCKED_PRE_LLM"
                        else "below reranker threshold"
                    ),
                    "highlight_score": entry.get("highlight_score"),
                    "agg_score": agg,
                    "ce_score": ce,
                    "composite_score": composite,
                    "evidence_type": entry.get("evidence_type"),
                    "evidence_confidence": entry.get("evidence_confidence"),
                    "evidence_reason": entry.get("evidence_reason"),
                }

    return side

def compare(rr_data, nr_data):
    rr = build_side(rr_data)
    nr = build_side(nr_data)

    # Union of passage keys. Matching is ONLY:
    # (ci_id, page_start, first 80 chars of underlying passage text)
    keys = list(dict.fromkeys(list(rr.keys()) + list(nr.keys())))

    rows = []
    for key in keys:
        ci_id, page, _ = key
        r = rr.get(key)
        n = nr.get(key)
        base = r or n

        if r and n:
            if r["result_category"] == "FINAL_HIT" and n["result_category"] == "FINAL_HIT":
                final_in = "BOTH"
            elif r["result_category"] == "FINAL_HIT":
                final_in = "WITH_RERANK_ONLY"
            elif n["result_category"] == "FINAL_HIT":
                final_in = "NO_RERANK_ONLY"
            else:
                final_in = "NEITHER_FINAL"
        elif r:
            final_in = "WITH_RERANK_ONLY" if r["result_category"] == "FINAL_HIT" else "NEITHER_FINAL"
        else:
            final_in = "NO_RERANK_ONLY" if n["result_category"] == "FINAL_HIT" else "NEITHER_FINAL"

        row = {
            "ci_id": ci_id,
            "ci_type": base["ci_type"],
            "ci_text": base["ci_text"],
            "page": page,
            "section": base.get("section"),
            "highlight_text": base.get("highlight_text"),
            "final_hit_in": final_in,
        }

        fields = [
            "candidates_after_retrieve","candidates_passed_to_llm","rerank_time_s",
            "agg_score","ce_score","composite_score","result_category",
            "llm_verdict","llm_confidence","llm_reason","highlight_score",
            "llm_verify_time_s","evidence_type","evidence_confidence","evidence_reason",
        ]

        for prefix, rec in (("rr", r), ("nr", n)):
            for field in fields:
                row[f"{prefix}_{field}"] = rec.get(field) if rec else None

        rows.append(row)

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rr", required=True, help="Full/rerank JSON")
    ap.add_argument("--nr", required=True, help="Skip-rerank JSON")
    ap.add_argument("-o", "--output", default="comparison_rerank_vs_norerank.csv")
    args = ap.parse_args()

    rr_data = load_json(args.rr)
    nr_data = load_json(args.nr)
    df = compare(rr_data, nr_data)
    df.to_csv(args.output, index=False)

    print(f"Rows: {len(df)}")
    print(f"Unique passage keys: {df[['ci_id','page','highlight_text']].drop_duplicates().shape[0]}")
    print("\nfinal_hit_in:")
    print(df["final_hit_in"].value_counts().to_string())
    print(f"\nWrote: {Path(args.output).resolve()}")

if __name__ == "__main__":
    main()
