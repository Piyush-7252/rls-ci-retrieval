"""
eval_harness.py — Per-stage retrieval evaluation with annotated ground truth.
=============================================================================

Purpose
-------
Converts the current search_test probe into a quantitative evaluation suite.
For each CI in the ground-truth annotation file, the harness runs the full
pipeline and tracks whether the expected pages / chunks survive each stage.

Sub-commands
------------
  annotate   Run search for each CI, show results, prompt user to mark
             authoritative pages and ranked gold chunks.  Writes ground_truth.json.

  eval       Run search for each CI against saved annotations.
             Writes per-stage metrics to eval_results.json and prints report.

  report     Print a report from a previously-saved eval_results.json
             (no pipeline run required).

  delta      Compare two eval_results.json files (before/after a change).
             Prints per-stage and per-CI-type deltas with significance markers.

Key metrics
-----------
  Stage recall          Did at least one gold chunk survive this stage?
  Precision@1           Was the top result the highest-priority gold chunk?
  Precision@1 (any)     Was the top result any acceptable gold chunk?
  MRR                   Mean reciprocal rank of the first gold chunk.
  Per-CI-type           All metrics broken down by CI type.
  Confidence intervals  Wilson score 95% CI on every proportion.
  Score distribution    Top1 score, Top2 score, gap, mean/median gap per stage.
  FP taxonomy           False positive category breakdown (Wrong Drug, etc.).
  CI difficulty         Easy / Medium / Hard / Impossible classification.

Annotation format (ground_truth.json)  version "2"
----------------------------------------------------
  {
    "document_id": "...",
    "version":     "2",
    "annotations": [
      {
        "ci_id":      123,
        "ci_text":    "...",
        "ci_type":    "OBJECTIVE",
        "difficulty": "Medium",
        "notes":      "...",
        "gold_chunks": [
          {"chunk_id": "abc-def-123", "priority": 1, "label": "synopsis"},
          {"chunk_id": "ghi-jkl-456", "priority": 2, "label": "objectives_section"}
        ],
        "expected_pages":  [14, 15],
        "expected_chunks": ["abc-def-123"]
      }
    ]
  }

Usage
-----
  python tests/eval_harness.py annotate \\
    --ci-file localfiles/ci/ahmedCis.json \\
    --out     localfiles/eval/ground_truth.json

  python tests/eval_harness.py eval \\
    --ci-file      localfiles/ci/ahmedCis.json \\
    --ground-truth localfiles/eval/ground_truth.json \\
    --out          localfiles/eval/eval_results_$(date +%Y%m%d_%H%M).json

  python tests/eval_harness.py report \\
    --results localfiles/eval/eval_results_20260726_1430.json

  python tests/eval_harness.py delta \\
    --before  localfiles/eval/eval_results_before.json \\
    --after   localfiles/eval/eval_results_after.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

search_test_path = ROOT / "tests" / "search_test.py"
_st_spec  = importlib.util.spec_from_file_location("search_test", search_test_path)
_st_mod   = importlib.util.module_from_spec(_st_spec)
_st_spec.loader.exec_module(_st_mod)

DOCUMENT_ID           = _st_mod.DOCUMENT_ID
OPENSEARCH_ENDPOINT   = _st_mod.OPENSEARCH_ENDPOINT
AWS_REGION            = _st_mod.AWS_REGION
_load                 = _st_mod._load
_inject_os            = _st_mod._inject_os
enrich_ci             = _st_mod.enrich_ci
_lookup_ci_from_index = _st_mod._lookup_ci_from_index
RETRIEVER_MAP         = _st_mod.RETRIEVER_MAP

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
logger = logging.getLogger("eval_harness")


# ─────────────────────────────────────────────────────────────────────────────
# False-positive taxonomy
# ─────────────────────────────────────────────────────────────────────────────

_FP_CATEGORY: dict[str, str] = {
    "drug_mismatch":        "Wrong Drug",
    "endpoint_mismatch":    "Wrong Endpoint",
    "phase_mismatch":       "Wrong Study",
    "missing_relation":     "Wrong Intent",
    "stmt_type_mismatch":   "Wrong Intent",
    "section_mismatch":     "Wrong Intent",
    "abbreviation_object":  "Abbreviation / Definition",
    "missing_fact_slot":    "Weak Semantic Match",
    "missing_section":      "Wrong Intent",
    "missing_statement":    "Wrong Intent",
    "low_ce_score":         "Weak Semantic Match",
    "dropped_by_reranker":  "Reranker Rejection",
    "not_in_candidates":    "Retrieval Miss",
    "not_top1":             "Ranked Below Top1",
    "ci_not_found":         "Annotation Error",
}

_DIFFICULTY_SIGNAL_PATTERNS = [
    r"\bpart\s+\d+\b",
    r"\bcohort\s+[A-Z\d]",
    r"\barm\s+[A-Z\b]",
    r"\bphase\s+\d",
    r"\brp2d\b",
    r"\bmrd\b",
    r"\bpharmacokinetics\b|\bpk\b",
    r"\b(ORR|PFS|OS|DOR|TTR|CBR|VGPR)\b",
]


def _fp_category(fail_reason: str) -> str:
    if not fail_reason:
        return ""
    key = fail_reason.split("(")[0].strip().split(" ")[0]
    return _FP_CATEGORY.get(key, "Other")


# ─────────────────────────────────────────────────────────────────────────────
# Wilson score confidence interval
# ─────────────────────────────────────────────────────────────────────────────

def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score CI for proportion k/n.  Returns (lower, upper)."""
    if n == 0:
        return 0.0, 1.0
    p      = k / n
    z2     = z * z
    denom  = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _ci_str(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = _wilson_ci(k, n)
    p = k / n
    hw = (hi - lo) / 2
    return f"{p:.1%} ±{hw:.1%}"


# ─────────────────────────────────────────────────────────────────────────────
# CI Difficulty classifier
# ─────────────────────────────────────────────────────────────────────────────

def _ci_difficulty(ci_text: str, ci_type: str = "") -> str:
    import re
    text  = ci_text.lower()
    score = 0
    for pat in _DIFFICULTY_SIGNAL_PATTERNS:
        if re.search(pat, text, re.I):
            score += 1
    _TYPE_PRIOR = {"PHARMACOKINETICS": 1, "DOSING": 1}
    score += _TYPE_PRIOR.get((ci_type or "").upper(), 0)
    if len(ci_text.split()) > 25:
        score += 1
    if   score == 0: return "Easy"
    elif score == 1: return "Medium"
    elif score == 2: return "Hard"
    else:            return "Impossible"


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_ground_truth(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"document_id": DOCUMENT_ID, "version": "2", "annotations": []}
    with p.open() as f:
        return json.load(f)


def _save_ground_truth(gt: dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(gt, f, indent=2)
    print(f"\nSaved ground truth → {p}")


def _gold_chunks(ann: dict) -> list[dict]:
    """Return [{chunk_id, priority, label}] — supports v2 and legacy formats."""
    if ann.get("gold_chunks"):
        return ann["gold_chunks"]
    return [{"chunk_id": c, "priority": 1, "label": ""}
            for c in ann.get("expected_chunks", [])]


def _gold_pages(ann: dict) -> list[int]:
    return ann.get("expected_pages", [])


def _pages_hit(hits: list[dict]) -> set[int]:
    pages: set[int] = set()
    for h in hits:
        for key in ("page_start", "page_end", "match_page"):
            v = h.get(key)
            if v:
                pages.add(v)
    return pages


def _chunks_hit(candidates: list[dict]) -> list[str]:
    return [c.get("chunk_id", "") for c in candidates if c.get("chunk_id")]


def _best_gold_rank(
    ranked_list: list[dict],
    gold:        list[dict],
    exp_pages:   list[int],
) -> tuple[int | None, int]:
    """Return (1-based rank, priority) of the first gold item in ranked_list."""
    gold_ids = {g["chunk_id"]: g["priority"] for g in gold if g.get("chunk_id")}
    for i, item in enumerate(ranked_list, 1):
        chunk = item.get("chunk_id", "")
        pages = {item.get("page_start"), item.get("page_end"),
                 item.get("match_page")} - {None}
        if chunk in gold_ids:
            return i, gold_ids[chunk]
        if exp_pages and pages & set(exp_pages):
            return i, 2
    return None, 0


# ─────────────────────────────────────────────────────────────────────────────
# Minimal pipeline runner — stage snapshots + score distributions
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline_for_eval(enriched_ci: dict, document_id: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "retriever_chunks": [], "retriever_pages": set(),
        "aggregator_candidates": [], "aggregator_chunks": [], "aggregator_pages": set(),
        "agg_top1_score": None, "agg_top2_score": None, "agg_score_gap": None,
        "reranker_ranked": [], "reranker_chunks": [], "reranker_pages": set(),
        "rnk_top1_score": None, "rnk_top2_score": None, "rnk_score_gap": None,
        "top1_chunk": None, "top1_page": None,
        "classifier_type": "?", "error": None,
    }
    try:
        import uuid
        req = {"search_id": f"eval-{uuid.uuid4().hex[:6]}",
               "document_id": document_id, "ci": enriched_ci, "document_context": {}}

        classifier = _load("search/classifier", "search_classifier")
        req        = classifier._process(req)
        result["classifier_type"] = req.get("classification", {}).get("ci_type", "?")

        strategies = req.get("classification", {}).get("strategies", list(RETRIEVER_MAP))
        ret_hits: list[dict] = []
        for strategy in strategies:
            if strategy not in RETRIEVER_MAP:
                continue
            mod = _load(RETRIEVER_MAP[strategy], f"search_{strategy}")
            _inject_os(mod)
            ret_hits.extend(mod._process(req).get("hits", []))

        result["retriever_chunks"] = list({h.get("chunk_id", "") for h in ret_hits})
        result["retriever_pages"]  = _pages_hit(ret_hits)
        req["retriever_results"]   = [{"hits": ret_hits}]

        agg   = _load("search/aggregator", "search_aggregator")
        req   = agg._process(req)
        cands = req.get("candidates", [])
        result["aggregator_candidates"] = cands
        result["aggregator_chunks"]     = _chunks_hit(cands)
        result["aggregator_pages"]      = _pages_hit(cands)

        agg_scores = [c.get("agg_score", 0.0) for c in cands]
        if agg_scores:
            result["agg_top1_score"] = agg_scores[0]
            result["agg_top2_score"] = agg_scores[1] if len(agg_scores) > 1 else None
            if len(agg_scores) > 1:
                result["agg_score_gap"] = round(agg_scores[0] - agg_scores[1], 4)

        if not cands:
            return result

        exp = _load("search/context_expander", "search_expander")
        _inject_os(exp)
        req = exp._process(req)

        reranker = _load("search/reranker", "search_reranker")
        req      = reranker._process(req)
        ranked   = req.get("ranked_candidates", [])

        result["reranker_ranked"] = ranked
        result["reranker_chunks"] = _chunks_hit(ranked)
        result["reranker_pages"]  = _pages_hit(ranked)

        rnk_scores = [r.get("cross_encoder_score", 0.0) for r in ranked]
        if rnk_scores:
            result["rnk_top1_score"] = rnk_scores[0]
            result["rnk_top2_score"] = rnk_scores[1] if len(rnk_scores) > 1 else None
            if len(rnk_scores) > 1:
                result["rnk_score_gap"] = round(rnk_scores[0] - rnk_scores[1], 4)

        if ranked:
            result["top1_chunk"] = ranked[0].get("chunk_id")
            result["top1_page"]  = ranked[0].get("page_start") or ranked[0].get("match_page")

    except Exception as exc:
        logger.exception("[eval] pipeline error: %s", exc)
        result["error"] = str(exc)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Failure attribution
# ─────────────────────────────────────────────────────────────────────────────

def _top_issue(candidates: list[dict], exp_pages: list[int],
               exp_chunks: list[str]) -> str:
    exp_set = set(exp_chunks)
    for c in candidates:
        if c.get("page_start") not in exp_pages and c.get("chunk_id","") not in exp_set:
            continue
        breakdown = c.get("score_breakdown") or {}
        detail    = breakdown.get("contra_detail") or []
        if detail:
            top = max(detail, key=lambda d: abs(d.get("weight", 0)))
            return f'{top["type"]} ({top["weight"]:.2f})'
        struct = breakdown.get("struct_detail") or {}
        if struct:
            top_k = max(struct, key=lambda k: abs(struct[k].get("score", 0)))
            return f'{top_k} (structural)'
    return "not_in_candidates"


# ─────────────────────────────────────────────────────────────────────────────
# Per-CI eval
# ─────────────────────────────────────────────────────────────────────────────

def _eval_one_ci(raw_ci: dict, ci_id: int, annotation: dict,
                 document_id: str) -> dict:
    print(f"\n  CI {ci_id}: {raw_ci.get('knownCI', '')[:80]}")

    enriched = _lookup_ci_from_index(raw_ci, ci_id)
    if enriched is None:
        enriched = enrich_ci(raw_ci, ci_id)

    stage = _run_pipeline_for_eval(enriched, document_id)

    gold       = _gold_chunks(annotation)
    exp_pages  = _gold_pages(annotation)
    exp_chunks = [g["chunk_id"] for g in gold if g.get("chunk_id")]
    p1_chunks  = [g["chunk_id"] for g in gold
                  if g.get("priority") == 1 and g.get("chunk_id")]

    ci_type    = annotation.get("ci_type", stage["classifier_type"]) or "?"
    difficulty = annotation.get("difficulty") or \
                 _ci_difficulty(raw_ci.get("knownCI", ""), ci_type)

    def _hit(pages: set[int], chunks: list[str]) -> bool:
        return bool((pages & set(exp_pages)) or (set(chunks) & set(exp_chunks)))

    retriever_hit  = _hit(stage["retriever_pages"],  stage["retriever_chunks"])
    aggregator_hit = _hit(stage["aggregator_pages"], stage["aggregator_chunks"])
    reranker_hit   = _hit(stage["reranker_pages"],   stage["reranker_chunks"])

    agg_rank, _    = _best_gold_rank(stage["aggregator_candidates"], gold, exp_pages)
    rnk_rank, rnk_priority = _best_gold_rank(stage["reranker_ranked"], gold, exp_pages)

    top1_chunk   = stage.get("top1_chunk") or ""
    top1_page    = stage.get("top1_page")
    p1_canonical = (top1_chunk in p1_chunks) or \
                   (bool(p1_chunks) and top1_page in exp_pages and rnk_priority == 1)
    p1_any       = (top1_chunk in exp_chunks) or \
                   (bool(exp_pages) and top1_page in exp_pages)

    if not retriever_hit:
        stage_label = "MISS_retriever";   fail_reason = "not_in_candidates"
    elif not aggregator_hit:
        stage_label = "MISS_aggregator";  fail_reason = _top_issue(
            stage["aggregator_candidates"], exp_pages, exp_chunks)
    elif not reranker_hit:
        stage_label = "MISS_reranker";    fail_reason = "dropped_by_reranker"
    elif p1_canonical:
        stage_label = "HIT_p1_canonical"; fail_reason = ""
    elif p1_any:
        stage_label = "HIT_p1_secondary"; fail_reason = "not_top1"
    else:
        stage_label = "HIT_not_p1";       fail_reason = "not_top1"

    fp_cat = _fp_category(fail_reason) if fail_reason else ""

    print(f"    [{difficulty:<10}]  type={stage['classifier_type']:<14}  "
          f"ret={'✓' if retriever_hit else '✗'}  "
          f"agg={'✓' if aggregator_hit else '✗'}  "
          f"rnk={'✓' if reranker_hit else '✗'}  "
          f"p@1={'✓' if p1_canonical else ('~' if p1_any else '✗')}  "
          f"rank={rnk_rank}  "
          f"{'→ '+fail_reason if fail_reason else ''}")

    return {
        "ci_id": ci_id, "ci_text": raw_ci.get("knownCI", ""),
        "ci_type": ci_type, "difficulty": difficulty,
        "gold_chunks": gold, "expected_pages": exp_pages,
        "classifier_type": stage["classifier_type"],
        "retriever_hit": retriever_hit, "aggregator_hit": aggregator_hit,
        "reranker_hit": reranker_hit,
        "precision_at_1": p1_canonical, "precision_at_1_any": p1_any,
        "agg_rank": agg_rank, "reranker_rank": rnk_rank, "rnk_priority": rnk_priority,
        "stage_label": stage_label, "fail_reason": fail_reason, "fp_category": fp_cat,
        "agg_top1_score": stage.get("agg_top1_score"),
        "agg_top2_score": stage.get("agg_top2_score"),
        "agg_score_gap":  stage.get("agg_score_gap"),
        "rnk_top1_score": stage.get("rnk_top1_score"),
        "rnk_top2_score": stage.get("rnk_top2_score"),
        "rnk_score_gap":  stage.get("rnk_score_gap"),
        "error": stage.get("error"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metrics computation
# ─────────────────────────────────────────────────────────────────────────────

def _score_dist(values: list) -> dict:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"mean": None, "median": None, "p25": None, "p75": None, "n": 0}
    s = sorted(clean)
    n = len(s)
    return {
        "mean":   round(statistics.mean(clean),   3),
        "median": round(statistics.median(clean), 3),
        "p25":    round(s[n // 4], 3),
        "p75":    round(s[3 * n // 4], 3),
        "n":      n,
    }


def _compute_metrics(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}

    def _k(key: str) -> int:
        return sum(1 for r in results if r.get(key))

    def _mrr(rank_key: str) -> float:
        rr = [1.0 / r[rank_key] for r in results if r.get(rank_key)]
        return round(sum(rr) / n, 3) if rr else 0.0

    ret_k = _k("retriever_hit"); agg_k = _k("aggregator_hit")
    rnk_k = _k("reranker_hit");  p1_k  = _k("precision_at_1")
    p1a_k = _k("precision_at_1_any")

    # Per-CI-type
    by_type: dict[str, list] = defaultdict(list)
    for r in results:
        by_type[r.get("ci_type", "UNKNOWN")].append(r)

    type_metrics: dict[str, dict] = {}
    for ci_type, tr in sorted(by_type.items()):
        nt  = len(tr)
        ret = sum(1 for r in tr if r["retriever_hit"])
        agg = sum(1 for r in tr if r["aggregator_hit"])
        rnk = sum(1 for r in tr if r["reranker_hit"])
        p1  = sum(1 for r in tr if r["precision_at_1"])
        type_metrics[ci_type] = {
            "n": nt,
            "retriever_recall":  _ci_str(ret, nt),
            "aggregator_recall": _ci_str(agg, nt),
            "reranker_recall":   _ci_str(rnk, nt),
            "precision_at_1":    _ci_str(p1,  nt),
            "mrr": round(sum(1.0 / r["reranker_rank"] for r in tr
                             if r.get("reranker_rank")) / nt, 3),
        }

    # Per-difficulty
    by_diff: dict[str, list] = defaultdict(list)
    for r in results:
        by_diff[r.get("difficulty", "Unknown")].append(r)

    diff_metrics: dict[str, dict] = {}
    for diff, dr in sorted(by_diff.items()):
        nd   = len(dr)
        p1d  = sum(1 for r in dr if r["precision_at_1"])
        rnkd = sum(1 for r in dr if r["reranker_hit"])
        diff_metrics[diff] = {
            "n": nd,
            "reranker_recall": _ci_str(rnkd, nd),
            "precision_at_1":  _ci_str(p1d,  nd),
        }

    # FP taxonomy
    fp_counts: dict[str, int] = defaultdict(int)
    for r in results:
        cat = r.get("fp_category", "")
        if cat:
            fp_counts[cat] += 1

    return {
        "n": n,
        "retriever_recall":   _ci_str(ret_k, n),
        "aggregator_recall":  _ci_str(agg_k, n),
        "reranker_recall":    _ci_str(rnk_k, n),
        "precision_at_1":     _ci_str(p1_k,  n),
        "precision_at_1_any": _ci_str(p1a_k, n),
        "mrr":                _mrr("reranker_rank"),
        "_raw": {"ret_k": ret_k, "agg_k": agg_k, "rnk_k": rnk_k,
                 "p1_k": p1_k, "p1a_k": p1a_k, "n": n},
        "by_ci_type":   type_metrics,
        "by_difficulty": diff_metrics,
        "fp_taxonomy":  dict(sorted(fp_counts.items(), key=lambda x: -x[1])),
        "score_distribution": {
            "agg_top1": _score_dist([r.get("agg_top1_score") for r in results]),
            "agg_gap":  _score_dist([r.get("agg_score_gap")  for r in results]),
            "rnk_top1": _score_dist([r.get("rnk_top1_score") for r in results]),
            "rnk_gap":  _score_dist([r.get("rnk_score_gap")  for r in results]),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_report(metrics: dict, run_meta: dict | None = None) -> None:
    w = 70
    print(f"\n{'═' * w}")
    print(f"  EVAL REPORT")
    if run_meta:
        print(f"  {run_meta.get('timestamp','')[:19]}  ·  {run_meta.get('ci_file','')}")
    print(f"{'═' * w}")
    m = metrics
    print(f"  CIs evaluated          : {m.get('n', 0)}")
    print(f"  Retriever recall       : {m.get('retriever_recall',  'n/a')}")
    print(f"  Aggregator recall      : {m.get('aggregator_recall', 'n/a')}")
    print(f"  Reranker recall        : {m.get('reranker_recall',   'n/a')}")
    print(f"  Precision@1 (canonical): {m.get('precision_at_1',    'n/a')}")
    print(f"  Precision@1 (any gold) : {m.get('precision_at_1_any','n/a')}")
    print(f"  MRR (reranker)         : {m.get('mrr', 0):.3f}")

    # Per CI-type
    print(f"\n{'─' * w}")
    print(f"  Per CI-type  (95% Wilson CI shown as ±)")
    print(f"{'─' * w}")
    print(f"  {'Type':<22}  {'N':>3}  {'Ret':>14}  {'Agg':>14}  {'Rnk':>14}  {'P@1':>14}")
    for ci_type, tm in sorted(m.get("by_ci_type", {}).items()):
        print(f"  {ci_type:<22}  {tm['n']:>3}  "
              f"{tm['retriever_recall']:>14}  {tm['aggregator_recall']:>14}  "
              f"{tm['reranker_recall']:>14}  {tm['precision_at_1']:>14}")

    # Per difficulty
    diff = m.get("by_difficulty", {})
    if diff:
        print(f"\n{'─' * w}")
        print(f"  Per CI difficulty")
        print(f"{'─' * w}")
        print(f"  {'Difficulty':<12}  {'N':>3}  {'Rnk recall':>14}  {'P@1':>14}")
        for d in ["Easy", "Medium", "Hard", "Impossible", "Unknown"]:
            dm = diff.get(d)
            if dm:
                print(f"  {d:<12}  {dm['n']:>3}  "
                      f"{dm['reranker_recall']:>14}  {dm['precision_at_1']:>14}")

    # FP taxonomy
    fp = m.get("fp_taxonomy", {})
    if fp:
        total_fp = sum(fp.values())
        print(f"\n{'─' * w}")
        print(f"  False-positive taxonomy  (total misses: {total_fp})")
        print(f"{'─' * w}")
        for cat, cnt in sorted(fp.items(), key=lambda x: -x[1])[:12]:
            bar = "█" * min(cnt, 30)
            pct = cnt / total_fp if total_fp else 0
            print(f"  {cat:<30}  {cnt:>3}  {pct:>5.1%}  {bar}")

    # Score distribution
    sd = m.get("score_distribution", {})
    if sd:
        print(f"\n{'─' * w}")
        print(f"  Score distribution")
        print(f"{'─' * w}")
        print(f"  {'Metric':<24}  {'Mean':>7}  {'Median':>7}  {'P25':>7}  {'P75':>7}")
        for key, label in [
            ("agg_top1", "Agg Top1 score"),
            ("agg_gap",  "Agg Top1−Top2 gap"),
            ("rnk_top1", "Reranker Top1 score"),
            ("rnk_gap",  "Reranker Top1−Top2 gap"),
        ]:
            d = sd.get(key, {})
            if d.get("n", 0) == 0:
                continue
            print(f"  {label:<24}  {d['mean']:>7.3f}  {d['median']:>7.3f}  "
                  f"{d['p25']:>7.3f}  {d['p75']:>7.3f}")

    print(f"{'═' * w}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Delta report
# ─────────────────────────────────────────────────────────────────────────────

def _delta_str(bp: float, ap: float, n: int) -> str:
    diff = ap - bp
    _, hi = _wilson_ci(round(bp * n), n)
    lo, _ = _wilson_ci(round(bp * n), n)
    hw = (hi - lo) / 2
    sign   = "+" if diff >= 0 else ""
    marker = "**" if abs(diff) > 2 * hw else (" *" if abs(diff) > hw else "  ")
    return f"{sign}{diff:+.1%}{marker}"


def _print_delta(before: dict, after: dict) -> None:
    bm = before.get("metrics", {}); am = after.get("metrics", {})
    brun = before.get("run", {}); arun = after.get("run", {})
    n    = bm.get("_raw", {}).get("n", 1)

    w = 72
    print(f"\n{'═' * w}")
    print(f"  DELTA REPORT")
    print(f"  BEFORE: {brun.get('timestamp','?')[:19]}")
    print(f"  AFTER:  {arun.get('timestamp','?')[:19]}")
    print(f"{'═' * w}")

    def _raw(m: dict, key: str) -> int:
        return m.get("_raw", {}).get(key, 0)

    print(f"\n  {'Metric':<24}  {'Before':>10}  {'After':>10}  {'Delta':>10}")
    print(f"  {'─'*24}  {'─'*10}  {'─'*10}  {'─'*10}")
    for label, key in [
        ("Retriever recall",  "ret_k"),
        ("Aggregator recall", "agg_k"),
        ("Reranker recall",   "rnk_k"),
        ("Precision@1",       "p1_k"),
        ("Precision@1 (any)", "p1a_k"),
    ]:
        bk = _raw(bm, key); ak = _raw(am, key)
        bp = bk / n; ap = ak / n
        print(f"  {label:<24}  {bp:>10.1%}  {ap:>10.1%}  {_delta_str(bp, ap, n):>10}")

    # Per CI-type P@1 delta
    bt = bm.get("by_ci_type", {}); at = am.get("by_ci_type", {})
    if bt or at:
        print(f"\n{'─' * w}")
        print(f"  Precision@1 per CI-type  (* suggestive, ** likely real)")
        print(f"{'─' * w}")
        print(f"  {'Type':<22}  {'Before':>8}  {'After':>8}  {'Delta':>10}")

        def _pct(s: str) -> float:
            try: return float(s.split("%")[0]) / 100.0
            except: return 0.0

        for ci_type in sorted(set(bt) | set(at)):
            brow = bt.get(ci_type, {}); arow = at.get(ci_type, {})
            nt   = brow.get("n", arow.get("n", 1))
            bp   = _pct(brow.get("precision_at_1", "0%"))
            ap   = _pct(arow.get("precision_at_1", "0%"))
            print(f"  {ci_type:<22}  {bp:>8.1%}  {ap:>8.1%}  {_delta_str(bp, ap, nt):>10}")

    # FP taxonomy delta
    bfp = bm.get("fp_taxonomy", {}); afp = am.get("fp_taxonomy", {})
    if bfp or afp:
        print(f"\n{'─' * w}")
        print(f"  False-positive taxonomy delta")
        print(f"{'─' * w}")
        print(f"  {'Category':<30}  {'Before':>6}  {'After':>6}  {'Δ':>6}")
        for cat in sorted(set(bfp) | set(afp),
                          key=lambda c: -(bfp.get(c, 0) + afp.get(c, 0))):
            bv = bfp.get(cat, 0); av = afp.get(cat, 0); d = av - bv
            print(f"  {cat:<30}  {bv:>6}  {av:>6}  {'+' if d>0 else ''}{d:>5}")

    print(f"\n  Legend: * = likely real   ** = very likely real")
    print(f"{'═' * w}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Sub-commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_annotate(args) -> None:
    with open(args.ci_file) as f:
        raw_cis = json.load(f)

    gt = _load_ground_truth(args.out)
    gt.setdefault("document_id", args.document_id)
    gt["version"] = "2"
    existing_ids  = {a["ci_id"] for a in gt.get("annotations", [])}

    ci_list = raw_cis if isinstance(raw_cis, list) else raw_cis.get("cis", [])
    max_cis = args.max_cis or len(ci_list)

    for idx, raw_ci in enumerate(ci_list[:max_cis]):
        ci_id   = raw_ci.get("id", idx)
        ci_text = raw_ci.get("knownCI", f"CI#{ci_id}")

        if ci_id in existing_ids and not args.force:
            print(f"\n  CI {ci_id}: already annotated (--force to re-annotate)")
            continue

        print(f"\n{'─' * 68}")
        print(f"  CI {ci_id}: {ci_text}")
        print(f"{'─' * 68}")

        enriched = _lookup_ci_from_index(raw_ci, ci_id)
        if enriched is None:
            enriched = enrich_ci(raw_ci, ci_id)

        stage = _run_pipeline_for_eval(enriched, args.document_id)

        print(f"\n  Type: {stage['classifier_type']}")
        print(f"  Top aggregator candidates:")
        for i, c in enumerate(stage["aggregator_candidates"][:5], 1):
            obj  = c.get("matched_object") or {}
            text = (obj.get("text") or "")[:120].replace("\n", " ")
            pg   = f"p{c.get('page_start')}" + (
                f"–{c['page_end']}" if c.get("page_end") != c.get("page_start") else "")
            print(f"    #{i}  {pg}  agg={c['agg_score']:.3f}  "
                  f"chunk={c.get('chunk_id','')[:28]}")
            print(f"         {text}…")

        print(f"\n  Top reranker results:")
        for i, r in enumerate(stage["reranker_ranked"][:5], 1):
            obj  = r.get("matched_object") or {}
            text = (obj.get("text") or "")[:120].replace("\n", " ")
            pg   = f"p{r.get('page_start')}" + (
                f"–{r['page_end']}" if r.get("page_end") != r.get("page_start") else "")
            print(f"    #{i}  {pg}  score={r.get('cross_encoder_score',0):.2f}  "
                  f"chunk={r.get('chunk_id','')[:28]}")
            print(f"         {text}…")

        exp_pages_str = input(
            "\n  Expected pages (comma-separated, blank=skip): ").strip()
        if not exp_pages_str:
            print("  Skipped.")
            continue

        exp_pages = [int(p.strip()) for p in exp_pages_str.split(",")
                     if p.strip().isdigit()]

        print("  Gold chunks — enter chunk IDs in priority order (blank=done).")
        print("  Format: <chunk_id> [label]")
        gold: list[dict] = []
        prio = 1
        while True:
            line = input(f"    Priority {prio} (blank=done): ").strip()
            if not line:
                break
            parts = line.split(None, 1)
            gold.append({"chunk_id": parts[0], "priority": prio,
                         "label": parts[1] if len(parts) > 1 else ""})
            prio += 1

        ci_type    = input(f"  CI type [{stage['classifier_type']}]: ").strip() \
                     or stage["classifier_type"]
        auto_diff  = _ci_difficulty(ci_text, ci_type)
        difficulty = input(
            f"  Difficulty [{auto_diff}] (Easy/Medium/Hard/Impossible, blank=auto): "
        ).strip() or None
        notes = input("  Notes (optional): ").strip()

        ann = {
            "ci_id": ci_id, "ci_text": ci_text, "ci_type": ci_type,
            "difficulty": difficulty, "gold_chunks": gold,
            "expected_pages": exp_pages,
            "expected_chunks": [g["chunk_id"] for g in gold],
            "notes": notes,
        }

        gt["annotations"] = [a for a in gt.get("annotations", []) if a["ci_id"] != ci_id]
        gt["annotations"].append(ann)
        existing_ids.add(ci_id)
        _save_ground_truth(gt, args.out)

    print(f"\nAnnotation complete.  {len(gt['annotations'])} CI(s) annotated.")


def cmd_eval(args) -> None:
    with open(args.ci_file) as f:
        raw_cis = json.load(f)

    ci_list  = raw_cis if isinstance(raw_cis, list) else raw_cis.get("cis", [])
    ci_by_id = {c.get("id", i): c for i, c in enumerate(ci_list)}

    gt   = _load_ground_truth(args.ground_truth)
    anns = gt.get("annotations", [])
    if not anns:
        print("No annotations found.  Run `annotate` first.")
        return

    max_cis = args.max_cis or len(anns)
    anns    = anns[:max_cis]
    print(f"\nEvaluating {len(anns)} annotated CI(s) against {args.document_id} …\n")

    per_ci: list[dict] = []

    _err_template: dict = {
        "retriever_hit": False, "aggregator_hit": False, "reranker_hit": False,
        "precision_at_1": False, "precision_at_1_any": False, "classifier_type": "?",
        "agg_rank": None, "reranker_rank": None, "rnk_priority": 0,
        "agg_top1_score": None, "agg_top2_score": None, "agg_score_gap": None,
        "rnk_top1_score": None, "rnk_top2_score": None, "rnk_score_gap": None,
    }

    def _run(ann: dict) -> dict:
        ci_id  = ann["ci_id"]
        raw_ci = ci_by_id.get(ci_id)
        if raw_ci is None:
            return {**_err_template, "ci_id": ci_id, "ci_text": ann.get("ci_text",""),
                    "ci_type": ann.get("ci_type","?"),
                    "difficulty": ann.get("difficulty","Unknown"),
                    "gold_chunks": [], "expected_pages": [],
                    "stage_label": "ERROR", "fail_reason": "ci_not_found",
                    "fp_category": "Annotation Error", "error": "CI not found"}
        return _eval_one_ci(raw_ci, ci_id, ann, args.document_id)

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed({ex.submit(_run, a): a for a in anns}):
                per_ci.append(fut.result())
    else:
        for ann in anns:
            per_ci.append(_run(ann))

    metrics = _compute_metrics(per_ci)
    _print_report(metrics)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({
            "run": {
                "timestamp":   datetime.now().isoformat(),
                "document_id": args.document_id,
                "ci_file":     str(args.ci_file),
                "ground_truth": str(args.ground_truth),
            },
            "metrics": metrics,
            "per_ci":  per_ci,
        }, f, indent=2, default=str)
    print(f"Results saved → {out_path}")


def cmd_report(args) -> None:
    with open(args.results) as f:
        data = json.load(f)
    _print_report(data["metrics"], run_meta=data.get("run"))


def cmd_delta(args) -> None:
    with open(args.before) as f: before = json.load(f)
    with open(args.after)  as f: after  = json.load(f)
    _print_delta(before, after)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eval harness — per-stage retrieval evaluation.")
    parser.add_argument("--document-id", default=DOCUMENT_ID)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ann = sub.add_parser("annotate")
    p_ann.add_argument("--ci-file",  required=True)
    p_ann.add_argument("--out",      default="localfiles/eval/ground_truth.json")
    p_ann.add_argument("--max-cis",  type=int, default=0)
    p_ann.add_argument("--force",    action="store_true")

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--ci-file",      required=True)
    p_eval.add_argument("--ground-truth", required=True)
    p_eval.add_argument("--out",
        default=f"localfiles/eval/eval_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    p_eval.add_argument("--max-cis",  type=int, default=0)
    p_eval.add_argument("--workers",  type=int, default=4)

    p_rep = sub.add_parser("report")
    p_rep.add_argument("--results", required=True)

    p_del = sub.add_parser("delta")
    p_del.add_argument("--before", required=True)
    p_del.add_argument("--after",  required=True)

    args = parser.parse_args()
    {"annotate": cmd_annotate, "eval": cmd_eval,
     "report": cmd_report, "delta": cmd_delta}[args.command](args)


if __name__ == "__main__":
    main()
