"""
search_new.py — Stage-Parallel Search Pipeline
===============================================
Runs the same pipeline as search.py but with a different parallelism model:

  search.py     → CI-level parallelism  (each CI runs full pipeline in one thread)
                  stage timings = summed CPU time across all workers (misleading)

  search_new.py → Stage-level parallelism (all CIs run each stage together)
                  stage timings = real wall-clock (max CI time at that stage)

Execution flow:
  ① Load enriched CIs (parallel OpenSearch lookups)
  ② Classify       — all CIs in parallel
  ③ Retrieve       — all CIs in parallel  (I/O bound, great for parallelism)
  ④ Aggregate      — all CIs in parallel
  ⑤ Context Expand — all CIs in parallel  (slowest stage, ~120s wall-clock)
  ⑥ Rerank         — all CIs in parallel  (Bedrock)
  ⑦ LLM Verify     — all CIs in parallel  (Bedrock, fresh module per CI)
  ⑧ Highlight Ext  — all CIs in parallel
  ⑨ Merge          — all CIs in parallel
  ⑩ Evidence Class — all CIs in parallel  (Bedrock)

Output JSON: same format as search.py + per-stage wall_clock_s in timing_summary.

Usage:
    python tests/search_new.py --max-cis 34 --workers 10
    python tests/search_new.py --ci-index 0 --verbose
"""
from __future__ import annotations

import importlib.util
import sys
import threading
import time
import types
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

# ── Import shared config + helpers from search.py ─────────────────────────────
import search as _S

_load                    = _S._load
_build_os_client         = _S._build_os_client
_inject_os               = _S._inject_os
_lookup_ci_from_index    = _S._lookup_ci_from_index
_classify_evidence       = _S._classify_evidence
_classify_evidence_batch = _S._classify_evidence_batch
_calibrate_evidence      = _S._calibrate_evidence
_numeric_gate_pattern = _S._numeric_gate_pattern
_NUMERIC_GATE_TYPES   = _S._NUMERIC_GATE_TYPES
RETRIEVER_MAP         = _S.RETRIEVER_MAP
_print_cost_estimate  = _S._print_cost_estimate
parse_args            = _S.parse_args
_clean_result         = _S._clean_result
_object_type_stats    = _S._object_type_stats

logger         = _S.logger
VERIFIER_MODEL = _S.VERIFIER_MODEL
OPENSEARCH_ENDPOINT = _S.OPENSEARCH_ENDPOINT

_HAIKU_INPUT_PRICE_PER_TOKEN  = _S._HAIKU_INPUT_PRICE_PER_TOKEN
_HAIKU_OUTPUT_PRICE_PER_TOKEN = _S._HAIKU_OUTPUT_PRICE_PER_TOKEN
_EST_INPUT_TOKENS_PER_CAND    = _S._EST_INPUT_TOKENS_PER_CAND
_EST_OUTPUT_TOKENS_PER_CAND   = _S._EST_OUTPUT_TOKENS_PER_CAND
_EST_EC_INPUT_TOKENS_PER_HIT  = _S._EST_EC_INPUT_TOKENS_PER_HIT
_EST_EC_OUTPUT_TOKENS_PER_HIT = _S._EST_EC_OUTPUT_TOKENS_PER_HIT


# ── Fresh (non-cached) module loader — for thread-safe monkey-patching ────────
def _load_fresh(rel_path: str) -> types.ModuleType:
    """Load a fresh uncached module instance so each thread can safely monkey-patch."""
    lf_path = ROOT / "lambdas" / rel_path / "lambda_function.py"
    alias   = f"_fresh_{rel_path.replace('/', '_')}_{threading.get_ident()}"
    spec    = importlib.util.spec_from_file_location(alias, lf_path)
    mod     = importlib.util.module_from_spec(spec)
    lf_dir  = str(lf_path.parent)
    if lf_dir not in sys.path:
        sys.path.insert(0, lf_dir)
    spec.loader.exec_module(mod)
    return mod


# ── Module-level helpers (replicated from run_search for stage functions) ──────
_EVIDENCE_RANK: dict[str, int] = {
    "DIRECT": 0, "SUPPORTING": 1,
    "RELATED_OBJECTIVE": 2, "RELATED_PROTOCOL": 2,
    "RELATED_DOSE": 3, "RELATED_POPULATION": 3,
    "RELATED_SAFETY": 3, "RELATED_EFFICACY": 3,
    "RELATED_DEFINITION": 4,
    "SAME_STUDY": 1, "SAME_PROTOCOL": 2, "SAME_OBJECTIVE": 2,
    "SAME_ENDPOINT": 3, "SAME_POPULATION": 3, "SAME_MECHANISM": 3,
    "BACKGROUND": 4, "UNRELATED": 9,
}

def _is_related(ev: str) -> bool:
    return ev.startswith("SAME_") or ev.startswith("RELATED_") or ev == "BACKGROUND"

_CONF_THRESHOLD = 0.2

def _candidate_confidence(c: dict) -> float:
    agg_s   = c.get("agg_score", 0.0)
    zero_id = c.get("zero_id_pen",    0.0)
    zero_en = c.get("zero_enrich_pen", 0.0)
    return round(
        0.5 * min(max(agg_s * 2.0, 0.0), 1.0)
        + 0.3 * max(0.0, 1.0 + zero_id / 0.4)
        + 0.2 * max(0.0, 1.0 + zero_en / 0.25),
        3,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage functions — each takes req dict, updates req["_st"], returns req
# ─────────────────────────────────────────────────────────────────────────────

import gzip as _gzip


def _s4_cache_path(args, n_cis: int) -> "Path":
    key = f"{Path(args.ci_file).stem}__{args.document_id}__{n_cis}"
    return ROOT / ".cache" / "search_expand" / f"{key}.json.gz"


def _save_s4_cache(all_reqs: list[dict], path: "Path") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save = []
    for r in all_reqs:
        r2 = {k: v for k, v in r.items()
              if k not in ("_st", "_failed", "_early_exit", "_ci_idx")}
        if isinstance(r2.get("_diagnose_ci_ids"), set):
            r2["_diagnose_ci_ids"] = list(r2["_diagnose_ci_ids"])
        save.append(r2)
    with _gzip.open(path, "wt", encoding="utf-8") as fh:
        import json as _j
        _j.dump(save, fh)
    print(f"  [cache] saved {len(save)} expanded CIs → {path.name}")


def _load_s4_cache(path: "Path") -> list[dict]:
    with _gzip.open(path, "rt", encoding="utf-8") as fh:
        import json as _j
        saved = _j.load(fh)
    all_reqs = []
    for r in saved:
        r["_st"]              = {}
        r["_failed"]          = False
        r["_early_exit"]      = False
        r["_ci_idx"]          = 0
        if isinstance(r.get("_diagnose_ci_ids"), list):
            r["_diagnose_ci_ids"] = set(r["_diagnose_ci_ids"])
        all_reqs.append(r)
    return all_reqs


def _s1_classify(req: dict) -> dict:
    if req.get("_failed") or req.get("_early_exit"):
        return req
    t0  = time.perf_counter()
    mod = _load("search/classifier", "search_classifier")
    req = mod._process(req)
    req["_st"]["classifier"] = round(time.perf_counter() - t0, 3)
    return req


def _s2_retrieve(req: dict) -> dict:
    if req.get("_failed") or req.get("_early_exit"):
        return req
    strategies = req.get("classification", {}).get("strategies", list(RETRIEVER_MAP.keys()))
    valid = [(s, RETRIEVER_MAP[s]) for s in strategies if s in RETRIEVER_MAP]
    if not valid:
        req["retriever_results"]       = []
        req["_st"]["retrievers"]       = {}
        req["_st"]["retrievers_total"] = 0.0
        return req

    def _run_one(s_path: tuple[str, str]) -> tuple[str, dict, float]:
        strategy, path = s_path
        mod = _load(path, f"search_{strategy}")
        _inject_os(mod)
        t0 = time.perf_counter()
        result = mod._process(req)
        elapsed = round(time.perf_counter() - t0, 3)
        return strategy, result, elapsed

    with ThreadPoolExecutor(max_workers=len(valid)) as pool:
        raw = list(pool.map(_run_one, valid))

    timings: dict[str, float] = {}
    retriever_results = []
    for strategy, result, elapsed in raw:
        timings[strategy] = elapsed
        # Merge vector sub-timings (body/heading/chunk) directly into the timings dict
        for k_sub, v_sub in (result.pop("_sub_timings", None) or {}).items():
            timings[k_sub] = v_sub
        retriever_results.append(result)
    req["retriever_results"]       = retriever_results
    req["_st"]["retrievers"]       = timings
    req["_st"]["retrievers_total"] = round(sum(timings.values()), 3)
    return req


def _s3_aggregate(req: dict) -> dict:
    if req.get("_failed") or req.get("_early_exit"):
        return req
    t0  = time.perf_counter()
    mod = _load("search/aggregator", "search_aggregator")
    req = mod._process(req)
    req["_st"]["aggregator"] = round(time.perf_counter() - t0, 3)
    if not req.get("candidates"):
        req["_early_exit"] = True
        req.setdefault("final_hits", [])
        req.setdefault("verified_candidates", [])
    return req


def _s4_context_expand(req: dict) -> dict:
    if req.get("_failed") or req.get("_early_exit"):
        return req
    t0  = time.perf_counter()
    mod = _load("search/context_expander", "search_expander")
    _inject_os(mod)
    req = mod._process(req)
    req["_st"]["context_expander"] = round(time.perf_counter() - t0, 3)
    return req


def _s5_rerank(req: dict, skip_rerank: bool = False) -> dict:
    if req.get("_failed") or req.get("_early_exit"):
        return req
    t0 = time.perf_counter()
    if skip_rerank:
        expanded = req.get("expanded_candidates", [])
        req["ranked_candidates"] = [
            # Set score above MIN_RERANK_SCORE (3.0) so all candidates reach Claude
            {**c, "cross_encoder_score": 10.0}
            for c in expanded
        ]
    else:
        mod = _load("search/reranker", "search_reranker")
        req = mod._process(req)
    req["_st"]["reranker"] = round(time.perf_counter() - t0, 3)

    # ── 5.5 Numeric text-presence gate ──────────────────────────────────────
    ci_type = (req.get("classification") or {}).get("ci_type", "") or ""
    if ci_type in _NUMERIC_GATE_TYPES:
        pat = _numeric_gate_pattern(req["ci"])
        if pat is not None:
            passed, gated = [], []
            for c in req.get("ranked_candidates", []):
                txt = ((c.get("context") or {}).get("current_text", "") or c.get("snippet", ""))
                (passed if pat.search(txt) else gated).append(c)
            req["ranked_candidates"] = passed

    # ── 5.6 Chunk deduplication ──────────────────────────────────────────────
    by_chunk: dict[str, dict] = {}
    for c in req.get("ranked_candidates", []):
        cid = c.get("chunk_id") or c.get("id") or ""
        if cid not in by_chunk or c.get("agg_score", 0) > by_chunk[cid].get("agg_score", 0):
            by_chunk[cid] = c
    req["ranked_candidates"] = list(by_chunk.values())

    # ── 5.7 Candidate confidence gate ────────────────────────────────────────
    passed, gated = [], []
    for c in req.get("ranked_candidates", []):
        if _candidate_confidence(c) >= _CONF_THRESHOLD:
            passed.append(c)
        else:
            gated.append({**c, "verdict": "NO", "reason": "candidate_confidence_gate"})
    req["ranked_candidates"] = passed
    req.setdefault("skipped_hits", []).extend(gated)
    return req


def _s6_llm_verify(req: dict, skip_verify: bool = False) -> dict:
    if req.get("_failed") or req.get("_early_exit"):
        req["_st"].update({
            "n_candidates_to_verifier": 0,
            "per_verifier_call_s":      {},
            "actual_verifier_tokens":   {"input": 0, "output": 0},
            "llm_verifier":             0.0,
        })
        return req
    req["_st"]["n_candidates_to_verifier"] = len(req.get("ranked_candidates", []))
    t0 = time.perf_counter()

    if skip_verify:
        req["verified_candidates"] = [
            {**c, "verdict": "MAYBE", "reason": "skipped", "confidence": 0.5}
            for c in req.get("ranked_candidates", [])
        ]
        req["_st"]["per_verifier_call_s"]    = {}
        req["_st"]["actual_verifier_tokens"] = {"input": 0, "output": 0}
    else:
        # Load a FRESH verifier module per CI so each thread can safely monkey-patch
        # without racing against other CI threads patching the same shared module.
        verifier = _load_fresh("search/llm_verifier")
        _inject_os(verifier)
        call_times:  list[float] = []
        call_tokens: list[dict]  = []

        # Patch sequential _verify (fallback path)
        orig = verifier._verify
        def _timed(*a, **kw):
            _t = time.perf_counter()
            _r = orig(*a, **kw)
            call_times.append(round(time.perf_counter() - _t, 3))
            call_tokens.append(_r.get("_tokens", {"input": 0, "output": 0}))
            return _r
        verifier._verify = _timed

        # Patch _verify_batch to capture batch-level call timing + tokens
        orig_batch = verifier._verify_batch
        def _timed_batch(*a, **kw):
            _t = time.perf_counter()
            results = orig_batch(*a, **kw)
            call_times.append(round(time.perf_counter() - _t, 3))
            call_tokens.append({
                "input":  sum(r.get("_tokens", {}).get("input", 0) for r in results),
                "output": sum(r.get("_tokens", {}).get("output", 0) for r in results),
            })
            return results
        verifier._verify_batch = _timed_batch

        req = verifier._process(req)
        req["_st"]["per_verifier_call_s"]    = {i + 1: t for i, t in enumerate(call_times)}
        req["_st"]["actual_verifier_tokens"] = {
            "input":  sum(t["input"]  for t in call_tokens),
            "output": sum(t["output"] for t in call_tokens),
        }

    req["_st"]["llm_verifier"] = round(time.perf_counter() - t0, 3)
    return req


def _s7_highlight_extract(req: dict) -> dict:
    if req.get("_failed") or req.get("_early_exit"):
        req["_st"]["highlight_extractor"] = 0.0
        return req
    t0  = time.perf_counter()
    mod = _load("search/highlight_extractor", "search_highlight_extractor")
    req = mod._process(req)
    req["_st"]["highlight_extractor"] = round(time.perf_counter() - t0, 3)
    return req


def _s8_merge(req: dict) -> dict:
    if req.get("_failed") or req.get("_early_exit"):
        req["_st"]["merger"] = 0.0
        req.setdefault("final_hits", [])
        return req
    t0  = time.perf_counter()
    mod = _load("search/merger", "search_merger")
    req = mod._process(req)
    req["_st"]["merger"] = round(time.perf_counter() - t0, 3)
    return req


def _s9_evidence_classify(req: dict, skip_verify: bool = False) -> dict:
    if req.get("_failed") or req.get("_early_exit"):
        req["_st"].update({
            "evidence_classification": 0.0,
            "per_ec_call_s":           {},
            "n_ec_calls":              0,
            "actual_ec_tokens":        {"input": 0, "output": 0},
        })
        return req
    t0        = time.perf_counter()
    ec_times: list[float]         = []
    ec_tokens: dict[str, int]     = {"input": 0, "output": 0}
    hits      = req.get("final_hits", [])
    ci_text   = req["ci"].get("knownCI", "")
    doc_ctx   = req.get("document_context", {})

    if hits and not skip_verify:
        to_classify = [h for h in hits if h.get("verdict") in ("YES", "MAYBE")]
        if to_classify:
            _t_batch = time.perf_counter()
            classified = _classify_evidence_batch(ci_text, to_classify, doc_ctx)
            ec_times.append(round(time.perf_counter() - _t_batch, 3))
            for hit, ec in zip(to_classify, classified):
                tok = ec.pop("_ec_tokens", {"input": 0, "output": 0})
                ec_tokens["input"]  += tok["input"]
                ec_tokens["output"] += tok["output"]
                ec_clean = {k: ec[k] for k in ("evidence_type", "evidence_confidence", "evidence_reason") if k in ec}
                ec_clean = _calibrate_evidence(hit, ec_clean)
                hit.update(ec_clean)
                if _is_related(ec_clean["evidence_type"]):
                    hit["verdict"] = "RELATED"
                elif ec_clean["evidence_type"] == "UNRELATED":
                    hit["verdict"] = "NO"
        hits.sort(key=lambda h: (
            _EVIDENCE_RANK.get(h.get("evidence_type", "BACKGROUND"), 4),
            -h.get("evidence_confidence", 0.0),
            -h.get("cross_encoder_score", 0.0),
            -h.get("highlight_score", 0.0),
        ))
        req["final_hits"] = hits

    req["_st"].update({
        "evidence_classification": round(time.perf_counter() - t0, 3),
        "per_ec_call_s":           {i + 1: t for i, t in enumerate(ec_times)},
        "n_ec_calls":              len(ec_times),
        "actual_ec_tokens":        ec_tokens,
    })
    return req


# ─────────────────────────────────────────────────────────────────────────────
# Timing display + result saving
# ─────────────────────────────────────────────────────────────────────────────

def _print_stage_timing(stage_wall: dict[str, float], all_results: list[dict],
                        total_wall: float) -> None:
    """Print per-stage wall-clock table."""
    n = len(all_results) or 1
    all_t = [r.get("timings", {}) for r in all_results]

    W1, W2 = 28, 12
    print(f"\n{'═' * 46}")
    print(f"  STAGE WALL-CLOCK  ({n} CIs,  {total_wall:.1f}s total)")
    print(f"{'═' * 46}")
    print(f"  {'Stage':<{W1}}  {'Wall-clock':>{W2}}")
    print(f"  {'─' * W1}  {'─' * W2}")

    stage_map = [
        ("S1:classify",          "classifier"),
        ("S2:retrieve",          "retrievers_total"),
        ("S3:aggregate",         "aggregator"),
        ("S4:context_expand",    "context_expander"),
        ("S5:rerank",            "reranker"),
        ("S6:llm_verify",        "llm_verifier"),
        ("S7:highlight",         "highlight_extractor"),
        ("S8:merge",             "merger"),
        ("S9:evidence_classify", "evidence_classification"),
    ]
    for stage_key, _ in stage_map:
        wc = stage_wall.get(stage_key, 0.0)
        print(f"  {stage_key:<{W1}}  {wc:>{W2}.2f}s")
        if stage_key == "S2:retrieve":
            # Per-CI retriever timings table
            all_strategies = sorted({s for t in all_t for s in t.get("retrievers", {})})
            if all_strategies:
                col = 8
                header = f"    {'CI':<5}" + "".join(f"{s[:col]:>{col+2}}" for s in all_strategies)
                print(header)
                print(f"    {'─'*5}" + "".join(f"{'─'*(col+2)}" for _ in all_strategies))
                for idx, (t, r) in enumerate(zip(all_t, all_results), 1):
                    ci_id = (r.get("ci") or {}).get("id", idx)
                    row = f"    {ci_id!s:<5}"
                    for s in all_strategies:
                        v = t.get("retrievers", {}).get(s)
                        row += f"{(f'{v:.2f}s' if v is not None else '-'):>{col+2}}"
                    print(row)

    print(f"  {'─' * W1}  {'─' * W2}")
    print(f"  {'TOTAL':<{W1}}  {total_wall:>{W2}.2f}s")
    print(f"{'═' * 46}")


def _retrieval_origin_summary(all_results: list[dict]) -> dict[str, int]:
    """Count final hits by retrieval_origin across all CIs.

    Each key is  '<retrievers>/<retrieved_unit>'  e.g. 'vector/chunk', 'vector/paragraph',
    'bm25+literal/sentence'.  Sorted by count descending.

    Uses context_expander's retrieval_origin field ("direct_X" vs "via_chunk_X") to
    determine the retrieved unit type:
      via_chunk_*  → unit_type = "chunk"  (retriever found a chunk; CE expanded to object)
      direct_X     → unit_type = X        (retriever found this object directly from index)
    Falls back to retrieved_type (set by vector_retriever) or matched_object.type.
    This means 'vector/chunk' and 'vector/sentence' are now correctly distinguished:
      vector/chunk    = vector_search_chunks lane → CE assigned best sentence/para within chunk
      vector/sentence = _vector_search_objects returned a sentence object from semantic-objects
    """
    import collections
    counts: collections.Counter = collections.Counter()
    for r in all_results:
        for h in r.get("final_hits", []):
            obj       = h.get("matched_object") or {}
            sources   = h.get("sources", [])
            ce_origin = h.get("retrieval_origin", "")  # context_expander's value
            if ce_origin.startswith("via_chunk_"):
                unit_type = "chunk"
            elif ce_origin.startswith("direct_"):
                unit_type = ce_origin[len("direct_"):]
            else:
                unit_type = h.get("retrieved_type") or (obj.get("type") or "unknown")
            key = ("+".join(sorted(sources)) if sources else "unknown") + "/" + unit_type
            counts[key] += 1
    return dict(counts.most_common())


def _save_results_staged(all_results: list[dict], args, out_path: Path,
                         wall_time: float, stage_wall: dict[str, float],
                         rerank_summary: dict | None = None) -> None:
    """Write results JSON with per-stage wall_clock_s in timing_summary."""
    import json
    from pathlib import Path

    output = {
        "run": {
            "timestamp":      datetime.now().isoformat(),
            "document_id":    args.document_id,
            "ci_file":        str(args.ci_file),
            "opensearch":     OPENSEARCH_ENDPOINT,
            "skip_rerank":    args.skip_rerank,
            "skip_verify":    args.skip_verify,
            "parallelism":    "stage-parallel",
        },
        "summary": {
            "cis_searched":     len(all_results),
            "total_final_hits": sum(len(r.get("final_hits", [])) for r in all_results),
            "direct_hits":      sum(
                1 for r in all_results for h in r.get("final_hits", [])
                if h.get("evidence_type") == "DIRECT"
            ),
            "same_study_hits":  sum(
                1 for r in all_results for h in r.get("final_hits", [])
                if h.get("evidence_type") == "SAME_STUDY"
            ),
            "related_hits":     sum(
                1 for r in all_results for h in r.get("final_hits", [])
                if (h.get("evidence_type") or "").startswith("SAME_")
                or (h.get("evidence_type") or "").startswith("RELATED_")
                or h.get("evidence_type") == "BACKGROUND"
            ),
            "related_breakdown": {
                sub: sum(
                    1 for r in all_results for h in r.get("final_hits", [])
                    if h.get("evidence_type") == f"RELATED_{sub}"
                    or h.get("evidence_type") == f"SAME_{sub}"
                )
                for sub in (
                    "OBJECTIVE", "PROTOCOL", "DOSE", "POPULATION",
                    "SAFETY", "EFFICACY", "DEFINITION",
                )
            },
            "background_hits":  sum(
                1 for r in all_results for h in r.get("final_hits", [])
                if h.get("evidence_type") == "BACKGROUND"
            ),
            "no_hits":          sum(
                1 for r in all_results for h in r.get("final_hits", [])
                if h.get("verdict") in ("NO", "SKIP")
            ),
            "total_rejected":   sum(
                1 for r in all_results
                for v in r.get("verified_candidates", [])
                if v.get("verdict") == "NO"
            ),
            "total_skipped":    sum(
                1 for r in all_results
                for v in r.get("verified_candidates", [])
                if v.get("verdict") == "SKIP"
            ),
            "object_type_stats": _object_type_stats(all_results),
        },
        "reranker_summary": rerank_summary or {},
        # retrieval_origin_summary: for every final hit, which retriever-lane × object-type
        # combination found it.  Key format: "bm25/sentence", "vector/paragraph", etc.
        # Multi-source hits use "+"-joined source names: "bm25+vector/sentence".
        # Use this to compare Variant A vs B vs C — which lanes lost hits.
        "retrieval_origin_summary": _retrieval_origin_summary(all_results),
        "results": [_clean_result(r) for r in all_results],
    }

    all_t = [r.get("timings", {}) for r in all_results]
    n     = len(all_results) or 1

    output["timing_summary"] = {
        "classifier":              {"wall_clock_s": stage_wall.get("S1:classify", 0.0)},
        "retrievers_total":        {"wall_clock_s": stage_wall.get("S2:retrieve", 0.0)},
        "per_ci_retrievers": [
            {"ci_id": (r.get("ci") or {}).get("id", i), "retrievers": t.get("retrievers", {})}
            for i, (r, t) in enumerate(zip(all_results, all_t), 1)
        ],
        "aggregator":              {"wall_clock_s": stage_wall.get("S3:aggregate", 0.0)},
        "context_expander":        {"wall_clock_s": stage_wall.get("S4:context_expand", 0.0)},
        "reranker":                {"wall_clock_s": stage_wall.get("S5:rerank", 0.0)},
        "llm_verifier":            {"wall_clock_s": stage_wall.get("S6:llm_verify", 0.0)},
        "highlight_extractor":     {"wall_clock_s": stage_wall.get("S7:highlight", 0.0)},
        "merger":                  {"wall_clock_s": stage_wall.get("S8:merge", 0.0)},
        "evidence_classification": {"wall_clock_s": stage_wall.get("S9:evidence_classify", 0.0)},
        "total":                   {"wall_clock_s": round(wall_time, 3)},
    }

    # Cost estimate (identical logic to search.py)
    n_to_verifier  = sum(t.get("n_candidates_to_verifier", 0) for t in all_t)
    n_actual_calls = sum(len(t.get("per_verifier_call_s", {})) for t in all_t)

    v_actual_in  = sum(t.get("actual_verifier_tokens", {}).get("input",  0) for t in all_t)
    v_actual_out = sum(t.get("actual_verifier_tokens", {}).get("output", 0) for t in all_t)
    if v_actual_in > 0:
        v_in, v_out, v_label = v_actual_in, v_actual_out, "actual"
    else:
        v_in, v_out, v_label = (n_actual_calls * _EST_INPUT_TOKENS_PER_CAND,
                                 n_actual_calls * _EST_OUTPUT_TOKENS_PER_CAND, "est.")
    v_cost = v_in * _HAIKU_INPUT_PRICE_PER_TOKEN + v_out * _HAIKU_OUTPUT_PRICE_PER_TOKEN

    n_ec_calls    = sum(t.get("n_ec_calls", 0) for t in all_t)
    ec_actual_in  = sum(t.get("actual_ec_tokens", {}).get("input",  0) for t in all_t)
    ec_actual_out = sum(t.get("actual_ec_tokens", {}).get("output", 0) for t in all_t)
    if ec_actual_in > 0:
        ec_in, ec_out, ec_label = ec_actual_in, ec_actual_out, "actual"
    else:
        ec_in, ec_out, ec_label = (n_ec_calls * _EST_EC_INPUT_TOKENS_PER_HIT,
                                    n_ec_calls * _EST_EC_OUTPUT_TOKENS_PER_HIT, "est.")
    ec_cost = ec_in * _HAIKU_INPUT_PRICE_PER_TOKEN + ec_out * _HAIKU_OUTPUT_PRICE_PER_TOKEN

    output["cost_estimate"] = {
        "model": VERIFIER_MODEL,
        "llm_verifier": {
            "candidates_passed_to_verifier": n_to_verifier,
            "skipped_below_threshold":       n_to_verifier - n_actual_calls,
            "actual_bedrock_calls":          n_actual_calls,
            "input_tokens":                  v_in,
            "output_tokens":                 v_out,
            "total_tokens":                  v_in + v_out,
            "token_source":                  v_label,
            "est_cost_usd":                  round(v_cost, 4),
        },
        "evidence_classification": {
            "bedrock_calls":  n_ec_calls,
            "input_tokens":   ec_in,
            "output_tokens":  ec_out,
            "total_tokens":   ec_in + ec_out,
            "token_source":   ec_label,
            "est_cost_usd":   round(ec_cost, 4),
        },
        "combined_est_cost_usd": round(v_cost + ec_cost, 4),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _st_to_timings(st: dict) -> dict:
    """Convert stage-timings dict to the search.py-compatible timings format."""
    cpu_total = sum([
        st.get("classifier",            0.0),
        st.get("retrievers_total",      0.0),
        st.get("aggregator",            0.0),
        st.get("context_expander",      0.0),
        st.get("reranker",              0.0),
        st.get("llm_verifier",          0.0),
        st.get("highlight_extractor",   0.0),
        st.get("merger",                0.0),
        st.get("evidence_classification", 0.0),
    ])
    return {
        "classifier":              st.get("classifier",            0.0),
        "retrievers":              st.get("retrievers",            {}),
        "retrievers_total":        st.get("retrievers_total",      0.0),
        "aggregator":              st.get("aggregator",            0.0),
        "context_expander":        st.get("context_expander",      0.0),
        "reranker":                st.get("reranker",              0.0),
        "n_candidates_to_verifier": st.get("n_candidates_to_verifier", 0),
        "per_verifier_call_s":     st.get("per_verifier_call_s",  {}),
        "actual_verifier_tokens":  st.get("actual_verifier_tokens", {"input": 0, "output": 0}),
        "llm_verifier":            st.get("llm_verifier",          0.0),
        "highlight_extractor":     st.get("highlight_extractor",   0.0),
        "merger":                  st.get("merger",                0.0),
        "evidence_classification": st.get("evidence_classification", 0.0),
        "per_ec_call_s":           st.get("per_ec_call_s",         {}),
        "n_ec_calls":              st.get("n_ec_calls",            0),
        "actual_ec_tokens":        st.get("actual_ec_tokens",      {"input": 0, "output": 0}),
        "total":                   round(cpu_total, 3),
    }


def main() -> None:
    import json
    args = parse_args()

    # ── Load CIs ──────────────────────────────────────────────────────────────
    from pathlib import Path as _Path
    ci_path = _Path(args.ci_file)
    if not ci_path.exists():
        print(f"ERROR: CI file not found: {ci_path}")
        sys.exit(1)
    with ci_path.open() as fh:
        raw_cis = json.load(fh)
    if isinstance(raw_cis, dict):
        raw_cis = list(raw_cis.values())
    if not isinstance(raw_cis, list):
        raw_cis = [raw_cis]
    if args.ci_index is not None:
        raw_cis = [raw_cis[args.ci_index]]
    else:
        raw_cis = raw_cis[: args.max_cis]

    print(f"\nSearch Pipeline Test  [stage-parallel]")
    print(f"  Document  : {args.document_id}")
    print(f"  CIs loaded: {len(raw_cis)} from {ci_path.name}")
    print(f"  Skip rerank: {args.skip_rerank}   skip verify: {args.skip_verify}")

    doc_context = _S._load_document_context(args.document_id)
    if doc_context:
        print(f"  Doc drug  : {', '.join(doc_context.get('primary_drugs', [])[:2])}"
              f"  | Study: {', '.join(doc_context.get('study_ids', [])[:1])}")

    print("\nConnecting to OpenSearch …")
    os_client = _build_os_client()
    info = os_client.info()
    print(f"  Cluster: {info.get('cluster_name')}  version: {info['version']['number']}")

    n_workers = min(args.workers, len(raw_cis))
    print(f"  Workers   : {n_workers}")

    # ── Load enriched CIs (parallel) ─────────────────────────────────────────
    print(f"\n  [S0:load]  Loading {len(raw_cis)} enriched CIs …")
    t_load = time.perf_counter()

    def _load_ci(item: tuple[int, dict]) -> dict | None:
        idx, raw_ci = item
        ci_id    = raw_ci.get("id", idx)
        enriched = _lookup_ci_from_index(raw_ci, ci_id)
        if enriched is None:
            print(f"  [CI {ci_id}] not found in ci-objects — skipping")
            return None
        return {
            "search_id":        f"srch-{__import__('uuid').uuid4().hex[:8]}",
            "document_id":      args.document_id,
            "ci":               enriched,
            "document_context": doc_context or {},
            "_diagnose_ci_ids": set(),
            "_st":              {},
            "_failed":          False,
            "_early_exit":      False,
            "_ci_idx":          idx,
        }

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        loaded = list(pool.map(_load_ci, enumerate(raw_cis)))
    all_reqs = [r for r in loaded if r is not None]
    print(f"  [S0:load]  {len(all_reqs)}/{len(raw_cis)} loaded  ({time.perf_counter()-t_load:.1f}s)")

    if not all_reqs:
        print("  No CIs loaded — exiting.")
        return

    # ── S4 cache: skip S1-S4 if --use-cache and cache exists ─────────────────
    use_cache = getattr(args, "use_cache", False)
    cache_path = _s4_cache_path(args, len(all_reqs))
    s4_loaded_from_cache = False
    if use_cache and cache_path.exists():
        print(f"  [cache] loading S4 from {cache_path.name} — skipping S1-S4")
        all_reqs = _load_s4_cache(cache_path)
        s4_loaded_from_cache = True

    # ── Stage-parallel pipeline ───────────────────────────────────────────────
    # max_workers=1 for the reranker serializes CrossEncoder.predict() calls,
    # which prevents PyTorch/BLAS non-determinism from concurrent inference.
    STAGES = [
        ("S1:classify",          lambda r: _s1_classify(r),                         n_workers),
        ("S2:retrieve",          lambda r: _s2_retrieve(r),                         n_workers),
        ("S3:aggregate",         lambda r: _s3_aggregate(r),                        n_workers),
        ("S4:context_expand",    lambda r: _s4_context_expand(r),                   n_workers),
        ("S5:rerank",            lambda r: _s5_rerank(r, args.skip_rerank),         1),         # CrossEncoder is not thread-safe; load once, score sequentially
        ("S6:llm_verify",        lambda r: _s6_llm_verify(r, args.skip_verify),     n_workers),
        ("S7:highlight",         lambda r: _s7_highlight_extract(r),                n_workers),
        ("S8:merge",             lambda r: _s8_merge(r),                            n_workers),
        ("S9:evidence_classify", lambda r: _s9_evidence_classify(r, args.skip_verify), n_workers),
    ]

    stage_wall: dict[str, float] = {}
    rerank_summary_dict: dict    = {}
    _t_wall_start = time.perf_counter()

    for stage_key, stage_fn, stage_workers in STAGES:
        active = sum(1 for r in all_reqs if not r.get("_failed") and not r.get("_early_exit"))
        # Skip S1-S4 when loaded from cache
        if s4_loaded_from_cache and stage_key in ("S1:classify","S2:retrieve","S3:aggregate","S4:context_expand"):
            stage_wall[stage_key] = 0.0
            continue
        if active == 0:
            print(f"  [{stage_key}]  (no active CIs — skip)")
            stage_wall[stage_key] = 0.0
            continue
        t_stage = time.perf_counter()
        workers_note = "  [serial]" if stage_workers == 1 else ""
        print(f"  [{stage_key}]  {active} CIs …{workers_note}", flush=True)
        with ThreadPoolExecutor(max_workers=stage_workers) as pool:
            all_reqs = list(pool.map(stage_fn, all_reqs))
        elapsed = round(time.perf_counter() - t_stage, 3)
        stage_wall[stage_key] = elapsed
        done_early = sum(1 for r in all_reqs if r.get("_early_exit"))
        print(f"  [{stage_key}]  done in {elapsed:.1f}s"
              + (f"  ({done_early} early-exit)" if done_early else ""))

        # Save S4 output to cache after context expansion completes
        if stage_key == "S4:context_expand" and not s4_loaded_from_cache:
            _save_s4_cache(all_reqs, cache_path)

        if stage_key == "S2:retrieve":
            # Quick per-CI retriever breakdown for early debugging
            for req in all_reqs:
                ret = req.get("_st", {}).get("retrievers", {})
                if not ret:
                    continue
                ci_id = (req.get("ci") or {}).get("id", "?")
                parts = "  ".join(f"{k}={v:.2f}s" for k, v in sorted(ret.items()))
                print(f"    CI {ci_id}: {parts}")

        if stage_key == "S5:rerank" and not args.skip_rerank:
            try:
                rerank_mod = _load("search/reranker", "search_reranker")
                rerank_summary_dict = rerank_mod.get_rerank_summary_dict()
                summary = rerank_mod.get_rerank_summary()
                if summary:
                    print(summary, flush=True)
            except Exception as _exc:
                logger.debug("rerank summary unavailable: %s", _exc)

    wall_time = time.perf_counter() - _t_wall_start

    # ── Convert _st → timings (compatible with search.py helpers) ────────────
    all_results: list[dict] = []
    for req in all_reqs:
        if req.get("_failed"):
            continue
        req["timings"]              = _st_to_timings(req.pop("_st", {}))
        req["classification"]       = req.get("classification", {})
        req["verified_candidates"]  = req.get("verified_candidates", [])
        req.pop("_failed",     None)
        req.pop("_early_exit", None)
        req.pop("_ci_idx",     None)
        req.pop("_diagnose_ci_ids", None)
        all_results.append(req)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  SUMMARY: {len(all_results)} CI(s) searched  (wall-clock {wall_time:.1f}s)")
    print(f"  Total confirmed hits: {sum(len(r.get('final_hits', [])) for r in all_results)}")

    _print_stage_timing(stage_wall, all_results, wall_time)
    _print_cost_estimate(all_results, wall_s={
        "llm_verifier":            stage_wall.get("S6:llm_verify", 0.0),
        "evidence_classification": stage_wall.get("S9:evidence_classify", 0.0),
    })

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = (
        Path(args.output) if args.output else
        ROOT / "localfiles" / "search_results" /
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_staged_{Path(args.ci_file).stem}_{args.document_id}.json"
    )
    _save_results_staged(all_results, args, out_path, wall_time, stage_wall,
                         rerank_summary=rerank_summary_dict)
    print(f"\n  Results saved: {out_path}")
    print()


if __name__ == "__main__":
    main()
