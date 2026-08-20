"""
Search Worker Lambda
====================
Processes a batch of pre-enriched CIs against a document using the
stage-parallel pipeline (mirrors search_new.py, no file I/O).

Input
-----
{
    "search_id":        str,
    "batch_idx":        int,
    "cis":              list[dict],   # enriched CIs (from ci-objects index)
    "document_id":      str,
    "document_context": dict,
    "skip_rerank":      bool,
    "skip_verify":      bool,
    "workers":          int           # per-stage thread count (default: len(cis))
}

Output
------
{
    "search_id":  str,
    "batch_idx":  int,
    "results":    list[dict],         # per-CI dict with final_hits + timings
    "stage_wall": dict[str, float]
}

Env vars
--------
  OPENSEARCH_ENDPOINT     — host only (no https://)
  OPENSEARCH_INDEX        — default: document-chunks
  SEMANTIC_OBJECTS_INDEX  — default: semantic-objects
  OPENSEARCH_CI_INDEX     — default: ci-objects
  AWS_REGION
  VERIFIER_MODEL          — Bedrock model ID for LLM verifier + evidence classifier
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import threading
import time
import types
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar, copy_context
from datetime import datetime
from pathlib import Path
from typing import Any


# ── Logging Context Variables (thread-safe for concurrent execution) ───────
_ctx_tenant = ContextVar("tenant", default="-")
_ctx_document_id = ContextVar("document_id", default="-")
_ctx_search_id = ContextVar("search_id", default="-")
_ctx_batch_idx = ContextVar("batch_idx", default="-")
_ctx_ci_id = ContextVar("ci_id", default="-")


class SearchContextFilter(logging.Filter):
    """Logging filter that injects search context into all log records.
    
    Automatically adds [tenant=...] [document=...] [search=...] [batch=...] [ci=...]
    prefix to every log from any module, without requiring manual propagation.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        tenant = _ctx_tenant.get()
        document_id = _ctx_document_id.get()
        search_id = _ctx_search_id.get()
        batch_idx = _ctx_batch_idx.get()
        ci_id = _ctx_ci_id.get()
        
        prefix = f"[tenant={tenant}] [document={document_id}] [search={search_id}]"
        if batch_idx != "-":
            prefix += f" [batch={batch_idx}]"
        if ci_id != "-":
            prefix += f" [ci={ci_id}]"
        
        record.msg = f"{prefix} {record.msg}"
        return True


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Configure root logger so all dynamically loaded modules inherit the context filter
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Add context filter to all existing handlers (preserve Lambda's CloudWatch handler)
context_filter = SearchContextFilter()
for handler in root_logger.handlers:
    handler.addFilter(context_filter)

# If no handlers exist, create one (e.g., local testing)
if not root_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(name)s] %(message)s'
    )
    handler.setFormatter(formatter)
    handler.addFilter(context_filter)
    root_logger.addHandler(handler)

_task_root_env = os.environ.get("LAMBDA_TASK_ROOT")
if _task_root_env and (Path(_task_root_env) / "lambdas").exists():
    ROOT = Path(_task_root_env)
else:
    ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── LLM cost estimation constants (Anthropic Haiku-4.5 base rates) ──────────
_HAIKU_INPUT_PRICE_PER_TOKEN  = 0.80 / 1_000_000   # USD per token
_HAIKU_OUTPUT_PRICE_PER_TOKEN = 4.00 / 1_000_000   # USD per token
# llm_verifier: full structured prompt with CI text, doc profile, excerpt, identity dims
# Output includes verdict + reason (~15 words) + confidence + 7-field identity block
_EST_INPUT_TOKENS_PER_CAND    = 418   # measured ~408 from real runs
_EST_OUTPUT_TOKENS_PER_CAND   = 85    # was 30 — identity block (same_drug/study/…) adds ~55 tokens
# _classify_evidence: CI text + match_span (400 chars) + 9-label classification list
# Output is short JSON: evidence_type + confidence + reason (~15 words)
_EST_EC_INPUT_TOKENS_PER_HIT  = 400   # was 300 — measured ~394 (long label list)
_EST_EC_OUTPUT_TOKENS_PER_HIT = 40    # was 60  — measured ~41 (shorter response)


# ── Env config ─────────────────────────────────────────────────────────────────
OPENSEARCH_ENDPOINT    = os.environ.get("OPENSEARCH_ENDPOINT", "localhost")
OPENSEARCH_INDEX       = os.environ.get("OPENSEARCH_INDEX", "document-chunks")
SEMANTIC_OBJECTS_INDEX = os.environ.get("SEMANTIC_OBJECTS_INDEX", "semantic-objects")
OPENSEARCH_CI_INDEX    = os.environ.get("OPENSEARCH_CI_INDEX", "ci-objects")
OPENSEARCH_TIMEOUT     = int(os.environ.get("OPENSEARCH_TIMEOUT", "30"))
OPENSEARCH_MAXSIZE     = int(os.environ.get("OPENSEARCH_MAXSIZE", "256"))  # Connection pool size (RETRIEVER_WORKERS × max concurrent CIs + buffer)
AWS_REGION             = os.environ.get("AWS_REGION", "us-east-1")
BEDROCK_REGION         = os.environ.get("BEDROCK_REGION", AWS_REGION)
VERIFIER_MODEL         = os.environ.get("VERIFIER_MODEL",
                                         "eu.anthropic.claude-haiku-4-5-20251001-v1:0")
EMBEDDING_MODEL        = os.environ.get("EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
# Concurrency control (prevent overwhelming OpenSearch with nested thread pools)
SEARCH_CI_WORKERS      = int(os.environ.get("SEARCH_CI_WORKERS", "5"))      # CIs per Worker
RETRIEVER_WORKERS      = int(os.environ.get("RETRIEVER_WORKERS", "4"))      # Retrievers per CI
SEARCH_RESULTS_DEBUG_BUCKET = os.environ.get("SEARCH_RESULTS_DEBUG_BUCKET", "rls-file-bucket-eu")
RESULTS_DEBUG_PREFIX   = os.environ.get("RESULTS_DEBUG_PREFIX", "search-results")
# ── Lazy singletons ────────────────────────────────────────────────────────────
_loaded: dict[str, types.ModuleType] = {}
_evidence_classifier = None  # Lazy load for evidence classification


def _get_evidence_classifier():
    """Lazy-load evidence classification module."""
    global _evidence_classifier
    if _evidence_classifier is None:
        _evidence_classifier = _load("search/evidence_classification", "evidence_classifier")
    return _evidence_classifier


def _get_os():
    from shared.opensearch_client import get_opensearch_client
    return get_opensearch_client()


def _load(rel_path: str, alias: str) -> types.ModuleType:
    if alias in _loaded:
        return _loaded[alias]
    lf_path = ROOT / "lambdas" / rel_path / "lambda_function.py"
    spec    = importlib.util.spec_from_file_location(alias, lf_path)
    mod     = importlib.util.module_from_spec(spec)
    lf_dir  = str(lf_path.parent)
    if lf_dir not in sys.path:
        sys.path.insert(0, lf_dir)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    _loaded[alias] = mod
    return mod


def _load_fresh(rel_path: str) -> types.ModuleType:
    """Fresh uncached module so each thread can safely monkey-patch."""
    lf_path = ROOT / "lambdas" / rel_path / "lambda_function.py"
    alias   = f"_fresh_{rel_path.replace('/', '_')}_{threading.get_ident()}"
    spec    = importlib.util.spec_from_file_location(alias, lf_path)
    mod     = importlib.util.module_from_spec(spec)
    lf_dir  = str(lf_path.parent)
    if lf_dir not in sys.path:
        sys.path.insert(0, lf_dir)
    spec.loader.exec_module(mod)
    return mod


def _inject_os(mod: types.ModuleType) -> None:
    mod.OPENSEARCH_ENDPOINT    = OPENSEARCH_ENDPOINT
    mod.OPENSEARCH_INDEX       = OPENSEARCH_INDEX
    mod.SEMANTIC_OBJECTS_INDEX = SEMANTIC_OBJECTS_INDEX
    mod.AWS_REGION             = AWS_REGION
    if hasattr(mod, "_os_client"):
        mod._os_client = _get_os()


# ── Retriever map ──────────────────────────────────────────────────────────────
RETRIEVER_MAP: dict[str, str] = {
    "literal":  "search/literal_retriever",
    "bm25":     "search/bm25_retriever",
    "vector":   "search/vector_retriever",
    "ner":      "search/ner_retriever",
    "fact":     "search/fact_retriever",
    "ontology": "search/ontology_retriever",
    "regex":    "search/regex_retriever",
    "numeric":  "search/numeric_retriever",
}

# ── Evidence helpers ───────────────────────────────────────────────────────────
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

_NUMERIC_GATE_TYPES: frozenset[str] = frozenset({
    "NUMERIC_SAMPLE_SIZE", "CONFIDENCE_INTERVAL", "P_VALUE",
    "HAZARD_RATIO", "ODDS_RATIO", "NUMERIC_PERCENTAGE", "MEDIAN",
    "NUMERIC", "STATISTICAL",
})

_CONF_THRESHOLD = 0.2


def _is_related(ev: str) -> bool:
    return ev.startswith("SAME_") or ev.startswith("RELATED_") or ev == "BACKGROUND"


def _numeric_gate_pattern(ci: dict):
    """
    Return a pattern whose .search(text) must be truthy for a candidate to pass
    the numeric gate.  Returns None when no meaningful constraint can be derived
    (in which case the gate is skipped and all candidates pass through).

    Uses statistical_identity.type + value when available; falls back to a
    direct regex on the CI text for common forms like "n = 8".
    """
    import re as _re

    si      = ci.get("statistical_identity") or {}
    si_type = si.get("type")
    ci_text = ci.get("knownCI", "")

    def _tok(v) -> str:
        try:
            return str(int(v)) if float(v) == int(float(v)) else str(v)
        except (TypeError, ValueError):
            return str(v)

    if si_type == "sample_size" and si.get("sample_size") is not None:
        n = _tok(si["sample_size"])
        # Match N=X regardless of whether the CI used =, ≥, >, ≤, < —
        # the document will always express the measured value as N=X.
        return _re.compile(rf'[Nn]\s*[=\u2265\u2264><]=?\s*{_re.escape(n)}\b')

    if si_type == "confidence_interval":
        lo = si.get("lower_ci")
        hi = si.get("upper_ci")
        if lo is not None and hi is not None:
            lo_p = _re.compile(rf'\b{_re.escape(_tok(lo))}\b')
            hi_p = _re.compile(rf'\b{_re.escape(_tok(hi))}\b')
            class _Both:
                def search(self, t): return lo_p.search(t) and hi_p.search(t)
            return _Both()

    if si_type == "p_value" and si.get("p_value") is not None:
        return _re.compile(rf'\b{_re.escape(str(si["p_value"]))}\b')

    if si_type == "hazard_ratio" and si.get("hazard_ratio") is not None:
        return _re.compile(rf'\b{_re.escape(_tok(si["hazard_ratio"]))}\b')

    if si_type == "odds_ratio" and si.get("odds_ratio") is not None:
        return _re.compile(rf'\b{_re.escape(_tok(si["odds_ratio"]))}\b')

    # Fallback: "n = 8" / "N>=8" / "N≥8" style CI text
    import re as _re2
    m = _re2.match(r'[Nn]\s*[=\u2265\u2264><]=?\s*(\d+)', ci_text.strip())
    if m:
        return _re2.compile(rf'[Nn]\s*[=\u2265\u2264><]=?\s*{m.group(1)}\b')

    return None   # no constraint determinable → don't filter


def _candidate_confidence(c: dict) -> float:
    return round(
        0.5 * min(max(c.get("agg_score", 0.0) * 2.0, 0.0), 1.0)
        + 0.3 * max(0.0, 1.0 + c.get("zero_id_pen",    0.0) / 0.4)
        + 0.2 * max(0.0, 1.0 + c.get("zero_enrich_pen", 0.0) / 0.25),
        3,
    )


def _calibrate_evidence(hit: dict, ec: dict) -> dict:
    mq     = hit.get("highlight_score", 1.0)
    method = hit.get("match_method", "")
    ev_t   = ec.get("evidence_type", "RELATED_EFFICACY")
    ev_c   = ec.get("evidence_confidence", 0.5)
    result = dict(ec)
    if method in ("text_fallback", "text_fallback_skipped") and mq < 0.15 and ev_t.startswith("RELATED_"):
        result["evidence_type"]   = "SUPPORTING"
        result["evidence_reason"] = result.get("evidence_reason", "") + f" [downgraded: mq={mq:.3f}]"
    if method == "text_fallback_skipped":
        qf = 0.25
    elif mq > 0:
        qf = mq / (mq + 0.05)
    else:
        qf = 0.10
    result["evidence_confidence"] = round(ev_c * qf, 3)
    return result


# ── Stage functions ────────────────────────────────────────────────────────────

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

    # Propagate ContextVar to nested retriever threads so [ci=...] appears in their logs
    with ThreadPoolExecutor(max_workers=min(RETRIEVER_WORKERS, len(valid))) as pool:
        futures = [
            pool.submit(copy_context().run, _run_one, s_path)
            for s_path in valid
        ]
        raw = [f.result() for f in futures]

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
    """Reranking stage — currently always skips reranker for cold start reduction."""
    if req.get("_failed") or req.get("_early_exit"):
        return req
    t0 = time.perf_counter()
    # Skip reranker entirely (skip_rerank always True) — set uniform score
    expanded = req.get("expanded_candidates", [])
    req["ranked_candidates"] = [
        # Set score above MIN_RERANK_SCORE (3.0) so all candidates reach Claude
        {**c, "cross_encoder_score": 10.0}
        for c in expanded
    ]
    req["_st"]["reranker"] = round(time.perf_counter() - t0, 3)

    # 5.5 Numeric gate
    ci_type = (req.get("classification") or {}).get("ci_type", "") or ""
    if ci_type in _NUMERIC_GATE_TYPES:
        pat = _numeric_gate_pattern(req["ci"])
        if pat is not None:
            passed, gated = [], []
            for c in req.get("ranked_candidates", []):
                txt = ((c.get("context") or {}).get("current_text", "") or c.get("snippet", ""))
                (passed if pat.search(txt) else gated).append(c)
            req["ranked_candidates"] = passed

    # 5.6 Chunk dedup
    by_chunk: dict[str, dict] = {}
    for c in req.get("ranked_candidates", []):
        cid = c.get("chunk_id") or c.get("id") or ""
        if cid not in by_chunk or c.get("agg_score", 0) > by_chunk[cid].get("agg_score", 0):
            by_chunk[cid] = c
    req["ranked_candidates"] = list(by_chunk.values())

    # 5.7 Confidence gate
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
            "per_verifier_call_s": {},
            "actual_verifier_tokens": {"input": 0, "output": 0},
            "llm_verifier": 0.0,
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
        verifier    = _load_fresh("search/llm_verifier")
        _inject_os(verifier)
        call_times: list[float] = []
        call_tokens: list[dict] = []
        orig = verifier._verify

        def _timed(*a, **kw):
            _t = time.perf_counter()
            _r = orig(*a, **kw)
            call_times.append(round(time.perf_counter() - _t, 3))
            call_tokens.append(_r.get("_tokens", {"input": 0, "output": 0}))
            return _r

        verifier._verify = _timed
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
            classifier = _get_evidence_classifier()
            classified = classifier._classify_evidence_batch(ci_text, to_classify, doc_ctx)
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


# ── Pipeline ───────────────────────────────────────────────────────────────────

def _safe_stage_wrapper(stage_key: str, stage_fn, req: dict) -> dict:
    """Wrap stage functions to catch exceptions and mark CI as failed.
    
    Sets CI context for all logs during this CI's stage processing.
    """
    if req.get("_failed") or req.get("_early_exit"):
        return req
    
    ci_id = req["ci"].get("id")
    token = _ctx_ci_id.set(ci_id)
    
    try:
        return stage_fn(req)
    except Exception as exc:
        logger.error("[SearchWorker] failed in stage %s: %s", stage_key, exc)
        req["_failed"] = True
        req["_failure"] = {
            "stage": stage_key,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        return req
    finally:
        _ctx_ci_id.reset(token)  # Properly restore previous context


def _run_pipeline(all_reqs: list[dict], skip_rerank: bool, skip_verify: bool,
                  n_workers: int, tenant_name: str) -> tuple[list[dict], dict[str, float]]:
    """Run stage-parallel pipeline, return (all_reqs, stage_wall)."""
    # Reranker uses max_workers=1 to serialise CrossEncoder.predict() calls.
    STAGES = [
        ("S1:classify",          lambda r: _s1_classify(r),                         n_workers),
        ("S2:retrieve",          lambda r: _s2_retrieve(r),                         n_workers),
        ("S3:aggregate",         lambda r: _s3_aggregate(r),                        n_workers),
        ("S4:context_expand",    lambda r: _s4_context_expand(r),                   n_workers),
        ("S5:rerank",            lambda r: _s5_rerank(r, skip_rerank),         1),
        ("S6:llm_verify",        lambda r: _s6_llm_verify(r, skip_verify),     n_workers),
        ("S7:highlight",         lambda r: _s7_highlight_extract(r),                n_workers),
        ("S8:merge",             lambda r: _s8_merge(r),                            n_workers),
        ("S9:evidence_classify", lambda r: _s9_evidence_classify(r, skip_verify), n_workers),
    ]

    stage_wall: dict[str, float] = {}
    for stage_key, stage_fn, stage_workers in STAGES:
        active = sum(1 for r in all_reqs if not r.get("_failed") and not r.get("_early_exit"))
        if active == 0:
            stage_wall[stage_key] = 0.0
            continue
        t_stage = time.perf_counter()
        with ThreadPoolExecutor(max_workers=stage_workers) as pool:
            all_reqs = list(pool.map(
                lambda r: _safe_stage_wrapper(stage_key, stage_fn, r),
                all_reqs
            ))
        stage_wall[stage_key] = round(time.perf_counter() - t_stage, 3)
        logger.info("[SearchWorker] stage %s done in %.1fs (%d active)",
                    stage_key, stage_wall[stage_key], active)
    
    logger.info("[SearchWorker] stage wall summary: %s", stage_wall)
    return all_reqs, stage_wall



# ─────────────────────────────────────────────────────────────────────────────
# Result serialization
# ─────────────────────────────────────────────────────────────────────────────

def _save_results_debug_s3(all_results: list[dict], event, wall_time: float = 0.0,
                           n_cis_total: int = 0, n_completed: int = 0, 
                           n_failed: int = 0, ci_failures: list[dict] = None) -> str:
    """Write a clean, human-readable results JSON — strips large vectors/context."""
    if ci_failures is None:
        ci_failures = []
    
    batch_idx = event.get('batch_idx')
    search_id = event.get('search_id')
    document_id = event.get('document_id')
    tenant_name = event.get("tenant_name", "")
    debug_json = {
        "run": {
            "timestamp":   datetime.now().isoformat(),
            "document_id": document_id,
            "opensearch":  OPENSEARCH_ENDPOINT,
            "skip_rerank": event.get("skip_rerank", False),
            "skip_verify": event.get("skip_verify", False),
        },
        "concurrency": {
            "ci_workers":      SEARCH_CI_WORKERS,
            "retriever_workers": RETRIEVER_WORKERS,
            "os_maxsize":      OPENSEARCH_MAXSIZE,
            "theoretical_max_concurrent_retrievers": SEARCH_CI_WORKERS * RETRIEVER_WORKERS,
        },
        "batch": {
            "batch_idx":      int(batch_idx),
            "expected_cis":   n_cis_total,
            "completed_cis":  n_completed,
            "failed_cis":     n_failed,
            "ci_failures":    ci_failures if ci_failures else [],
        },
        "summary": {
            "cis_searched":     len(all_results),
            "total_final_hits": sum(len(r.get("final_hits", [])) for r in all_results),
            "direct_hits":    sum(
                1 for r in all_results for h in r.get("final_hits", [])
                if h.get("evidence_type") == "DIRECT"
            ),
            "same_study_hits": sum(
                1 for r in all_results for h in r.get("final_hits", [])
                if h.get("evidence_type") == "SAME_STUDY"
            ),
            "related_hits":   sum(
                1 for r in all_results for h in r.get("final_hits", [])
                if (h.get("evidence_type") or "").startswith("SAME_")
                or h.get("evidence_type") == "BACKGROUND"
            ),
            "related_breakdown": {
                sub: sum(
                    1 for r in all_results for h in r.get("final_hits", [])
                    if h.get("evidence_type") == f"SAME_{sub}"
                )
                for sub in (
                    "PROTOCOL", "OBJECTIVE", "ENDPOINT",
                    "POPULATION", "MECHANISM"
                )
            },
            "background_hits": sum(
                1 for r in all_results for h in r.get("final_hits", [])
                if h.get("evidence_type") == "BACKGROUND"
            ),
            "no_hits":        sum(
                1 for r in all_results for h in r.get("final_hits", [])
                if h.get("verdict") in ("NO", "SKIP")
            ),
            "total_rejected": sum(
                1 for r in all_results
                for v in r.get("verified_candidates", [])
                if v.get("verdict") == "NO"
            ),
            "total_skipped": sum(
                1 for r in all_results
                for v in r.get("verified_candidates", [])
                if v.get("verdict") == "SKIP"
            ),
            "object_type_stats": _object_type_stats(all_results),
        },
        "results": [_clean_result(r) for r in all_results],
    }

    # ── Timing + cost summary ─────────────────────────────────────────────────
    all_t = [r.get("timings", {}) for r in all_results]
    n     = len(all_results) or 1

    def _agg_t(key: str) -> dict:
        vals = [t.get(key, 0.0) for t in all_t]
        return {"total_s": round(sum(vals), 3), "avg_s": round(sum(vals) / n, 3)}

    retriever_keys: list[str] = []
    for _tr in all_t:
        for _k in _tr.get("retrievers", {}):
            if _k not in retriever_keys:
                retriever_keys.append(_k)

    debug_json["timing_summary"] = {
        "classifier":              _agg_t("classifier"),
        "retrievers_total":        _agg_t("retrievers_total"),
        "retrievers": {
            rk: {
                "total_s": round(sum(t.get("retrievers", {}).get(rk, 0) for t in all_t), 3),
                "avg_s":   round(sum(t.get("retrievers", {}).get(rk, 0) for t in all_t) / n, 3),
            }
            for rk in retriever_keys
        },
        "aggregator":              _agg_t("aggregator"),
        "context_expander":        _agg_t("context_expander"),
        "reranker":                _agg_t("reranker"),
        "llm_verifier":            _agg_t("llm_verifier"),
        "highlight_extractor":     _agg_t("highlight_extractor"),
        "merger":                  _agg_t("merger"),
        "evidence_classification": _agg_t("evidence_classification"),
        # total_cpu_s = sum of all per-CI pipeline times (CIs run in parallel,
        # so this is >  wall-clock time when workers > 1)
        "total": {
            "wall_clock_s":  round(wall_time, 3),
        },
        "_note": "per-stage total_s = cumulative CPU time across all parallel workers",
    }

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

    n_ec_calls = sum(t.get("n_ec_calls", 0) for t in all_t)
    ec_actual_in  = sum(t.get("actual_ec_tokens", {}).get("input",  0) for t in all_t)
    ec_actual_out = sum(t.get("actual_ec_tokens", {}).get("output", 0) for t in all_t)
    if ec_actual_in > 0:
        ec_in, ec_out, ec_label = ec_actual_in, ec_actual_out, "actual"
    else:
        ec_in, ec_out, ec_label = (n_ec_calls * _EST_EC_INPUT_TOKENS_PER_HIT,
                                    n_ec_calls * _EST_EC_OUTPUT_TOKENS_PER_HIT, "est.")
    ec_cost = ec_in * _HAIKU_INPUT_PRICE_PER_TOKEN + ec_out * _HAIKU_OUTPUT_PRICE_PER_TOKEN

    debug_json["cost_estimate"] = {
        "model": VERIFIER_MODEL,
        "llm_verifier": {
            "candidates_passed_to_verifier": n_to_verifier,
            "skipped_below_threshold":        n_to_verifier - n_actual_calls,
            "actual_bedrock_calls":           n_actual_calls,
            "input_tokens":                  v_in,
            "output_tokens":                 v_out,
            "total_tokens":                  v_in + v_out,
            "token_source":                  v_label,
            "est_cost_usd":                  round(v_cost, 4),
        },
        "evidence_classification": {
            "bedrock_calls":   n_ec_calls,
            "input_tokens":    ec_in,
            "output_tokens":   ec_out,
            "total_tokens":    ec_in + ec_out,
            "token_source":    ec_label,
            "est_cost_usd":    round(ec_cost, 4),
        },
        "combined_est_cost_usd": round(v_cost + ec_cost, 4),
    }

    s3_url = _upload_debug_json_to_s3(debug_json, search_id, batch_idx, document_id, tenant_name)
    return s3_url



def _object_type_stats(all_results: list[dict]) -> dict:
    """Per-object-type breakdown: how many candidates were retrieved, passed, rejected, skipped."""
    from collections import defaultdict
    retrieved: dict[str, int] = defaultdict(int)
    final_yes: dict[str, int] = defaultdict(int)
    rejected:  dict[str, int] = defaultdict(int)
    skipped:   dict[str, int] = defaultdict(int)

    for r in all_results:
        for v in r.get("verified_candidates", []):
            obj_type = (v.get("matched_object") or {}).get("type") or "chunk"
            retrieved[obj_type] += 1
            verdict = v.get("verdict", "")
            if verdict in ("YES", "MAYBE"):
                final_yes[obj_type] += 1
            elif verdict == "NO":
                rejected[obj_type] += 1
            elif verdict == "SKIP":
                skipped[obj_type] += 1

    all_types = sorted(set(list(retrieved.keys())))
    return {
        t: {
            "retrieved": retrieved[t],
            "final_yes": final_yes[t],
            "rejected":  rejected[t],
            "skipped":   skipped[t],
        }
        for t in all_types
    }


def _base_object_id(object_id: str | None) -> str:
    """Normalize retriever object-id variants like *_s0/*_s1 to a stable base id."""
    import re as _re
    oid = str(object_id or "")
    return _re.sub(r"_s\d+$", "", oid)


def _hit_with_provenance(hit: dict) -> dict:
    """Add retrieval provenance fields to a final hit; strip the raw matched_object."""
    obj            = hit.get("matched_object") or {}
    obj_type       = obj.get("type") or "unknown"
    sources        = hit.get("sources", [])

    # context_expander sets retrieval_origin as "{direct|via_chunk}_{object_type}":
    #   direct_sentence   → retriever returned a sentence object from the semantic-objects index
    #   direct_paragraph  → retriever returned a paragraph object directly
    #   via_chunk_sentence → retriever returned a CHUNK; context_expander extracted a sentence
    # We use this to determine what the retriever ACTUALLY retrieved, not what was expanded to.
    ce_origin = hit.get("retrieval_origin", "")  # context_expander's value (will be overwritten)
    if ce_origin.startswith("via_chunk_"):
        # retriever found a chunk; context_expander chose best object within it
        retrieved_unit = "chunk"
    elif ce_origin.startswith("direct_"):
        # retriever found this object type directly in the index
        retrieved_unit = ce_origin[len("direct_"):]
    else:
        # no context_expander info → use retrieved_type (set by vector_retriever) or matched_object.type
        retrieved_unit = hit.get("retrieved_type") or obj_type

    # retrieval_origin: analytics format "sources/retrieved_unit"
    # e.g. "vector/chunk", "bm25/sentence", "vector/paragraph"
    origin_str = ("+".join(sorted(sources)) if sources else "unknown") + "/" + retrieved_unit
    object_id = obj.get("object_id")
    parent_chunk_id = obj.get("parent_chunk_id")
    retrieval_chunk_id = parent_chunk_id or hit.get("chunk_id")
    extra = {
        "retrieval_object_type": obj_type,
        # retrieved_type: the unit the retriever actually fetched from the index.
        # "chunk" = vector_search_chunks fallback; context_expander assigned the object.
        # "sentence"/"paragraph"/etc = object fetched directly from semantic-objects index.
        "retrieved_type":        retrieved_unit,
        # expansion_origin: context_expander's raw value preserved for debugging.
        # Format: "direct_{type}" or "via_chunk_{type}".
        "expansion_origin":      ce_origin or None,
        "retrieval_object_id":   object_id,
        "retrieval_object_id_base": _base_object_id(object_id),
        "retrieval_parent_chunk_id": parent_chunk_id,
        "retrieval_chunk_id": retrieval_chunk_id,
        "retrieval_heading_path": obj.get("heading_path"),
        "retrieval_section":     obj.get("section_category") or obj.get("section"),
        "retrieval_origin":      origin_str,
        "selection_reason":      hit.get("selection_reason"),
        "literal_match_count":   hit.get("literal_match_count"),
        "context_strategy":      hit.get("context_strategy"),
        "matched_distance":      hit.get("matched_distance"),
        "distance_ratio":        hit.get("distance_ratio"),
        "current_text_chars":    hit.get("current_text_chars"),
        "agg_score":             hit.get("agg_score"),
        "score_breakdown":       hit.get("score_breakdown"),
        "agg_score_breakdown":   hit.get("agg_score_breakdown"),
        "indexed_object":        _indexed_object(hit),
    }
    # Remove embedding vectors and other unnecessary large fields
    vectors_to_exclude = {
        "matched_object",  # Already handled separately
        "dense_vector",
        "embedding",
        "sparse_vector",
        "vector",
        "dense_embedding",
        "context",  # Context expanded separately in indexed_object
    }
    base = {k: v for k, v in hit.items() if k not in vectors_to_exclude}
    return {**base, **extra}


def _indexed_object(v: dict) -> dict | None:
    """
    Return the full indexed data for a candidate — everything stored in OpenSearch
    for this semantic object, minus large vector fields.
    Included in every hit/rejected/skipped entry so reviewers can audit
    exactly what was indexed (entities, facts, clinical_relations, etc.).
    """
    obj = v.get("matched_object")
    if not obj:
        return None
    ctx_text = (v.get("context") or {}).get("current_text", "")
    return {
        # Identity
        "object_id":          obj.get("object_id"),
        "parent_chunk_id":    obj.get("parent_chunk_id"),
        "type":               obj.get("type"),
        # Location
        "page":               obj.get("page"),
        "bbox":               obj.get("bbox"),
        "position":           obj.get("position"),
        "global_position":    obj.get("global_position"),
        "document_position":  obj.get("document_position"),
        # Text content
        "text":               obj.get("text"),
        "paragraph_text":     obj.get("paragraph_text"),
        "prev_sentence_text": obj.get("prev_sentence_text"),
        "next_sentence_text": obj.get("next_sentence_text"),
        "context_chunk_text": ctx_text or None,
        # Section / heading
        "section_category":   obj.get("section_category"),
        "heading_path":       obj.get("heading_path"),
        "semantic_path":      obj.get("semantic_path"),
        "section_confidence": obj.get("section_confidence"),
        # NER
        "entities":           obj.get("entities", []),
        # Clinical fact extraction
        "facts":              obj.get("facts", {}),
        "statement_type":     obj.get("statement_type"),
        "study_context":      obj.get("study_context"),
        "clinical_relations": obj.get("clinical_relations", []),
    }


def _ci_metadata(ci: dict) -> dict:
    """
    Return all enriched CI fields suitable for the result JSON.
    Dense and sparse embedding vectors are intentionally excluded (large + irrelevant to reviewers).
    """
    emb  = ci.get("embedding", {})
    norm = ci.get("normalization", {})
    ner  = ci.get("ner", {})
    onto = ci.get("ontology", {})
    cls  = ci.get("classification", {})

    return {
        # ── Identity ─────────────────────────────────────────────────────────
        "id":                   ci.get("id"),
        "text":                 ci.get("knownCI", ""),
        "category":             ci.get("category") or ci.get("type"),
        # ── Normalization ─────────────────────────────────────────────────────
        "normalized_text":      norm.get("normalized_text", ""),
        "tokens":               norm.get("tokens", []),
        "abbreviations":        norm.get("abbreviations_found", {}),
        # ── NER ───────────────────────────────────────────────────────────────
        "entities":             ner.get("entities", []),
        "ner_model":            ner.get("model"),
        # ── Ontology ──────────────────────────────────────────────────────────
        "ontology_expansions":  onto.get("expansions", []),
        "ontology_synonyms":    onto.get("synonyms", {}),
        "regex_patterns":       onto.get("regex_patterns", []),
        # ── Embedding metadata (no vectors) ───────────────────────────────────
        "embedding_model":      emb.get("model"),
        "embedding_dimensions": emb.get("dimensions"),
        # ── Classification ────────────────────────────────────────────────────
        "ci_type":              cls.get("ci_type"),
        "strategies":           cls.get("strategies", []),
        "classification_reason": cls.get("reason"),
        # ── Clinical facts ────────────────────────────────────────────────────
        "facts":                ci.get("facts", {}),
        "own_facts":            ci.get("own_facts", {}),
        "effective_facts":      ci.get("effective_facts", {}),
        "inherited_slots":      ci.get("inherited_slots", []),
        "slot_provenance":      ci.get("slot_provenance", {}),
        "study_hierarchy":      ci.get("study_hierarchy", {}),
        "clinical_identity":    ci.get("clinical_identity", {}),
        "study_context":        ci.get("study_context"),
        "statement_type":       ci.get("statement_type"),
        "modality":             ci.get("modality"),
        "negated_slots":        ci.get("negated_slots", []),
        "treatment_identity":   ci.get("treatment_identity", {}),
        "endpoint_identity":    ci.get("endpoint_identity", {}),
        "population_identity":  ci.get("population_identity", {}),
        "temporal_context":     ci.get("temporal_context", {}),
        "clinical_relations":   ci.get("clinical_relations", []),
        # ── Source metadata (passthrough from raw CI file) ────────────────────
        "justification_text":   ci.get("justificationText"),
        "assets":               ci.get("assets", []),
    }


def _full_candidate_record(v: dict) -> dict:
    """
    Comprehensive per-candidate record saved for every verified_candidate
    (verdict = YES / MAYBE / NO / SKIP).

    Captures all pipeline stages: aggregator scores, reranker score,
    verifier verdict + reason, highlight extraction, and the full indexed
    object (prev_sentence_text, text, next_sentence_text, paragraph_text,
    context_chunk_text, heading_path, section, NER, facts, relations).
    """
    sb  = v.get("score_breakdown") or {}
    obj = v.get("matched_object") or {}
    ctx_text = (v.get("context") or {}).get("current_text", "")
    object_id = obj.get("object_id")
    parent_chunk_id = obj.get("parent_chunk_id")
    retrieval_chunk_id = parent_chunk_id or v.get("chunk_id")
    return {
        # ── Identity ─────────────────────────────────────────────────────
        "chunk_id":             v.get("chunk_id", ""),
        "page_start":           v.get("page_start"),
        "page_end":             v.get("page_end"),
        "sources":              v.get("sources", []),
        "retriever":            v.get("retriever", ""),
        "retrieval_origin":     v.get("retrieval_origin", "direct_unknown"),
        "selection_reason":     v.get("selection_reason"),
        "literal_match_count":  v.get("literal_match_count"),
        "context_strategy":     v.get("context_strategy"),
        "matched_distance":     v.get("matched_distance"),
        "distance_ratio":       v.get("distance_ratio"),
        "current_text_chars":   v.get("current_text_chars"),
        # ── Verifier outcome ─────────────────────────────────────────────
        "verdict":              v.get("verdict"),
        "confidence":           v.get("confidence"),
        "verifier_reason":      v.get("reason", ""),
        "verifier_identity":    v.get("identity", {}),
        "verifier_supporting_sentences":  v.get("supporting_sentences", []),
        "verifier_highlight_type":        v.get("highlight_type", "sentence"),
        "verifier_primary_support_index": v.get("primary_support_index", 0),
        "verifier_tokens":      v.get("_tokens"),   # actual input/output token counts from Bedrock
        # ── Reranker scores ──────────────────────────────────────────────
        "cross_encoder_score":  v.get("cross_encoder_score") or sb.get("ce"),
        "agg_score":            v.get("agg_score"),
        "score_breakdown":      sb,
        "agg_score_breakdown":  v.get("agg_score_breakdown"),
        # ── Highlight / match (populated for YES/MAYBE) ──────────────────
        "match_span":           v.get("match_span", ""),
        "context_sentence":     v.get("context_sentence", ""),
        "highlight_score":      v.get("highlight_score"),
        "match_method":         v.get("match_method", ""),
        "match_page":           v.get("match_page"),
        # ── Retrieval provenance ─────────────────────────────────────────
        "retrieval_object_type":  obj.get("type"),
        "retrieval_object_id":    object_id,
        "retrieval_object_id_base": _base_object_id(object_id),
        "retrieval_parent_chunk_id": parent_chunk_id,
        "retrieval_chunk_id":     retrieval_chunk_id,
        "retrieval_heading_path": obj.get("heading_path"),
        "retrieval_section":      obj.get("section_category") or obj.get("section"),
        # ── Full indexed object ──────────────────────────────────────────
        "indexed_object": {
            # Identity
            "object_id":          obj.get("object_id"),
            "parent_chunk_id":    obj.get("parent_chunk_id"),
            "type":               obj.get("type"),
            # Location
            "page":               obj.get("page"),
            "bbox":               obj.get("bbox"),
            "position":           obj.get("position"),
            "global_position":    obj.get("global_position"),
            "document_position":  obj.get("document_position"),
            # ── Text context (all layers) ─────────────────────────────────
            "prev_sentence_text": obj.get("prev_sentence_text", ""),
            "text":               obj.get("text", ""),
            "next_sentence_text": obj.get("next_sentence_text", ""),
            "paragraph_text":     obj.get("paragraph_text", ""),
            "context_chunk_text": ctx_text or obj.get("context_chunk_text", ""),
            # ── Document structure ────────────────────────────────────────
            "section_category":   obj.get("section_category"),
            "heading_path":       obj.get("heading_path"),
            "semantic_path":      obj.get("semantic_path"),
            "section_confidence": obj.get("section_confidence"),
            # ── NER ───────────────────────────────────────────────────────
            "entities":           obj.get("entities", []),
            # ── Clinical facts ────────────────────────────────────────────
            "facts":              obj.get("facts", {}),
            "effective_facts":    obj.get("effective_facts"),
            "statement_type":     obj.get("statement_type"),
            "study_context":      obj.get("study_context"),
            "clinical_relations": obj.get("clinical_relations", []),
        },
    }


def _clean_result(result: dict) -> dict:
    """Strip embedding vectors, raw context blobs — keep only what matters."""
    ci = result.get("ci", {})

    # Candidates the verifier rejected (verdict=NO). Included here so evaluators
    # can review what was retrieved-but-rejected and label false negatives.
    def _provenance(v: dict) -> dict:
        obj = (v.get("matched_object") or {})
        object_id = obj.get("object_id")
        parent_chunk_id = obj.get("parent_chunk_id")
        retrieval_chunk_id = parent_chunk_id or v.get("chunk_id")
        return {
            "retrieval_object_type": obj.get("type"),
            "retrieval_object_id":   object_id,
            "retrieval_object_id_base": _base_object_id(object_id),
            "retrieval_parent_chunk_id": parent_chunk_id,
            "retrieval_chunk_id":   retrieval_chunk_id,
            "retrieval_heading_path": obj.get("heading_path"),
            "retrieval_section":     obj.get("section_category") or obj.get("section"),
            "retrieval_origin":      v.get("retrieval_origin", "direct_unknown"),
            "selection_reason":      v.get("selection_reason"),
            "literal_match_count":   v.get("literal_match_count"),
            "context_strategy":      v.get("context_strategy"),
            "matched_distance":      v.get("matched_distance"),
            "distance_ratio":        v.get("distance_ratio"),
            "current_text_chars":    v.get("current_text_chars"),
        }

    rejected_hits = [
        {
            "chunk_id":      v.get("chunk_id", ""),
            "page_start":    v.get("page_start"),
            "page_end":      v.get("page_end"),
            "text":          (v.get("context", {}).get("current_text", "") or v.get("text", ""))[:500],
            "verdict":       v.get("verdict"),
            "confidence":    v.get("confidence"),
            "reason":        v.get("reason", ""),
            "retriever":     v.get("retriever", ""),
            "sources":       v.get("sources", []),
            "agg_score":     v.get("agg_score"),
            "score_breakdown": v.get("score_breakdown"),
            "agg_score_breakdown": v.get("agg_score_breakdown"),
            "indexed_object": _indexed_object(v),
            **_provenance(v),
        }
        for v in result.get("verified_candidates", [])
        if v.get("verdict") == "NO"
    ]

    # Candidates that never reached Claude (verdict=SKIP: cross_encoder_score below threshold).
    # Included so evaluators can see what the reranker filtered out before LLM verification.
    skipped_hits = [
        {
            "chunk_id":            v.get("chunk_id", ""),
            "page_start":          v.get("page_start"),
            "page_end":            v.get("page_end"),
            "text":                (v.get("context", {}).get("current_text", "") or v.get("text", ""))[:500],
            "verdict":             v.get("verdict"),
            "cross_encoder_score": v.get("cross_encoder_score"),
            "agg_score":           v.get("agg_score"),
            "reason":              v.get("reason", ""),
            "sources":             v.get("sources", []),
            "score_breakdown":     v.get("score_breakdown"),
            "agg_score_breakdown": v.get("agg_score_breakdown"),
            "indexed_object":      _indexed_object(v),
            **_provenance(v),
        }
        for v in result.get("verified_candidates", [])
        if v.get("verdict") == "SKIP"
    ]


    return {
        "search_id":        result.get("search_id"),
        "ci_id":            ci.get("id"),
        "ci_text":          ci.get("knownCI", ""),
        "ci_type":          result.get("classification", {}).get("ci_type"),
        "strategies":       result.get("classification", {}).get("strategies", []),
        # Full enriched CI metadata (everything except dense/sparse vectors)
        "ci":               _ci_metadata(ci),
        "object_type_stats": _object_type_stats([result]),
        "candidates_found": len(result.get("candidates", [])),
        # ── Per-candidate detail (all verdicts) ───────────────────────────────
        # One entry per verified_candidate — gives full candidate-level
        # granularity for the CSV exporter and for manual review.
        # Replaces the separate rejected_hits / skipped_hits split for
        # downstream tools; both are kept below for backward compatibility.
        "candidates": [_full_candidate_record(v) for v in result.get("verified_candidates", [])],
        "final_hits":       [_hit_with_provenance(h) for h in result.get("final_hits", [])],
        "rejected_hits":    rejected_hits,
        "skipped_hits":     skipped_hits,
        "ce_histogram":     result.get("ce_histogram"),
        "timings":          result.get("timings", {}),
    }


def _build_result(req: dict) -> dict:
    """Trim the request to a serialisable result dict (simple return to orchestrator)."""
    return {
        "ci_id":          req["ci"].get("id"),
        "search_id":      req.get("search_id"),
        "final_hits":     req.get("final_hits", []),
    }


def _upload_debug_json_to_s3(debug_json: dict, search_id: str, batch_idx: int, document_id: str, tenant_name: str) -> str:
    """Upload detailed debug JSON to S3 and return the S3 URL."""
    try:
        import boto3
        s3 = boto3.client("s3", region_name=AWS_REGION)
        bucket = SEARCH_RESULTS_DEBUG_BUCKET
        
        from datetime import datetime as _dt
        s3_key = f"{RESULTS_DEBUG_PREFIX}/{tenant_name}/{search_id}/{document_id}/batch/{batch_idx}/debug.json"
        
        s3.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=json.dumps(debug_json, indent=2, ensure_ascii=False),
            ContentType="application/json",
        )
        s3_url = f"s3://{bucket}/{s3_key}"
        logger.info("[S3] debug results uploaded to %s", s3_url)
        return s3_url
    except Exception as exc:
        logger.warning("[S3] failed to upload debug JSON: %s", exc)
        return ""


# ── Lambda handler ─────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    search_id   = event.get("search_id", str(uuid.uuid4()))
    batch_idx   = event.get("batch_idx", 0)
    enriched_cis = event.get("cis", [])
    document_id  = event.get("document_id", "")
    tenant_name  = event.get("tenant_name", "")
    doc_context  = event.get("document_context", {})
    skip_rerank  = bool(event.get("skip_rerank", False))
    skip_verify  = bool(event.get("skip_verify",  False))
    
    n_workers    = SEARCH_CI_WORKERS

    # ── Set context variables for all logs (automatically injected by SearchContextFilter) ────────────────────────────────
    _ctx_tenant.set(tenant_name)
    _ctx_document_id.set(document_id)
    _ctx_search_id.set(search_id)
    _ctx_batch_idx.set(str(batch_idx))
    
    # Test log to verify this version is deployed
    logger.info("🔥 SEARCH WORKER VERSION 2026-08-20 — context vars active")
    logger.info("[SearchWorker] start cis=%d", len(enriched_cis))

    all_reqs = [
        {
            "search_id":        f"{search_id}-{i}",
            "document_id":      document_id,
            "ci":               ci,
            "document_context": doc_context,
            "_st":              {},
            "_failed":          False,
            "_early_exit":      False,
            "_ci_idx":          i,
        }
        for i, ci in enumerate(enriched_cis)
    ]

    t_total = time.perf_counter()
    all_reqs, stage_wall = _run_pipeline(all_reqs, skip_rerank, skip_verify, n_workers, tenant_name)
    wall_time = round(time.perf_counter() - t_total, 3)

    # ── Track CI-level success/failure ─────────────────────────────────────────
    failed_cis = [r for r in all_reqs if r.get("_failed")]
    completed_cis = [r for r in all_reqs if not r.get("_failed")]
    
    # Extract detailed failure info for each failed CI
    ci_failures = []
    for req in failed_cis:
        ci_failures.append({
            "ci_id": req["ci"].get("id"),
            "ci_text": req["ci"].get("knownCI", ""),
            "error_type": req.get("_failure", {}).get("error_type", "unknown"),
            "error": req.get("_failure", {}).get("error", ""),
            "stage": req.get("_failure", {}).get("stage", ""),
        })
    
    # NOW save debug JSON with complete failure information
    s3_url = _save_results_debug_s3(all_reqs, event, wall_time, len(enriched_cis), 
                                     len(completed_cis), len(failed_cis), ci_failures)
    
    # ── Simple return to orchestrator ──────────────────────────────────────────
    results = [_build_result(r) for r in completed_cis]  # Only return completed CIs
    total_hits = sum(len(r.get("final_hits", [])) for r in results)
    
    logger.info(
        "[SearchWorker] done wall=%.1fs "
        "cis_total=%d completed=%d failed=%d hits=%d",
        wall_time,
        len(enriched_cis), len(completed_cis), len(failed_cis), total_hits
    )

    return {
        "document_id":   document_id,
        "search_id":     search_id,
        "batch_idx":     batch_idx,
        "n_cis":         len(enriched_cis),
        "completed_cis": len(completed_cis),
        "failed_cis":    len(failed_cis),
        "ci_failures":   ci_failures,  # NEW: detailed failure info
        "results":       results,
        "stage_wall":    stage_wall,
        "wall_time":     wall_time,
        "debug_s3_url":  s3_url,
    }
