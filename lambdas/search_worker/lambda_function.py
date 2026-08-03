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
import re
import sys
import threading
import time
import types
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_task_root_env = os.environ.get("LAMBDA_TASK_ROOT")
if _task_root_env and (Path(_task_root_env) / "lambdas").exists():
    ROOT = Path(_task_root_env)
else:
    ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Env config ─────────────────────────────────────────────────────────────────
OPENSEARCH_ENDPOINT    = os.environ.get("OPENSEARCH_ENDPOINT", "localhost")
OPENSEARCH_INDEX       = os.environ.get("OPENSEARCH_INDEX", "document-chunks")
SEMANTIC_OBJECTS_INDEX = os.environ.get("SEMANTIC_OBJECTS_INDEX", "semantic-objects")
OPENSEARCH_CI_INDEX    = os.environ.get("OPENSEARCH_CI_INDEX", "ci-objects")
AWS_REGION             = os.environ.get("AWS_REGION", "us-east-1")
BEDROCK_REGION         = os.environ.get("BEDROCK_REGION", AWS_REGION)
VERIFIER_MODEL         = os.environ.get("VERIFIER_MODEL",
                                         "eu.anthropic.claude-haiku-4-5-20251001-v1:0")
EMBEDDING_MODEL        = os.environ.get("EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")

# ── Lazy singletons ────────────────────────────────────────────────────────────
_loaded: dict[str, types.ModuleType] = {}
_os_client = None
_aws: dict = {}


def _get(service: str, region: str | None = None):
    key = f"{service}:{region or ''}"
    if key not in _aws:
        import boto3
        _aws[key] = boto3.client(service, region_name=region) if region else boto3.client(service)
    return _aws[key]


def _get_os():
    global _os_client
    if _os_client is None:
        import boto3
        from opensearchpy import OpenSearch, RequestsHttpConnection
        from requests_aws4auth import AWS4Auth
        frozen  = boto3.Session().get_credentials().get_frozen_credentials()
        awsauth = AWS4Auth(frozen.access_key, frozen.secret_key, AWS_REGION, "es",
                          session_token=frozen.token)
        _os_client = OpenSearch(
            hosts=[{"host": OPENSEARCH_ENDPOINT, "port": 443}],
            http_auth=awsauth, use_ssl=True, verify_certs=True,
            connection_class=RequestsHttpConnection,
        )
    return _os_client


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
    text = ci.get("knownCI", "")
    nums = re.findall(r"\b\d+(?:\.\d+)?(?:%|ms|kg|mg|ml|L|mmol|μg)?\b", text)
    if not nums:
        return None
    patterns = [re.escape(n) for n in nums if len(n) >= 2]
    return re.compile("|".join(patterns), re.IGNORECASE) if patterns else None


def _candidate_confidence(c: dict) -> float:
    return round(
        0.5 * min(max(c.get("agg_score", 0.0) * 2.0, 0.0), 1.0)
        + 0.3 * max(0.0, 1.0 + c.get("zero_id_pen",    0.0) / 0.4)
        + 0.2 * max(0.0, 1.0 + c.get("zero_enrich_pen", 0.0) / 0.25),
        3,
    )


# ── Evidence classification (Bedrock) ─────────────────────────────────────────

def _strip_fence(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$",          "", text.strip())
    brace = text.find("{")
    return text[brace:] if brace > 0 else text


def _classify_evidence(ci_text: str, hit: dict, doc_ctx: dict) -> dict:
    span    = (hit.get("match_span") or hit.get("text", ""))[:400]
    doc_tag = ""
    if doc_ctx:
        drugs   = ", ".join(doc_ctx.get("primary_drugs", [])[:2])
        studies = ", ".join(doc_ctx.get("study_ids", [])[:1])
        if drugs or studies:
            doc_tag = f"Document — Drug: {drugs} | Study: {studies}\n\n"

    prompt = (
        f"You are a clinical evidence analyst.\n\n"
        f"{doc_tag}"
        f"Confidential Information (CI):\n\"{ci_text}\"\n\n"
        f"Matched excerpt (page {hit.get('match_page', '?')}):\n\"{span}\"\n\n"
        f"Classify using exactly one label: DIRECT, SUPPORTING, RELATED_OBJECTIVE, "
        f"RELATED_PROTOCOL, RELATED_DOSE, RELATED_POPULATION, RELATED_SAFETY, "
        f"RELATED_EFFICACY, RELATED_DEFINITION\n\n"
        f"Reply ONLY with valid JSON:\n"
        f"{{\"evidence_type\": \"<label>\", \"confidence\": <0.0-1.0>, "
        f"\"reason\": \"<one sentence>\"}}"
    )
    try:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 160,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp      = _get("bedrock-runtime", BEDROCK_REGION).invoke_model(
            modelId=VERIFIER_MODEL, contentType="application/json",
            accept="application/json", body=json.dumps(body).encode(),
        )
        resp_body = json.loads(resp["body"].read())
        text      = _strip_fence(resp_body["content"][0]["text"].strip())
        usage     = resp_body.get("usage", {})
        parsed    = json.loads(text)
        ev = parsed.get("evidence_type", "RELATED_EFFICACY")
        if ev == "RELATED":
            ev = "RELATED_EFFICACY"
        return {
            "evidence_type":       ev,
            "evidence_confidence": float(parsed.get("confidence", 0.5)),
            "evidence_reason":     parsed.get("reason", ""),
            "_ec_tokens":          {"input": usage.get("input_tokens", 0),
                                    "output": usage.get("output_tokens", 0)},
        }
    except Exception as exc:
        return {
            "evidence_type": "RELATED_EFFICACY", "evidence_confidence": 0.0,
            "evidence_reason": f"classification failed: {exc}",
            "_ec_tokens": {"input": 0, "output": 0},
        }


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
    results: list[dict] = []
    timings: dict[str, float] = {}

    def _run(strat: str) -> tuple[str, dict, float]:
        mod = _load(RETRIEVER_MAP[strat], f"search_{strat}")
        _inject_os(mod)
        t0     = time.perf_counter()
        result = mod._process(req)
        return strat, result, round(time.perf_counter() - t0, 3)

    valid = [s for s in strategies if s in RETRIEVER_MAP]
    with ThreadPoolExecutor(max_workers=len(valid) or 1) as pool:
        for strat, result, elapsed in [f.result() for f in
                                        [pool.submit(_run, s) for s in valid]]:
            timings[strat] = elapsed
            results.append(result)

    req["retriever_results"]       = results
    req["_st"]["retrievers"]       = timings
    req["_st"]["retrievers_total"] = round(max(timings.values(), default=0), 3)
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
        req["ranked_candidates"] = [
            {**c, "cross_encoder_score": c.get("agg_score", 0.0)}
            for c in req.get("expanded_candidates", [])
        ]
    else:
        mod = _load("search/reranker", "search_reranker")
        req = mod._process(req)
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
            "per_ec_call_s": {}, "n_ec_calls": 0,
            "actual_ec_tokens": {"input": 0, "output": 0},
        })
        return req
    t0        = time.perf_counter()
    ec_times: list[float]     = []
    ec_tokens: dict[str, int] = {"input": 0, "output": 0}
    hits      = req.get("final_hits", [])
    ci_text   = req["ci"].get("knownCI", "")
    doc_ctx   = req.get("document_context", {})

    if hits and not skip_verify:
        for hit in hits:
            if hit.get("verdict") in ("YES", "MAYBE"):
                _t  = time.perf_counter()
                ec  = _classify_evidence(ci_text, hit, doc_ctx)
                ec_times.append(round(time.perf_counter() - _t, 3))
                tok = ec.pop("_ec_tokens", {"input": 0, "output": 0})
                ec_tokens["input"]  += tok["input"]
                ec_tokens["output"] += tok["output"]
                ec = _calibrate_evidence(hit, ec)
                hit.update(ec)
                if _is_related(ec["evidence_type"]):
                    hit["verdict"] = "RELATED"
                elif ec["evidence_type"] == "UNRELATED":
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

def _run_pipeline(all_reqs: list[dict], skip_rerank: bool, skip_verify: bool,
                  n_workers: int) -> tuple[list[dict], dict[str, float]]:
    """Run stage-parallel pipeline, return (all_reqs, stage_wall)."""
    # Reranker uses max_workers=1 to serialise CrossEncoder.predict() calls.
    STAGES: list[tuple[str, Any, int]] = [
        ("S1:classify",          lambda r: _s1_classify(r),                            n_workers),
        ("S2:retrieve",          lambda r: _s2_retrieve(r),                            n_workers),
        ("S3:aggregate",         lambda r: _s3_aggregate(r),                           n_workers),
        ("S4:context_expand",    lambda r: _s4_context_expand(r),                      n_workers),
        ("S5:rerank",            lambda r: _s5_rerank(r, skip_rerank),                 1),
        ("S6:llm_verify",        lambda r: _s6_llm_verify(r, skip_verify),             n_workers),
        ("S7:highlight",         lambda r: _s7_highlight_extract(r),                   n_workers),
        ("S8:merge",             lambda r: _s8_merge(r),                               n_workers),
        ("S9:evidence_classify", lambda r: _s9_evidence_classify(r, skip_verify),      n_workers),
    ]

    stage_wall: dict[str, float] = {}
    for stage_key, stage_fn, stage_workers in STAGES:
        active = sum(1 for r in all_reqs if not r.get("_failed") and not r.get("_early_exit"))
        if active == 0:
            stage_wall[stage_key] = 0.0
            continue
        t_stage = time.perf_counter()
        with ThreadPoolExecutor(max_workers=stage_workers) as pool:
            all_reqs = list(pool.map(stage_fn, all_reqs))
        stage_wall[stage_key] = round(time.perf_counter() - t_stage, 3)
        logger.info("[SearchWorker] %s done in %.1fs (%d active)",
                    stage_key, stage_wall[stage_key], active)

    return all_reqs, stage_wall


def _build_result(req: dict) -> dict:
    """Trim the request to a serialisable result dict."""
    st = req.get("_st", {})
    return {
        "ci_id":          req["ci"].get("id"),
        "search_id":      req.get("search_id"),
        "final_hits":     req.get("final_hits", []),
        "skipped_hits":   req.get("skipped_hits", []),
        "timings": {
            "classifier":              st.get("classifier",            0.0),
            "retrievers_total":        st.get("retrievers_total",      0.0),
            "aggregator":              st.get("aggregator",            0.0),
            "context_expander":        st.get("context_expander",      0.0),
            "reranker":                st.get("reranker",              0.0),
            "llm_verifier":            st.get("llm_verifier",          0.0),
            "highlight_extractor":     st.get("highlight_extractor",   0.0),
            "merger":                  st.get("merger",                0.0),
            "evidence_classification": st.get("evidence_classification", 0.0),
            "n_candidates_to_verifier": st.get("n_candidates_to_verifier", 0),
            "actual_verifier_tokens":  st.get("actual_verifier_tokens", {"input": 0, "output": 0}),
            "actual_ec_tokens":        st.get("actual_ec_tokens", {"input": 0, "output": 0}),
        },
    }


# ── Lambda handler ─────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    search_id   = event.get("search_id", str(uuid.uuid4()))
    batch_idx   = event.get("batch_idx", 0)
    enriched_cis = event.get("cis", [])
    document_id  = event.get("document_id", "")
    doc_context  = event.get("document_context", {})
    skip_rerank  = bool(event.get("skip_rerank", False))
    skip_verify  = bool(event.get("skip_verify",  False))
    n_workers    = int(event.get("workers", len(enriched_cis))) or 1

    logger.info("[SearchWorker] start search_id=%s batch_idx=%d cis=%d doc=%s",
                search_id, batch_idx, len(enriched_cis), document_id)

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
    all_reqs, stage_wall = _run_pipeline(all_reqs, skip_rerank, skip_verify, n_workers)
    wall_time = round(time.perf_counter() - t_total, 3)

    results = [_build_result(r) for r in all_reqs]
    logger.info("[SearchWorker] done search_id=%s batch_idx=%d wall=%.1fs hits=%d",
                search_id, batch_idx, wall_time,
                sum(len(r.get("final_hits", [])) for r in results))

    return {
        "search_id":  search_id,
        "batch_idx":  batch_idx,
        "results":    results,
        "stage_wall": stage_wall,
        "wall_time":  wall_time,
    }
