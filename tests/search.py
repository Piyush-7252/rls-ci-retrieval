"""
Search Pipeline Test — Real AWS, No Mocks
==========================================
Runs the full search pipeline against real AWS services using an already-indexed
document and enriches CIs inline via the CI pipeline before searching.

Pipeline flow executed sequentially
-------------------------------------
  CI Enrichment (inline):
    raw CI  →  ci/normalize  →  ci/ner  →  ci/ontology  →  ci/embedding
  
  Search:
    enriched CI  →  search/classifier
                →  [selected retrievers in sequence, simulating parallel]
                →  search/aggregator
                →  search/context_expander
                →  search/reranker          (Bedrock Claude)
                →  search/llm_verifier      (Bedrock Claude)
                →  search/merger
                →  print final_hits

Services used (all real, no mocks)
------------------------------------
  Comprehend Medical  CI NER (eu-west-1)
  Bedrock Titan       CI embedding (eu-west-1)
  Bedrock Claude      reranker + LLM verifier (eu-west-1)
  OpenSearch          rls-dev cluster — document-chunks index (eu-west-1)

Pre-requisites
--------------
  The document pipeline must have run at least once so that
  "document-chunks" in OpenSearch contains chunks for DOCUMENT_ID.
  (Run s3_pipeline_test.py first if needed.)

  AWS credentials must be exported in the calling shell:
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    export AWS_SESSION_TOKEN=...

Usage
-----
    python tests/search_test.py
    python tests/search_test.py --ci-file localfiles/ci/ahmedCis.json --max-cis 3
    python tests/search_test.py --ci-index 0            # single CI by index
    python tests/search_test.py --skip-rerank           # skip Bedrock reranker
    python tests/search_test.py --skip-verify           # skip Bedrock LLM verifier
    python tests/search_test.py --verbose
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import logging
import math
import os
import sys
import threading
import time
import types
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

# ─── project root on sys.path ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DOCUMENT_ID          = "Combined_REDACTED_CSR-Full-co-jnj-64407564"
OPENSEARCH_ENDPOINT  = (
    "search-rls-dev-rhitzxwnctmuyq2l4kny5kwelu.eu-west-1.es.amazonaws.com"
)
AWS_REGION           = "eu-west-1"
EMBEDDING_MODEL      = "amazon.titan-embed-text-v2:0"
RERANKER_MODEL       = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
VERIFIER_MODEL       = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_CI_FILE      = str(ROOT / "localfiles" / "ci" / "ahmedCis.json")

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
DOCUMENT_ASSETS_FILE = str(ROOT / "localfiles" / "assets" / "document_assets.json")

# ─── env vars (set before any Lambda module is imported) ─────────────────────
os.environ.update(
    {
        "AWS_DEFAULT_REGION":       AWS_REGION,
        "AWS_REGION":               AWS_REGION,
        "BEDROCK_REGION":           AWS_REGION,
        "OPENSEARCH_ENDPOINT":      OPENSEARCH_ENDPOINT,
        "OPENSEARCH_INDEX":         "document-chunks",
        "OPENSEARCH_CI_INDEX":      os.environ.get("OPENSEARCH_CI_INDEX", "ci-objects"),
        "SEMANTIC_OBJECTS_INDEX":   "semantic-objects",
        "NER_MODEL":                "comprehend-medical",
        "EMBEDDING_MODEL":          EMBEDDING_MODEL,
        "RERANKER_MODEL":           RERANKER_MODEL,
        "VERIFIER_MODEL":           VERIFIER_MODEL,
        "RETRIEVER_TOP_K":          "10",
        "LLM_VERIFY_TOP_N":        "5",
        "MIN_RERANK_SCORE":        "2.0",   # low threshold for test (index is small)
        # Fan-out ARNs unused in sequential mode
        "CI_NER_LAMBDA_ARN":        "",
        "CI_ONTOLOGY_LAMBDA_ARN":   "",
        "CI_EMBEDDING_LAMBDA_ARN":  "",
        "CI_STORE_LAMBDA_ARN":      "",
        "CONTEXT_EXPANDER_LAMBDA_ARN": "",
        "RERANKER_LAMBDA_ARN":      "",
        "LLM_VERIFIER_LAMBDA_ARN":  "",
        "MERGER_LAMBDA_ARN":        "",
    }
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("search_test")


# ─────────────────────────────────────────────────────────────────────────────
# Module loader  (same pattern as s3_pipeline_test.py)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Real OpenSearch client
# ─────────────────────────────────────────────────────────────────────────────

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
        timeout          = 60,
        max_retries      = 2,
        retry_on_timeout = True,
    )
    return _os_client


def _inject_os(module):
    """Inject the shared OS client into a Lambda module's singleton."""
    if hasattr(module, "_os_client"):
        module._os_client = _build_os_client()


# ───────────────────────────────────────────────────────────────────────────────
# Document fingerprint helpers
# ───────────────────────────────────────────────────────────────────────────────

def _load_document_context(document_id: str) -> dict:
    """Load the document fingerprint from document_assets.json for the given document_id."""
    path = Path(DOCUMENT_ASSETS_FILE)
    if not path.exists():
        return {}
    try:
        with path.open() as fh:
            all_contexts = json.load(fh)
        return all_contexts.get(document_id, {})
    except Exception as exc:
        logger.warning("[document_context] failed to load %s: %s", DOCUMENT_ASSETS_FILE, exc)
        return {}


def _lookup_ci_from_index(raw_ci: dict, ci_id: int) -> dict | None:
    """Try fetching an already-enriched CI from the ci-objects OpenSearch index.

    Returns a fully reconstructed enriched CI dict if found, preserving the raw
    CI’s original fields (assets, category, etc.) via shallow merge.  Returns
    None if the CI is not yet indexed so the caller can fall back to inline enrichment.
    """
    try:
        client   = _build_os_client()
        ci_index = os.environ.get("OPENSEARCH_CI_INDEX", "ci-objects")
        resp     = client.get(index=ci_index, id=str(ci_id), ignore=[404])
        if not resp.get("found"):
            return None
        doc = resp["_source"]
        # Merge: raw CI fields first (keeps assets, justificationText, etc.),
        # then overwrite with the enriched pipeline fields from the index.
        #
        # Enrichment fields (facts, effective_facts, clinical_identity, …) must be
        # restored so the reranker comparators and enrichment_status diagnostics
        # receive a fully populated CI object — not just the NLP pipeline structs.
        from shared.opensearch_enrichment import ENRICHMENT_DEFAULTS
        enrichment_fields = {k: doc.get(k, default) for k, default in ENRICHMENT_DEFAULTS.items()}
        return {
            **raw_ci,
            **enrichment_fields,           # facts, effective_facts, clinical_identity, treatment_identity, …
            "entities": doc.get("entities", []),   # top-level alias expected by reranker diagnostics
            "knownCI": doc.get("known_ci", raw_ci.get("knownCI", "")),
            "normalization": {
                "normalized_text":     doc.get("normalized_text", ""),
                "tokens":              doc.get("tokens", []),
                "abbreviations_found": {},
            },
            "ner": {
                "entities": doc.get("entities", []),
                "model":    doc.get("ner_model") or doc.get("ner", {}).get("model") or "gliner",
            },
            "ontology": {
                "expansions":     doc.get("ontology_expansions", []),
                "synonyms":       doc.get("ontology_synonyms", {}),
                "regex_patterns": doc.get("regex_patterns", []),
            },
            "embedding": {
                "dense_vector":  doc.get("dense_vector", []),
                "sparse_vector": doc.get("sparse_vector", {}),
                "model":         doc.get("embedding_model", EMBEDDING_MODEL),
                "dimensions":    len(doc.get("dense_vector", [])),
            },
        }
    except Exception as exc:
        logger.warning("[lookup_ci] failed ci_id=%s: %s", ci_id, exc)
        return None


def _classify_evidence(ci_text: str, hit: dict, doc_ctx: dict) -> dict:
    """Classify a verified YES/MAYBE hit into evidence tiers.

    DIRECT      — excerpt explicitly states the CI (same facts, same drug, same data)
    SUPPORTING  — excerpt provides data/context that supports the CI concept
    RELATED_OBJECTIVE   — same study objective or endpoint structure, different drug/arm
    RELATED_PROTOCOL    — same study design / part / phase reference, no direct CI match
    RELATED_DOSE        — dose levels, RP2D, dose escalation context; not the CI itself
    RELATED_POPULATION  — patient demographics, enrollment criteria, baseline characteristics
    RELATED_SAFETY      — adverse events, toxicity, safety profile data
    RELATED_EFFICACY    — efficacy outcomes (ORR, DOR, PFS, OS) not directly addressing the CI
    RELATED_DEFINITION  — abbreviation legends, glossary entries, figure keys
    """
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
        f"Classify the evidence relationship using exactly one of these labels:\n"
        f"- DIRECT: excerpt explicitly states or reproduces the CI (same facts, same drug, same numbers)\n"
        f"- SUPPORTING: excerpt provides data or context that supports the CI concept\n"
        f"- RELATED_OBJECTIVE: related through a shared study objective or endpoint, different drug/arm\n"
        f"- RELATED_PROTOCOL: related through study design, part/phase structure, or protocol reference\n"
        f"- RELATED_DOSE: related through dose levels, RP2D selection, or dose escalation context\n"
        f"- RELATED_POPULATION: related through patient demographics, eligibility, or baseline data\n"
        f"- RELATED_SAFETY: related through adverse events, toxicity, or safety profile\n"
        f"- RELATED_EFFICACY: related through efficacy outcomes (ORR, DOR, PFS, OS) not directly addressing the CI\n"
        f"- RELATED_DEFINITION: abbreviation legend, glossary entry, figure key, or acronym definition\n\n"
        f"Reply ONLY with valid JSON:\n"
        f"{{\"evidence_type\": \"DIRECT\"|\"SUPPORTING\"|\"RELATED_OBJECTIVE\"|\"RELATED_PROTOCOL\"|"
        f"\"RELATED_DOSE\"|\"RELATED_POPULATION\"|\"RELATED_SAFETY\"|\"RELATED_EFFICACY\"|"
        f"\"RELATED_DEFINITION\", "
        f"\"confidence\": <0.0-1.0>, \"reason\": \"<one sentence>\"}}"
    )
    try:
        import boto3 as _boto3
        br   = _boto3.client("bedrock-runtime", region_name=AWS_REGION)
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 160,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp      = br.invoke_model(
            modelId=VERIFIER_MODEL, contentType="application/json",
            accept="application/json", body=json.dumps(body).encode(),
        )
        import re as _re
        resp_body    = json.loads(resp["body"].read())
        text         = resp_body["content"][0]["text"].strip()
        ec_usage     = resp_body.get("usage", {})
        ec_in_tok    = ec_usage.get("input_tokens", 0)
        ec_out_tok   = ec_usage.get("output_tokens", 0)
        text = _re.sub(r"^```(?:json)?\s*", "", text)
        text = _re.sub(r"\s*```$", "", text.strip())
        brace = text.find("{")
        if brace > 0:
            text = text[brace:]
        parsed = json.loads(text)
        ev = parsed.get("evidence_type", "RELATED_EFFICACY")
        # Normalise any plain RELATED to RELATED_EFFICACY as safe default
        if ev == "RELATED":
            ev = "RELATED_EFFICACY"
        return {
            "evidence_type":       ev,
            "evidence_confidence": float(parsed.get("confidence", 0.5)),
            "evidence_reason":     parsed.get("reason", ""),
            "_ec_tokens":          {"input": ec_in_tok, "output": ec_out_tok},
        }
    except Exception as exc:
        return {
            "evidence_type":       "RELATED_EFFICACY",
            "evidence_confidence": 0.0,
            "evidence_reason":     f"classification failed: {exc}",
            "_ec_tokens":          {"input": 0, "output": 0},
        }


# ─────────────────────────────────────────────────────────────────────────────
# Evidence calibration  (Issues 1 + 2)
# ─────────────────────────────────────────────────────────────────────────────

# highlight_score thresholds
_WEAK_MQ      = 0.15   # below → weak coverage (partial sentence match)
_VERY_WEAK_MQ = 0.05   # below → very weak (barely any highlight)


def _calibrate_evidence(hit: dict, ec: dict) -> dict:
    """
    Reconcile LLM evidence classification with highlight highlight_score.

    Issue 1 — Confidence: the LLM classified whatever span was passed in;
    if that span was a poor match (low highlight_score) its confidence is
    over-stated.  Apply a quality-proportional penalty so highlight_score
    and evidence_confidence tell a coherent story.

      Formula: quality_factor = mq / (mq + 0.05)  [Michaelis-Menten]
        mq = 0.047  →  factor ≈ 0.48   (e.g. old token-overlap fallback)
        mq = 0.15   →  factor ≈ 0.75
        mq = 0.65   →  factor ≈ 0.93
        mq = 0.95   →  factor ≈ 0.95   (ExactScorer object hit)

    Issue 2 — Evidence type: text_fallback with mq < _WEAK_MQ only
    partially covers the CI.  Downgrade RELATED_* → SUPPORTING so
    reviewers see it as contextual support, not corroborating evidence.
    """
    mq     = hit.get("highlight_score", 1.0)
    method = hit.get("match_method", "")
    ev_t   = ec.get("evidence_type", "RELATED_EFFICACY")
    ev_c   = ec.get("evidence_confidence", 0.5)

    result = dict(ec)

    # Issue 2: downgrade RELATED_* → SUPPORTING for weak text_fallback hits
    is_fallback = method in ("text_fallback", "text_fallback_skipped")
    if is_fallback and mq < _WEAK_MQ and ev_t.startswith("RELATED_"):
        result["evidence_type"]   = "SUPPORTING"
        result["evidence_reason"] = (
            result.get("evidence_reason", "")
            + f" [downgraded from {ev_t}: low match quality {mq:.3f}]"
        )

    # Issue 1: quality-proportional confidence penalty
    if method == "text_fallback_skipped":
        # No highlight was extracted at all (exact object hit dominated);
        # apply a strong fixed penalty — the span the LLM saw was the raw
        # chunk text, not a precise match.
        quality_factor = 0.25
    elif mq > 0:
        quality_factor = mq / (mq + 0.05)
    else:
        quality_factor = 0.10   # mq = 0 but not skipped (edge case)

    result["evidence_confidence"] = round(ev_c * quality_factor, 3)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Numeric text-presence gate
# ─────────────────────────────────────────────────────────────────────────────

_NUMERIC_GATE_TYPES: frozenset[str] = frozenset({
    "NUMERIC_SAMPLE_SIZE", "CONFIDENCE_INTERVAL", "P_VALUE",
    "HAZARD_RATIO", "ODDS_RATIO", "NUMERIC_PERCENTAGE", "MEDIAN",
    "NUMERIC", "STATISTICAL",
})


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


# ─────────────────────────────────────────────────────────────────────────────
# Search Pipeline  (sequential execution of all stages)
# ─────────────────────────────────────────────────────────────────────────────

# Retriever key → Lambda module path
RETRIEVER_MAP = {
    "literal":  "search/literal_retriever",
    "bm25":     "search/bm25_retriever",
    "vector":   "search/vector_retriever",
    "ontology": "search/ontology_retriever",
    "regex":    "search/regex_retriever",
    "ner":      "search/ner_retriever",
    "fact":     "search/fact_retriever",
    "numeric":  "search/numeric_retriever",
}


def run_search(
    enriched_ci:      dict,
    document_id:      str,
    document_context: dict | None = None,
    skip_rerank:      bool = False,
    skip_verify:      bool = False,
    verbose:          bool = False,
    diagnose_ids:     set | None = None,
) -> dict:
    search_id = f"srch-{uuid.uuid4().hex[:8]}"
    ci_text   = enriched_ci.get("knownCI", "?")
    print(f"\n{'\u2500' * 60}")
    print(f"  SEARCH  search_id={search_id}  CI=\"{ci_text}\"")
    print(f"{'\u2500' * 60}")

    req = {
        "search_id":            search_id,
        "document_id":          document_id,
        "ci":                   enriched_ci,
        "document_context":     document_context or {},
        "document_page_count":  int((document_context or {}).get("total_pages", 0)),
        "_diagnose_ci_ids":     diagnose_ids or set(),
    }
    _t_search_start = time.perf_counter()
    _timings: dict[str, object] = {}

    # ── Stage 1: Classify ────────────────────────────────────────────────────
    _t0        = time.perf_counter()
    classifier = _load("search/classifier", "search_classifier")
    req        = classifier._process(req)
    _timings["classifier"] = round(time.perf_counter() - _t0, 3)
    classification = req.get("classification", {})
    print(f"  → classifier: type={classification.get('ci_type')}  "
          f"strategies={classification.get('strategies')}  "
          f"reason={classification.get('reason')}  ({_timings['classifier']:.2f}s)")

    # ── Stage 2: Retrieve (parallel — all retrievers run concurrently) ────────
    strategies           = classification.get("strategies", list(RETRIEVER_MAP.keys()))
    retriever_results:   list[dict] = []
    _retriever_timings:  dict[str, float] = {}

    # Pre-load + inject OS before spawning threads (avoids module-cache races)
    _strat_mods: dict[str, Any] = {}
    for strategy in strategies:
        if strategy not in RETRIEVER_MAP:
            continue
        mod = _load(RETRIEVER_MAP[strategy], f"search_{strategy}")
        _inject_os(mod)
        _strat_mods[strategy] = mod

    def _run_retriever(strat: str) -> tuple[str, dict, float]:
        _t0 = time.perf_counter()
        result = _strat_mods[strat]._process(req)
        return strat, result, round(time.perf_counter() - _t0, 3)

    with ThreadPoolExecutor(max_workers=len(_strat_mods)) as _pool:
        for strat, result, elapsed in [f.result() for f in
                                        as_completed(_pool.submit(_run_retriever, s)
                                                     for s in _strat_mods)]:
            _retriever_timings[strat] = elapsed
            n_hits = len(result.get("hits", []))
            print(f"  → {strat:10s}: {n_hits} hits  ({elapsed:.2f}s)")
            if verbose and n_hits:
                for h in result["hits"][:3]:
                    print(f"      chunk={h['chunk_id']}  score={h['score']:.3f}  "
                          f"pages={h.get('page_start')}-{h.get('page_end')}")
            retriever_results.append(result)

    req["retriever_results"] = retriever_results
    _timings["retrievers"] = _retriever_timings
    _timings["retrievers_total"] = round(max(_retriever_timings.values(), default=0), 3)

    # ── Stage 3: Aggregate ───────────────────────────────────────────────────
    _t0        = time.perf_counter()
    aggregator = _load("search/aggregator", "search_aggregator")
    req        = aggregator._process(req)
    _timings["aggregator"] = round(time.perf_counter() - _t0, 3)
    candidates = req.get("candidates", [])
    print(f"  → aggregator: {len(candidates)} unique candidates  ({_timings['aggregator']:.2f}s)")
    if verbose:
        for c in candidates[:5]:
            print(f"      chunk={c['chunk_id']}  agg_score={c['agg_score']:.3f}  "
                  f"sources={c['sources']}")
    # Diagnostic score table — printed when --diagnose-ci matches this CI's id
    diagnose_ids = req.get("_diagnose_ci_ids", set())
    ci_id_int    = enriched_ci.get("id")
    if diagnose_ids and ("all" in diagnose_ids or ci_id_int in diagnose_ids):
        _print_score_table(candidates, ci_id_int, enriched_ci.get("knownCI", ""))

    if not candidates:
        print("  ✗ No candidates found — search complete with zero hits.")
        _timings["total"] = round(time.perf_counter() - _t_search_start, 3)
        _timings.setdefault("n_candidates_to_verifier", 0)
        req["timings"] = _timings
        return {**req, "final_hits": []}

    # ── Stage 4: Context Expand ──────────────────────────────────────────────
    _t0      = time.perf_counter()
    expander = _load("search/context_expander", "search_expander")
    _inject_os(expander)
    req      = expander._process(req)
    _timings["context_expander"] = round(time.perf_counter() - _t0, 3)
    expanded = req.get("expanded_candidates", [])
    print(f"  → context expander: {len(expanded)} expanded candidates  ({_timings['context_expander']:.2f}s)")
    if verbose:
        for ec in expanded[:3]:
            txt = ec.get("context", {}).get("current_text", "")[:120].replace("\n", " ")
            print(f"      chunk={ec['chunk_id']}  text=\"{txt}…\"")

    # ── Stage 5: Rerank (Bedrock Claude) ─────────────────────────────────────
    _t0 = time.perf_counter()
    if skip_rerank:
        print("  → reranker: SKIPPED")
        # promote agg_score to cross_encoder_score
        req["ranked_candidates"] = [
            {**ec, "cross_encoder_score": ec.get("agg_score", 0.0)}
            for ec in expanded
        ]
    else:
        reranker = _load("search/reranker", "search_reranker")
        req      = reranker._process(req)
        ranked   = req.get("ranked_candidates", [])
        print(f"  → reranker: {len(ranked)} ranked  "
              f"top_score={ranked[0]['cross_encoder_score']:.2f}" if ranked else "")
        if verbose:
            for r in ranked[:5]:
                print(f"      chunk={r['chunk_id']}  cross_score={r['cross_encoder_score']:.2f}  "
                      f"pages={r.get('page_start')}-{r.get('page_end')}")
    _timings["reranker"] = round(time.perf_counter() - _t0, 3)

    # ── Stage 5.5: Numeric text-presence gate ────────────────────────────────
    # Eliminate candidates where the key numeric value is simply absent from
    # the text — e.g. "Day 8" or "PROMIS 8c" must not reach the LLM verifier
    # when the CI is "n = 8" (sample size).
    _gate_ci_type = classification.get("ci_type", "") or ""
    if _gate_ci_type in _NUMERIC_GATE_TYPES:
        _gate_pat = _numeric_gate_pattern(req["ci"])
        if _gate_pat is not None:
            _passed, _gated = [], []
            for _c in req.get("ranked_candidates", []):
                _txt = ((_c.get("context") or {}).get("current_text", "")
                        or _c.get("snippet", ""))
                if _gate_pat.search(_txt):
                    _passed.append(_c)
                else:
                    _gated.append(_c)
            if _gated:
                print(f"  → numeric gate: {len(_passed)} passed, "
                      f"{len(_gated)} eliminated (key value absent from text)")
                if verbose:
                    for _g in _gated:
                        print(f"      eliminated: chunk={_g['chunk_id']}  "
                              f"xscore={_g.get('cross_encoder_score', 0):.2f}")
            req["ranked_candidates"] = _passed

    # ── Stage 5.6: Chunk deduplication ────────────────────────────────────────
    # The same chunk_id can arrive from multiple retrieval strategies (e.g.
    # bm25 + fact_retriever both return chunk_0230).  Keep only the highest-
    # agg_score copy per chunk so the LLM verifier never sees duplicate text.
    _cand_by_chunk: dict[str, dict] = {}
    for _c in req.get("ranked_candidates", []):
        _cid = _c.get("chunk_id") or _c.get("id") or ""
        if _cid not in _cand_by_chunk or _c.get("agg_score", 0) > _cand_by_chunk[_cid].get("agg_score", 0):
            _cand_by_chunk[_cid] = _c
    _before_dedup = len(req.get("ranked_candidates", []))
    req["ranked_candidates"] = list(_cand_by_chunk.values())
    if len(req["ranked_candidates"]) < _before_dedup:
        print(f"  → chunk dedup: {_before_dedup} → {len(req['ranked_candidates'])} (removed {_before_dedup - len(req['ranked_candidates'])} duplicates)")

    # ── Stage 5.7: Candidate confidence gate ──────────────────────────────────
    # Combines three aggregator quality signals into a single confidence metric.
    # Candidates below the threshold are blocked before the expensive LLM call.
    #
    # confidence = 0.5 * agg_norm + 0.3 * identity_ok + 0.2 * enrich_ok
    #   agg_norm    = min(max(agg_score * 2, 0), 1)           — 0.5→1.0; <0→0
    #   identity_ok = max(0, 1 + zero_id_pen / 0.4)          — -0.4→0.0; 0.0→1.0
    #   enrich_ok   = max(0, 1 + zero_enrich_pen / 0.25)     — -0.25→0.0; 0.0→1.0
    #
    # Threshold 0.2 blocks candidates with negative agg_score AND identity/
    # enrichment veto both fired — the structured pipeline unanimously disagrees.
    _CONF_THRESHOLD = 0.2
    def _candidate_confidence(c: dict) -> float:
        agg_s    = c.get("agg_score", 0.0)
        zero_id  = c.get("zero_id_pen",    0.0)
        zero_en  = c.get("zero_enrich_pen", 0.0)
        agg_norm   = min(max(agg_s * 2.0, 0.0), 1.0)
        id_ok      = max(0.0, 1.0 + zero_id / 0.4)
        enrich_ok  = max(0.0, 1.0 + zero_en / 0.25)
        return round(0.5 * agg_norm + 0.3 * id_ok + 0.2 * enrich_ok, 3)

    _conf_passed, _conf_gated = [], []
    for _c in req.get("ranked_candidates", []):
        if _candidate_confidence(_c) >= _CONF_THRESHOLD:
            _conf_passed.append(_c)
        else:
            _conf_gated.append({**_c, "verdict": "NO", "reason": "candidate_confidence_gate"})
    if _conf_gated:
        print(f"  → confidence gate: {len(_conf_passed)} passed, "
              f"{len(_conf_gated)} blocked (structured pipeline veto)")
        if verbose:
            for _g in _conf_gated:
                agg_s = _g.get("agg_score", 0)
                zid   = _g.get("zero_id_pen", 0)
                zen   = _g.get("zero_enrich_pen", 0)
                print(f"      blocked: chunk={_g.get('chunk_id','')[:45]}  "
                      f"agg={agg_s:.3f}  zero_id={zid}  zero_enrich={zen}")
    req["ranked_candidates"] = _conf_passed
    req.setdefault("skipped_hits", []).extend(_conf_gated)

    # ── Stage 6: LLM Verify (Bedrock Claude) ─────────────────────────────────
    _n_to_verifier = len(req.get("ranked_candidates", []))
    _timings["n_candidates_to_verifier"] = _n_to_verifier
    _t0 = time.perf_counter()
    if skip_verify:
        print("  → llm_verifier: SKIPPED")
        req["verified_candidates"] = [
            {**c, "verdict": "MAYBE", "reason": "skipped", "confidence": 0.5}
            for c in req.get("ranked_candidates", [])
        ]
    else:
        verifier = _load("search/llm_verifier", "search_verifier")
        # Wrap _verify to capture per-call timing + actual token usage
        _verify_call_times:  list[float] = []
        _verify_call_tokens: list[dict]  = []
        _orig_verify = verifier._verify
        def _timed_verify(*_a, **_kw):
            _t = time.perf_counter()
            _r = _orig_verify(*_a, **_kw)
            _verify_call_times.append(round(time.perf_counter() - _t, 3))
            _verify_call_tokens.append(_r.get("_tokens", {"input": 0, "output": 0}))
            return _r
        verifier._verify = _timed_verify
        req = verifier._process(req)
        verifier._verify = _orig_verify   # restore
        _timings["per_verifier_call_s"] = {i + 1: t for i, t in enumerate(_verify_call_times)}
        _timings["actual_verifier_tokens"] = {
            "input":  sum(t["input"]  for t in _verify_call_tokens),
            "output": sum(t["output"] for t in _verify_call_tokens),
        }
        verified = req.get("verified_candidates", [])
        yes      = sum(1 for v in verified if v.get("verdict") == "YES")
        maybe    = sum(1 for v in verified if v.get("verdict") == "MAYBE")
        _verifier_s = time.perf_counter() - _t0
        _v_times = _verify_call_times
        _v_summary = (
            f"  min={min(_v_times):.2f}s  max={max(_v_times):.2f}s  "
            f"avg={sum(_v_times)/len(_v_times):.2f}s"
        ) if _v_times else ""
        print(f"  → llm_verifier: YES={yes}  MAYBE={maybe}  "
              f"SKIP={len(verified) - yes - maybe}  ({_verifier_s:.1f}s total,"
              f" {len(_v_times)} Bedrock calls{_v_summary})")
        if verbose:
            for v in verified[:5]:
                print(f"      chunk={v['chunk_id']}  verdict={v['verdict']}  "
                      f"conf={v.get('confidence', 0):.2f}  reason={v.get('reason', '')[:80]}")
    _timings["llm_verifier"] = round(time.perf_counter() - _t0, 3)

    # ── Stage 6.5: Highlight Extraction ──────────────────────────────────────
    _t0                 = time.perf_counter()
    highlight_extractor = _load("search/highlight_extractor", "search_highlight_extractor")
    req                 = highlight_extractor._process(req)
    _timings["highlight_extractor"] = round(time.perf_counter() - _t0, 3)
    spans_found         = sum(
        1 for v in req.get("verified_candidates", [])
        if v.get("match_span") and v.get("verdict") in ("YES", "MAYBE")
    )
    print(f"  → highlight_extractor: {spans_found} highlights extracted")

    # ── Stage 7: Merge ───────────────────────────────────────────────────────
    _t0    = time.perf_counter()
    merger = _load("search/merger", "search_merger")
    req    = merger._process(req)
    _timings["merger"] = round(time.perf_counter() - _t0, 3)
    hits   = req.get("final_hits", [])

    # ── Stage 8: Evidence Classification → drives final verdict + ranking ──────
    # DIRECT / SUPPORTING → YES    RELATED_* → RELATED (contextual only)
    _EVIDENCE_STARS: dict[str, str] = {
        # Current taxonomy (RELATED_*) — matches _classify_evidence prompt
        "DIRECT":               "★★★★★",
        "SUPPORTING":           "★★★★",
        "RELATED_OBJECTIVE":    "★★★",
        "RELATED_PROTOCOL":     "★★★",
        "RELATED_DOSE":         "★★",
        "RELATED_POPULATION":   "★★",
        "RELATED_SAFETY":       "★★",
        "RELATED_EFFICACY":     "★★",
        "RELATED_DEFINITION":   "★",
        # Legacy taxonomy (SAME_*) — kept for backward compatibility
        "SAME_STUDY":           "★★★★",
        "SAME_PROTOCOL":        "★★★",
        "SAME_OBJECTIVE":       "★★★",
        "SAME_ENDPOINT":        "★★",
        "SAME_POPULATION":      "★★",
        "SAME_MECHANISM":       "★★",
        "BACKGROUND":           "★",
        "UNRELATED":            "✗",
    }
    # Sort order: lower rank = better
    _EVIDENCE_RANK: dict[str, int] = {
        # Current taxonomy
        "DIRECT":               0,
        "SUPPORTING":           1,
        "RELATED_OBJECTIVE":    2,
        "RELATED_PROTOCOL":     2,
        "RELATED_DOSE":         3,
        "RELATED_POPULATION":   3,
        "RELATED_SAFETY":       3,
        "RELATED_EFFICACY":     3,
        "RELATED_DEFINITION":   4,
        # Legacy taxonomy
        "SAME_STUDY":           1,
        "SAME_PROTOCOL":        2,
        "SAME_OBJECTIVE":       2,
        "SAME_ENDPOINT":        3,
        "SAME_POPULATION":      3,
        "SAME_MECHANISM":       3,
        "BACKGROUND":           4,
        "UNRELATED":            9,
    }

    def _is_related(ev: str) -> bool:
        return ev.startswith("SAME_") or ev.startswith("RELATED_") or ev == "BACKGROUND"

    _t0 = time.perf_counter()
    _ec_call_times: list[float] = []
    _ec_actual_tokens: dict[str, int] = {"input": 0, "output": 0}
    if hits and not skip_verify:
        doc_ctx = req.get("document_context", {})
        print(f"\n  Classifying evidence for {len(hits)} hit(s) …")
        for hit in hits:
            if hit.get("verdict") in ("YES", "MAYBE"):
                _t_ec = time.perf_counter()
                ec = _classify_evidence(ci_text, hit, doc_ctx)
                _ec_call_times.append(round(time.perf_counter() - _t_ec, 3))
                ec_tok = ec.pop("_ec_tokens", {"input": 0, "output": 0})
                _ec_actual_tokens["input"]  += ec_tok["input"]
                _ec_actual_tokens["output"] += ec_tok["output"]
                ec = _calibrate_evidence(hit, ec)   # Issues 1+2: penalty + type downgrade
                hit.update(ec)
                # Evidence type overrides the verifier verdict
                if _is_related(ec["evidence_type"]):
                    hit["verdict"] = "RELATED"
                elif ec["evidence_type"] == "UNRELATED":
                    hit["verdict"] = "NO"
                # DIRECT / SAME_STUDY keep verdict = YES
                stars = _EVIDENCE_STARS.get(ec["evidence_type"], "?")
                # Show identity breakdown if available
                ident = hit.get("identity", {})
                id_str = ""
                if ident:
                    id_score = ident.get("identity_score", "?")
                    sem_score = ident.get("semantic_score", "?")
                    id_str = f"  id={id_score:.0%} sem={sem_score:.0%}" if isinstance(id_score, float) else ""
                print(f"    p{hit.get('match_page', '?')} → {stars} {ec['evidence_type']:<18}"
                      f"  conf={ec['evidence_confidence']:.2f}{id_str}  {ec['evidence_reason'][:55]}")

        # Sort: DIRECT → SUPPORTING → RELATED_* (by sub-rank) → BACKGROUND → UNRELATED
        # then by evidence_confidence desc, cross_encoder_score desc, highlight_score desc
        hits.sort(key=lambda h: (
            _EVIDENCE_RANK.get(h.get("evidence_type", "BACKGROUND"), 4),
            -h.get("evidence_confidence", 0.0),
            -h.get("cross_encoder_score", 0.0),
            -h.get("highlight_score", 0.0),
        ))
        req["final_hits"] = hits

    _timings["evidence_classification"] = round(time.perf_counter() - _t0, 3)
    _timings["per_ec_call_s"]       = {i + 1: t for i, t in enumerate(_ec_call_times)}
    _timings["n_ec_calls"]          = len(_ec_call_times)
    _timings["actual_ec_tokens"]    = _ec_actual_tokens
    direct_n    = sum(1 for h in hits if h.get("evidence_type") == "DIRECT")
    same_study_n = sum(1 for h in hits if h.get("evidence_type") == "SAME_STUDY")
    related_n   = sum(1 for h in hits if _is_related(h.get("evidence_type", "")))
    unclass_n   = len(hits) - direct_n - same_study_n - related_n

    # Related sub-type breakdown for the summary line
    rel_subtypes = {}
    for h in hits:
        ev = h.get("evidence_type", "")
        if _is_related(ev):
            sub = ev.replace("SAME_", "")
            rel_subtypes[sub] = rel_subtypes.get(sub, 0) + 1
    rel_detail = "  ".join(f"{k}={v}" for k, v in rel_subtypes.items()) if rel_subtypes else ""

    print(f"\n  ✓ FINAL: {len(hits)} hit(s)  "
          f"[★★★★★ DIRECT={direct_n}  ★★★★ SAME_STUDY={same_study_n}  "
          f"RELATED={related_n}" +
          (f" ({rel_detail})" if rel_detail else "") +
          (f"  unclassified={unclass_n}" if unclass_n else "") + "]")

    prev_tier = None
    for i, hit in enumerate(hits, 1):
        ev    = hit.get("evidence_type", "")
        stars = _EVIDENCE_STARS.get(ev, "")
        tier  = "RELATED" if _is_related(ev) else ev

        # Print tier header only when the top-level tier changes
        if tier != prev_tier:
            if tier == "DIRECT":
                print(f"\n  {'━'*56}")
                print(f"  ★★★★★ DIRECT EVIDENCE")
                print(f"  {'━'*56}")
            elif tier == "SAME_STUDY":
                print(f"\n  {'━'*56}")
                print(f"  ★★★★ SAME STUDY")
                print(f"  {'━'*56}")
            elif tier == "RELATED":
                print(f"\n  {'─'*56}")
                print(f"  RELATED CONTEXT")
                print(f"  {'─'*56}")
            prev_tier = tier

        mq      = hit.get('highlight_score', 0)
        mq_lbl  = "✓ good" if mq >= 0.4 else ("~ partial" if mq >= 0.2 else "✗ weak")
        method  = hit.get('match_method', '?')
        econf   = hit.get('evidence_confidence', 0)
        xscore  = hit.get('cross_encoder_score', 0)
        sboost  = hit.get('section_boost', 1.0)
        section = (hit.get('matched_object') or {}).get('section') or \
                  (hit.get('matched_object') or {}).get('parent_heading') or ''
        section_s = f"  § {section[:60]}" if section else ""
        print(f"  #{i} Page {hit['page_start']}" +
              (f"–{hit['page_end']}" if hit['page_end'] != hit['page_start'] else "") +
              f"  |  {stars} {ev}  ({econf*100:.0f}%)"
              f"  xscore={xscore:.2f}  boost=×{sboost}{section_s}")
        print(f"     Span:    [{mq_lbl}  q={mq:.2f}  via={method}]  "
              f"\"{hit.get('match_span', '')[:180]}\"")
        print(f"     Reason:  {hit.get('evidence_reason', '')[:100]}")
        print(f"     Sources: {', '.join(hit['sources'])}")

    _timings["total"] = round(time.perf_counter() - _t_search_start, 3)
    req["timings"] = _timings
    return req


# ─────────────────────────────────────────────────────────────────────────────
# Result serialization
# ─────────────────────────────────────────────────────────────────────────────

def _save_results(all_results: list[dict], args, out_path: Path, wall_time: float = 0.0) -> None:
    """Write a clean, human-readable results JSON — strips large vectors/context."""
    output = {
        "run": {
            "timestamp":   datetime.now().isoformat(),
            "document_id": args.document_id,
            "ci_file":     str(args.ci_file),
            "opensearch":  OPENSEARCH_ENDPOINT,
            "skip_rerank": args.skip_rerank,
            "skip_verify": args.skip_verify,
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

    output["timing_summary"] = {
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

    output["cost_estimate"] = {
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

    with out_path.open("w") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)


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
    extra = {
        "retrieval_object_type": obj_type,
        # retrieved_type: the unit the retriever actually fetched from the index.
        # "chunk" = vector_search_chunks fallback; context_expander assigned the object.
        # "sentence"/"paragraph"/etc = object fetched directly from semantic-objects index.
        "retrieved_type":        retrieved_unit,
        # expansion_origin: context_expander's raw value preserved for debugging.
        # Format: "direct_{type}" or "via_chunk_{type}".
        "expansion_origin":      ce_origin or None,
        "retrieval_object_id":   obj.get("object_id"),
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
    base = {k: v for k, v in hit.items() if k != "matched_object"}
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
        "retrieval_object_id":    obj.get("object_id"),
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
        return {
            "retrieval_object_type": obj.get("type"),
            "retrieval_object_id":   obj.get("object_id"),
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


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Aggregator score breakdown table
# ─────────────────────────────────────────────────────────────────────────────

def _print_score_table(candidates: list[dict], ci_id: int | None, ci_text: str) -> None:
    """
    Print a per-component score table for all aggregator candidates of one CI.

    Columns
    -------
    Candidate   : truncated chunk_id  (+ object type if available)
    Vec/BM25/Lit: raw scores from each retriever  (0-1)
    Fact-R      : raw score from the fact retriever (0 when not invoked)
    Relation    : _relation_score output (0-1)
    Entity      : entity overlap Jaccard (0-1)
    Fact-O      : fact slot overlap Jaccard (0-1)
    Sect        : section multiplier (1.00 or 1.25)
    Ctx         : study-context multiplier (0.70 or 1.00)
    Final       : agg_score
    """
    short_text = ci_text[:70].replace("\n", " ")
    print(f"\n{'='*130}")
    print(f"  SCORE BREAKDOWN  CI={ci_id}  \"{short_text}{'...' if len(ci_text)>70 else ''}\"")
    print(f"{'='*130}")

    # header
    hdr = (f"  {'Candidate':<28}  {'Vec':>5}  {'BM25':>5}  {'Lit':>5}  "
           f"{'FRet':>5}  {'Rel':>5}  {'Ent':>5}  {'FOlp':>5}  "
           f"{'Ctra':>6}  {'Sect':>5}  {'Ctx':>5}  {'Raw':>6}  {'Final':>7}  Sources")
    print(hdr)
    print(f"  {'-'*126}")

    for c in candidates:
        chunk_short = c["chunk_id"].split("_chunk_")[-1]
        obj         = c.get("matched_object") or {}
        obj_type    = obj.get("type", "")[:3]
        label       = f"chunk_{chunk_short}/{obj_type}" if obj_type else f"chunk_{chunk_short}"
        bd          = c.get("score_breakdown", {})

        vec     = bd.get("vector",       0.0)
        bm25    = bd.get("bm25",         0.0)
        lit     = bd.get("literal",      0.0)
        fret    = bd.get("fact_ret",     0.0)
        rel     = bd.get("relation",     0.0)
        ent     = bd.get("entity_olap",  0.0)
        folp    = bd.get("fact_olap",    0.0)
        contra  = bd.get("contradiction", 0.0)
        sect    = bd.get("sect_mult",    1.0)
        ctx     = bd.get("ctx_mult",     1.0)
        raw     = bd.get("raw",          c.get("agg_score", 0.0))
        final   = c.get("agg_score",     0.0)
        srcs    = ",".join(c.get("sources", []))

        flag = ""
        if sect > 1.0:  flag += "S"
        if ctx  < 1.0:  flag += "C"
        if fret > 0.0:  flag += "F"
        if contra < 0:  flag += "!"

        print(f"  {label:<28}  {vec:5.3f}  {bm25:5.3f}  {lit:5.3f}  "
              f"{fret:5.3f}  {rel:5.3f}  {ent:5.3f}  {folp:5.3f}  "
              f"{contra:6.3f}  {sect:5.2f}  {ctx:5.2f}  {raw:6.4f}  {final:7.4f}  "
              f"{srcs}{' ['+flag+']' if flag else ''}")

    print(f"  {'-'*126}")
    print("  Legend: Vec=vector  BM25=bm25  Lit=literal  FRet=fact-retriever  "
          "Rel=relation  Ent=entity-overlap  FOlp=fact-slot-overlap  Ctra=contradiction")
    print("          Sect=section-boost  Ctx=context-penalty  Raw=pre-mult  Final=agg_score")
    print("          [S]=section boosted  [C]=context penalised  [F]=fact retriever fired  [!]=contradiction")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Timing + cost console helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_timing_summary(all_results: list[dict], wall_time: float) -> None:
    """Print a formatted per-stage timing table after all CIs are processed."""
    n = len(all_results)
    if not n:
        return
    all_t = [r.get("timings", {}) for r in all_results]

    def _tot(key: str) -> float:
        return sum(t.get(key, 0.0) for t in all_t)

    retriever_keys: list[str] = []
    for _t in all_t:
        for _k in _t.get("retrievers", {}):
            if _k not in retriever_keys:
                retriever_keys.append(_k)

    total_pipeline = _tot("total")
    denom = total_pipeline if total_pipeline > 0 else 1.0
    W1, W2, W3, W4 = 30, 10, 10, 8

    def _row(label: str, tot: float, ind: str = "") -> None:
        avg = tot / n
        pct = 100.0 * tot / denom
        print(f"  {ind}{label:<{W1 - len(ind)}}  {tot:>{W2}.2f}s  {avg:>{W3}.2f}s  {pct:>{W4}.1f}%")

    print(f"\n{'═' * 66}")
    print(f"  TIMING BREAKDOWN  ({n} CI{'s' if n != 1 else ''},  "
          f"wall-clock {wall_time:.1f}s,  pipeline {total_pipeline:.1f}s)")
    print(f"{'═' * 66}")
    print(f"  {'Stage':<{W1}}  {'Total':>{W2}}  {'Avg/CI':>{W3}}  {'Share':>{W4}}")
    print(f"  {'─' * W1}  {'─' * W2}  {'─' * W3}  {'─' * W4}")

    _row("Classifier",              _tot("classifier"))
    _row("Retrievers",              _tot("retrievers_total"))
    for rk in retriever_keys:
        _row(rk, sum(t.get("retrievers", {}).get(rk, 0.0) for t in all_t), ind="  ")
    _row("Aggregator",              _tot("aggregator"))
    _row("Context expander",        _tot("context_expander"))
    _row("Reranker",                _tot("reranker"))
    _row("LLM verifier",            _tot("llm_verifier"))
    _row("Highlight extractor",     _tot("highlight_extractor"))
    _row("Merger",                  _tot("merger"))
    _row("Evidence classification", _tot("evidence_classification"))

    print(f"  {'─' * W1}  {'─' * W2}  {'─' * W3}  {'─' * W4}")
    # Wall-clock is the meaningful "total" for a parallel run
    wall_avg = wall_time / n
    print(f"  {'Wall-clock (actual)':<{W1}}  {wall_time:>{W2}.2f}s  {wall_avg:>{W3}.2f}s  {'':>{W4}}")
    print(f"  {'CPU total (all workers)':<{W1}}  {total_pipeline:>{W2}.2f}s  {total_pipeline/n:>{W3}.2f}s  {'100.0%':>{W4}}")
    print(f"{'═' * 66}")


def _print_cost_estimate(all_results: list[dict],
                         wall_s: "dict | None" = None) -> None:
    """Print LLM cost estimate table — verifier and evidence classification separately.

    wall_s: optional real wall-clock seconds per stage
            {"llm_verifier": float, "evidence_classification": float}.
            When supplied (stage-parallel run) the rows show "Wall clock".
            When absent (CI-parallel run) they show "CPU total" (sum across workers).
    """
    all_t = [r.get("timings", {}) for r in all_results]
    n_to_verifier  = sum(t.get("n_candidates_to_verifier", 0) for t in all_t)
    n_actual_calls = sum(len(t.get("per_verifier_call_s", {})) for t in all_t)
    n_ec_calls     = sum(t.get("n_ec_calls", 0) for t in all_t)
    if not n_to_verifier and not n_ec_calls:
        return

    # Use actual token counts from Bedrock usage field when available
    v_actual_in  = sum(t.get("actual_verifier_tokens", {}).get("input",  0) for t in all_t)
    v_actual_out = sum(t.get("actual_verifier_tokens", {}).get("output", 0) for t in all_t)
    if v_actual_in > 0:
        v_in, v_out, v_label = v_actual_in, v_actual_out, "actual"
    else:
        v_in, v_out, v_label = (n_actual_calls * _EST_INPUT_TOKENS_PER_CAND,
                                 n_actual_calls * _EST_OUTPUT_TOKENS_PER_CAND, "est.")

    ec_actual_in  = sum(t.get("actual_ec_tokens", {}).get("input",  0) for t in all_t)
    ec_actual_out = sum(t.get("actual_ec_tokens", {}).get("output", 0) for t in all_t)
    if ec_actual_in > 0:
        ec_in, ec_out, ec_label = ec_actual_in, ec_actual_out, "actual"
    else:
        ec_in, ec_out, ec_label = (n_ec_calls * _EST_EC_INPUT_TOKENS_PER_HIT,
                                    n_ec_calls * _EST_EC_OUTPUT_TOKENS_PER_HIT, "est.")

    v_cost  = v_in  * _HAIKU_INPUT_PRICE_PER_TOKEN + v_out  * _HAIKU_OUTPUT_PRICE_PER_TOKEN
    ec_cost = ec_in * _HAIKU_INPUT_PRICE_PER_TOKEN + ec_out * _HAIKU_OUTPUT_PRICE_PER_TOKEN
    if wall_s:
        verifier_time  = wall_s.get("llm_verifier", 0.0)
        ec_time        = wall_s.get("evidence_classification", 0.0)
        v_time_label   = "Wall clock"
        ec_time_label  = "Wall clock"
    else:
        verifier_time  = sum(t.get("llm_verifier", 0.0) for t in all_t)
        ec_time        = sum(t.get("evidence_classification", 0.0) for t in all_t)
        v_time_label   = "CPU total"
        ec_time_label  = "CPU total"

    all_v_times: list[float] = []
    for t in all_t:
        all_v_times.extend(t.get("per_verifier_call_s", {}).values())
    all_ec_times: list[float] = []
    for t in all_t:
        all_ec_times.extend(t.get("per_ec_call_s", {}).values())

    W1, W2 = 30, 14

    def _row(label: str, value: str) -> None:
        print(f"  │ {label:<{W1}} │ {value:>{W2}} │")

    sep  = f"  ├{'─' * (W1 + 2)}┼{'─' * (W2 + 2)}┤"
    top  = f"  ┌{'─' * (W1 + 2)}┬{'─' * (W2 + 2)}┐"
    bot  = f"  └{'─' * (W1 + 2)}┴{'─' * (W2 + 2)}┘"

    print(f"\n{top}")
    _row("COST ANALYSIS", "")
    print(sep)
    _row("CIs searched",                str(len(all_results)))
    print(sep)

    # ── LLM Verifier section ──────────────────────────────────────────────
    _row(f"LLM VERIFIER  [{v_label}]", "")
    _row("  Candidates passed in",      f"{n_to_verifier:,}")
    _row("  Skipped (below threshold)",  f"{n_to_verifier - n_actual_calls:,}")
    _row("  Actual Bedrock calls",       f"{n_actual_calls:,}")
    _row(f"  Input tokens  [{v_label}]",  f"{v_in:,}")
    _row(f"  Output tokens [{v_label}]",  f"{v_out:,}")
    _row("  Cost",                        f"${v_cost:.4f}")
    _row(f"  {v_time_label}",             f"{verifier_time:.1f}s")
    if all_v_times:
        _row("  Per-call min/avg/max",
             f"{min(all_v_times):.2f}s / {sum(all_v_times)/len(all_v_times):.2f}s / {max(all_v_times):.2f}s")
    print(sep)

    # ── Evidence Classification section ──────────────────────────────────
    _row(f"EVIDENCE CLASSIFICATION  [{ec_label}]", "")
    _row("  Bedrock calls (YES/MAYBE hits)", f"{n_ec_calls:,}")
    _row(f"  Input tokens  [{ec_label}]",   f"{ec_in:,}")
    _row(f"  Output tokens [{ec_label}]",   f"{ec_out:,}")
    _row("  Cost",                          f"${ec_cost:.4f}")
    _row(f"  {ec_time_label}",              f"{ec_time:.1f}s")
    if all_ec_times:
        _row("  Per-call min/avg/max",
             f"{min(all_ec_times):.2f}s / {sum(all_ec_times)/len(all_ec_times):.2f}s / {max(all_ec_times):.2f}s")
    print(sep)

    _row("COMBINED cost",                  f"${v_cost + ec_cost:.4f}")
    print(bot)
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search pipeline real AWS test")
    p.add_argument("--ci-file",     default=DEFAULT_CI_FILE,
                   help=f"Path to CI JSON file  (default: {DEFAULT_CI_FILE})")
    p.add_argument("--max-cis",     type=int, default=2,
                   help="Maximum number of CIs to search for  (default: 2)")
    p.add_argument("--ci-index",    type=int, default=None,
                   help="Run a single CI by its 0-based index in the CI file")
    p.add_argument("--document-id", default=DOCUMENT_ID,
                   help=f"OpenSearch document_id to search  (default: {DOCUMENT_ID})")
    p.add_argument("--skip-rerank", action="store_true",
                   help="Skip Bedrock cross-encoder reranker")
    p.add_argument("--skip-verify", action="store_true",
                   help="Skip Bedrock LLM verifier")
    p.add_argument("--verbose",     action="store_true",
                   help="Print intermediate results in detail")
    p.add_argument("--workers",     type=int, default=4,
                   help="Max concurrent CI searches (default: 4)")
    p.add_argument("--output",      default=None,
                   help="Path to save results JSON (default: localfiles/search_results/<timestamp>.json)")
    p.add_argument("--diagnose-ci", default=None,
                   help="Print aggregator score table for these CI ids (comma-separated, or 'all')")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # Parse --diagnose-ci into a set of ints (or {"all"})
    diagnose_ids: set = set()
    if args.diagnose_ci:
        raw_diag = args.diagnose_ci.strip()
        if raw_diag.lower() == "all":
            diagnose_ids = {"all"}
        else:
            diagnose_ids = {int(x.strip()) for x in raw_diag.split(",") if x.strip().isdigit()}

    # ── Load CIs ─────────────────────────────────────────────────────────────
    ci_path = Path(args.ci_file)
    if not ci_path.exists():
        print(f"ERROR: CI file not found: {ci_path}")
        sys.exit(1)

    with ci_path.open() as fh:
        raw_cis = json.load(fh)

    if isinstance(raw_cis, dict):
        raw_cis = list(raw_cis.values())  # some files are {id: ci_obj, ...}
    if not isinstance(raw_cis, list):
        raw_cis = [raw_cis]

    if args.ci_index is not None:
        raw_cis = [raw_cis[args.ci_index]]
    else:
        raw_cis = raw_cis[: args.max_cis]

    print(f"\nSearch Pipeline Test")
    print(f"  Document  : {args.document_id}")
    print(f"  OpenSearch: {OPENSEARCH_ENDPOINT}")
    print(f"  CIs loaded: {len(raw_cis)} from {ci_path.name}")
    print(f"  Skip rerank: {args.skip_rerank}   skip verify: {args.skip_verify}")

    # ── Load document fingerprint (drug / study context) ────────────────────────
    doc_context = _load_document_context(args.document_id)
    if doc_context:
        print(f"  Doc drug  : {', '.join(doc_context.get('primary_drugs', [])[:2])}  "
              f"| Study: {', '.join(doc_context.get('study_ids', [])[:1])}")
    else:
        print(f"  Doc drug  : (no fingerprint found for {args.document_id})")

    # ── Build OS client once ──────────────────────────────────────────────────
    print("\nConnecting to OpenSearch …")
    os_client = _build_os_client()
    info = os_client.info()
    print(f"  Cluster: {info.get('cluster_name')}  version: {info['version']['number']}")

    # ── Process CIs in parallel ───────────────────────────────────────────────
    n_workers = min(args.workers, len(raw_cis))
    print(f"  Workers   : {n_workers}")

    all_results: list[dict | None] = [None] * len(raw_cis)
    _print_lock   = threading.Lock()
    _t_wall_start = time.perf_counter()

    def _search_one_ci(raw_ci: dict, idx: int) -> tuple[int, str, dict | None]:
        """Enrich + search one CI; captures all stdout to avoid interleaving."""
        ci_id   = raw_ci.get("id", idx)
        ci_text = raw_ci.get("knownCI", raw_ci.get("name", f"ci-{idx}"))
        buf    = io.StringIO()
        result = None
        with contextlib.redirect_stdout(buf):
            print(f"\n{'═' * 60}")
            print(f"  CI #{idx + 1}/{len(raw_cis)}  id={ci_id}  text=\"{ci_text}\"")
            try:
                enriched = _lookup_ci_from_index(raw_ci, ci_id)
                if enriched is None:
                    print(f"  [CI {ci_id}] ✗ not found in ci-objects — run index_cis.py first")
                    return idx, buf.getvalue(), None
                print(f"  [CI {ci_id}] loaded from ci-objects")

                result = run_search(
                    enriched,
                    document_id      = args.document_id,
                    document_context = doc_context,
                    skip_rerank      = args.skip_rerank,
                    skip_verify      = args.skip_verify,
                    verbose          = args.verbose,
                    diagnose_ids     = diagnose_ids,
                )
            except Exception as exc:
                logger.exception("Search failed for CI %s", ci_id)
                print(f"  ✗ ERROR: {exc}")
        return idx, buf.getvalue(), result

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_search_one_ci, raw_ci, idx): idx
            for idx, raw_ci in enumerate(raw_cis)
        }
        for future in as_completed(futures):
            orig_idx, output, result = future.result()
            with _print_lock:
                print(output, end="")  # flush entire CI block atomically
            if result is not None:
                all_results[orig_idx] = result

    all_results = [r for r in all_results if r is not None]
    _wall_time  = time.perf_counter() - _t_wall_start

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  SUMMARY: {len(all_results)} CI(s) searched  (wall-clock {_wall_time:.1f}s)")
    total_hits = sum(len(r.get("final_hits", [])) for r in all_results)
    print(f"  Total confirmed hits: {total_hits}")

    _print_timing_summary(all_results, _wall_time)
    _print_cost_estimate(all_results)

    # ── Save results ─────────────────────────────────────────────────────────
    out_path = Path(args.output) if args.output else (
        ROOT / "localfiles" / "search_results" /
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{Path(args.ci_file).stem}_{args.document_id}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_results(all_results, args, out_path, wall_time=_wall_time)
    print(f"  Results saved: {out_path}")
    print()


if __name__ == "__main__":
    main()
