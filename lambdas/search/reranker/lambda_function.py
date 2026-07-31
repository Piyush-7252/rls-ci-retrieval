"""
Search Pipeline — Stage 5: Cross-Encoder Re-ranker
====================================================
Deterministic, explainable reranker built around a local cross-encoder model.
No API calls. No non-determinism.

Architecture
------------
  OpenSearch retrieval
    ↓
  Cross Encoder (BAAI/bge-reranker-base) — semantic relevance
    ↓
  CI-type feature profiles   — per-CI-type weights; irrelevant features zeroed
    ↓
  Drug identity + Fact-slot  — entity-level precision signals
    ↓
  Intent alignment           — section category vs CI intent (background-aware)
    ↓
  Structural validation      — abbreviation tables, missing relations, absent slots
    ↓
  Semantic contradiction     — drug/endpoint/phase/population/biomarker/arm mismatches
    ↓
  Interaction scoring        — drug × fact concordance/conflict synergy
    ↓
  CE contradiction veto      — caps composite when hard contradictions confirmed
    ↓
  Explainable decision       — matched[], conflicts[], decision sentence

Scoring
-------
  Weights are looked up from _FEATURE_PROFILES[ci_type] so each CI type activates
  a different feature set.  Study ID weight is zeroed at runtime when the CI
  carries no extractable study IDs (returns a constant otherwise — pure noise).

  Additive terms on top of the weighted sum:
    • pos_bonus      — document-position alignment (max +0.30)
    • interaction    — drug × fact co-evidence (±0.50)
    • struct_penalty — structural incompatibilities (≤ 0)
    • contra_penalty — semantic value conflicts (≤ 0, caps composite when severe)

Model: BAAI/bge-reranker-base  (1.1 GB, cached locally)
       Override with RERANKER_CE_MODEL env var.
Input:  context-expanded search request  (must have "expanded_candidates")
Appends: "ranked_candidates": list[RankedCandidate]

RankedCandidate = ExpandedCandidate + {
    "cross_encoder_score": float,
    "score_breakdown": {
        ..feature scores..,
        "contra_detail": dict,        # {slot: {weight, severity, evidence}} per conflict
        "struct_detail": dict,        # {issue: {weight, severity, evidence}} per structural issue
        "profile": str,               # which _FEATURE_PROFILES entry was used
        "clinical_reasoning": {
            "matched":   list[str],   # signals that supported the candidate
            "conflicts": list[str],   # signals that argued against it
            "decision":  str,         # Accepted / Rejected / Partial / Marginal
        },
    },
}
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ── ensure shared/ is importable ─────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Comparator package: shared types, outcome constants, individual comparators ──
# The comparators/ subdirectory is in the same folder as this file.
# Ensure the lambda directory is on sys.path so relative package imports work
# whether invoked by Lambda or by the local test runner.
_LAMBDA_DIR = str(Path(__file__).resolve().parent)
if _LAMBDA_DIR not in sys.path:
    sys.path.insert(0, _LAMBDA_DIR)

from comparators import (                              # noqa: E402
    _SEV_NONE, _SEV_LOW, _SEV_MEDIUM, _SEV_HIGH, _SEV_FATAL,
    _SEV_RANK, _RANK_SEV,
    CMP_MATCH, CMP_RELATED, CMP_SPECIALIZATION, CMP_GENERALIZATION,
    CMP_UNKNOWN, CMP_CONFLICT,
    ComparisonResult,
    ClinicalContext, _build_ci_context, _build_cand_context,
    _CONFLICT_METADATA, _CONFLICT_WEIGHTS,
    _DRUG_CONTRA_WEIGHTS, _MODALITY_GROUP, _TEMPORAL_FIELD_SEVERITY,
    _ENDPOINT_FAMILY, _ep_family,
    _COMPARATORS,
)

try:
    from shared.query_normalizer import normalize_query as _normalize_query, QueryNorm as _QueryNorm
except ImportError:
    from dataclasses import dataclass as _dc, field as _f
    @_dc
    class _QueryNorm:  # type: ignore[no-redef]
        original: str = ""
        expanded_text: str = ""
        entities: list = _f(default_factory=list)
        label_families: frozenset = frozenset()
        labels: frozenset = frozenset()
        normalized_terms: frozenset = frozenset()
    def _normalize_query(text: str) -> "_QueryNorm":  # type: ignore[misc]
        return _QueryNorm(original=text, expanded_text=text)
    logger.warning("[Reranker] shared.query_normalizer not found — entity overlap disabled")

try:
    from shared.drug_identity import (
        DrugRelation as _DrugRelation,
        DRUG_RELATION_SCORE as _DRUG_RELATION_SCORE,
        best_drug_relation as _best_drug_relation_fn,
    )
    _DRUG_IDENTITY_AVAILABLE = True
except ImportError:
    _DRUG_IDENTITY_AVAILABLE = False
    _DrugRelation = None  # type: ignore
    _DRUG_RELATION_SCORE = {}  # type: ignore

LLM_VERIFIER_LAMBDA_ARN = os.environ.get("LLM_VERIFIER_LAMBDA_ARN", "")
RERANK_TOP_N             = int(os.environ.get("RERANK_TOP_N", "20"))
CE_MODEL_NAME            = os.environ.get("RERANKER_CE_MODEL", "BAAI/bge-reranker-base")

import threading

_aws: dict      = {}
_ce_model       = None   # lazy-loaded cross-encoder
_ce_model_lock  = threading.Lock()

def _get(service: str):
    if service not in _aws:
        import boto3
        _aws[service] = boto3.client(service)
    return _aws[service]


def _get_ce_model():
    """Thread-safe lazy-load of the cross-encoder (double-checked locking)."""
    global _ce_model
    if _ce_model is None:
        with _ce_model_lock:
            if _ce_model is None:   # re-check inside lock
                from sentence_transformers import CrossEncoder
                logger.info("[Reranker] loading cross-encoder model: %s", CE_MODEL_NAME)
                _ce_model = CrossEncoder(CE_MODEL_NAME)
                logger.info("[Reranker] model loaded")
    return _ce_model


# ─── Section-aware ranking ────────────────────────────────────────────────────────────
# Two sources used in order of preference:
#   1. section_category (canonical, set by section_chunker) — precise
#   2. section / parent_heading text keyword match          — fallback

# Expected document-position ranges by category (lo, hi) for position bonus.
# Protocols are structured: Objectives near the front, Appendices at the back.
_CATEGORY_POSITION_RANGES: dict[str, tuple[float, float]] = {
    "SYNOPSIS":       (0.00, 0.15),
    "OBJECTIVES":     (0.05, 0.25),
    "ENDPOINTS":      (0.05, 0.25),
    "DESIGN":         (0.10, 0.35),
    "BACKGROUND":     (0.05, 0.25),
    "ELIGIBILITY":    (0.15, 0.40),
    "TREATMENT":      (0.25, 0.55),
    "PROCEDURES":     (0.25, 0.60),
    "STATISTICS":     (0.35, 0.65),
    "SAFETY":         (0.25, 0.75),
    "EFFICACY":       (0.25, 0.65),
    "PK":             (0.30, 0.65),
    "POPULATION":     (0.25, 0.55),
    "BIOMARKER":      (0.30, 0.70),
    "BENEFIT_RISK":   (0.30, 0.65),
    "APPENDIX":       (0.65, 1.00),
    "ADMINISTRATIVE": (0.00, 0.10),
}
_POSITION_BONUS_MAX  = 0.30   # max additive bonus when position is in expected range
_POSITION_DECAY_SPAN = 0.30   # distance from range edge over which bonus decays to 0


def _document_position_bonus(cand: dict) -> float:
    """Small additive bonus when a chunk's document position matches its category's expected range."""
    obj = cand.get("matched_object") or {}
    pos = obj.get("document_position")
    cat = obj.get("section_category")
    if pos is None or not cat:
        return 0.0
    rng = _CATEGORY_POSITION_RANGES.get(cat)
    if not rng:
        return 0.0
    lo, hi = rng
    if lo <= pos <= hi:
        return _POSITION_BONUS_MAX
    dist = min(abs(pos - lo), abs(pos - hi))
    return round(max(0.0, _POSITION_BONUS_MAX * (1.0 - dist / _POSITION_DECAY_SPAN)), 3)


_CATEGORY_BOOSTS: dict[str, float] = {
    "OBJECTIVES":   1.50,
    "ENDPOINTS":    1.40,
    "SYNOPSIS":     1.35,
    "DESIGN":       1.25,
    "ELIGIBILITY":  1.15,
    "EFFICACY":     1.10,
    "BENEFIT_RISK": 1.05,
    "SAFETY":       1.00,
    "TREATMENT":    0.95,
    "POPULATION":   0.90,
    "PROCEDURES":   0.85,
    "STATISTICS":   0.85,
    "PK":           0.80,
    "BIOMARKER":    0.80,
    "BACKGROUND":   0.70,
    "APPENDIX":     0.60,
    "OTHER":        0.90,
}

_SECTION_BOOSTS: list[tuple[str, float]] = [
    ("objective",         1.50),
    ("executive summary", 1.40),
    ("study design",      1.25),
    ("endpoint",          1.15),
    ("result",            1.00),
    ("discussion",        0.95),
    ("conclusion",        0.90),
    ("appendix",          0.80),
    ("reference",         0.70),
    ("footer",            0.50),
]
_DEFAULT_SECTION_BOOST = 1.00
_MIN_BOOST             = 0.50   # lowest possible
_MAX_BOOST             = 1.50   # highest possible


def _section_boost(candidate: dict) -> float:
    """Boost multiplier from section context.  Uses section_category when present."""
    obj = candidate.get("matched_object") or {}
    # Prefer the canonical category (set by section_chunker at index time)
    cat = obj.get("section_category")
    if cat and cat in _CATEGORY_BOOSTS:
        return _CATEGORY_BOOSTS[cat]
    # Fall back to keyword match on the section heading text
    section = (obj.get("section") or obj.get("parent_heading") or "").lower()
    if not section:
        return _DEFAULT_SECTION_BOOST
    for keyword, boost in _SECTION_BOOSTS:
        if keyword in section:
            return boost
    return _DEFAULT_SECTION_BOOST


# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    search_id = event.get("search_id", "unknown")
    logger.info("[Reranker] start search_id=%s candidates=%d",
                search_id, len(event.get("expanded_candidates", [])))
    try:
        result = _process(event)
    except Exception as exc:
        logger.error("[Reranker] failed search_id=%s error=%s", search_id, exc)
        raise
    logger.info("[Reranker] done search_id=%s top_score=%.2f",
                search_id, result["ranked_candidates"][0]["cross_encoder_score"]
                if result["ranked_candidates"] else 0.0)
    if LLM_VERIFIER_LAMBDA_ARN:
        _get("lambda").invoke(
            FunctionName   = LLM_VERIFIER_LAMBDA_ARN,
            InvocationType = "Event",
            Payload        = json.dumps(result).encode(),
        )
    return result


def _process(req: dict) -> dict:
    ci        = req["ci"]
    ci_text   = ci.get("knownCI", "")
    ci_assets = ci.get("assets", [])
    ci_type      = req.get("classification", {}).get("ci_type")   # e.g. "OBJECTIVE"
    ci_confidence = float(req.get("classification", {}).get("confidence", 1.0))  # 0–1
    ci_entities: list[dict] = ci.get("ner", {}).get("entities", [])
    ci_facts:      dict       = ci.get("effective_facts") or ci.get("facts", {})
    ci_relations:  list       = ci.get("clinical_relations", [])
    candidates = req.get("expanded_candidates", [])

    # Normalize query once: expand abbreviations, extract entity metadata
    query_norm = _normalize_query(ci_text)
    logger.info("[Reranker] query_norm families=%s", sorted(query_norm.label_families))

    # Build chunk_idx → candidate map for sibling context expansion
    chunk_by_idx: dict[int, dict] = {}
    for c in candidates:
        idx = (c.get("matched_object") or {}).get("chunk_idx")
        if idx is not None:
            chunk_by_idx[idx] = c

    # Build (query, passage) pairs for batch cross-encoder inference
    passages = [_expanded_candidate_text(c, chunk_by_idx) for c in candidates]
    pairs    = [(ci_text, p) for p in passages]

    ce_logits: list[float]
    if pairs:
        try:
            ce_logits = _get_ce_model().predict(pairs).tolist()
        except Exception as exc:
            logger.warning("[Reranker] cross-encoder failed, using zeros: %s", exc)
            ce_logits = [0.0] * len(pairs)
    else:
        ce_logits = []

    ranked = []
    for cand, logit in zip(candidates, ce_logits):
        score, breakdown = _composite_score(
            cand, logit, ci_assets, query_norm, ci_type, ci_text,
            ci_entities, ci_confidence, ci_facts, ci_relations, ci
        )
        ranked.append({
            **cand,
            "cross_encoder_score":  score,
            "section_boost":        breakdown["section_mult"],
            # Preserve the aggregator's score_breakdown (has contradiction / contra_detail)
            # under a separate key before the reranker's breakdown overwrites it.
            "agg_score_breakdown":  cand.get("score_breakdown", {}),
            "score_breakdown":      breakdown,
        })

    ranked.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

    return {
        **req,
        "ranked_candidates": ranked[:RERANK_TOP_N],
    }


def _candidate_text(cand: dict) -> str:
    """
    Build the passage string passed to BGE cross-encoder.

    Prefixes the passage with a section breadcrumb so that BGE sees:
      [Category: OBJECTIVES | Section: 2.1 Primary Objective]
      The primary objective of this study is to evaluate...

    This disambiguates otherwise similar text that appears in different
    sections (e.g. "ORR" in Objectives vs. Background vs. Safety).
    """
    ctx = cand.get("context", {})
    obj = cand.get("matched_object") or {}

    # Build breadcrumb prefix from section metadata
    parts: list[str] = []
    cat      = obj.get("section_category")
    sem_path = obj.get("semantic_path")   # "OBJECTIVES > Primary Objective"
    path     = obj.get("heading_path")
    sec      = obj.get("section") or obj.get("parent_heading")
    if cat and cat not in ("OTHER", "ADMINISTRATIVE"):
        parts.append(f"Category: {cat}")
    if sem_path:
        parts.append(f"Meaning: {sem_path}")
    elif path:
        parts.append(f"Section: {path}")
    elif sec:
        parts.append(f"Section: {sec}")
    prefix = f"[{' | '.join(parts)}]\n" if parts else ""

    body = "\n".join(filter(None, [
        ctx.get("parent_text",  "")[:300],   # parent heading context (shorter)
        ctx.get("prev_text",    ""),
        ctx.get("current_text", ""),
        ctx.get("next_text",    ""),
    ]))
    return (prefix + body)[:3000]


def _expanded_candidate_text(cand: dict, chunk_by_idx: dict) -> str:
    """
    Like ``_candidate_text`` but pads small chunks with adjacent-sibling context
    retrieved in the same batch, so BGE can rank them meaningfully.

    Only expands when the base text is short (< 80 words); larger chunks
    already have enough context.
    """
    base = _candidate_text(cand)
    obj  = cand.get("matched_object") or {}
    if not chunk_by_idx or len(base.split()) >= 80:
        return base[:3000]

    parts: list[str] = []
    prev_idx = obj.get("prev_chunk_idx")
    if prev_idx is not None and prev_idx in chunk_by_idx:
        prev_text = _candidate_text(chunk_by_idx[prev_idx])[:350]
        parts.append(f"[preceding section]\n{prev_text}")
    parts.append(base)
    next_idx = obj.get("next_chunk_idx")
    if next_idx is not None and next_idx in chunk_by_idx:
        next_text = _candidate_text(chunk_by_idx[next_idx])[:350]
        parts.append(f"[following section]\n{next_text}")
    return "\n---\n".join(parts)[:3000]


def _entity_family_overlap(query_families: frozenset, cand_entities: list[dict],
                            cand_facts: dict | None = None) -> float:
    """
    Recall of query entity families in the candidate chunk, scaled 0-10.

    Falls back to deriving families from fact-slot keys when the entity list
    is empty (chunk-level hits without a matched semantic object).
    """
    if not query_families:
        return 5.0
    # Derive family from the entity's label field (entities use 'label', not 'family')
    cand_families = {
        e.get("label") or e.get("family") or "OTHER"
        for e in cand_entities
        if e.get("label") or e.get("family")
    }
    # Fallback: derive from fact-slot keys when entities aren't available
    if not cand_families and cand_facts:
        _FACT_TO_FAMILY: dict[str, str] = {
            "drug":             "MEDICATION",
            "treatment_regimen": "TREATMENT_NAME",
            "endpoint":         "CLINICAL_ENDPOINT",
            "adverse_event":    "ADVERSE_EVENT",
            "dose":             "DOSAGE",
            "study_arm":        "STUDY_ARM",
            "biomarker":        "BIOMARKER",
            "assessment":       "QUESTIONNAIRE",
            "population":       "STUDY_POPULATION",
        }
        cand_families = {
            _FACT_TO_FAMILY[k] for k in cand_facts
            if k in _FACT_TO_FAMILY and cand_facts[k]
        }
    if not cand_families:
        return 0.0
    recall = len(query_families & cand_families) / len(query_families)
    return round(recall * 10.0, 3)


def _entity_term_overlap(query_terms: frozenset, cand_entities: list[dict]) -> float:
    """
    Recall of query normalized terms in candidate entity text/normalized fields, scaled 0-10.

    Returns 5.0 (neutral) when query has no normalized terms.
    """
    if not query_terms:
        return 5.0
    cand_terms: set[str] = set()
    for e in cand_entities:
        for v in (e.get("text",""), e.get("normalized",""), e.get("canonical",""), e.get("abbreviation","")):
            if v:
                cand_terms.add(v.lower())
    if not cand_terms:
        return 0.0
    matched = sum(1 for qt in query_terms if qt in cand_terms)
    recall  = matched / len(query_terms)
    return round(recall * 10.0, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Clinical structured signals  (Gaps 1–4: drug identity, study ID, intent)
# ─────────────────────────────────────────────────────────────────────────────

# Study / protocol ID pattern.  Matches:
#   NCT01234567    ClinicalTrials.gov
#   64407564       JNJ numeric IDs
#   MMY1001        Janssen protocol codes
#   64407564MMY3002 compound IDs
_STUDY_ID_RE = re.compile(
    r'\b(?:'
    r'NCT\d{6,9}'
    r'|\b\d{8}\b'
    r'|[A-Z]{2,6}\d{3,4}\b'
    r')\b',
    re.IGNORECASE,
)

_DRUG_ENTITY_TYPES = frozenset({"MEDICATION", "TREATMENT_NAME", "BRAND_NAME"})

# Expected section categories per CI intent type.
# First set = strong match (+10).  Second set = weak match (+6).
_CI_TYPE_SECTIONS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "OBJECTIVE":        (frozenset({"OBJECTIVES", "ENDPOINTS", "SYNOPSIS"}),
                         frozenset({"EFFICACY", "DESIGN"})),
    "EFFICACY":         (frozenset({"EFFICACY", "ENDPOINTS", "OBJECTIVES"}),
                         frozenset({"RESULTS", "SYNOPSIS"})),
    "SAFETY":           (frozenset({"SAFETY", "ADVERSE_EVENTS", "TOLERABILITY"}),
                         frozenset({"BENEFIT_RISK", "ELIGIBILITY"})),
    "DOSING":           (frozenset({"TREATMENT", "PROCEDURES"}),
                         frozenset({"DESIGN", "PK"})),
    "POPULATION":       (frozenset({"ELIGIBILITY", "POPULATION"}),
                         frozenset({"DEMOGRAPHICS", "DESIGN"})),
    "PHARMACOKINETICS": (frozenset({"PK", "PK_PD"}),
                         frozenset({"PHARMACOLOGY", "PROCEDURES"})),
    "PHARMACODYNAMICS": (frozenset({"PK", "PK_PD"}),
                         frozenset({"PHARMACOLOGY", "BIOMARKER"})),
    "PROTOCOL":         (frozenset({"DESIGN", "SYNOPSIS"}),
                         frozenset({"PROCEDURES", "OBJECTIVES"})),
    "STUDY_DESIGN":     (frozenset({"DESIGN", "SYNOPSIS"}),
                         frozenset({"PROCEDURES", "OBJECTIVES"})),
    "BIOMARKER":        (frozenset({"BIOMARKER", "TRANSLATIONAL"}),
                         frozenset({"EFFICACY", "PROCEDURES"})),
    "PHRASE":           (frozenset(), frozenset()),   # neutral — full retrieval suite
}

_BACKGROUND_CATS = frozenset({"BACKGROUND", "APPENDIX", "ADMINISTRATIVE", "REFERENCE"})

# Sections that are structurally incompatible with a CI type — stronger penalty than
# a neutral mismatch (2.0), weaker than a background appendix (1.5).
_STRUCTURALLY_INCOMPATIBLE_CATS: dict[str, frozenset] = {
    "OBJECTIVE": frozenset({"PROCEDURES", "MANUFACTURING", "CMC", "FORMULATION",
                             "CHEMISTRY", "CHARACTERIZATION"}),
    "EFFICACY":  frozenset({"PROCEDURES", "MANUFACTURING", "CMC"}),
    "DOSING":    frozenset({"MANUFACTURING", "CMC", "CHARACTERIZATION"}),
    "SAFETY":    frozenset({"MANUFACTURING", "CMC"}),
}

# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical endpoint sub-intent taxonomy
# ─────────────────────────────────────────────────────────────────────────────
# Within OBJECTIVE and EFFICACY CI types, section-level intent scoring is too
# coarse — "Evaluate ORR" and "Evaluate PFS" both map to OBJECTIVES section
# but describe different clinical aims.
#
# All clinical knowledge lives in:
#   shared/data/clinical_concepts.json     — canonical
#   lambdas/search/reranker/data/clinical_concepts.json  — Lambda mirror
#
# Generated by tools/build_endpoint_ontology.py using NCI Thesaurus concept
# IDs + curated additional_terms (49 categories, 457+ synonyms).
# The Lambda contains ZERO hardcoded synonym strings.
#
# If the JSON is missing or corrupt the classifier returns None for every
# query (graceful degradation: sub-intent refinement is skipped, all other
# scoring continues normally).

# Loaded once at cold start from clinical_concepts.json.
# {category_label: frozenset[synonym_str]}
_ENDPOINT_CONCEPTS: dict[str, frozenset[str]] = {}


def _load_endpoint_concepts() -> dict[str, frozenset[str]]:
    """
    Load the ontology-backed synonym table from clinical_concepts.json.

    Built by tools/build_endpoint_ontology.py (NCIt anchor IDs → subtree walk
    → synonym expansion, merged with additional_terms for NCIt gaps).
    Covers 49 categories: endpoints, response criteria, diseases, drugs,
    targets, AE types, lab tests, and statistical methods.

    OS guard: the raw JSON stores "os" as a bare token.  We space-pad it
    (" os ") here to prevent substring collisions with "those", "dose", etc.

    Returns {category: frozenset[synonym]}.
    Logs a warning and returns {} on any failure — safe degradation.
    """
    # Resolution order (stops at first hit):
    #   1. shared/data/clinical_concepts.json   — canonical
    #   2. local data/clinical_concepts.json    — Lambda package mirror
    #   3. shared/data/endpoint_concepts.json   — legacy name (backward compat)
    #   4. local data/endpoint_concepts.json    — legacy mirror
    _shared = Path(__file__).parent.parent.parent.parent / "shared" / "data"
    _local  = Path(__file__).parent / "data"
    _candidates = [
        _shared / "clinical_concepts.json",
        _local  / "clinical_concepts.json",
        _shared / "endpoint_concepts.json",
        _local  / "endpoint_concepts.json",
    ]
    json_path = next((p for p in _candidates if p.exists()), None)
    if json_path is None:
        logger.warning(
            "[Reranker] clinical_concepts.json not found — "
            "sub-intent refinement disabled.  "
            "Run tools/build_endpoint_ontology.py to generate it."
        )
        return {}

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        raw: dict[str, list[str]] = data.get("categories", {})
        if not raw:
            raise ValueError("'categories' key missing or empty")
    except Exception as exc:
        logger.warning("[Reranker] %s unreadable (%s) — "
                       "sub-intent refinement disabled.", json_path.name, exc)
        return {}

    result: dict[str, frozenset[str]] = {}
    for cat, syns in raw.items():
        kws: set[str] = set()
        for s in syns:
            # OS guard: space-pad the bare "os" token to avoid substring
            # collision with "those", "dose", "close", etc.
            if cat == "os" and s.strip() == "os":
                kws.add(" os ")
            else:
                kws.add(s)
        result[cat] = frozenset(kws)

    built_at = data.get("built_at", "unknown")
    source   = data.get("source",   "unknown")
    total    = sum(len(v) for v in result.values())
    logger.info(
        "[Reranker] endpoint_concepts loaded: %d synonyms, %d categories "
        "(built %s, source: %s)",
        total, len(result), built_at, source,
    )
    return result


# Initialise at module load (Lambda cold start — fast, JSON is ~4 KB).
_ENDPOINT_CONCEPTS = _load_endpoint_concepts()


def _classify_endpoint_subintent(text: str) -> str | None:
    """
    Classify text to an endpoint sub-intent category using _ENDPOINT_CONCEPTS.

    _ENDPOINT_CONCEPTS is loaded from clinical_concepts.json (generated by
    tools/build_endpoint_ontology.py using NCIt anchor concept IDs).  The
    Lambda itself contains no hardcoded synonym strings.

    Returns the single matching category label, or None when:
    - _ENDPOINT_CONCEPTS is empty (JSON missing — graceful degradation), or
    - no category keywords appear in the text (unclear), or
    - multiple categories match (ambiguous — never penalise ambiguity).
    """
    if not _ENDPOINT_CONCEPTS:
        return None
    # Space-pad the text so that keywords like " os " match at string
    # boundaries (start/end) without requiring surrounding spaces in the
    # original text.  Prevents substring collision with "those", "dose", etc.
    tl = " " + text.lower() + " "
    matches = [label for label, kws in _ENDPOINT_CONCEPTS.items()
               if any(kw in tl for kw in kws)]
    return matches[0] if len(matches) == 1 else None


# Slot weights for fact overlap (mirrors fact_retriever / aggregator values)
_FACT_SLOT_WEIGHTS: dict[str, float] = {
    "drug":               4.0,
    "study_id":           4.0,
    "study_arm":          3.0,
    "endpoint":           2.5,
    "adverse_event":      2.0,
    "biomarker":          2.0,
    "study_population":   1.5,
    "dose":               1.5,
    "phase":              1.2,
    "response_criterion": 1.0,
    "statistical_method": 1.0,
}

# Object-type granularity bonus — conditional on CI type.
#
# Design principle: reward object types that fit the CI intent; never penalise.
# A PK CI should welcome a table row; an Objective CI should not.
# No negative values — preserves recall for cases where tables/lists hold the only evidence.
#
# Values are in the 0.00–0.30 range (raw) and are further divided by 20 to produce
# a 0.000–0.015 multiplier factor in _composite_score — a true tiebreaker that
# cannot leapfrog a semantically stronger candidate (a 0.14-point semantic gap
# requires only a 1.6% relative difference to remain decisive).
#
# A special "_default" row applies when ci_type is absent or unrecognised.
_CI_TYPE_OBJECT_BONUS: dict[str, dict[str, float]] = {
    # Narrative-first CI types — heading and paragraph slightly preferred for OBJECTIVE
    # (the paragraph under a heading carries the evidence, not the heading alone)
    "OBJECTIVE":        {"sentence": 0.30, "paragraph": 0.30, "heading": 0.25, "chunk": 0.05, "table_row": 0.00, "list": 0.00},
    "ENDPOINT":         {"sentence": 0.30, "paragraph": 0.25, "heading": 0.25, "chunk": 0.05, "table_row": 0.10, "list": 0.00},
    "SAFETY":           {"sentence": 0.25, "paragraph": 0.20, "heading": 0.15, "chunk": 0.05, "table_row": 0.05, "list": 0.00},
    "EFFICACY":         {"sentence": 0.25, "paragraph": 0.20, "heading": 0.15, "chunk": 0.05, "table_row": 0.10, "list": 0.00},
    "PROTOCOL":         {"heading": 0.25, "paragraph": 0.20, "sentence": 0.20, "chunk": 0.05, "table_row": 0.05, "list": 0.00},
    "STUDY_DESIGN":     {"heading": 0.25, "paragraph": 0.20, "sentence": 0.20, "chunk": 0.05, "table_row": 0.05, "list": 0.00},
    "BIOMARKER":        {"sentence": 0.25, "paragraph": 0.20, "heading": 0.15, "chunk": 0.05, "table_row": 0.10, "list": 0.00},
    # Structured-data-first CI types — tables and lists are genuinely useful
    "PHARMACOKINETICS": {"table_row": 0.30, "sentence": 0.25, "paragraph": 0.15, "heading": 0.10, "chunk": 0.05, "list": 0.05},
    "PHARMACODYNAMICS": {"table_row": 0.25, "sentence": 0.25, "paragraph": 0.15, "heading": 0.10, "chunk": 0.05, "list": 0.05},
    "DOSING":           {"table_row": 0.25, "sentence": 0.25, "paragraph": 0.15, "heading": 0.10, "chunk": 0.05, "list": 0.10},
    "POPULATION":       {"list": 0.25, "sentence": 0.25, "paragraph": 0.20, "heading": 0.10, "chunk": 0.05, "table_row": 0.10},
    # Default and PHRASE: small uniform reward to narrative types; no penalty elsewhere
    "_default":         {"sentence": 0.15, "paragraph": 0.15, "heading": 0.15, "chunk": 0.05, "table_row": 0.00, "list": 0.00},
    "PHRASE":           {"sentence": 0.15, "paragraph": 0.15, "heading": 0.15, "chunk": 0.05, "table_row": 0.00, "list": 0.00},
}


def _granularity_bonus(ci_type: str | None, obj_type: str) -> float:
    """
    Raw conditional bonus (0.00–0.30) for (CI type, object type) pair.

    Rewards object types that fit the CI intent without penalising others.
    Caller divides by 6 and multiplies by CI confidence to get the final
    multiplicative factor (max ~0.05), keeping it a gentle prior.
    """
    row = _CI_TYPE_OBJECT_BONUS.get(
        (ci_type or "").upper(),
        _CI_TYPE_OBJECT_BONUS["_default"],
    )
    return row.get(obj_type, 0.0)


def _fact_slot_overlap(ci_facts: dict, cand_facts: dict) -> float:
    """
    Slot-weighted Jaccard fact overlap, scaled to 0–10.

    Returns 5.0 (neutral) when either facts dict is empty so that candidates
    indexed before enrich_object was added to the pipeline are not unfairly
    penalised — consistent with the neutral-default pattern used by
    _drug_identity_score and _study_id_score.

    The key advantage over entity text overlap: two candidates that both
    mention “Teclistamab” score differently if one has endpoint=ORR
    (what the CI wants) and the other has adverse_event=Headache.
    """
    if not ci_facts:
        return 5.0   # neutral — CI has no fact expectations
    if not cand_facts:
        # CI expects structured facts; candidate provides none.
        # 3.0 (below neutral 5.0) rather than 0.0 to preserve recall for
        # objects not yet enriched with enrich_object.
        return 3.0

    total_weight = 0.0
    score        = 0.0
    for slot, ci_values in ci_facts.items():
        weight = _FACT_SLOT_WEIGHTS.get(slot)
        if weight is None or not ci_values:
            continue
        total_weight += weight
        cand_values = cand_facts.get(slot, [])
        if not cand_values:
            continue
        ci_set   = {v.lower() for v in ci_values if v}
        cand_set = {v.lower() for v in cand_values if v}
        if ci_set and cand_set:
            jaccard = len(ci_set & cand_set) / len(ci_set | cand_set)
            score  += weight * jaccard

    if total_weight == 0.0:
        return 5.0
    return round(score / total_weight * 10.0, 3)


def _extract_study_ids(text: str) -> frozenset[str]:
    return frozenset(m.lower() for m in _STUDY_ID_RE.findall(text))


def _drug_identity_score(ci_entities: list[dict], cand_entities: list[dict],
                          cand: dict | None = None) -> float:
    """
    0–10 score based on MEDICATION entity overlap between CI and candidate.

    Fallback chain when cand_entities is empty (chunk-level hit / un-enriched object):
      1. matched_object.facts.drug  (populated by enrich_object)
      2. Substring search in the raw candidate text:
         → 3.0 if the CI drug appears anywhere (mentioned but not primary subject)
         → 0.0 if the CI drug is absent (confirmed mismatch)
    """
    ci_drugs = {
        e.get("text", "").lower().strip()
        for e in ci_entities
        if e.get("label") in _DRUG_ENTITY_TYPES or e.get("type") in _DRUG_ENTITY_TYPES
    } - {""}

    if not ci_drugs:
        return 5.0   # neutral — CI does not specify a drug

    cand_drugs = {
        e.get("text", "").lower().strip()
        for e in cand_entities
        if e.get("label") in _DRUG_ENTITY_TYPES or e.get("type") in _DRUG_ENTITY_TYPES
    } - {""}

    # Fallback 1: use matched_object.facts.drug when entity list is empty
    if not cand_drugs and cand is not None:
        obj = cand.get("matched_object") or {}
        cand_drugs = {v.lower().strip() for v in obj.get("facts", {}).get("drug", []) if v}

    # Fallback 2: substring search in candidate text
    if not cand_drugs and cand is not None:
        text = _candidate_text(cand).lower()
        drug_in_text = any(d in text for d in ci_drugs if len(d) > 4) or any(
            f" {d} " in f" {text} " for d in ci_drugs if len(d) <= 4
        )
        return 3.0 if drug_in_text else 0.0

    if not cand_drugs:
        return 5.0   # truly no candidate context — stay neutral

    def _matches(ci_d: str, cand_set: set[str]) -> bool:
        if len(ci_d) <= 4:
            return ci_d in cand_set
        return any(ci_d in cd or cd in ci_d for cd in cand_set)

    matched = sum(1 for d in ci_drugs if _matches(d, cand_drugs))
    if matched == 0:
        # No text-level overlap.  Use the drug identity graph for partial credit:
        #   COMBINATION  → 4.0  (one drug is part of the other's regimen)
        #   SAME_FAMILY  → 2.0  (same mechanism class, different compound)
        #   RELATED      → 1.5  (different class, same clinical space)
        #   DIFFERENT    → 0.0  (confirmed drug mismatch — hard zero)
        if _DRUG_IDENTITY_AVAILABLE:
            rel = _best_drug_relation_fn(list(ci_drugs), list(cand_drugs))
            if rel == _DrugRelation.EXACT:
                pass  # shouldn't happen here (text already said no overlap)
            return _DRUG_RELATION_SCORE.get(rel, 0.0)
        return 0.0   # identity graph unavailable — confirmed mismatch
    recall    = matched / len(ci_drugs)
    precision = matched / len(cand_drugs)
    f1        = (2.0 * recall * precision) / (recall + precision)
    return round(f1 * 10.0, 3)

def _study_id_score(ci_text: str, cand_text: str, cand_doc_id: str = "") -> float:
    """
    0–10 score based on study / protocol ID overlap.

    Neutral (5.0) when CI text contains no recognisable study IDs.
    Bonus when candidate contains the same IDs as the CI.
    Confirmed mismatch (0.0) when candidate has IDs and none match the CI.

    ``cand_doc_id`` is the candidate's document_id field (e.g.
    "10993-co-jnj-64407564").  Protocol numbers embedded in document IDs
    are included so cross-document mismatches are caught even when chunk
    text does not repeat the full protocol number.
    """
    ci_ids   = _extract_study_ids(ci_text)
    # Candidate IDs from text + document-level identifier
    cand_ids = _extract_study_ids(cand_text)
    if cand_doc_id:
        cand_ids = cand_ids | _extract_study_ids(cand_doc_id.replace("-", " "))

    if not ci_ids:
        return 5.0   # neutral — CI does not specify a study

    if not cand_ids:
        return 4.0   # slight disadvantage — candidate has no identifiable IDs

    overlap = ci_ids & cand_ids
    if overlap:
        recall = len(overlap) / len(ci_ids)
        return round(4.0 + recall * 6.0, 3)   # 4.0–10.0
    else:
        return 0.0   # confirmed mismatch — candidate identifies a different study


def _intent_alignment_score(ci_type: str | None, cand: dict) -> float:
    """
    0–10 score based on alignment between CI intent type and candidate section category.

    Strong bonus for expected sections (e.g. OBJECTIVE CI → OBJECTIVES section).
    Background sections are penalised at CI-type-dependent severity: correct answers
    for DOSING/SAFETY/PK CIs frequently live in prior-study BACKGROUND sections, so
    a universal hard penalty here was anti-correlated with correctness in practice.
    APPENDIX / ADMINISTRATIVE / REFERENCE remain hard-penalised for all CI types.
    """
    obj = cand.get("matched_object") or {}
    cat = (obj.get("section_category") or "").upper()

    # Background/appendix penalty — severity is CI-type dependent.
    # Empirical finding: final hits average intent 3.00 vs rejected 4.55 (inverted!).
    # Root cause: correct answers often live in BACKGROUND sections; a universal
    # return 1.5 was causing the feature to score wrong candidates HIGHER than correct ones.
    if cat in _BACKGROUND_CATS:
        if cat in {"APPENDIX", "ADMINISTRATIVE", "REFERENCE"}:
            return 1.5   # structural dead-ends — low for all CI types
        # BACKGROUND: severity depends on CI type
        if ci_type and ci_type.upper() in {"OBJECTIVE", "EFFICACY"}:
            return 2.5   # primary objectives rarely backed by background-only paragraphs
        return 4.0       # DOSING/SAFETY/PK/PHRASE — evidence often in prior-study background

    if not ci_type:
        return 5.0

    entry = _CI_TYPE_SECTIONS.get(ci_type.upper())
    if not entry:
        return 5.0

    strong_cats, weak_cats = entry
    if not strong_cats:   # PHRASE or unmapped
        return 5.0
    if cat in strong_cats:
        score = 10.0
    elif cat in weak_cats:
        score = 6.0
    elif cat in _STRUCTURALLY_INCOMPATIBLE_CATS.get(ci_type.upper(), frozenset()):
        score = 2.0   # structurally wrong for this CI type (was generic 3.5)
    else:
        score = 3.5   # section mismatch but not structurally incompatible

    # Statement-type modifier: if the candidate's own object-level statement_type
    # is GENERAL or PROTOCOL for an assertion-type CI, apply a downgrade.
    # This fires even when the section is in the strong set (e.g. a GENERAL-typed
    # object that happens to live in the OBJECTIVES section).
    stmt = (obj.get("statement_type") or "").upper()
    if ci_type.upper() in {"OBJECTIVE", "EFFICACY"} and stmt in {"GENERAL", "PROTOCOL", "BACKGROUND"}:
        score = min(score, 2.5)

    return score


# ── Structural penalty — additive on the 0–10 composite scale ───────────────────────
# Fires for confirmed structural incompatibilities not captured by the
# weighted feature scores above.  Values are on the 0–10 scale so they
# materially shift the composite without requiring weight tuning.
#
# Design principles:
#  • Never penalise absence of evidence (null / not-enriched stays neutral)
#  • Only penalise when a *specific* structural violation is confirmed
#  • Cap at -3.0 so a single signal cannot eliminate a candidate alone

_STRUCT_PENALTY_CAP = -3.0

# Object subtypes that signal non-assertive content for assertion-type CIs
_NON_ASSERTIVE_SUBTYPES: dict[str, float] = {
    "ABBREVIATION_TABLE": -2.0,   # glossary — no clinical assertions
    "DEFINITION":          -1.0,   # single-term definition
    "SCHEDULE_MATRIX":     -0.8,   # visit/cycle schedule — not evidence
}


# ─── Clinical Validation Engine ───────────────────────────────────────────────
# Architecture: Detect → Validate → Score → Explain.
#
# All semantic validation flows through _ValidationResult.  Every conflict that
# appears in clinical_reasoning.conflicts MUST also appear in contra_penalty
# (and vice-versa) — they derive from the same detection object.
#
# Responsibility boundaries:
#   struct_penalty   — section/type/object-hierarchy issues   (_structural_penalty)
#   contra_penalty   — clinical contradictions                (_detect_semantic_conflicts)
#   agg contradiction — retrieval-stage penalties             (aggregator, not here)


# Combination rules: frozensets of conflict types that together constitute FATAL severity
# even when individual per-issue severities are lower.
# Example: drug (HIGH) + endpoint (MEDIUM) → categorically wrong candidate → FATAL.
_FATAL_CONFLICT_COMBOS: tuple[frozenset[str], ...] = (
    frozenset({"drug", "endpoint"}),
)


class ValidationIssue:
    """
    A single conflict or warning carrying its type, severity, weight, and evidence.

    Replaces plain strings in _ValidationResult.conflicts / .warnings so that
    the explanation layer can render evidence without re-deriving anything.

    To add a new conflict type:
        1. Add one entry to _CONFLICT_METADATA or _STRUCT_ISSUE_METADATA.
        2. Create a ValidationIssue in the corresponding detect function.
        3. Nothing else changes — severity(), scoring, and explanation all
           derive from the metadata automatically.
    """
    __slots__ = ("type", "severity", "weight", "evidence", "outcome")

    def __init__(
        self,
        type:     str,
        severity: str,
        weight:   float = 0.0,
        evidence: dict | None = None,
        outcome:  str | None = None,
    ) -> None:
        self.type     = type
        self.severity = severity
        self.weight   = weight
        self.evidence = evidence if evidence is not None else {}
        self.outcome  = outcome

    def __repr__(self) -> str:
        return (
            f"ValidationIssue(type={self.type!r}, severity={self.severity!r}, "
            f"weight={self.weight}, evidence={self.evidence!r})"
        )


class _ValidationResult:
    """Single source of truth for clinical semantic and structural validation findings."""
    __slots__ = ("matched", "conflicts", "warnings", "missing", "comparator_trace")

    def __init__(self) -> None:
        self.matched:          list[str]               = []   # slot names that agree
        self.conflicts:        list["ValidationIssue"] = []   # hard violations with metadata
        self.warnings:         list["ValidationIssue"] = []   # soft violations with metadata
        self.missing:          list[str]               = []   # absent required CI fact slots
        self.comparator_trace: list                    = []   # all ComparisonResult objects (MATCH+UNKNOWN+RELATED+CONFLICT)

    def merge(self, other: "_ValidationResult") -> "_ValidationResult":
        """Return a new result combining findings from both sources."""
        out = _ValidationResult()
        out.matched          = self.matched   + other.matched
        out.conflicts        = self.conflicts + other.conflicts
        out.warnings         = self.warnings  + other.warnings
        out.missing          = self.missing   + other.missing
        out.comparator_trace = self.comparator_trace  # only semantic side carries traces
        return out

    def severity(self) -> str:
        """
        Derive overall validation severity — driven by per-issue severity levels
        in _CONFLICT_METADATA / _STRUCT_ISSUE_METADATA and _FATAL_CONFLICT_COMBOS.

        Adding a new conflict type requires only updating the metadata dict and
        (if needed) _FATAL_CONFLICT_COMBOS.  severity() itself needs no changes.

          FATAL  — cap at 3.0  (e.g. abbreviation_table, or drug + endpoint)
          HIGH   — cap at 4.5  (e.g. drug alone, or 3+ distinct conflict types)
          MEDIUM — cap at 6.5  (e.g. 2 conflicts with per-issue severity ≤ MEDIUM)
          LOW    — no cap      (single low-severity conflict, or warnings only)
          NONE   — no cap      (nothing detected)
        """
        types = {i.type for i in self.conflicts}

        # Per-issue FATAL (e.g. abbreviation_table — definitionally non-clinical)
        for issue in self.conflicts:
            if issue.severity == _SEV_FATAL:
                return _SEV_FATAL

        # Combination rules: pairs/groups that escalate to FATAL together
        for combo in _FATAL_CONFLICT_COMBOS:
            if combo <= types:
                return _SEV_FATAL

        # 3+ distinct conflict types → escalate to HIGH regardless of per-issue level
        if len(types) >= 3:
            return _SEV_HIGH

        # 2 conflict types → max(per-issue severity, MEDIUM floor)
        if len(types) == 2:
            max_rank = max((_SEV_RANK[i.severity] for i in self.conflicts), default=0)
            return _RANK_SEV[max(max_rank, _SEV_RANK[_SEV_MEDIUM])]

        # 1 conflict type → use its own per-issue severity
        if self.conflicts:
            return self.conflicts[0].severity

        # Warnings only (no conflicts) → LOW
        if self.warnings:
            return _SEV_LOW

        return _SEV_NONE




# Structural issue metadata — single source of truth for weight AND per-issue severity.
# Adding a new structural check: one entry here; _detect_structural_issues creates the
# ValidationIssue; severity() and _score_structural derive everything from it.
_STRUCT_ISSUE_METADATA: dict[str, dict] = {
    "abbreviation_table":   {"weight": -2.0,  "severity": _SEV_FATAL},   # no clinical content
    "definition_object":    {"weight": -1.0,  "severity": _SEV_HIGH},    # single-term definition
    "schedule_matrix":      {"weight": -0.8,  "severity": _SEV_LOW},     # visit/cycle schedule
    "missing_relation":     {"weight": -0.6,  "severity": _SEV_LOW},     # CI has relations; cand has none
    "non_assertive_stmt":   {"weight": -0.4,  "severity": _SEV_LOW},     # GENERAL/BACKGROUND stmt
    "missing_primary_slot": {"weight": -0.25, "severity": _SEV_LOW},     # absent key slot
    "no_object":            {"weight": -0.5,  "severity": _SEV_LOW},     # chunk-level hit
}
# Weight-only view — kept for backward compat; not used internally after refactor.
_STRUCT_ISSUE_WEIGHTS: dict[str, float] = {k: v["weight"] for k, v in _STRUCT_ISSUE_METADATA.items()}
_STRUCT_MISSING_SLOT_WEIGHT = -0.4  # applied once per missing required CI fact slot



# High-signal CI slots: if CI has them and candidate completely lacks them,
# the candidate is unlikely to be the specific evidence sought.
_REQUIRED_CI_SLOTS: dict[str, list[str]] = {
    "OBJECTIVE":        ["dose", "study_arm"],
    "EFFICACY":         ["dose", "study_arm"],
    "DOSING":           ["dose", "study_arm"],
    "SAFETY":           ["dose"],
    "PHARMACOKINETICS": ["dose"],
}

# ─────────────────────────────────────────────────────────────────────────────
# CI-type feature profiles
# ─────────────────────────────────────────────────────────────────────────────
# Instead of one universal weighted sum, each CI type activates a different
# feature profile.  Features irrelevant to a CI type carry weight 0.0 so
# they cannot contribute noise.  Every profile sums to 1.00.
#
# Rationale by type:
#   OBJECTIVE  — intent + drug + endpoint matter; CE is semantic anchor
#   EFFICACY   — drug + endpoint most critical; fact slot carries more
#   DOSING     — drug is primary discriminator; dose facts essential
#   SAFETY     — drug + population entities; AE-class entity families
#   PK         — drug + PK-parameter facts; similar to EFFICACY
#   PHRASE     — no structured intent; CE + entity term dominate; drug/intent minimal
_FEATURE_PROFILES: dict[str, dict[str, float]] = {
    "OBJECTIVE": {
        "ce": 0.27, "drug": 0.18, "agg": 0.10,
        "fact": 0.10, "ent_term": 0.10, "ent_fam": 0.08,
        "intent": 0.10, "source": 0.04, "study": 0.02, "asset": 0.01,
    },
    "EFFICACY": {
        "ce": 0.26, "drug": 0.20, "agg": 0.10,
        "fact": 0.15, "ent_term": 0.09, "ent_fam": 0.07,
        "intent": 0.05, "source": 0.04, "study": 0.03, "asset": 0.01,
    },
    "DOSING": {
        "ce": 0.27, "drug": 0.25, "agg": 0.12,
        "fact": 0.15, "ent_term": 0.08, "ent_fam": 0.05,
        "intent": 0.03, "source": 0.02, "study": 0.02, "asset": 0.01,
    },
    "SAFETY": {
        "ce": 0.26, "drug": 0.22, "agg": 0.12,
        "fact": 0.14, "ent_term": 0.10, "ent_fam": 0.08,
        "intent": 0.03, "source": 0.03, "study": 0.01, "asset": 0.01,
    },
    "PHARMACOKINETICS": {
        "ce": 0.27, "drug": 0.20, "agg": 0.11,
        "fact": 0.14, "ent_term": 0.10, "ent_fam": 0.07,
        "intent": 0.04, "source": 0.04, "study": 0.02, "asset": 0.01,
    },
    "PHRASE": {
        # No structured clinical intent — CE + entity term dominate.
        # Drug/intent/study weights zeroed: they are irrelevant noise for free-text phrases.
        "ce": 0.38, "drug": 0.06, "agg": 0.16,
        "fact": 0.06, "ent_term": 0.16, "ent_fam": 0.10,
        "intent": 0.02, "source": 0.05, "study": 0.00, "asset": 0.01,
    },
}

# Default profile used when CI type is unknown or not listed above.
# Mirrors the empirically-calibrated global weights.
_DEFAULT_PROFILE: dict[str, float] = {
    "ce": 0.30, "drug": 0.16, "agg": 0.12,
    "fact": 0.11, "ent_term": 0.09, "ent_fam": 0.07,
    "intent": 0.05, "source": 0.05, "study": 0.04, "asset": 0.01,
}


def _detect_structural_issues(
    ci_type:      str | None,
    ci_facts:     dict,
    ci_relations: list,
    cand:         dict,
) -> "_ValidationResult":
    """
    Detect structural incompatibilities between CI and candidate.
    Returns a _ValidationResult — pure detection, no weights applied here.
    Score with _score_structural() afterwards.

    Issues map:
      conflicts — hard disqualifiers  (abbreviation_table, definition_object)
      warnings  — soft penalties      (missing_relation, non_assertive_stmt, …)
      missing   — absent required CI fact slots
    """
    result   = _ValidationResult()
    ci_upper = (ci_type or "").upper()
    obj      = cand.get("matched_object") or {}

    _ASSERTION_TYPES = {"OBJECTIVE", "EFFICACY", "DOSING", "SAFETY", "PHARMACOKINETICS"}

    # 1. Object subtype is definitional / non-assertive
    if ci_upper in _ASSERTION_TYPES:
        obj_subtype = (obj.get("object_subtype") or "").upper()
        if obj_subtype == "ABBREVIATION_TABLE":
            _m = _STRUCT_ISSUE_METADATA["abbreviation_table"]
            result.conflicts.append(ValidationIssue(
                type="abbreviation_table", severity=_m["severity"], weight=_m["weight"],
                evidence={"object_subtype": obj_subtype},
            ))
        elif obj_subtype == "DEFINITION":
            _m = _STRUCT_ISSUE_METADATA["definition_object"]
            result.conflicts.append(ValidationIssue(
                type="definition_object", severity=_m["severity"], weight=_m["weight"],
                evidence={"object_subtype": obj_subtype},
            ))
        elif obj_subtype == "SCHEDULE_MATRIX":
            _m = _STRUCT_ISSUE_METADATA["schedule_matrix"]
            result.warnings.append(ValidationIssue(
                type="schedule_matrix", severity=_m["severity"], weight=_m["weight"],
                evidence={"object_subtype": obj_subtype},
            ))
        else:
            # Text-pattern fallback for objects not yet re-indexed with object_subtype.
            # Fires when the candidate text has a high density of ABBREV = definition
            # patterns (e.g. ORR = overall response rate, PFS = progression-free survival).
            _abbrev_text = (obj.get("text") or "")
            _n_abbrevs   = len(re.findall(r'\b[A-Z]{2,8}\s*[=:]\s*\w', _abbrev_text))
            _n_words     = max(len(_abbrev_text.split()), 1)
            if _n_abbrevs >= 4 or (_n_abbrevs >= 2 and _n_abbrevs / _n_words > 0.08):
                _m = _STRUCT_ISSUE_METADATA["abbreviation_table"]
                result.conflicts.append(ValidationIssue(
                    type="abbreviation_table", severity=_m["severity"], weight=_m["weight"],
                    evidence={"pattern": "abbreviation_density", "abbreviations_found": _n_abbrevs, "word_count": _n_words},
                ))

    # 2. CI has clinical relations (structured assertions); candidate has none.
    # Only fire when the candidate object has been enriched (has facts, entities,
    # or statement_type) — absence of relations in an un-enriched object is a
    # data gap, not a confirmed structural mismatch.
    # Use effective_facts for missing_fact_slot and relation checks so that
    # facts inherited from the heading/paragraph context count as present.
    cand_facts       = obj.get("effective_facts") or obj.get("facts", {})
    cand_relations   = obj.get("clinical_relations", [])
    cand_is_enriched = bool(cand_facts or obj.get("entities", []) or obj.get("statement_type"))
    if ci_relations and not cand_relations and cand_is_enriched:
        _m = _STRUCT_ISSUE_METADATA["missing_relation"]
        result.warnings.append(ValidationIssue(
            type="missing_relation", severity=_m["severity"], weight=_m["weight"],
            evidence={"ci_relations": len(ci_relations), "candidate_relations": 0},
        ))

    # 3. CI specifies high-signal required slots; candidate completely lacks them.
    for slot in _REQUIRED_CI_SLOTS.get(ci_upper, []):
        if ci_facts.get(slot) and not cand_facts.get(slot):
            result.missing.append(slot)

    # 4. Statement type is non-assertive for assertion-type CIs.
    # NOTE: conservative (-0.4) — many correct answers in BACKGROUND sections
    # carry statement_type=GENERAL (data gap, not semantic general-ness).
    if ci_upper in _ASSERTION_TYPES:
        stmt = (obj.get("statement_type") or "").upper()
        if stmt in {"GENERAL", "BACKGROUND"}:
            _m = _STRUCT_ISSUE_METADATA["non_assertive_stmt"]
            result.warnings.append(ValidationIssue(
                type="non_assertive_stmt", severity=_m["severity"], weight=_m["weight"],
                evidence={"statement_type": stmt},
            ))

    # 5. Candidate has structured facts but entirely lacks the expected primary slot.
    # Fires without requiring CI facts — CI type implies the expectation.
    _TYPE_EXPECTED_SLOTS: dict[str, str] = {
        "OBJECTIVE":        "endpoint",
        "EFFICACY":         "endpoint",
        "DOSING":           "dose",
        "SAFETY":           "dose",
        "PHARMACOKINETICS": "endpoint",
    }
    expected_slot = _TYPE_EXPECTED_SLOTS.get(ci_upper)
    if expected_slot and obj and cand_facts and not cand_facts.get(expected_slot):
        _m = _STRUCT_ISSUE_METADATA["missing_primary_slot"]
        result.warnings.append(ValidationIssue(
            type="missing_primary_slot", severity=_m["severity"], weight=_m["weight"],
            evidence={"expected_slot": expected_slot},
        ))

    # 6. No semantic object at all (chunk-level hit) for an assertion-type CI.
    # Without matched_object, statement_type / facts / relations are all unknown.
    if not obj and ci_upper in _ASSERTION_TYPES:
        _m = _STRUCT_ISSUE_METADATA["no_object"]
        result.warnings.append(ValidationIssue(
            type="no_object", severity=_m["severity"], weight=_m["weight"],
            evidence={"reason": "no matched_object for assertion-type CI"},
        ))

    return result


def _score_structural(vr: "_ValidationResult") -> tuple[float, dict]:
    """
    Convert a structural _ValidationResult into (penalty, detail).
    Reads weight directly from each ValidationIssue — no external lookup needed.
    struct_detail now carries weight, severity, and evidence per issue,
    making the breakdown fully self-documenting.
    """
    penalty = 0.0
    detail: dict = {}
    for issue in (*vr.conflicts, *vr.warnings):
        if issue.weight:
            penalty += issue.weight
            detail[issue.type] = {
                "weight":   round(issue.weight, 3),
                "severity": issue.severity,
                "evidence": issue.evidence,
            }
    if vr.missing:
        slot_pen = len(vr.missing) * _STRUCT_MISSING_SLOT_WEIGHT
        penalty  += slot_pen
        detail["missing_slots"] = {
            "weight":   round(slot_pen, 3),
            "severity": _SEV_LOW,
            "evidence": {"missing_slots": vr.missing},
        }
    return round(max(penalty, _STRUCT_PENALTY_CAP), 3), detail





def _detect_semantic_conflicts(
    ci_ctx:   "ClinicalContext",
    cand_ctx: "ClinicalContext",
) -> "_ValidationResult":
    """
    Orchestrate all clinical comparators and merge their results.

    Each comparator returns ComparisonResult(slot, outcome, score, severity, evidence).
    Outcomes map to _ValidationResult slots:
      CMP_MATCH          -> result.matched   (positive evidence)
      CMP_RELATED        -> result.warnings  (soft — same space, not identical)
      CMP_SPECIALIZATION -> result.warnings  (soft — candidate is specific instance of CI class)
      CMP_GENERALIZATION -> result.warnings  (soft — candidate is broader than CI asked for)
      CMP_CONFLICT       -> result.conflicts (hard — confirmed semantic mismatch)
      CMP_UNKNOWN        -> (nothing — absence of data != contradiction)

    To add a new comparator: write the function and add it to _COMPARATORS.
    This function needs no changes.
    """
    result = _ValidationResult()
    _SOFT_OUTCOMES = frozenset({CMP_RELATED, CMP_SPECIALIZATION, CMP_GENERALIZATION})
    for cmp_fn in _COMPARATORS:
        cmp = cmp_fn(ci_ctx, cand_ctx)
        result.comparator_trace.append(cmp)   # always record — MATCH and UNKNOWN included
        if cmp.outcome == CMP_MATCH:
            result.matched.append(cmp.slot)
        elif cmp.outcome == CMP_CONFLICT:
            result.conflicts.append(ValidationIssue(
                type=cmp.slot, severity=cmp.severity,
                weight=cmp.score, evidence=cmp.evidence, outcome=cmp.outcome,
            ))
        elif cmp.outcome in _SOFT_OUTCOMES:
            result.warnings.append(ValidationIssue(
                type=cmp.slot, severity=cmp.severity,
                weight=cmp.score, evidence=cmp.evidence, outcome=cmp.outcome,
            ))
        # CMP_UNKNOWN: no entry added — absence of data != contradiction
    return result

def _score_semantic_conflicts(vr: "_ValidationResult") -> tuple[float, dict]:
    """
    Convert a semantic _ValidationResult into (contra_penalty, contra_detail).

    Scores both conflicts (CMP_CONFLICT) and warnings (CMP_RELATED) — both
    carry penalty weights.  The 'outcome' key in each detail entry distinguishes
    them for the explanation layer without affecting the penalty arithmetic.
    """
    penalty = 0.0
    detail: dict = {}
    for issue in vr.conflicts:
        if issue.weight:
            penalty += issue.weight
            detail[issue.type] = {
                "weight":   round(issue.weight, 3),
                "severity": issue.severity,
                "outcome":  issue.outcome or CMP_CONFLICT,
                "evidence": issue.evidence,
            }
    for issue in vr.warnings:
        if issue.weight:
            penalty += issue.weight
            detail[issue.type] = {
                "weight":   round(issue.weight, 3),
                "severity": issue.severity,
                "outcome":  issue.outcome or CMP_RELATED,
                "evidence": issue.evidence,
            }
    return round(penalty, 3), detail


def _interaction_bonus(
    drug_norm: float,
    fact_norm: float,
    ci_facts:  dict,
) -> float:
    """
    Additive interaction term for drug × fact co-evidence.

    Independent drug and fact scores miss the clinical interaction:
    concordant evidence (drug AND facts both match) should be worth more
    than the sum of parts; drug context with absent/contradicting facts should
    be penalised beyond the individual fact penalty alone.

    Returns an additive delta in the range [−0.30, +0.50].
    Thresholds are conservative — the primary weighted scores still dominate.
    """
    has_ci_structure = any(ci_facts.get(s) for s in ("drug", "endpoint", "dose"))

    # Strong concordant evidence — both drug identity and fact slot are high
    if drug_norm >= 6.5 and fact_norm >= 5.5:
        return +0.50   # genuine co-evidence bonus

    # Moderate concordant evidence
    if drug_norm >= 4.0 and fact_norm >= 3.5:
        return +0.20

    # Drug context confirmed but CI's expected facts are entirely absent
    # (only applies when CI has structured facts to compare against)
    if drug_norm >= 5.0 and fact_norm <= 1.0 and has_ci_structure:
        return -0.30   # drug found, facts expected but missing — suspicious gap

    # Neither drug nor facts found when CI expects them — double absence
    if drug_norm <= 1.5 and fact_norm <= 1.0 and has_ci_structure:
        return -0.20

    return 0.0


def _composite_score(
    cand: dict,
    ce_logit: float,
    ci_assets: list,
    query_norm: "_QueryNorm",
    ci_type: str | None = None,
    ci_text: str = "",
    ci_entities: list[dict] | None = None,
    ci_confidence: float = 1.0,
    ci_facts: dict | None = None,
    ci_relations: list | None = None,
    ci: dict | None = None,
) -> tuple[float, dict]:
    """
    CI-type-conditional composite score — all components normalised to 0–10.

    Feature weights are looked up from _FEATURE_PROFILES[ci_type] so that
    irrelevant signals are zeroed rather than contributing constant noise.
    When ci_type is unknown the _DEFAULT_PROFILE is used (empirically calibrated).

    Additive terms (not included in the profile sum):
      • struct_penalty  — structural incompatibility flag (≤ 0)
      • pos_bonus       — document-position alignment bonus (max +0.30)
      • interaction     — drug × fact co-evidence synergy/penalty (±0.50)

    Study ID weight is set to 0.0 at runtime when the CI has no extractable
    study IDs (returns a constant 5.0 in that case — pure noise).
    """
    if ci_entities is None:
        ci_entities = []
    if ci_facts is None:
        ci_facts = {}
    if ci_relations is None:
        ci_relations = []
    # Build ClinicalContext objects so _detect_semantic_conflicts stays signature-stable.
    # ci (the raw CI dict) supplies treatment_identity, temporal_context, negated_slots,
    # modality etc. that aren't forwarded as individual positional args.
    _ci_ctx   = _build_ci_context(ci or {}, ci_entities)
    _cand_ctx = _build_cand_context(cand)

    # 1. Cross-encoder: sigmoid(logit) → 0–1 → 0–10
    ce_norm = _sigmoid(ce_logit) * 10.0

    # 2. Retrieval score
    agg_norm = min(cand.get("agg_score", 0.0) * 2.0, 10.0)

    # 3. Entity family overlap
    matched_obj   = cand.get("matched_object") or {}
    cand_entities = matched_obj.get("entities", [])
    # effective_facts: inherited heading/paragraph context fills slots the object
    # didn't explicitly mention (e.g. drug from heading, endpoint from sentence).
    # Fall back to facts for objects indexed before inheritance was added.
    cand_facts_early = (matched_obj.get("effective_facts") or
                        matched_obj.get("facts") or {})
    # query_norm.label_families is populated by shared.query_normalizer; when the
    # module is absent (local test), fall back to families derived from CI entities.
    ci_entities_for_fam = ci_entities if ci_entities else []
    ent_fam_query = query_norm.label_families or frozenset(
        e.get("label") or e.get("family") or "OTHER"
        for e in ci_entities_for_fam
        if e.get("label") or e.get("family")
    )
    ent_fam_norm  = _entity_family_overlap(ent_fam_query, cand_entities, cand_facts_early)

    # 4. Normalized term overlap
    ent_term_norm = _entity_term_overlap(query_norm.normalized_terms, cand_entities)

    # 5. Source richness
    sources = cand.get("sources", [])
    src_pts = sum([
        4.0 if "literal"  in sources else 0.0,
        3.0 if "ontology" in sources else 0.0,
        2.0 if "bm25"     in sources else 0.0,
        1.0 if "vector"   in sources else 0.0,
    ])
    source_norm = min(src_pts * 10.0 / 10.0, 10.0)

    # 6. Asset match
    text_lower = _candidate_text(cand).lower()
    asset_hit  = _asset_match(text_lower, ci_assets)
    asset_norm = 10.0 if asset_hit else 0.0

    # 7. Section score
    boost_mult   = _section_boost(cand)
    section_norm = (boost_mult - _MIN_BOOST) / (_MAX_BOOST - _MIN_BOOST) * 10.0

    # 8. Document-position alignment bonus (additive, max +0.30)
    pos_bonus = _document_position_bonus(cand)

    # ── New structured clinical signals ──────────────────────────────────────
    # 9. Drug / medication identity (Gap 1 — biggest precision gap)
    drug_norm = _drug_identity_score(ci_entities, cand_entities, cand)

    # 10. Study / protocol ID overlap (Gap 2)
    # Prefer matched_object.document_id; fall back to chunk-level candidate
    # fields (always present) so protocol numbers embedded in document/chunk
    # IDs are compared even when the matched object has no document_id field.
    cand_doc_id = (
        matched_obj.get("document_id")
        or cand.get("document_id")
        or cand.get("chunk_id", "")
    )
    study_norm = _study_id_score(ci_text, text_lower, cand_doc_id)

    # 11. Intent / section alignment + background penalty (Gaps 3, 4, 8)
    intent_norm = _intent_alignment_score(ci_type, cand)

    # Sub-intent refinement A: eval-vs-comparison mismatch.
    # If the CI evaluates one specific treatment but the candidate compares arms,
    # the section match (both OBJECTIVES) is superficially correct but misaligned.
    if intent_norm >= 7.0 and ci_type and ci_type.upper() in {"OBJECTIVE", "EFFICACY"}:
        _eval_kws = frozenset({"evaluate", "assess", "measure", "determine"})
        _cmp_kws  = frozenset({"compare", "versus", " vs ", "superiority",
                                "non-inferior", "combination", "all arms"})
        ci_lower    = ci_text.lower()
        ci_is_eval  = any(kw in ci_lower for kw in _eval_kws)
        ci_is_cmp   = any(kw in ci_lower for kw in _cmp_kws)
        cand_is_cmp = any(kw in text_lower for kw in _cmp_kws)
        if ci_is_eval and not ci_is_cmp and cand_is_cmp:
            intent_norm = max(intent_norm - 3.5, 0.0)

    # Sub-intent refinement B: endpoint sub-type (ORR vs PFS vs OS vs DOR …).
    # Within OBJECTIVE / EFFICACY, the section score is too coarse — both an
    # "evaluate ORR" CI and an "evaluate PFS" CI map to OBJECTIVES section.
    # When both CI and candidate can be unambiguously classified to a specific
    # endpoint sub-intent:
    #   • Same sub-intent  → +2.0 bonus  (confirmed hierarchical match)
    #   • Different        → −2.5 penalty (confirmed hierarchical mismatch)
    # Ambiguous or unclassifiable texts get no adjustment (no-penalty / no-bonus
    # principle: do not penalise what we cannot determine).
    if ci_type and ci_type.upper() in {"OBJECTIVE", "EFFICACY"} and intent_norm >= 2.0:
        _ci_sub   = _classify_endpoint_subintent(ci_text)
        _cand_sub = _classify_endpoint_subintent(text_lower)
        if _ci_sub and _cand_sub:
            if _ci_sub == _cand_sub:
                intent_norm = min(intent_norm + 2.0, 10.0)   # hierarchical match
            else:
                intent_norm = max(intent_norm - 2.5, 0.0)    # hierarchical mismatch

    # 12. Slot-weighted fact overlap — separates drug+endpoint from drug+AE
    # effective_facts is used so that inherited context (drug from heading)
    # is included — prevents false 0-overlap when the object itself omits the drug.
    cand_facts  = cand_facts_early   # already resolved to effective_facts above
    fact_norm   = _fact_slot_overlap(ci_facts, cand_facts)

    # 13. Structural issues — Detect → Score (mirrors semantic validation pattern).
    #     _detect_structural_issues returns a _ValidationResult; _score_structural
    #     applies _STRUCT_ISSUE_WEIGHTS to produce (penalty, detail).  The same
    #     _struct_vr is later merged into _validation for severity computation.
    _struct_vr                    = _detect_structural_issues(ci_type, ci_facts, ci_relations, cand)
    struct_penalty, struct_detail = _score_structural(_struct_vr)

    # 12. Object-type granularity prior — conditional on CI type, scaled by confidence.
    #     Applied as a small additive bonus so a weak candidate cannot leapfrog a
    #     semantically stronger one purely due to object type.
    #     gran_factor = raw_bonus (0–0.30) / 2 * confidence → 0.000–0.150 additive range.
    #     At max: a sentence for a PK CI adds 0.15 to base — a meaningful tiebreaker.
    obj_type     = (cand.get("matched_object") or {}).get("type", "")
    raw_gran     = _granularity_bonus(ci_type, obj_type)
    gran_factor  = (raw_gran / 2.0) * min(max(ci_confidence, 0.0), 1.0)

    # ── Feature profile lookup ────────────────────────────────────────────────
    ci_upper_key = (ci_type or "").upper()
    profile      = dict(_FEATURE_PROFILES.get(ci_upper_key, _DEFAULT_PROFILE))

    # Study ID: when CI has no extractable study IDs the score is constant
    # (returns 5.0 for ~97% of decisions — pure noise at that point).
    # Zero its weight and redistribute to CE so the formula stays calibrated.
    if not _extract_study_ids(ci_text):
        freed = profile["study"]
        profile["study"] = 0.0
        profile["ce"]    = round(profile["ce"] + freed, 3)

    # Drug × fact interaction — captures concordance that additive sum misses
    interaction = _interaction_bonus(drug_norm, fact_norm, ci_facts)

    # ── Clinical Validation: Detect → Score ────────────────────────────────────
    # _detect_semantic_conflicts returns a _ValidationResult (pure detection).
    # _score_semantic_conflicts converts it to (contra_penalty, contra_detail)
    # using _CONFLICT_WEIGHTS as the single source of truth.
    # clinical_reasoning below reads from the same _semantic_vr so the
    # explanation and the penalty are always in sync.
    _semantic_vr                  = _detect_semantic_conflicts(_ci_ctx, _cand_ctx)
    contra_penalty, contra_detail = _score_semantic_conflicts(_semantic_vr)

    base = (
        profile["ce"]       * ce_norm
        + profile["drug"]   * drug_norm
        + profile["agg"]    * agg_norm
        + profile["fact"]   * fact_norm
        + profile["ent_term"] * ent_term_norm
        + profile["ent_fam"]  * ent_fam_norm
        + profile["intent"] * intent_norm
        + profile["source"] * source_norm
        + profile["study"]  * study_norm
        + profile["asset"]  * asset_norm
        + pos_bonus
        + struct_penalty    # additive, ≤0; structural absence penalties
        + interaction       # additive, ±0.50; drug×fact concordance
        + contra_penalty    # additive, ≤0; semantic value conflicts
    )
    composite = min(round(base + gran_factor, 3), 10.0)

    # CE veto — severity-based composite caps.
    # Merge semantic + structural findings into one _ValidationResult; derive
    # a single severity level from it.  No more scattered slot-name checks —
    # the severity() method encodes the cap rules in one auditable place.
    _validation = _semantic_vr.merge(_struct_vr)
    _sev        = _validation.severity()
    _pre_veto   = composite
    if _sev == _SEV_FATAL:
        composite = min(composite, 3.0)
    elif _sev == _SEV_HIGH:
        composite = min(composite, 4.2)   # FIX-5: cap below RELATED floor (4.5) so HIGH truly rejects
    elif _sev == _SEV_MEDIUM:
        composite = min(composite, 6.5)
    composite = round(composite, 3)

    # ── Clinical reasoning summary ─────────────────────────────────────────────
    # Human-readable explanation of the scoring decision: lists what matched,
    # what conflicted, and a one-line decision rationale.
    _matched:   list[str] = []
    _conflicts: list[str] = []
    # Score-based match signals
    if drug_norm    >= 6.0:  _matched.append("drug")
    if fact_norm    >= 5.0:  _matched.append("facts")
    if ce_norm      >= 6.0:  _matched.append("semantic")
    if intent_norm  >= 7.0:  _matched.append("section")
    if interaction  >= 0.40: _matched.append("co-evidence")
    # Score-based conflict signals (these affect struct_penalty / interaction,
    # not contra_penalty — they are clinically distinct from semantic conflicts)
    if fact_norm      <= 1.0 and ci_facts: _conflicts.append("facts")
    if struct_penalty <= -1.0:             _conflicts.append("structure")  # → struct_penalty
    if interaction    <= -0.20:            _conflicts.append("drug-fact gap")
    # Semantic conflicts and matches — derived from the SAME _ValidationResult
    # that produced contra_penalty / contra_detail.  This is the single source
    # of truth: it is now impossible for clinical_reasoning to report a conflict
    # that is absent from contra_detail, or vice versa.
    for _issue in _semantic_vr.conflicts:
        if _issue.type not in _conflicts:
            _conflicts.append(_issue.type)
    for _slot in _semantic_vr.matched:
        _lbl = _slot + " match"
        if _lbl not in _matched:
            _matched.append(_lbl)
    if not _matched and not _conflicts:
        _decision = "Marginal: weak evidence across all signals"
    elif _conflicts and not _matched:
        _decision = "Rejected: " + ", ".join(_conflicts) + " conflict"
    elif _matched and not _conflicts:
        _decision = "Accepted: " + ", ".join(_matched) + " aligned"
    else:
        _decision = ("Partial: matched [" + ", ".join(_matched)
                     + "] but conflicts [" + ", ".join(_conflicts) + "]")

    # ── Enrichment status + diagnostics (computed before breakdown dict) ───────
    _ci_raw = ci or {}

    # Boolean presence map — immediately shows which normalized fields were
    # populated at index time for CI and candidate.  One glance tells you
    # whether a comparator returned UNKNOWN because of an enrichment gap.
    _enrichment_status: dict = {
        "ci": {
            "entities":            bool(_ci_raw.get("entities")),
            "facts":               bool(_ci_raw.get("facts")),
            "effective_facts":     bool(ci_facts),
            "clinical_identity":   bool(_ci_raw.get("clinical_identity")),
            "treatment_identity":  bool(_ci_raw.get("treatment_identity")),
            "endpoint_identity":   bool(_ci_raw.get("endpoint_identity")),
            "population_identity": bool(_ci_raw.get("population_identity")),
            "clinical_relations":  bool(ci_relations),
            "statement_type":      bool(_ci_raw.get("statement_type")),
            "modality":            bool(_ci_raw.get("modality")),
        },
        "candidate": {
            "entities":            bool(cand_entities),
            "facts":               bool(matched_obj.get("facts")),
            "effective_facts":     bool(matched_obj.get("effective_facts")),
            "clinical_identity":   bool(matched_obj.get("clinical_identity")),
            "treatment_identity":  bool(matched_obj.get("treatment_identity")),
            "endpoint_identity":   bool(matched_obj.get("endpoint_identity")),
            "population_identity": bool(matched_obj.get("population_identity")),
            "clinical_relations":  bool(matched_obj.get("clinical_relations")),
            "statement_type":      bool(matched_obj.get("statement_type")),
            "modality":            bool(matched_obj.get("modality")),
        },
    }

    # Stage-level enrichment diagnosis for CI and candidate.
    # Pipeline order: ner → fact_extractor → effective_facts_propagator → identity_builder → complete
    # The first missing stage is reported so you know exactly where to look when a
    # comparator returns UNKNOWN.  missing_slots lists every absent field regardless
    # of which stage is blamed, so a single glance shows the full gap.
    def _diagnose_obj(obj: dict) -> dict:
        _missing = [
            f for f in (
                "entities", "facts", "effective_facts",
                "clinical_identity", "treatment_identity",
                "endpoint_identity", "population_identity",
                "clinical_relations",
            )
            if not obj.get(f)
        ]
        # Sentence objects have their own entities (sentence-level filtered subset)
        # but inherit facts / effective_facts / clinical_identity from their parent
        # paragraph via build_enrichment_fields(parent).  An empty entities list
        # does NOT mean NER failed — it can simply mean the sentence's own text
        # contains no recognised entities while the parent paragraph does.
        has_own_ents = bool(obj.get("entities"))
        has_facts    = bool(obj.get("facts"))
        has_eff      = bool(obj.get("effective_facts"))

        if not has_own_ents and not has_facts and not has_eff:
            _stg, _sta = "ner", "FAILED"
            _rsn = "no entities, facts, or effective_facts — object fully unenriched"
        elif has_own_ents and not has_facts and not has_eff:
            _stg, _sta = "fact_extractor", "FAILED"
            _rsn = "entities present but no facts or effective_facts extracted"
        elif not has_eff:
            _stg, _sta = "effective_facts_propagator", "FAILED"
            _rsn = "facts present but effective_facts not propagated from heading context"
        elif any(not obj.get(f) for f in (
                "clinical_identity", "treatment_identity",
                "endpoint_identity",  "population_identity")):
            _stg, _sta = "identity_builder", "PARTIAL"
            _rsn = "effective_facts present but one or more identity fields absent"
        else:
            _stg, _sta = "complete", "OK"
            _rsn = "all expected enrichment fields present"
        return {"stage": _stg, "status": _sta, "reason": _rsn, "missing_slots": _missing}

    _enrichment_diag: dict = {
        "ci":        _diagnose_obj(_ci_raw),
        "candidate": _diagnose_obj(matched_obj),
    }

    breakdown = {
        "ce":            round(ce_norm,       3),
        "retrieval":     round(agg_norm,      3),
        "drug_identity": round(drug_norm,     3),
        "intent_align":  round(intent_norm,   3),
        "entity_family": round(ent_fam_norm,  3),
        "fact_slot":     round(fact_norm,     3),
        "study_id":      round(study_norm,    3),
        "source":        round(source_norm,   3),
        "entity_term":   round(ent_term_norm, 3),
        "asset":         round(asset_norm,    3),
        "pos_bonus":     round(pos_bonus,     3),
        "struct_penalty":  round(struct_penalty,  3),
        "struct_detail":   struct_detail,
        "contra_penalty":  round(contra_penalty,  3),
        "contra_detail":   contra_detail,
        "interaction":     round(interaction,     3),
        "gran_factor":     round(gran_factor,     4),
        "section_mult":  boost_mult,
        "profile":       ci_upper_key or "DEFAULT",
        "composite":     composite,
        # ── FP debug fields ──────────────────────────────────────────────────
        # All comparator outcomes including MATCH and UNKNOWN — invisible in
        # contra_detail because they carry no penalty.  Essential for diagnosing
        # FPs where no conflict fired due to empty facts on one or both sides.
        "comparator_trace": {
            cmp.slot: {
                "outcome":   cmp.outcome,
                "score":     round(cmp.score, 3),
                "severity":  cmp.severity,
                "evidence":  cmp.evidence,
            }
            for cmp in _semantic_vr.comparator_trace
        },
        # Merged severity level that drove the CE veto cap (NONE/LOW/MEDIUM/HIGH/FATAL).
        "validation_severity": _sev,
        # Composite before the veto was applied — shows how much the veto moved the score.
        "pre_veto_composite":  round(_pre_veto, 3),
        # Facts both sides brought — lets you see why a comparator returned UNKNOWN
        # (e.g. candidate had empty effective_facts.drug).
        "ci_facts":   {k: v for k, v in ci_facts.items()  if v},
        "cand_facts": {k: v for k, v in cand_facts.items() if v},
        # Which normalized enrichment fields were populated at index/CI build time.
        # False on effective_facts/clinical_identity/treatment_identity means the
        # enrichment pipeline didn't run or failed for that object — immediate root
        # cause for a cluster of UNKNOWN comparator outcomes.
        "enrichment_status":      _enrichment_status,
        # Per-slot root cause for every comparator that returned UNKNOWN.
        # reason: both_empty | ci_empty | cand_empty | insufficient
        # Use this to distinguish: comparator logic issue vs enrichment gap.
        "enrichment_diagnostics": _enrichment_diag,
        # ────────────────────────────────────────────────────────────────────
        "clinical_reasoning": {
            "matched":   _matched,
            "conflicts": _conflicts,
            "decision":  _decision,
        },
    }
    return composite, breakdown
def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _asset_match(text_lower: str, ci_assets: list) -> bool:
    """Return True if any CI asset name or alias appears in the candidate text."""
    for asset in (ci_assets or []):
        if not asset:
            continue
        for term in [asset.get("name", ""), asset.get("genericName", ""), *asset.get("aliases", [])]:
            if term and len(term) > 2 and term.lower() in text_lower:
                return True
    return False


# _cross_encode (Claude/Bedrock) removed — replaced by local CrossEncoder model above.
