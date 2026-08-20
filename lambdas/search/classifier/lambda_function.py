"""
Search Pipeline — Stage 1: CI Classifier
==========================================
Determines what *type* of CI it is and which retrieval strategies
should be activated.  This makes the search adaptive — not every
retriever runs for every CI.

The POC rule: every CI is different.
  NCT03456789     → Regex is almost perfect.  Embeddings are overkill.
  Dr John Smith   → NER + Literal + Fuzzy.
  Principal Inv.  → Ontology + Vector + Cross Encoder.
  Protocol Number → Section-awareness + Regex + Metadata.

Input
------
{
    "search_id":   str,
    "document_id": str,
    "ci":          dict   # enriched CI object (must have normalization, ner, ontology, embedding)
}

Appends
-------
"classification": {
    "ci_type":   str,     # PERSON | IDENTIFIER | CLINICAL_ROLE | ORGANIZATION | PHRASE
    "strategies": list[str],  # which retriever keys to run
    "reason":    str,
}
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ─── CI type → retrieval strategies (text-inference fallback) ────────────────
_STRATEGIES: dict[str, list[str]] = {
    "PERSON":        ["literal", "ner", "bm25", "vector"],
    "IDENTIFIER":    ["regex", "bm25", "literal"],
    "CLINICAL_ROLE": ["ontology", "vector", "bm25"],
    "ORGANIZATION":  ["literal", "ner", "vector"],
    "PHRASE":        ["bm25", "vector", "ontology", "literal"],
    # Numeric/statistical CIs bypass vector/BM25 entirely — the number IS the secret.
    # The numeric retriever pre-filters to documents containing the exact numeric
    # tokens (sample size, CI bounds, p-value) rather than doing semantic search.
    "NUMERIC_SAMPLE_SIZE":  ["numeric", "literal"],
    "CONFIDENCE_INTERVAL":  ["numeric", "literal"],
    "P_VALUE":              ["numeric", "literal"],
    "HAZARD_RATIO":         ["numeric", "literal"],
    "ODDS_RATIO":           ["numeric", "literal"],
    "NUMERIC_PERCENTAGE":   ["numeric", "literal"],
    "MEDIAN":               ["numeric", "literal"],
    # Legacy coarse types — kept for backward compatibility
    "NUMERIC":              ["numeric", "literal"],
    "STATISTICAL":          ["numeric", "literal"],
}

# ─── Human-assigned category code → ci_type + strategies ────────────────────
# Keyed by category.code (U+2017 ‗ delimited slug as stored in the database).
# Takes strict precedence over the text-based classifier — human metadata is
# always more accurate than inferred intent.
#
# Adding "fact" to strategies for structured clinical categories means the
# Fact Retriever is invoked for objectives/endpoints/dosing CIs, which is
# exactly where pre-computed facts.* and clinical_relations fields help.
_CATEGORY_MAP: dict[str, dict] = {
    # ── Objectives & Endpoints ────────────────────────────────────────────────
    "primary\u2017objectives\u2017and\u2017endpoints": {
        "ci_type": "OBJECTIVE",
        "strategies": ["bm25", "vector", "ontology", "literal", "fact"],
    },
    "primary\u2017objectives": {
        "ci_type": "OBJECTIVE",
        "strategies": ["bm25", "vector", "ontology", "literal", "fact"],
    },
    "endpoints": {
        "ci_type": "EFFICACY",
        "strategies": ["bm25", "vector", "ontology", "literal", "fact"],
    },
    "why\u2017is\u2017this\u2017study\u2017being\u2017done?": {
        "ci_type": "STUDY_DESIGN",
        "strategies": ["bm25", "vector", "ontology", "literal", "fact"],
    },
    "exploratory": {
        "ci_type": "EFFICACY",
        "strategies": ["bm25", "vector", "ontology", "literal", "fact"],
    },
    # ── Dosing ────────────────────────────────────────────────────────────────
    "analysis\u2017of\u2017clinical\u2017information\u2017relevant\u2017to\u2017dosing\u2017recomendations": {
        "ci_type": "DOSING",
        "strategies": ["bm25", "vector", "ontology", "literal", "fact"],
    },
    "justification\u2017for\u2017dose": {
        "ci_type": "DOSING",
        "strategies": ["bm25", "vector", "ontology", "literal", "fact"],
    },
    "dosage": {
        "ci_type": "DOSING",
        "strategies": ["bm25", "vector", "ontology", "literal", "fact"],
    },
    "dosage\u2017&\u2017administration\u2017(clinical\u2017use)": {
        "ci_type": "DOSING",
        "strategies": ["bm25", "vector", "ontology", "literal", "fact"],
    },
    "sample\u2017size": {
        "ci_type": "EFFICACY",
        "strategies": ["bm25", "vector", "ontology", "literal", "fact"],
    },
    # ── Study Design / Protocol ───────────────────────────────────────────────
    "study\u2017design": {
        "ci_type": "STUDY_DESIGN",
        "strategies": ["bm25", "vector", "ontology", "literal", "fact"],
    },
    "product\u2017development\u2017rationale": {
        "ci_type": "STUDY_DESIGN",
        "strategies": ["bm25", "vector", "ontology", "literal"],
    },
    "schedule\u2017of\u2017events\u2017(timepoints)": {
        "ci_type": "PROTOCOL",
        "strategies": ["bm25", "vector", "literal"],
    },
    "business\u2017regulatory\u2017strategy": {
        "ci_type": "STUDY_DESIGN",
        "strategies": ["bm25", "vector", "ontology", "literal"],
    },
    # ── Formulation / Manufacturing ───────────────────────────────────────────
    "formulation\u2017development": {
        "ci_type": "MANUFACTURING",
        "strategies": ["bm25", "literal", "vector"],
    },
    "formulation": {
        "ci_type": "MANUFACTURING",
        "strategies": ["bm25", "literal", "vector"],
    },
    "novel\u2017manufacturing\u2017method/process": {
        "ci_type": "MANUFACTURING",
        "strategies": ["bm25", "literal", "vector"],
    },
    "product\u2017characteristics": {
        "ci_type": "IDENTIFIER",
        "strategies": ["regex", "bm25", "literal", "vector"],
    },
    # ── Pharmacokinetics / Bioanalytical ──────────────────────────────────────
    "biopharmaceutics": {
        "ci_type": "PHARMACOKINETICS",
        "strategies": ["bm25", "vector", "ontology", "literal", "fact"],
    },
    "bioanalytical\u2017method": {
        "ci_type": "PHARMACOKINETICS",
        "strategies": ["bm25", "literal", "vector"],
    },
    # ── Vendor / Organization ─────────────────────────────────────────────────
    "vendor\u2017name\u2017and\u2017details": {
        "ci_type": "ORGANIZATION",
        "strategies": ["literal", "ner", "vector"],
    },
}

# Regex patterns that signal IDENTIFIER type
_IDENTIFIER_PATTERNS = [
    r"\bNCT\d{6,}\b",
    r"\b[A-Z]{2,}-\d{4,}\b",          # protocol codes like RXP-2024-001
    r"\b\d{5,}\b",                      # long numeric IDs
    r"\bProtocol\s+(?:Number|No\.?)\b",
    r"\b(?:IND|NDA|BLA)\s*[:\s]\s*\d+",
]

# Terms that signal CLINICAL_ROLE type
_ROLE_TERMS = {
    "investigator", "principal investigator", "sub-investigator", "sponsor",
    "monitor", "cra", "coordinator", "crc", "pi",
}

# Person name detector — kept TIGHT.
# Only fires on short CI text (≤ 80 chars, no newlines) that looks like a real name.
# Prevents "International Myeloma Working Group" from triggering a false PERSON match.
_PERSON_RE = re.compile(
    r"^\s*Dr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s*$"   # Dr. Firstname [Lastname]
    r"|"
    r"^\s*(?:Prof\.?|Mr\.?|Ms\.?|Mrs\.?)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s*$"
    r"|"
    r"^\s*[A-Z][a-z]+\s+[A-Z][a-z]+\s*$",                # bare First Last (short only)
)

# ─── Numeric / Statistical CI detection ───────────────────────────────────────────────────────────────────────────
# These are checked BEFORE category-map routing.  A CI that is predominantly
# a number is always better served by the numeric retriever than by semantic
# search, regardless of what human-assigned category label it carries.

_SAMPLE_SIZE_NUM_RE = re.compile(
    r'\b[nN]\s*=\s*\d+'
    r'|\b\d+\s+(?:subjects?|patients?|participants?|individuals?|volunteers?)\b'
    r'|\(\d+\s+(?:subjects?|patients?)\s+in\b',
    re.I,
)
_CONFIDENCE_INTERVAL_NUM_RE = re.compile(
    r'\b\d+\s*%\s+CI\b',
    re.I,
)
_P_VALUE_NUM_RE = re.compile(
    r'\bp\s*[<>=\u2264\u2265]\s*0\.\d+'
    r'|\bp\s*-?\s*value\b',
    re.I,
)
_HR_NUM_RE = re.compile(
    r'\bHR\s*=\s*[\d.]+'
    r'|\bhazard\s+ratio\b',
    re.I,
)
_OR_NUM_RE = re.compile(
    r'\bOR\s*=\s*[\d.]+'
    r'|\bodds\s+ratio\b',
    re.I,
)
# Isolated percentage: entire short CI is just a percentage value
_PURE_PERCENTAGE_NUM_RE  = re.compile(r'^\s*\d+(?:\.\d+)?%\s*$')
# Comparative percentages: "30% of placebo ... 67%", "from X% to Y%"
_COMP_PERCENTAGE_NUM_RE = re.compile(
    r'\b\d+(?:\.\d+)?%\s+of\b'
    r'|\bfrom\s+\d+(?:\.\d+)?%\s+to\s+\d+(?:\.\d+)?%\b'
    r'|\bcompared\s+to\s+\d+(?:\.\d+)?%\b',
    re.I,
)


def _classify_numeric(text: str) -> str | None:
    """
    Return the specific numeric CI subtype when the CI text is predominantly a
    numeric/statistical value, else None (fall through to normal routing).

    Returns one of:
        CONFIDENCE_INTERVAL  — "95% CI: 27–48 days"
        P_VALUE              — "p<0.0001"
        HAZARD_RATIO         — "HR = 0.82"
        ODDS_RATIO           — "OR = 1.23"
        NUMERIC_PERCENTAGE   — "73%" or "30% of placebo ... 67%"
        NUMERIC_SAMPLE_SIZE  — "n = 8", "254 subjects"
        None                 — not a numeric CI

    Takes precedence over the category-map so that a CI labelled
    "Product Characteristics" but containing only "p<0.0001" is routed to
    the numeric retriever rather than vector+BM25.

    Each subtype maps to a different statistical_identity.type value, which
    the numeric retriever uses as a filter clause to eliminate false positives
    (e.g. \"dose level 8\" cannot match a \"sample_size\" query).
    """
    if _CONFIDENCE_INTERVAL_NUM_RE.search(text):
        return "CONFIDENCE_INTERVAL"
    if _P_VALUE_NUM_RE.search(text):
        return "P_VALUE"
    if _HR_NUM_RE.search(text):
        return "HAZARD_RATIO"
    if _OR_NUM_RE.search(text):
        return "ODDS_RATIO"
    if _PURE_PERCENTAGE_NUM_RE.match(text) or _COMP_PERCENTAGE_NUM_RE.search(text):
        return "NUMERIC_PERCENTAGE"
    if _SAMPLE_SIZE_NUM_RE.search(text):
        return "NUMERIC_SAMPLE_SIZE"
    return None


# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    search_id = event.get("search_id", "unknown")
    ci_id     = event.get("ci", {}).get("id", "unknown")
    logger.info("[Classifier] start search_id=%s ci_id=%s", search_id, ci_id)

    try:
        result = _process(event)
    except Exception as exc:
        logger.error("[Classifier] failed search_id=%s error=%s", search_id, exc)
        raise

    logger.info("[Classifier] done search_id=%s type=%s strategies=%s",
                search_id,
                result["classification"]["ci_type"],
                result["classification"]["strategies"])

    return result


def _process(req: dict) -> dict:
    ci_text  = req["ci"].get("knownCI", "")
    entities = req["ci"].get("ner", {}).get("entities", [])
    category = req["ci"].get("category") or {}
    cat_code = (category.get("code") or "").strip()

    # Numeric/statistical detection takes precedence over all other routing.
    # A CI that is purely a number is always better served by the numeric
    # retriever than by semantic search, regardless of its category label.
    _numeric_type = _classify_numeric(ci_text)
    if _numeric_type:
        ci_type    = _numeric_type
        strategies = _STRATEGIES[_numeric_type]
        reason     = f"Numeric/statistical pattern detected: {_numeric_type}"
    # Human-assigned category takes precedence over text inference
    elif cat_code and cat_code in _CATEGORY_MAP:
        mapping    = _CATEGORY_MAP[cat_code]
        ci_type    = mapping["ci_type"]
        strategies = mapping["strategies"]
        reason     = (f"Category '{category.get('name', cat_code)}' "
                      f"(id={category.get('id')}) → {ci_type}")
    else:
        ci_type, reason = _classify(ci_text, entities)
        strategies      = _STRATEGIES.get(ci_type, _STRATEGIES["PHRASE"])

    # Enrich the CI with facts, clinical_relations, statement_type, and object_subtype
    # if they are not already present.  This makes ci_facts and ci_relations available
    # to the aggregator, reranker, and contradiction scorer so that slot-aware scoring
    # and structural penalties can actually fire.
    ci = req["ci"]
    if ci_text and "facts" not in ci:
        try:
            from shared.clinical_fact_extractor import enrich_object
            enrichment = enrich_object(
                text             = ci_text,
                entities         = entities,
                section_category = ci.get("section_category", "") or "",
                heading_path     = ci.get("heading_path") or [],
            )
            ci = {**ci, **enrichment}
        except Exception as exc:
            logger.debug("[Classifier] CI enrichment skipped: %s", exc)

    return {
        **req,
        "ci": ci,
        "classification": {
            "ci_type":     ci_type,
            "strategies":  strategies,
            "reason":      reason,
            "category_id": category.get("id"),
            "category_code": cat_code or None,
        },
    }


def _classify(text: str, entities: list[dict]) -> tuple[str, str]:
    """Rule-based classifier. Returns (ci_type, reason)."""

    # Long multi-sentence / multi-line text is always a PHRASE — skip name checks
    is_paragraph = len(text) > 80 or "\n" in text or text.count(".") >= 2

    # 1. Identifier patterns take priority — highest precision strategy
    for pat in _IDENTIFIER_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return "IDENTIFIER", f"Matches identifier pattern: {pat}"

    # 2. Person name — only for short, single-line CI text
    if not is_paragraph:
        ner_labels = {e.get("label", "") for e in entities}
        ner_types  = {e.get("sub_type", "") for e in entities}
        if "PROTECTED_HEALTH_INFORMATION" in ner_labels and "NAME" in ner_types:
            return "PERSON", "Comprehend Medical detected a PHI NAME entity"
        if _PERSON_RE.search(text):
            return "PERSON", "Person name pattern detected (Dr. or First Last)"

    # 3. Clinical role (short or embedded in text)
    text_lower = text.lower()
    for term in _ROLE_TERMS:
        if term in text_lower:
            return "CLINICAL_ROLE", f"Clinical role term '{term}' found"

    # 4. Organization (from NER, short text only)
    if not is_paragraph:
        ner_labels = {e.get("label", "") for e in entities}
        if "ORGANIZATION" in ner_labels:
            return "ORGANIZATION", "NER detected ORGANIZATION entity"

    # 5. Default — general phrase (use full retrieval suite)
    return "PHRASE", "Long clinical phrase — full retrieval suite"
