"""
Ontology Lambda  (Unified — CI and Document)
=============================================
Routes on ``event["source_type"]``:
  "ci"       → full expansion: abbreviations + role synonyms + drug/disease
               synonyms + asset drug regexes + regex_patterns for CI matching
  "document" → standard expansion: abbreviations + role synonyms

Both paths share abbreviation and role-synonym logic.  The CI path adds
the richer regex_pattern generation required for CI-level matching.

Fan-out
-------
  CI path       : EMBEDDING_LAMBDA_ARN
  Document path : EMBEDDING_LAMBDA_ARN
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from shared.clinical_ontology import (
    ROLE_SYNONYMS,
    build_regex_patterns,
    expand_abbreviation,
    get_drug_synonyms,
    get_disease_synonyms,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EMBEDDING_LAMBDA_ARN = os.environ.get("EMBEDDING_LAMBDA_ARN", "")

# ─── lazy AWS clients ─────────────────────────────────────────────────────────
_aws: dict = {}

def _get(service: str):
    if service not in _aws:
        import boto3
        _aws[service] = boto3.client(service)
    return _aws[service]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    source_type = event.get("source_type", "document")

    if source_type == "ci":
        ci_id = event.get("id", "unknown")
        logger.info("[Ontology] start source=ci ci_id=%s", ci_id)
        try:
            result = _process_ci(event)
        except Exception as exc:
            logger.error("[Ontology] failed source=ci ci_id=%s error=%s", ci_id, exc)
            raise
        logger.info("[Ontology] done source=ci ci_id=%s patterns=%d",
                    ci_id, len(result["ontology"]["regex_patterns"]))
        if EMBEDDING_LAMBDA_ARN:
            _get("lambda").invoke(
                FunctionName   = EMBEDDING_LAMBDA_ARN,
                InvocationType = "Event",
                Payload        = json.dumps(result).encode(),
            )
        return result

    else:
        chunk_id = event.get("chunk_id", "unknown")
        logger.info("[Ontology] start source=document chunk_id=%s", chunk_id)
        try:
            result = _process_document(event)
        except Exception as exc:
            logger.error("[Ontology] failed source=document chunk_id=%s error=%s", chunk_id, exc)
            raise
        logger.info("[Ontology] done source=document chunk_id=%s expansions=%d",
                    chunk_id, len(result["ontology"]["expansions"]))
        if EMBEDDING_LAMBDA_ARN:
            _get("lambda").invoke(
                FunctionName   = EMBEDDING_LAMBDA_ARN,
                InvocationType = "Event",
                Payload        = json.dumps(result).encode(),
            )
        return result


# ─────────────────────────────────────────────────────────────────────────────
# CI path  (richer: regex_patterns + drug/disease synonyms + asset regexes)
# ─────────────────────────────────────────────────────────────────────────────

def _process_ci(ci: dict) -> dict:
    original_text       = ci.get("knownCI", "")
    abbreviations_found = ci["normalization"].get("abbreviations_found", {})
    entities            = ci["ner"].get("entities", [])
    ci_assets           = ci.get("assets", [])

    expansions     = _expand_abbreviations(abbreviations_found)
    synonyms       = _resolve_role_synonyms(entities)
    regex_patterns = build_regex_patterns(original_text)

    # Abbreviation-expanded forms
    for exp in expansions:
        for expanded_text in exp["expanded"]:
            regex_patterns.append(re.escape(expanded_text))

    # Role synonyms
    for syns in synonyms.values():
        for syn in syns:
            regex_patterns.append(re.escape(syn))

    # Drug / disease / abbreviation expansions from NER entities
    for ent in entities:
        text = (ent.get("text") or "").strip()
        if not text:
            continue
        for syn in get_drug_synonyms(text):
            regex_patterns.append(re.escape(syn))
        for syn in get_disease_synonyms(text):
            regex_patterns.append(re.escape(syn))
        if text.isupper() and len(text) <= 8:
            for exp in expand_abbreviation(text):
                regex_patterns.append(re.escape(exp))

    # Drug names from linked CI assets
    for pat in _generate_drug_regexes(ci_assets):
        regex_patterns.append(pat)

    # Deduplicate, preserving order
    seen:   set[str]  = set()
    unique: list[str] = []
    for p in regex_patterns:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return {
        **ci,
        "ontology": {
            "expansions":    expansions,
            "synonyms":      synonyms,
            "regex_patterns": unique,
        },
    }


def _generate_drug_regexes(ci_assets: list) -> list[str]:
    """Build regex patterns from drug names/codes in the CI's linked assets."""
    patterns: list[str] = []
    for asset in ci_assets:
        for field in ("name", "genericName"):
            val = (asset.get(field) or "").strip()
            if val and len(val) > 2:
                patterns.append(re.escape(val))
        code = (asset.get("code") or "").strip()
        if code and len(code) > 3 and "\u2017" not in code:
            patterns.append(re.escape(code))
    return patterns


# ─────────────────────────────────────────────────────────────────────────────
# Document path  (standard: abbreviations + role synonyms only)
# ─────────────────────────────────────────────────────────────────────────────

def _process_document(chunk: dict) -> dict:
    abbreviations_found = chunk["normalization"].get("abbreviations_found", {})
    entities            = chunk["ner"].get("entities", [])

    return {
        **chunk,
        "ontology": {
            "expansions": _expand_abbreviations(abbreviations_found),
            "synonyms":   _resolve_role_synonyms(entities),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Shared logic
# ─────────────────────────────────────────────────────────────────────────────

def _expand_abbreviations(abbreviations_found: dict) -> list[dict]:
    return [
        {"original": abbr, "expanded": forms, "type": "abbreviation"}
        for abbr, forms in abbreviations_found.items()
    ]


def _resolve_role_synonyms(entities: list[dict]) -> dict[str, list[str]]:
    synonyms: dict[str, list[str]] = {}
    for ent in entities:
        key = ent.get("text", "").lower()
        if key in ROLE_SYNONYMS:
            synonyms[ent["text"]] = ROLE_SYNONYMS[key]
    return synonyms
