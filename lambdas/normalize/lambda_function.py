"""
Normalize Lambda  (Unified — CI and Document)
==============================================
Routes on ``event["source_type"]``:
  "ci"       → text from  ci["knownCI"]
  "document" → text from  chunk["extraction"]["raw_text"]

Both paths apply identical unicode normalisation, tokenisation, and
abbreviation lookup.

Fan-out
-------
  CI path       : NER_LAMBDA_ARN  (single downstream)
  Document path : NER_LAMBDA_ARN  (sequential: NER → Ontology → Embedding → Index)
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from typing import Any

from shared.clinical_ontology import ABBREVIATIONS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

NER_LAMBDA_ARN       = os.environ.get("NER_LAMBDA_ARN", "")

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
        logger.info("[Normalize] start source=ci ci_id=%s", ci_id)
        try:
            result = _process_ci(event)
        except Exception as exc:
            logger.error("[Normalize] failed source=ci ci_id=%s error=%s", ci_id, exc)
            raise
        logger.info("[Normalize] done source=ci ci_id=%s tokens=%d abbreviations=%d",
                    ci_id,
                    len(result["normalization"]["tokens"]),
                    len(result["normalization"]["abbreviations_found"]))
        if NER_LAMBDA_ARN:
            _get("lambda").invoke(
                FunctionName   = NER_LAMBDA_ARN,
                InvocationType = "Event",
                Payload        = json.dumps(result).encode(),
            )
        return result

    else:
        chunk_id = event.get("chunk_id", "unknown")
        logger.info("[Normalize] start source=document chunk_id=%s", chunk_id)
        try:
            result = _process_document(event)
        except Exception as exc:
            logger.error("[Normalize] failed source=document chunk_id=%s error=%s", chunk_id, exc)
            raise
        logger.info("[Normalize] done source=document chunk_id=%s tokens=%d abbreviations=%d",
                    chunk_id,
                    len(result["normalization"]["tokens"]),
                    len(result["normalization"]["abbreviations_found"]))
        # Document fans out to NER only (sequential pipeline: NER → Ontology → Embedding → Index)
        if NER_LAMBDA_ARN:
            _get("lambda").invoke(
                FunctionName   = NER_LAMBDA_ARN,
                InvocationType = "Event",
                Payload        = json.dumps(result).encode(),
            )
        return result


# ─────────────────────────────────────────────────────────────────────────────
# CI path
# ─────────────────────────────────────────────────────────────────────────────

def _process_ci(ci: dict) -> dict:
    text = ci.get("knownCI", "")
    return {
        **ci,
        "normalization": _build_normalization(text),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Document path
# ─────────────────────────────────────────────────────────────────────────────

def _process_document(chunk: dict) -> dict:
    text = chunk["extraction"]["raw_text"]
    return {
        **chunk,
        "normalization": _build_normalization(text),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Shared logic (identical for CI and document)
# ─────────────────────────────────────────────────────────────────────────────

def _build_normalization(text: str) -> dict:
    normalized = _normalize_text(text)
    return {
        "normalized_text":     normalized,
        "tokens":              _tokenize(normalized),
        "abbreviations_found": _find_abbreviations(text),
    }


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    return text.strip()


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text) if t]


def _find_abbreviations(text: str) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for abbr, expansions in ABBREVIATIONS.items():
        if re.search(r"\b" + re.escape(abbr) + r"\b", text):
            found[abbr] = expansions
    return found
