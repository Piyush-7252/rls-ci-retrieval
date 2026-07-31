"""
Search Pipeline — Stage 6.5: Span Extractor
============================================
Finds the *exact* matching sentence within a retrieved chunk.

Strategy (in priority order)
-----------------------------
1. **Embedding match** (if sentence embeddings were stored at index time):
   Dot-product the CI's dense vector against each sentence vector.
   Returns the highest-scoring sentence with its page number and bbox.

2. **Text fallback** (if no sentence embeddings — old chunks or re-index pending):
   Best sentence by token overlap + SequenceMatcher, same as before.

Input:  req with "verified_candidates" (each must have "sentences" from context_expander)
        req["ci"]["embedding"]["dense_vector"]  — CI embedding (from CI enrichment)

Output: each candidate enriched with:
        - "match_span"       (str)            — exact sentence text from the document
        - "context_sentence" (str)            — surrounding sentence (same for now)
        - "match_page"       (int)            — exact page containing the match
        - "match_bbox"       (list[float])    — [x1, y1, x2, y2] for PDF highlighting
        - "highlight_score"    (float 0–1)      — word-overlap of CI vs match_span
        - "match_method"     (str)            — "embedding" | "text_fallback"
"""

from __future__ import annotations

import difflib
import logging
import math
import re
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_aws: dict = {}


def _get(service: str):
    if service not in _aws:
        import boto3
        _aws[service] = boto3.client(service)
    return _aws[service]


# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    search_id = event.get("search_id", "unknown")
    logger.info("[SpanExtractor] start search_id=%s", search_id)
    try:
        result = _process(event)
    except Exception as exc:
        logger.error("[SpanExtractor] failed search_id=%s error=%s", search_id, exc)
        raise
    return result


def _process(req: dict) -> dict:
    ci          = req.get("ci", {})
    ci_text     = ci.get("knownCI", "")
    ci_vec      = ci.get("embedding", {}).get("dense_vector", [])
    verified    = req.get("verified_candidates", [])

    enriched = []
    for cand in verified:
        if cand.get("verdict") in ("YES", "MAYBE"):
            objects    = cand.get("objects", [])   # semantic objects: paragraph/row/field
            ctx        = cand.get("context", {})
            chunk_text = ctx.get("current_text", "")

            if objects and ci_vec:
                # Best object by embedding dot-product (whole paragraph vs CI vector)
                best_obj = _best_by_embedding(ci_vec, objects)
                method   = "embedding"
            else:
                # Fallback: text overlap against raw chunk text
                best_obj = _best_by_text(ci_text, chunk_text)
                method   = "text_fallback"

            # The best display_span within that object (spaCy sentence or whole row)
            best_span   = _best_display_span(ci_text, best_obj)
            mq          = _highlight_score(ci_text, best_obj.get("text", ""))
            cand = {
                **cand,
                # The matched semantic object
                "match_object_id":  best_obj.get("object_id"),
                "match_object_type": best_obj.get("type"),
                # The display span (sentence / row / field) for the UI
                "match_span":       best_span.get("text", best_obj.get("text", "")),
                "match_span_start": best_span.get("start", 0),
                "match_span_end":   best_span.get("end", 0),
                "context_sentence": best_obj.get("text", ""),  # full object = reviewer context
                "match_page":       best_obj.get("page", cand.get("page_start")),
                "match_bbox":       best_obj.get("bbox", []),
                "highlight_score":    round(mq, 3),
                "match_method":     method,
            }
        enriched.append(cand)

    return {**req, "verified_candidates": enriched}


def _best_display_span(ci_text: str, obj: dict) -> dict:
    """
    Pick the best display_span within a matched semantic object.
    For paragraphs: the sentence with highest token overlap vs the CI.
    For atomic types (table_row, heading, list_item): the single span.
    Returns a display_span dict: {type, text, start, end}
    """
    spans = obj.get("display_spans", [])
    if not spans:
        text = obj.get("text", "")
        return {"type": obj.get("type", "paragraph"), "text": text, "start": 0, "end": len(text)}
    if len(spans) == 1:
        return spans[0]
    # Multiple spans (paragraph with multiple sentences) — pick best by token overlap
    return max(spans, key=lambda s: _token_overlap(ci_text, s.get("text", "")))


# ─────────────────────────────────────────────────────────────────────────────
# Embedding-based matching
# ─────────────────────────────────────────────────────────────────────────────

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _best_by_embedding(ci_vec: list[float], sentences: list[dict]) -> dict:
    """Return the sentence with highest cosine similarity to the CI vector."""
    best_score = -1.0
    best_sent  = sentences[0]

    for sent in sentences:
        vec = sent.get("embedding", [])
        if not vec:
            continue
        score = _dot(ci_vec, vec)   # vectors are L2-normalised by Titan
        if score > best_score:
            best_score = score
            best_sent  = sent

    return best_sent


# ─────────────────────────────────────────────────────────────────────────────
# Text-fallback matching (no embeddings available)
# ─────────────────────────────────────────────────────────────────────────────

_SENT_RE     = re.compile(r'(?<=[.!?])\s+|\n{2,}')
_MIN_CHARS   = 15


def _split_sentences(text: str) -> list[str]:
    parts = _SENT_RE.split(text)
    return [p.strip() for p in parts if len(p.strip()) >= _MIN_CHARS]


def _token_overlap(a: str, b: str) -> float:
    wa = set(re.findall(r'\b\w+\b', a.lower()))
    wb = set(re.findall(r'\b\w+\b', b.lower()))
    return len(wa & wb) / max(len(wa), 1)


def _best_by_text(ci_text: str, chunk_text: str) -> dict:
    """Return best matching sentence as a fake sentence dict (no page/bbox)."""
    if not chunk_text:
        return {"text": ci_text[:200], "page": None, "bbox": []}

    # Exact substring
    ci_norm    = " ".join(ci_text.split())
    chunk_norm = " ".join(chunk_text.split())
    idx = chunk_norm.lower().find(ci_norm.lower())
    if idx >= 0:
        return {"text": chunk_norm[idx: idx + len(ci_norm)], "page": None, "bbox": []}

    # Best sentence by token overlap
    sents = _split_sentences(chunk_norm) or [chunk_norm[:400]]
    best  = max(sents, key=lambda s: _token_overlap(ci_text, s))
    return {"text": best, "page": None, "bbox": []}


# ─────────────────────────────────────────────────────────────────────────────
# Match quality
# ─────────────────────────────────────────────────────────────────────────────

def _highlight_score(ci_text: str, match_span: str) -> float:
    """Fraction of CI words found in match_span (0 = noise, 1 = perfect)."""
    return _token_overlap(ci_text, match_span)

