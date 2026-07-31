"""
Search Pipeline — Stage 7 (terminal): Merger
==============================================
Groups adjacent verified chunks, deduplicates overlapping page ranges, and
produces the final human-readable hit list for the reviewer.

Logic
-----
1. Keep only "YES" and "MAYBE" candidates.
2. Sort by page_start.
3. Merge chunks whose page ranges overlap or are adjacent (gap ≤ 1 page).
4. Build final_hits with merged page range, combined text, sources, confidence.

Input:  verified search request  (must have "verified_candidates")
Appends: "final_hits": list[FinalHit]

FinalHit schema
---------------
{
    "ci_id":        int | str,
    "ci_text":      str,
    "page_start":   int,
    "page_end":     int,
    "text":         str,          # merged context text
    "sources":      list[str],
    "verdict":      str,          # "YES" | "MAYBE"
    "confidence":   float,        # max confidence across merged chunks
    "chunk_ids":    list[str],    # all chunk IDs contributing to this hit
}
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PAGE_GAP_THRESHOLD = int(os.environ.get("MERGE_PAGE_GAP", "1"))

_aws: dict = {}

def _get(service: str):
    if service not in _aws:
        import boto3
        _aws[service] = boto3.client(service)
    return _aws[service]


# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    search_id = event.get("search_id", "unknown")
    logger.info("[Merger] start search_id=%s", search_id)
    try:
        result = _process(event)
    except Exception as exc:
        logger.error("[Merger] failed search_id=%s error=%s", search_id, exc)
        raise
    logger.info("[Merger] done search_id=%s final_hits=%d",
                search_id, len(result["final_hits"]))
    return result


def _process(req: dict) -> dict:
    ci       = req.get("ci", {})
    ci_id    = ci.get("id", ci.get("ci_id", ""))
    ci_text  = ci.get("knownCI", "")
    verified = req.get("verified_candidates", [])

    # Filter: keep YES and MAYBE
    accepted = [c for c in verified if c.get("verdict") in ("YES", "MAYBE")]

    final_hits = _merge_groups(ci_id, ci_text, accepted)

    return {
        **req,
        "final_hits": final_hits,
    }


def _merge_groups(ci_id, ci_text: str, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []

    # Sort by page_start
    candidates = sorted(candidates, key=lambda c: c.get("page_start", 0))

    groups: list[list[dict]] = []
    current_group: list[dict] = [candidates[0]]

    for cand in candidates[1:]:
        prev_end    = current_group[-1].get("page_end",   0)
        this_start  = cand.get("page_start", 0)
        if this_start <= prev_end + PAGE_GAP_THRESHOLD:
            current_group.append(cand)
        else:
            groups.append(current_group)
            current_group = [cand]
    groups.append(current_group)

    hits = []
    for group in groups:
        merged = _merge_group(ci_id, ci_text, group)
        hits.append(merged)

    return hits


def _merge_group(ci_id, ci_text: str, group: list[dict]) -> dict:
    page_start  = min(c.get("page_start", 0) for c in group)
    page_end    = max(c.get("page_end",   0) for c in group)
    chunk_ids   = list({c["chunk_id"] for c in group})
    sources     = list({s for c in group for s in c.get("sources", [])})
    confidence  = max(c.get("confidence", 0.0) for c in group)
    verdict     = "YES" if any(c.get("verdict") == "YES" for c in group) else "MAYBE"

    # Pick the span from the highest highlight_score candidate that has one
    span_cands = [c for c in group if c.get("match_span")]
    if span_cands:
        best = max(span_cands, key=lambda c: c.get("highlight_score", 0.0))
    else:
        best = max(group, key=lambda c: c.get("confidence", 0.0))
    match_span       = best.get("match_span", "")
    context_sentence = best.get("context_sentence", "")
    highlight_score    = best.get("highlight_score", 0.0)
    match_page       = best.get("match_page") or page_start
    match_bbox       = best.get("match_bbox", [])
    match_method     = best.get("match_method", "text_fallback")

    # Build merged text from current_text of each chunk in page order
    texts: list[str] = []
    for c in group:
        txt = c.get("context", {}).get("current_text", c.get("snippet", ""))
        if txt and txt not in texts:
            texts.append(txt)
    merged_text = "\n\n---\n\n".join(texts)

    return {
        "ci_id":            ci_id,
        "ci_text":          ci_text,
        "page_start":       page_start,
        "page_end":         page_end,
        "text":             merged_text,
        "match_span":       match_span,
        "context_sentence": context_sentence,
        "highlight_score":    round(highlight_score, 3),
        "match_page":       match_page,
        "match_bbox":       match_bbox,      # [x1,y1,x2,y2] for PDF highlight
        "match_method":     match_method,
        "sources":          sorted(sources),
        "verdict":          verdict,
        "confidence":       round(confidence, 3),
        "chunk_ids":        chunk_ids,
        # Retrieval provenance — which semantic object was matched and how
        "matched_object":   best.get("matched_object"),
        "agg_score":        round(best.get("agg_score", 0.0), 4),
        # Scoring breakdowns from reranker (score_breakdown) and aggregator (agg_score_breakdown)
        "score_breakdown":      best.get("score_breakdown"),
        "agg_score_breakdown":  best.get("agg_score_breakdown"),
        "span_debug":           best.get("span_debug"),
    }
