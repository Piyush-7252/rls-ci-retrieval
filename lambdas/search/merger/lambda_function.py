"""
Search Pipeline — Stage 7 (terminal): Merger
==============================================
Groups adjacent verified chunks, deduplicates overlapping page ranges, and
produces the final human-readable hit list for the reviewer.

PRODUCTION GEOMETRY CONTRACT
============================
This merger enforces critical invariants for PDF geometry preservation:

1. REJECTION INVARIANT (CRITICAL)
   - NO candidates cannot contribute any fields (text, geometry, sources, etc.)
   - Formally: rejected_only_rects ∩ final_hit_rects = ∅
   - Validated by: highlight_spans contains ONLY accepted candidates

2. SHARED GEOMETRY INVARIANT
   - Shared rectangles between sentences are ALLOWED if both owners are accepted
   - NOT a bug: consequence of line-level Apryse geometry
   - Merger correctly preserves ALL accepted-candidate geometry
   - For example:
     * Sentence A owns rects [R1, R2]
     * Sentence B owns rects [R2, R3, R4]  (R2 shared with A)
     * If B is rejected but A is accepted: final hit has [R1, R2]
     * R2 is preserved because A owns it and A is accepted

3. ORDERING INVARIANT
   - Document order is deterministic and preserved throughout merging
   - chunk_ids and highlight_spans ordered by (page_start, position_in_doc, confidence)
   - No set() deduplication that loses ordering

4. GEOMETRY SOURCE METADATA
   - Every final geometry has match_geometry_source field
   - Values: "apryse_span" | "object_bbox" | "none"
   - Enables production debugging and traceability

Logic
-----
1. Keep only "YES" and "MAYBE" candidates (rejection invariant).
2. Sort by page_start, then by document position.
3. Merge chunks whose page ranges overlap or are adjacent (gap ≤ 1 page).
4. Build final_hits preserving:
   - all accepted-candidate geometries in highlight_spans
   - deterministic document order
   - geometry_source metadata

Input:  verified search request  (must have "verified_candidates")
Appends: "final_hits": list[FinalHit]

FinalHit schema
---------------
{
    "ci_id":            int | str,
    "ci_text":          str,
    "page_start":       int,
    "page_end":         int,
    "text":             str,          # merged context text
    "sources":          list[str],
    "verdict":          str,          # "YES" | "MAYBE"
    "confidence":       float,        # max confidence across merged chunks
    "chunk_ids":        list[str],    # all chunk IDs (ORDERED by document position)
    "match_geometry_source": str,     # "apryse_span" | "object_bbox" | "none"
    "match_rects":      list[list[float]],  # primary/best highlight geometry
    "highlight_spans":  list[{
        "chunk_id": str,
        "match_rects": list[list[float]],  # per-line geometry from Apryse
        "match_geometry_source": str,      # "apryse_span" | "object_bbox" | "none"
        "confidence": float,
        ...
    }],                 # ALL accepted candidate geometries (preserves shared rects)
}

KEY PRODUCTION RULES
====================
- NEVER modify geometry to remove "duplicate" or "shared" rectangles
- NEVER use set() for chunk_ids or highlight_spans (breaks ordering)
- ALWAYS preserve match_geometry_source in all outputs
- ALWAYS validate that rejected candidates have zero contribution to final_hit
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PAGE_GAP_THRESHOLD = int(os.environ.get("MERGE_PAGE_GAP", "1"))

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
    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 1: Preserve document order
    # ─────────────────────────────────────────────────────────────────────────────
    # Sort accepted candidates by page_start, then by position in document.
    # Preserve order to avoid reordering A, B, C into C, A, B.
    
    page_start  = min(c.get("page_start", 0) for c in group)
    page_end    = max(c.get("page_end",   0) for c in group)
    
    # Preserve document order: sort by page, then by document position or score
    sorted_group = sorted(
        group,
        key=lambda c: (
            c.get("page_start", 0),
            c.get("position_in_doc", 0),  # Document position (if available)
            -c.get("confidence", 0.0),     # Fallback: higher score first
        )
    )
    
    # Extract chunk_ids in order (BEFORE set conversion)
    chunk_ids = [c["chunk_id"] for c in sorted_group]
    
    # Sources and confidence from all (deduped but ordered)
    sources_set = {s for c in group for s in c.get("sources", [])}
    sources = sorted(sources_set)  # Sort for consistency
    confidence = max(c.get("confidence", 0.0) for c in group)
    verdict = "YES" if any(c.get("verdict") == "YES" for c in group) else "MAYBE"

    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 2: Pick best candidate for primary highlight
    # ─────────────────────────────────────────────────────────────────────────────
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
    match_rects      = best.get("match_rects", [])
    match_geometry_source = best.get("match_geometry_source", "none")
    match_method     = best.get("match_method", "text_fallback")

    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 3: Build highlight_spans to preserve ALL accepted candidate geometries
    # ─────────────────────────────────────────────────────────────────────────────
    # This allows UI to show ALL relevant geometry regions, not just the best.
    # CRITICALLY: rejected candidates (not in group) have NO geometry here.
    # Shared rects between accepted candidates are preserved (expected with
    # line-level geometry; sentence boundaries may fall within a PDF line).
    
    highlight_spans = []
    for c in sorted_group:
        candidate_rects = c.get("match_rects", [])
        candidate_source = c.get("match_geometry_source", "none")
        
        # Only add if it has geometry
        if candidate_rects:
            highlight_spans.append({
                "chunk_id": c["chunk_id"],
                "match_span": c.get("match_span", ""),
                "match_page": c.get("match_page") or c.get("page_start", 0),
                "match_rects": candidate_rects,
                "match_geometry_source": candidate_source,
                "confidence": c.get("confidence", 0.0),
                "highlight_score": c.get("highlight_score", 0.0),
            })

    # Build merged text from current_text of each chunk in page order
    texts: list[str] = []
    for c in sorted_group:
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
        "match_bbox":       match_bbox,      # [x1,y1,x2,y2] for PDF highlight (legacy)
        "match_rects":      match_rects,     # list of [x1,y1,x2,y2] rects (new: per-line geometry)
        "match_geometry_source": match_geometry_source,  # "apryse_span" | "object_bbox" | "none"
        "highlight_spans":  highlight_spans,  # All accepted candidate geometries
        "match_method":     match_method,
        "sources":          sources,
        "verdict":          verdict,
        "confidence":       round(confidence, 3),
        "chunk_ids":        chunk_ids,  # Preserved document order
        # Retrieval provenance — which semantic object was matched and how
        "matched_object":      best.get("matched_object"),
        "retrieval_origin":    best.get("retrieval_origin", "direct_unknown"),
        "selection_reason":    best.get("selection_reason"),
        "literal_match_count": best.get("literal_match_count"),
        "context_strategy":    best.get("context_strategy"),
        "matched_distance":    best.get("matched_distance"),
        "distance_ratio":      best.get("distance_ratio"),
        "current_text_chars":  best.get("current_text_chars"),
        "agg_score":           round(best.get("agg_score", 0.0), 4),
        # Scoring breakdowns from reranker (score_breakdown) and aggregator (agg_score_breakdown)
        "score_breakdown":      best.get("score_breakdown"),
        "agg_score_breakdown":  best.get("agg_score_breakdown"),
        "span_debug":             best.get("span_debug"),
        "supporting_sentences":   best.get("supporting_sentences", []),
        "highlight_type":         best.get("highlight_type", "sentence"),
        "primary_support_index":  best.get("primary_support_index", 0),
    }
