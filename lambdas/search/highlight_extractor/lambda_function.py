"""
Search Pipeline — Stage 6.5: Highlight Extractor (Simplified)
==============================================================
Resolves highlight geometry from LLM-verified evidence.

No semantic re-ranking. No display span selection.
Just: consume verified evidence, resolve highlight geometry.

New Architecture
----------------
The LLM verifier produces verified_candidates with:

  - evidence: the authoritative verified text/span
  - literal_matches: optional precise sub-match (e.g. "N = 8")

HighlightExtractor now has a simple contract:

  1. If literal_matches exists
       → Find the first literal match within evidence.text
       → Use flexible matching (handles "N=8" vs "N = 8")
       → Calculate precise character offsets
       → Return with line-level geometry
  
  2. Otherwise
       → Use evidence span directly as-is
       → Return its existing page/char_start/char_end/rects

Geometry Model
--------------
  - evidence.text = original extracted text (immutable coordinates)
  - evidence.char_start/char_end = offsets in document
  - evidence.rects = line-level geometry (not character-level)
  
For literal matches:
  - char_start/char_end point to "N = 8" within the line
  - rects are still line-level (WebViewer will cover the whole line)
  - geometry_source = "literal_match_line_geometry" (honest about precision)

No semantic scoring. No scorer registry. No re-selection of spans.
All decisions are made by the LLM verifier upstream.
"""


from __future__ import annotations

import difflib
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# SentenceSpan deserialization (for geometry preservation)
# ─────────────────────────────────────────────────────────────────────────────

def _deserialize_sentence_span(span_dict: dict) -> dict | None:
    """
    Restore SentenceSpan from serialized dict if _sentence_span field exists.
    
    Returns a dict with geometry info if present, otherwise None.
    """
    if not isinstance(span_dict, dict):
        return None
    
    sentence_span = span_dict.get("_sentence_span")
    if not isinstance(sentence_span, dict):
        return None
    
    return {
        "text": sentence_span.get("text"),
        "page": sentence_span.get("page"),
        "char_start": sentence_span.get("char_start"),
        "char_end": sentence_span.get("char_end"),
        "rects": sentence_span.get("rects", []),
        "source_object_id": sentence_span.get("source_object_id"),
        "source_span_ids": sentence_span.get("source_span_ids", []),
        "span_type": sentence_span.get("span_type"),
        "geometry_source": sentence_span.get("geometry_source", "none"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scorer registry
# ─────────────────────────────────────────────────────────────────────────────

class BaseScorer:
    """
    Subclass to add a new matching strategy.

    ci_meta keys:   synonyms (list[str]), ent_types (set[str])
    span_meta keys: entities (list[dict]), literal_matches (list[dict])
    """
    name: str = ""

    def score(self, ci_text: str, ci_meta: dict,
              span_text: str, span_meta: dict) -> dict | None:
        """Return {"score": float, "reason": str} or None if not applicable."""
        raise NotImplementedError


class LiteralContainmentScorer(BaseScorer):
    """
    Fires when either text literally contains the other (bidirectional).

    Covers three distinct cases that all deserve an exact-match score:
      - CI ⊂ span  — short CI phrase appears inside the display sentence
      - span ⊂ CI  — display sentence is a verbatim excerpt of a long CI paragraph
      - operator regex  — "N=8" ↔ "N = 8" flexible whitespace/operator matching

    Whitespace is normalised before comparison so PDF line-breaks (\n) do not
    prevent a match when the extracted span uses spaces instead.
    """
    name = "exact"  # keep "exact" so is_authoritative logic in _process() is unchanged

    def score(self, ci_text, ci_meta, span_text, span_meta):
        ci_s = ci_text.strip()
        if not ci_s:
            return None

        # Normalise whitespace (newlines → spaces) so PDF extraction artefacts
        # don't block what is otherwise a verbatim match.
        _ws  = re.compile(r'\s+')
        ci_n = _ws.sub(' ', ci_s)
        sp_n = _ws.sub(' ', span_text)

        # Determine which text is longer; containment is checked as
        # shorter ∈ longer, which covers both CI⊂span and span⊂CI.
        if len(ci_n) >= len(sp_n):
            longer, shorter = ci_n, sp_n
        else:
            longer, shorter = sp_n, ci_n

        # Coverage bonus — Michaelis-Menten shape, 0 → +0.05.
        # Measures what fraction of the longer text the shorter covers;
        # symmetrically rewards both directions.
        #   coverage = 1 %  → +0.001
        #   coverage = 10 % → +0.025
        #   coverage = 50 % → +0.042
        #   coverage = 100% → +0.050
        coverage  = len(shorter) / max(len(longer), 1)
        cov_bonus = coverage / (coverage + 0.05) * 0.05

        # Strategy 1: bidirectional literal containment (whitespace-normalised).
        idx = longer.lower().find(shorter.lower())
        if idx >= 0:
            # Boundary bonus: +0.025 per side when the match aligns with a
            # word boundary (not preceded / followed by an alphanumeric char).
            pre_ok  = idx == 0 or not longer[idx - 1].isalnum()
            post_ok = (idx + len(shorter) >= len(longer)
                       or not longer[idx + len(shorter)].isalnum())
            boundary_bonus = 0.025 * int(pre_ok) + 0.025 * int(post_ok)
            return {"score": min(1.00, 0.90 + cov_bonus + boundary_bonus),
                    "reason": "exact"}

        # Strategy 2: flexible whitespace/operator regex.
        #   "N=8" → r'N\s*=\s*8' → matches "N = 8", "(N=8)", "N= 8"
        pat = _build_flexible_pattern(ci_n)
        if pat:
            m = pat.search(sp_n)
            if m:
                pre_ok  = m.start() == 0 or not sp_n[m.start() - 1].isalnum()
                post_ok = m.end() >= len(sp_n) or not sp_n[m.end()].isalnum()
                boundary_bonus = 0.025 * int(pre_ok) + 0.025 * int(post_ok)
                return {"score": min(1.00, 0.90 + cov_bonus + boundary_bonus),
                        "reason": "exact"}
        return None


class ExtractionEquivalentScorer(BaseScorer):
    """
    Fires when either text literally contains the other after all whitespace is
    removed.  Score: 0.89 (flat).

    Handles PDF text-extraction artefacts where word boundaries are lost
    regardless of cause (OCR, ligatures, font encoding, layout engine, etc.):
        "of the"        →  "ofthe"
        "dose response" →  "doseresponse"
        "anti CD38"     →  "antiCD38"

    Score is deliberately below LiteralContainmentScorer (0.90+): the texts
    differ by at least one tokenization boundary so they are not verbatim
    equals.  Score is at the top of the FuzzyScorer range: the underlying
    wording IS identical; only extraction damaged word boundaries.

    Both directions are checked (CI⊂span and span⊂CI) for the same reason as
    LiteralContainmentScorer.
    """
    name = "extraction_equivalent"

    def score(self, ci_text, ci_meta, span_text, span_meta):
        ci_s = ci_text.strip()
        if not ci_s:
            return None

        # Remove all whitespace but preserve punctuation and clinical symbols.
        # This collapses PDF word-boundary artefacts without destroying structure.
        _despace = re.compile(r'\s+')
        ci_ds = _despace.sub('', ci_s).lower()
        sp_ds = _despace.sub('', span_text).lower()

        if not ci_ds or not sp_ds:
            return None

        if len(ci_ds) >= len(sp_ds):
            longer, shorter = ci_ds, sp_ds
        else:
            longer, shorter = sp_ds, ci_ds

        if shorter in longer:
            return {"score": 0.89, "reason": "extraction_equivalent"}
        return None


class FuzzyScorer(BaseScorer):
    name = "fuzzy"

    def score(self, ci_text, ci_meta, span_text, span_meta):
        ci_n  = _normalise(ci_text)
        sp_n  = _normalise(span_text)
        ratio = difflib.SequenceMatcher(None, ci_n, sp_n, autojunk=False).ratio()
        if ratio < 0.65:
            return None
        # Map [0.65, 1.0] → [0.65, 0.89]
        score = 0.65 + (ratio - 0.65) / 0.35 * 0.24
        return {"score": score, "reason": "fuzzy"}


class OntologyScorer(BaseScorer):
    name = "ontology"

    def score(self, ci_text, ci_meta, span_text, span_meta):
        span_lower = span_text.lower()
        for syn in ci_meta.get("synonyms", []):
            if syn and syn.lower() in span_lower:
                score = 0.60 + min(0.14, len(syn) / 100.0)
                return {"score": score, "reason": "ontology"}
        return None


class NERScorer(BaseScorer):
    name = "ner"

    def score(self, ci_text, ci_meta, span_text, span_meta):
        ci_types   = ci_meta.get("ent_types", set())
        span_types = {e.get("type", "") for e in span_meta.get("entities", [])
                      if isinstance(e, dict)}
        if not ci_types or not span_types:
            return None
        overlap = ci_types & span_types
        if not overlap:
            return None
        ratio = len(overlap) / max(len(ci_types), 1)
        return {"score": 0.40 + ratio * 0.19, "reason": "ner"}


class TokenScorer(BaseScorer):
    name = "token"

    def score(self, ci_text, ci_meta, span_text, span_meta):
        return {"score": _token_overlap(ci_text, span_text) * 0.39, "reason": "token"}


# Registry — append instances here to extend the pipeline.
# Literal-retriever evidence is consumed directly in _process(); it is NOT
# a scorer.  Add new scorers here for non-literal evidence sources.
SCORERS: list[BaseScorer] = [
    LiteralContainmentScorer(),
    ExtractionEquivalentScorer(),
    FuzzyScorer(),
    OntologyScorer(),
    NERScorer(),
    TokenScorer(),
]


# ─────────────────────────────────────────────────────────────────────────────
# Highlight Geometry Resolution
# ─────────────────────────────────────────────────────────────────────────────

def _find_literal_match_position(literal_text: str, evidence_text: str) -> tuple[int, int] | None:
    """
    Find exact position of literal_text within evidence_text.
    Reuses LiteralContainmentScorer strategies:
      1. Case-insensitive exact substring
      2. Flexible whitespace/operator regex (handles "N=8" vs "N = 8")
    
    Returns (start_offset, end_offset) or None.
    """
    if not literal_text or not evidence_text:
        return None
    
    lit_s = literal_text.strip()
    if len(lit_s) > 100:
        return None
    
    # Strategy 1: case-insensitive exact substring
    idx = evidence_text.lower().find(lit_s.lower())
    if idx >= 0:
        return (idx, idx + len(lit_s))
    
    # Strategy 2: flexible whitespace/operator regex (from LiteralContainmentScorer)
    # "N=8" → r'N\s*=\s*8' → matches "N = 8", "N=8", "N  =  8", etc.
    pat = _build_flexible_pattern(lit_s)
    if pat:
        m = pat.search(evidence_text)
        if m:
            return (m.start(), m.end())
    
    return None


def _resolve_highlight(final_hit: dict) -> dict:
    """
    Resolve highlight geometry from final_hit.
    
    Returns both object-relative and page-relative coordinates:
      - char_start/char_end: Offsets within the evidence object  
      - page_char_start/page_char_end: Offsets within the page text
    
    PAGE-LEVEL COORDINATES ARE THE PRIMARY UI INTERFACE:
      page_text[page][page_char_start:page_char_end] == text
    
    PATH 1: Verified literal/numeric evidence (has literal_matches field)
            → Try each literal match until one is found in evidence text
            → Extract the original extracted text from evidence
            → Calculate char offsets within evidence
            → Calculate page offsets by adding evidence.page_char_start
            → Return with line-level geometry (honest about precision)
    
    PATH 2: Everything else (no literal_matches)
            → Use evidence span directly as-is
            → No recalculation or semantic scoring
    
    If evidence field is missing, build it from matched_object or indexed_object.
    """
    evidence = final_hit.get("evidence")
    
    # Build evidence from matched_object if not provided
    if not evidence:
        matched_obj = final_hit.get("matched_object")
        indexed_obj = final_hit.get("indexed_object")
        
        # Try matched_object first, fallback to indexed_object
        obj = matched_obj or indexed_obj
        if obj and obj.get("text"):
            evidence = {
                "text": obj.get("text", ""),
                "page": obj.get("page", 0),
                "char_start": 0,  # No precise offsets available from object
                "char_end": len(obj.get("text", "")),
                "page_char_start": 0,  # Default to 0 if not available
                "page_char_end": len(obj.get("text", "")),
                "rects": [],
                "bbox": obj.get("bbox", []),
            }
        else:
            # No evidence available
            return {
                "text": "",
                "char_start": 0,
                "char_end": 0,
                "page_char_start": 0,
                "page_char_end": 0,
                "page": 0,
                "rects": [],
                "geometry_source": "none",
            }
    
    evidence_text = evidence.get("text", "")
    evidence_char_start = evidence.get("char_start", 0)
    evidence_page_char_start = evidence.get("page_char_start", 0)
    
    # ─────────────────────────────────────────────────────────────────────────
    # PATH 1: Verified literal/numeric evidence
    # Try each literal match; use the first one found in evidence text
    # ─────────────────────────────────────────────────────────────────────────
    literal_matches = final_hit.get("literal_matches") or []
    for literal_match in literal_matches:
        match_text = literal_match.get("text", "").strip()
        if not match_text:
            continue
        
        match_pos = _find_literal_match_position(match_text, evidence_text)
        if not match_pos:
            continue
        
        local_start, local_end = match_pos
        
        # Return the original extracted text, not the literal match text
        # This preserves "N = 8" if the evidence contains "N = 8"
        # while still calculating correct offsets
        return {
            "text": evidence_text[local_start:local_end],
            "char_start": evidence_char_start + local_start,
            "char_end": evidence_char_start + local_end,
            "page_char_start": evidence_page_char_start + local_start,  # ← PAGE-RELATIVE FOR UI
            "page_char_end": evidence_page_char_start + local_end,      # ← PAGE-RELATIVE FOR UI
            "page": evidence.get("page", 0),
            "rects": evidence.get("rects", []),
            "geometry_source": "literal_match_line_geometry",
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # PATH 2: Authoritative evidence span (no literal_matches found)
    # ─────────────────────────────────────────────────────────────────────────
    return {
        "text": evidence_text,
        "char_start": evidence_char_start,
        "char_end": evidence.get(
            "char_end",
            evidence_char_start + len(evidence_text),
        ),
        "page_char_start": evidence_page_char_start,  # ← PAGE-RELATIVE FOR UI
        "page_char_end": evidence.get(
            "page_char_end",
            evidence_page_char_start + len(evidence_text),
        ),  # ← PAGE-RELATIVE FOR UI
        "page": evidence.get("page", 0),
        "rects": evidence.get("rects", []),
        "geometry_source": evidence.get(
            "geometry_source",
            "evidence_span",
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Handler
# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    search_id = event.get("search_id", "unknown")
    logger.info("[HighlightExtractor] start search_id=%s", search_id)
    try:
        result = _process(event)
    except Exception as exc:
        logger.error("[HighlightExtractor] failed search_id=%s error=%s", search_id, exc)
        raise
    return result


def _process(req: dict) -> dict:
    """
    Process verified candidates using evidence-based geometry resolution.
    
    For each verified candidate with verdict YES/MAYBE:
      1. Resolve highlight geometry via _resolve_highlight()
      2. Map result fields to legacy field names for downstream compatibility
      3. Build highlight_spans and metadata
    
    No semantic re-scoring. No display span selection.
    All decisions already made by LLM verifier upstream.
    """
    verified = req.get("verified_candidates", [])
    mtokens  = _extract_matched_tokens(req.get("ci", {}).get("knownCI", ""))
    enriched = []

    for cand in verified:
        verdict = cand.get("verdict")
        
        # Only process YES/MAYBE verdicts
        if verdict not in ("YES", "MAYBE"):
            enriched.append(cand)
            continue
        
        # Use evidence-based geometry resolution (no semantic scoring)
        # If evidence is missing, _resolve_highlight() builds it from indexed_object/matched_object
        geom = _resolve_highlight(cand)
        
        # Extract highlight text and spans
        highlight_text = geom.get("text", "")
        char_start = geom.get("char_start", 0)
        char_end = geom.get("char_end", 0)
        page_char_start = geom.get("page_char_start", 0)  # ← PAGE-RELATIVE FOR UI
        page_char_end = geom.get("page_char_end", 0)      # ← PAGE-RELATIVE FOR UI
        geom_source = geom.get("geometry_source", "evidence_span")
        
        # Authority is determined by whether evidence was verified by LLM or built from indexed_object
        # If highlight was resolved (not "none"), it's authoritative enough to include
        evidence = cand.get("evidence")
        is_authoritative = bool(evidence.get("text")) if evidence else bool(highlight_text)
        
        # Highlight score reflects precision of geometry resolution:
        #   1.0 = literal match with precise character-level offset (though rects are line-level)
        #   0.9 = evidence span with original geometry preserved
        #   0.0 = no highlight geometry resolved
        highlight_score = (
            1.0 if geom_source == "literal_match_line_geometry"
            else 0.9 if geom_source == "evidence_span"
            else 0.0
        )
        
        # Build highlight_spans list (for UI rendering)
        if highlight_text:
            highlight_spans = [
                {
                    "text": highlight_text,
                    "start": char_start,
                    "end": char_end,
                    "page_char_start": page_char_start,  # ← PAGE-RELATIVE FOR UI
                    "page_char_end": page_char_end,      # ← PAGE-RELATIVE FOR UI
                    "source": geom_source,
                    "retriever": cand.get("retriever", ""),
                }
            ]
        else:
            highlight_spans = []
        
        # Enrich candidate with resolved geometry
        # Extract bbox from indexed_object if available
        indexed_obj = cand.get("indexed_object", {})
        match_bbox = indexed_obj.get("bbox", []) if indexed_obj else []
        
        cand = {
            **cand,
            "match_span":        highlight_text,
            "match_span_start":  char_start,
            "match_span_end":    char_end,
            "match_page_char_start": page_char_start,  # ← PAGE-RELATIVE FOR UI HIGHLIGHTING
            "match_page_char_end": page_char_end,      # ← PAGE-RELATIVE FOR UI HIGHLIGHTING
            "highlight_spans":   highlight_spans,
            "context_sentence":  highlight_text,  # For backward compat; same as match_span for evidence-based
            "match_page":        geom.get("page", 0),
            "match_bbox":        match_bbox,  # Line-level bounding box from indexed_object
            "match_rects":       geom.get("rects", []),  # per-line geometry (line-level precision)
            "match_geometry_source": geom_source,  # "literal_match_line_geometry" | "evidence_span" | "none"
            "highlight_score":   highlight_score,
            "match_method":      "evidence_based",
            "match_reason":      geom_source,
            "is_authoritative":  is_authoritative,
            "matched_tokens":    mtokens,
            "span_debug": {
                "geometry_source": geom_source,
                "highlight_text": highlight_text,
                "char_start": char_start,
                "char_end": char_end,
                "page_char_start": page_char_start,
                "page_char_end": page_char_end,
                "is_authoritative": is_authoritative,
            },
        }
        enriched.append(cand)

    logger.info("[HighlightExtractor] done search_id=%s highlights=%d",
                req.get("search_id"), sum(1 for c in enriched if c.get("match_span")))
    return {**req, "verified_candidates": enriched}


# ─────────────────────────────────────────────────────────────────────────────
# Core span picker
# ─────────────────────────────────────────────────────────────────────────────

def _pick_best_span(ci_text: str, ci_meta: dict, obj: dict) -> dict:
    """
    Run every scorer against every display_span and return the span
    with the highest score across all scorers.

    obj  — the retrieval unit (semantic object from OpenSearch)
           obj["text"]          → embedded content (never scored here)
           obj["display_spans"] → UI-facing spans (scored here)

    Also scores prev_sentence_text / next_sentence_text so the returned
    span is the sentence most literally relevant to the CI, even when the
    indexed object itself is a neighbouring sentence (e.g. 11/35 vs N=35).
    
    Geometry preservation:
    - If a display_span has _sentence_span with rects, extract per-line geometry
    - Use rects instead of bbox for accurate multi-line highlighting
    """
    spans = obj.get("display_spans", [])
    if not spans:
        text = obj.get("text", "")
        spans = [{"type": obj.get("type", "paragraph"), "text": text,
                  "start": 0, "end": len(text), "bbox": obj.get("bbox", [])}]

    # Also consider adjacent sentences stored on the object — the indexed
    # sentence may be a contextual neighbour of the sentence that literally
    # contains the CI value (e.g. "11/35" sentence vs the "N=35" sentence).
    for neighbour_key in ("prev_sentence_text", "next_sentence_text"):
        nbr = obj.get(neighbour_key)
        if nbr and isinstance(nbr, str) and nbr.strip():
            spans = spans + [{"type": "sentence", "text": nbr,
                               "start": 0, "end": len(nbr), "bbox": obj.get("bbox", [])}]

    best: dict = {"text": "", "start": 0, "end": 0,
                  "bbox": [], "rects": [], "score": -1.0, "reason": "none"}
    all_scored: list[dict] = []

    for span in spans:
        span_text = span.get("text", "")
        if not span_text:
            continue
        span_meta = {"entities": obj.get("entities", [])}

        results = [
            s.score(ci_text, ci_meta, span_text, span_meta)
            for s in SCORERS
        ]
        top = max((r for r in results if r), key=lambda r: r["score"], default=None)
        all_scores_for_span = {
            r["reason"]: round(r["score"], 4)
            for r in results if r
        }
        all_scored.append({
            "text":     span_text[:200],
            "top_score": round(top["score"], 4) if top else 0.0,
            "top_reason": top["reason"] if top else "none",
            "scores":   all_scores_for_span,
        })
        if top and top["score"] > best["score"]:
            # Extract geometry from SentenceSpan if available (geometry preservation)
            sentence_span = _deserialize_sentence_span(span)
            rects = sentence_span.get("rects", []) if sentence_span else []
            geom_source = sentence_span.get("geometry_source", "none") if sentence_span else "none"
            
            best = {
                "text":   span_text,
                "start":  span.get("start", 0),
                "end":    span.get("end", 0),
                "bbox":   span.get("bbox", []),
                "rects":  rects,  # per-line geometry if available
                "score":  top["score"],
                "reason": top["reason"],
                "match_geometry_source": geom_source,  # "apryse_span" | "object_bbox" | "none"
            }
    if best["score"] < 0:
        text = obj.get("text", "")
        best = {"text": text, "start": 0, "end": len(text),
                "bbox": obj.get("bbox", []), "rects": [], "score": 0.0, "reason": "none",
                "match_geometry_source": "none"}
    best["all_scored_spans"] = all_scored
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Span utilities
# ─────────────────────────────────────────────────────────────────────────────

_SENT_RE   = re.compile(r'(?<=[.!?])\s+|\n{2,}')
_MIN_CHARS = 15


def _split_sentences(text: str) -> list[str]:
    return [p.strip() for p in _SENT_RE.split(text) if len(p.strip()) >= _MIN_CHARS]


def _chunk_to_synthetic_obj(chunk_text: str) -> dict:
    """
    Convert a plain chunk text into a synthetic obj compatible with _pick_best_span.

    Each sentence becomes a display_span so the full scorer pipeline
    (Exact → Fuzzy → Ontology → NER → Token) can rank them.  Token overlap
    is therefore the **last** fallback, not the only one.
    """
    sents = _split_sentences(chunk_text)
    if not sents:
        sents = [chunk_text[:400]] if chunk_text else []
    spans = [
        {"type": "sentence", "text": s, "start": 0, "end": len(s), "bbox": []}
        for s in sents
    ]
    return {
        "text":          chunk_text,
        "type":          "paragraph",
        "display_spans": spans,
        "entities":      [],
        "bbox":          [],
    }


def _find_exact_match(ci_text: str, span_text: str) -> str | None:
    """
    Return the minimal verbatim substring of span_text that matches ci_text.

    Strategy 1: case-insensitive exact substring.
    Strategy 2: flexible whitespace regex.
                  "n = 8" → r'n\\s*=\\s*8'  matches "N = 8", "(N = 8)", "n=8".

    Returns None when no match is found; caller falls back to the full span text.
    """
    if not span_text:
        return None
    ci_s = ci_text.strip()
    # Strategy 1 — case-insensitive exact
    idx = span_text.lower().find(ci_s.lower())
    if idx >= 0:
        return span_text[idx: idx + len(ci_s)]
    # Strategy 2 — flexible whitespace / punctuation
    pat = _build_flexible_pattern(ci_s)
    if pat:
        m = pat.search(span_text)
        if m:
            return m.group()
    return None


def _build_flexible_pattern(ci_text: str) -> re.Pattern | None:
    """
    Build a case-insensitive regex that allows flexible whitespace between tokens.

    Tokenises on *both* whitespace and operator boundaries so that CI text
    with or without spaces around operators produces the same pattern:
      "N=4"   → ["N","=","4"]  → r'N\\s*=\\s*4'  → matches "N=4","N = 4","N= 4"
      "N<3"   → ["N","<","3"]  → r'N\\s*<\\s*3'  → matches "N<3","N < 3","N< 3"
      "n = 8" → ["n","=","8"]  → r'n\\s*=\\s*8'  → matches "n=8","N = 8","(N=8)"
      "PK/PD" → ["PK/PD"]      → r'PK/PD'        (/ kept inside token — not an op)
    """
    ci_s = ci_text.strip()
    if not ci_s or len(ci_s) > 200:
        return None
    # Split on whitespace AND on comparison/equality operator boundaries,
    # keeping the operators themselves as separate tokens.
    tokens = [t for t in re.findall(r'[≥≤≠<>=]+|[^\s≥≤≠<>=]+', ci_s) if t.strip()]
    if not tokens:
        return None
    pattern = r'\s*'.join(re.escape(t) for t in tokens)
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        return None


_HIGHLIGHT_STOPWORDS = frozenset({
    "a", "an", "the", "in", "of", "and", "or", "to", "is", "for",
    "with", "at", "on", "was", "were", "by", "be", "this", "that",
})


def _extract_matched_tokens(ci_text: str) -> list[str]:
    """
    Return the significant (non-trivial) tokens from ci_text for UI highlighting.
    The front-end can use these to bold/underline terms within the display span.

    Preserves clinical symbols: µ, ≥, ≤, <, >, ×, ^, %, /, °.
    Single-char tokens are kept unless they are stopwords (e.g. 'N' in 'N = 8',
    '8' in 'n = 8', '<' in '< 8 g/dL' are all meaningful).
    """
    tokens = re.findall(r'[\w./%µ≥≤<>×^°]+', ci_text)
    return [t for t in tokens if t.lower() not in _HIGHLIGHT_STOPWORDS]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_PUNCT_RE = re.compile(r'[^\w\s]')
_SPACE_RE = re.compile(r'\s+')


def _normalise(text: str) -> str:
    return _SPACE_RE.sub(' ', _PUNCT_RE.sub(' ', text.lower())).strip()


def _token_overlap(a: str, b: str) -> float:
    wa = set(re.findall(r'\b\w+\b', a.lower()))
    wb = set(re.findall(r'\b\w+\b', b.lower()))
    return len(wa & wb) / max(len(wa | wb), 1)
