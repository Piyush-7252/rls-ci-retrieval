"""
Search Pipeline — Stage 6.5: Highlight Extractor
=================================================
Takes the already-retrieved semantic object and selects the best
display_span (sentence, table row, form field, bullet, signature, etc.)
to surface in the UI.

No embeddings.  No re-ranking.  All matching is text-only.

Design: scorer registry
-----------------------
Each scorer is an independent class that inspects a (CI, span) pair and
returns {"score": float, "reason": str} or None.

Score ranges are non-overlapping so max() gives natural priority:

  LiteralContainmentScorer      0.90 – 1.00 either text literally contains the other
  ExtractionEquivalentScorer    0.89       verbatim match after removing all whitespace
  FuzzyScorer                   0.65 – 0.89 difflib SequenceMatcher ≥ 0.65
  OntologyScorer            0.60 – 0.74 a CI synonym appears in span
  NERScorer                 0.40 – 0.59 CI entity-type overlap
  TokenScorer               0.00 – 0.39 Jaccard token fallback (always fires)

Highlight contract
------------------
Literal retriever hits carry character offsets in ``cand["literal_matches"]``.
They are consumed DIRECTLY in ``_process()`` — literal evidence is NOT routed
through the scorer registry.  No rediscovery.  No re-scoring.

  literal_matches present?
        │
   ┌────┴────┐
   YES        NO
    │          │
  offsets    scorer registry
  directly   (LiteralContainment→Extraction→Fuzzy→Onto→NER→Token)

Both paths call _pick_best_span() to select the context_sentence.

Output fields
-------------
  highlight_spans  — list[{text, start, end, source}], one entry per matched term
  match_span       — text of the primary (first) highlight (backward compat)
  context_sentence — best display sentence, always chosen by scorer registry

To add a new scorer:
  1. Subclass BaseScorer
  2. Append an instance to SCORERS

Nothing else changes.
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
    ci      = req.get("ci", {})
    ci_text = ci.get("knownCI", "")
    ci_meta = {
        "synonyms":  (ci.get("normalization", {}).get("expansions", [])
                      or list(ci.get("ontology", {}).get("synonyms", {}).keys())),
        "ent_types": {e.get("type", "") for e in ci.get("ner", {}).get("entities", [])},
    }
    verified = req.get("verified_candidates", [])
    mtokens  = _extract_matched_tokens(ci_text)

    # ── Pass 1: process object-based candidates ────────────────────────────────
    # Pre-computing avoids calling _pick_best_span twice per object candidate.
    precomputed: dict[int, dict] = {}
    has_authoritative_object = False

    for i, cand in enumerate(verified):
        if cand.get("verdict") not in ("YES", "MAYBE"):
            continue
        matched_obj = cand.get("matched_object")
        if not matched_obj:
            continue  # text_fallback — deferred to Pass 2

        # literal_matches are stored at the cluster (cand) level by the aggregator.
        lit  = cand.get("literal_matches", [])
        # Safety: if the selected object is a sentence that doesn't contain the
        # literal match text, promote to a context_object that does.  This guards
        # against stale candidates where context_expander chose on type priority
        # alone before literal-aware object selection was introduced.
        if lit and matched_obj.get("type") == "sentence":
            primary_lit = (lit[0].get("text") or "").lower()
            if primary_lit and primary_lit not in (matched_obj.get("text") or "").lower():
                for ctx_obj in cand.get("context_objects", []):
                    if primary_lit in (ctx_obj.get("text") or "").lower():
                        logger.debug(
                            "[HighlightExtractor] literal mismatch on sentence — "
                            "promoted to %s object_id=%s search_id=%s",
                            ctx_obj.get("type"), ctx_obj.get("object_id"),
                            req.get("search_id"),
                        )
                        matched_obj = ctx_obj
                        break
        # Always pick the best context sentence via the scorer registry.
        best = _pick_best_span(ci_text, ci_meta, matched_obj)

        if lit:
            # Retriever already produced character-accurate offsets.
            # Consume them directly — no scorer involvement for match_span.
            ret = cand.get("retriever", "")
            highlight_spans = [
                {"text": lm["text"], "start": lm["start"], "end": lm["end"],
                 "source": "literal", "retriever": ret}
                for lm in lit if lm.get("text")
            ]
            primary = highlight_spans[0]
            precomputed[i] = {
                "best":            best,
                "highlight_spans": highlight_spans,
                "match_span":      primary["text"],
                "match_start":     primary["start"],
                "match_end":       primary["end"],
                "highlight_score":   1.0,
                "match_reason":    "literal",
                "is_authoritative": True,
                "page":            matched_obj.get("page", cand.get("page_start")),
            }
        else:
            exact     = _find_exact_match(ci_text, best["text"])
            span_text = exact or best["text"]
            precomputed[i] = {
                "best":            best,
                "highlight_spans": [{"text": span_text, "start": best["start"],
                                     "end": best["end"], "source": best["reason"],
                                     "retriever": cand.get("retriever", "")}],
                "match_span":      span_text,
                "match_start":     best["start"],
                "match_end":       best["end"],
                "highlight_score":   best["score"],
                "match_reason":    best["reason"],
                "is_authoritative": best["reason"] == "exact",
                "page":            matched_obj.get("page", cand.get("page_start")),
            }

        # Suppress text-fallback candidates only when retrieval evidence is
        # authoritative: a literal match (retriever confirmed the term) or an
        # exact verbatim match.  OCR-equivalent matches are NOT authoritative
        # for suppression — whitespace removal is more permissive than exact.
        if bool(lit) or precomputed[i]["match_reason"] == "exact":
            has_authoritative_object = True

    if has_authoritative_object:
        logger.debug("[HighlightExtractor] authoritative object hit (literal/exact) — "
                     "skipping text_fallback candidates for search_id=%s",
                     req.get("search_id"))

    # ── Pass 2: build enriched list ────────────────────────────────────────────
    enriched = []
    for i, cand in enumerate(verified):
        if cand.get("verdict") in ("YES", "MAYBE"):
            matched_obj = cand.get("matched_object")

            if i in precomputed:
                # Object path — use precomputed result from Pass 1
                r    = precomputed[i]
                best = r["best"]
                cand = {
                    **cand,
                    "match_span":        r["match_span"],
                    "match_span_start":  r["match_start"],
                    "match_span_end":    r["match_end"],
                    "highlight_spans":   r["highlight_spans"],
                    "context_sentence":  best["text"],
                    "match_page":        r["page"],
                    "match_bbox":        best.get("bbox") or matched_obj.get("bbox", []),
                    "match_rects":       best.get("rects") or [],  # per-line geometry if available
                    "match_geometry_source": best.get("match_geometry_source", "none"),  # "apryse_span" | "object_bbox" | "none"
                    "highlight_score":     round(r["highlight_score"], 3),
                    "match_method":      "object",
                    "match_reason":      r["match_reason"],
                    "is_authoritative":  r["is_authoritative"],
                    "matched_tokens":    mtokens,
                    "span_debug": {
                        "best_text":    best["text"],
                        "best_score":   round(best["score"], 4),
                        "best_reason":  best["reason"],
                        "match_span":   r["match_span"],
                        "match_reason": r["match_reason"],
                        "highlight_score": round(r["highlight_score"], 3),
                        "is_authoritative": r["is_authoritative"],
                        "all_scored_spans": best.get("all_scored_spans", []),
                    },
                }

            elif has_authoritative_object:
                # An authoritative object hit (literal or exact) exists.
                # Text-fallback candidates cannot add stronger evidence.
                cand = {
                    **cand,
                    "match_span":       "",
                    "highlight_spans":  [],
                    "highlight_score":    0.0,
                    "match_method":     "text_fallback_skipped",
                    "match_reason":     "skipped:exact_object_hit",
                    "is_authoritative": False,
                    "matched_tokens":   mtokens,
                }

            else:
                # Text-fallback path — no matched_object.
                # Check for literal evidence first; fall back to scorer registry.
                chunk_text = cand.get("context", {}).get("current_text", "")
                lit  = cand.get("literal_matches", [])
                obj  = _chunk_to_synthetic_obj(chunk_text)
                page = cand.get("page_start")
                best = _pick_best_span(ci_text, ci_meta, obj)

                if lit:
                    ret = cand.get("retriever", "")
                    highlight_spans = [
                        {"text": lm["text"], "start": lm["start"], "end": lm["end"],
                         "source": "literal", "retriever": ret}
                        for lm in lit if lm.get("text")
                    ]
                    primary = highlight_spans[0]
                    cand = {
                        **cand,
                        "match_span":        primary["text"],
                        "match_span_start":  primary["start"],
                        "match_span_end":    primary["end"],
                        "highlight_spans":   highlight_spans,
                        "context_sentence":  best["text"],
                        "match_page":        page,
                        "match_bbox":        [],
                        "match_rects":       best.get("rects", []),  # geometry if available
                        "match_geometry_source": best.get("match_geometry_source", "none"),  # quality indicator
                        "highlight_score":     1.0,
                        "match_method":      "text_fallback",
                        "match_reason":      "literal",
                        "is_authoritative":  True,
                        "matched_tokens":    mtokens,
                        "span_debug": {
                            "best_text":    best["text"],
                            "best_score":   round(best["score"], 4),
                            "best_reason":  best["reason"],
                            "match_span":   primary["text"],
                            "match_reason": "literal",
                            "highlight_score": 1.0,
                            "is_authoritative": True,
                            "all_scored_spans": best.get("all_scored_spans", []),
                        },
                    }
                else:
                    exact     = _find_exact_match(ci_text, best["text"])
                    span_text = exact or best["text"]
                    cand = {
                        **cand,
                        "match_span":        span_text,
                        "match_span_start":  best["start"],
                        "match_span_end":    best["end"],
                        "highlight_spans":   [{"text": span_text, "start": best["start"],
                                               "end": best["end"], "source": best["reason"],
                                               "retriever": cand.get("retriever", "")}],
                        "context_sentence":  best["text"],
                        "match_page":        page,
                        "match_bbox":        [],
                        "match_rects":       best.get("rects", []),  # geometry if available
                        "match_geometry_source": best.get("match_geometry_source", "none"),  # quality indicator
                        "highlight_score":     round(best["score"], 3),
                        "match_method":      "text_fallback",
                        "match_reason":      best["reason"],
                        "is_authoritative":  best["reason"] == "exact",
                        "matched_tokens":    mtokens,
                        "span_debug": {
                            "best_text":    best["text"],
                            "best_score":   round(best["score"], 4),
                            "best_reason":  best["reason"],
                            "match_span":   span_text,
                            "match_reason": best["reason"],
                            "highlight_score": round(best["score"], 3),
                            "is_authoritative": best["reason"] == "exact",
                            "all_scored_spans": best.get("all_scored_spans", []),
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
