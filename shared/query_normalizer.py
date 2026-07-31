"""
Query Normalization and Expansion
===================================
Pre-retrieval query processor that normalizes clinical abbreviations and
expands queries with synonyms before embedding and retrieval.

Two entry points:
  normalize_query(text)    — detect entities, return expanded text + metadata
  expand_query_terms(text) — return flat list of all equivalent search terms

Usage:
    from shared.query_normalizer import normalize_query

    norm = normalize_query("PK sampling after CRS")
    print(norm.expanded_text)
    # "PK sampling after CRS Pharmacokinetics Cytokine Release Syndrome"
    print(norm.label_families)
    # frozenset({'ASSESSMENT', 'SAFETY'})
    print(norm.normalized_terms)
    # frozenset({'pharmacokinetics', 'pk', 'cytokine release syndrome', 'crs'})

The expanded_text is suitable for passing to the embedding model so that
"PK" embeds closer to "Pharmacokinetics" results already in the index.

The label_families and normalized_terms are used by the entity-aware reranker
to compute entity overlap scores between the query and each candidate chunk.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

# Ensure shared/ is importable when run from any directory
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryNorm:
    """Normalized and expanded representation of a user query."""
    original:         str
    expanded_text:    str               # original + appended synonyms / full-forms
    entities:         list[dict]        # dict entities detected in the query
    label_families:   frozenset[str]    # unique family buckets  (ASSESSMENT, RESPONSE, …)
    labels:           frozenset[str]    # unique fine-grained labels (PK_ASSESSMENT, …)
    normalized_terms: frozenset[str]    # lowercased canonical / normalized forms


# ─────────────────────────────────────────────────────────────────────────────
# Core normalizer
# ─────────────────────────────────────────────────────────────────────────────

def normalize_query(text: str) -> QueryNorm:
    """
    Detect clinical entities in *text* and return an expanded form for retrieval.

    Expansion strategy
    ------------------
    For each matched entity, append its full normalized name and abbreviation
    (when they differ from the matched text).  De-duplicate additions and avoid
    appending terms that are already present verbatim in the original query.

    Example
    -------
    Query  : "PK sampling after CRS"
    Matched: PK → PK_ASSESSMENT (Pharmacokinetics), CRS → ADVERSE_EVENT (Cytokine Release Syndrome)
    Extra  : ["Pharmacokinetics", "Cytokine Release Syndrome"]
    Result : "PK sampling after CRS Pharmacokinetics Cytokine Release Syndrome"
    """
    try:
        from shared.clinical_dict import match_entities
    except ImportError:
        from clinical_dict import match_entities  # type: ignore[no-redef]

    entities: list[dict] = match_entities(text)
    text_lower = text.lower()

    extra:      list[str] = []
    seen_extra: set[str]  = set()

    for e in entities:
        for term in (
            e.get("normalized",   ""),   # full name     e.g. "Pharmacokinetics"
            e.get("abbreviation", ""),   # abbreviation  e.g. "PK"
            e.get("canonical",    ""),   # canonical     e.g. "PFS"
        ):
            if not term:
                continue
            tl = term.lower()
            # Skip if already present verbatim in the original query or already queued
            if tl in text_lower or tl in seen_extra:
                continue
            seen_extra.add(tl)
            extra.append(term)

    expanded_text = (text + " " + " ".join(extra)).strip() if extra else text

    label_families   = frozenset(e.get("family", "OTHER") for e in entities)
    labels           = frozenset(e["label"]               for e in entities)
    normalized_terms = frozenset(
        n.lower()
        for e in entities
        for n in (e.get("normalized", ""), e.get("canonical", ""), e.get("abbreviation", ""))
        if n
    )

    return QueryNorm(
        original         = text,
        expanded_text    = expanded_text,
        entities         = entities,
        label_families   = label_families,
        labels           = labels,
        normalized_terms = normalized_terms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Synonym groups (bidirectional, free-text domain phrases typed by users)
# These supplement the clinical_dict coverage for informal query phrasings.
# ─────────────────────────────────────────────────────────────────────────────

_SYNONYM_GROUPS: list[tuple[str, ...]] = [
    ("bone marrow", "bone marrow aspirate", "bone marrow biopsy", "bma"),
    ("whole blood", "blood sample", "blood draw"),
    ("response assessment", "response evaluation", "response criteria"),
    ("pharmacokinetic", "pharmacokinetics", "pk sampling", "pk assessment"),
    ("pharmacodynamic", "pharmacodynamics", "pd assessment"),
    ("overall survival",              "os"),
    ("progression-free survival",     "pfs"),
    ("complete response",             "cr"),
    ("stringent complete response",   "scr"),
    ("partial response",              "pr"),
    ("very good partial response",    "vgpr"),
    ("minimal residual disease",      "mrd"),
    ("overall response rate",         "orr"),
    ("duration of response",          "dor"),
    ("cytokine release syndrome",     "crs"),
    ("adverse event",                 "ae", "side effect"),
    ("end of treatment",              "eot"),
    ("immunophenotyping",             "flow cytometry", "immunophenotype"),
    ("anti-drug antibody",            "ada", "immunogenicity"),
    ("kaplan-meier",                  "km", "survival curve"),
]

_SYNONYM_INDEX: dict[str, int] = {}
for _gi, _group in enumerate(_SYNONYM_GROUPS):
    for _term in _group:
        _SYNONYM_INDEX[_term.lower()] = _gi


def expand_query_terms(text: str) -> list[str]:
    """
    Return all equivalent search terms for *text*.

    Combines:
    - entity expansions from the clinical dictionary
    - synonym-group lookups for informal phrasings

    Useful for BM25 / multi-match text search alongside vector search.

    Example
    -------
        expand_query_terms("PK sampling")
        → ["PK sampling", "PK", "Pharmacokinetics", "pharmacokinetic", "pk assessment"]
    """
    norm  = normalize_query(text)
    terms: set[str] = {text}

    # Entity-derived terms (text, normalized, canonical, abbreviation)
    for e in norm.entities:
        for v in (e.get("text"), e.get("normalized"), e.get("canonical"), e.get("abbreviation")):
            if v:
                terms.add(v)

    # Synonym-group expansion for each accumulated term
    for term in list(terms):
        gi = _SYNONYM_INDEX.get(term.lower())
        if gi is not None:
            for syn in _SYNONYM_GROUPS[gi]:
                terms.add(syn)

    return [t for t in terms if t]
