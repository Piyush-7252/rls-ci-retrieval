"""
drug_identity.py
────────────────
Canonical drug name resolution and gradient drug-relation classification.

All clinical knowledge lives in   shared/data/drug_graph.json.
Rebuild the graph with:           python tools/build_drug_graph.py --ncit /tmp/ncit_owl/ncit.obo

Provides:
    canonical(name)            → str | None      normalised canonical ID
    drug_relation(a, b)        → DrugRelation    EXACT … DIFFERENT
    drug_relation_weight(a, b) → float           additive penalty 0.00 … −0.25
    best_drug_relation(ci_vals, cand_vals) → DrugRelation

DrugRelation severity (ordered; lower index = better):
    EXACT       = same drug (synonym-resolved)           → 0.00 penalty
    COMBINATION = one is a component of a combo regimen  → −0.08
    SAME_FAMILY = same mechanism class                   → −0.12
    RELATED     = different class, same clinical space   → −0.18
    DIFFERENT   = no clinical relationship               → −0.25
"""
from __future__ import annotations

import json, re
from enum import Enum
from pathlib import Path


# ── Relation enum & lookup tables ────────────────────────────────────────────

class DrugRelation(str, Enum):
    EXACT       = "EXACT"
    COMBINATION = "COMBINATION"
    SAME_FAMILY = "SAME_FAMILY"
    RELATED     = "RELATED"
    DIFFERENT   = "DIFFERENT"


_RELATION_ORDER: list[DrugRelation] = [
    DrugRelation.EXACT, DrugRelation.COMBINATION,
    DrugRelation.SAME_FAMILY, DrugRelation.RELATED, DrugRelation.DIFFERENT,
]

DRUG_RELATION_WEIGHTS: dict[DrugRelation, float] = {
    DrugRelation.EXACT:        0.00,
    DrugRelation.COMBINATION: -0.08,
    DrugRelation.SAME_FAMILY: -0.12,
    DrugRelation.RELATED:     -0.18,
    DrugRelation.DIFFERENT:   -0.25,
}

DRUG_RELATION_SCORE: dict[DrugRelation, float] = {
    DrugRelation.EXACT:       10.0,
    DrugRelation.COMBINATION:  4.0,
    DrugRelation.SAME_FAMILY:  2.0,
    DrugRelation.RELATED:      1.5,
    DrugRelation.DIFFERENT:    0.0,
}


# ── Load graph (once at import time) ─────────────────────────────────────────

_GRAPH_PATH = Path(__file__).resolve().parent / "data" / "drug_graph.json"
try:
    _GRAPH = json.loads(_GRAPH_PATH.read_text())
except FileNotFoundError:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "drug_graph.json not found at %s — run: python tools/build_drug_graph.py",
        _GRAPH_PATH,
    )
    _GRAPH = {"nodes": {}, "families": {}, "combos": []}

_NODES:    dict = _GRAPH.get("nodes",    {})
_FAMILIES: dict = _GRAPH.get("families", {})
_COMBOS:   list = _GRAPH.get("combos",   [])

# Flat alias map:  lower-cased alias → canonical_id
_ALIAS_MAP: dict[str, str] = {
    alias: node["canonical_id"]
    for node in _NODES.values()
    for alias in node.get("aliases", [])
}

# Compiled combo patterns
_PLUS_RE = re.compile(r"\s*[+/]\s*|\s+and\s+", re.I)
_COMBO_PATTERNS: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(c["pattern"], re.I), c["components"])
    for c in _COMBOS
]


# ── Public API ───────────────────────────────────────────────────────────────

def canonical(name: str) -> str | None:
    """Resolve a drug name string to its canonical_id, or None if unknown."""
    if not name:
        return None
    return _ALIAS_MAP.get(name.lower().strip())


def _decompose(name: str) -> frozenset[str] | None:
    """If name is a known combo regimen, return its canonical component IDs."""
    for pat, components in _COMBO_PATTERNS:
        if pat.search(name):
            return frozenset(components)
    if _PLUS_RE.search(name):
        parts = _PLUS_RE.split(name)
        cans  = {canonical(p.strip()) for p in parts}
        cans.discard(None)
        if len(cans) >= 2:
            return frozenset(cans)
    return None


def drug_relation(a: str, b: str) -> DrugRelation:
    """
    Return the relationship between two drug name strings.
    Always returns a value; never raises.
    """
    if not a or not b:
        return DrugRelation.DIFFERENT
    if a.lower().strip() == b.lower().strip():
        return DrugRelation.EXACT

    a_can, b_can = canonical(a), canonical(b)

    if a_can and b_can and a_can == b_can:
        return DrugRelation.EXACT

    a_combo, b_combo = _decompose(a), _decompose(b)

    if a_can and b_combo and a_can in b_combo:
        return DrugRelation.COMBINATION
    if b_can and a_combo and b_can in a_combo:
        return DrugRelation.COMBINATION
    if a_combo and b_combo and a_combo & b_combo:
        return DrugRelation.COMBINATION

    a_fam = _NODES.get(a_can, {}).get("family") if a_can else None
    b_fam = _NODES.get(b_can, {}).get("family") if b_can else None

    if a_fam and b_fam and a_fam == b_fam:
        return DrugRelation.SAME_FAMILY

    if a_fam and b_fam:
        if b_fam in _FAMILIES.get(a_fam, {}).get("related", []):
            return DrugRelation.RELATED

    return DrugRelation.DIFFERENT


def drug_relation_weight(a: str, b: str) -> float:
    """Additive contradiction penalty for drug pair (a, b)."""
    return DRUG_RELATION_WEIGHTS[drug_relation(a, b)]


def best_drug_relation(
    ci_vals:   list[str],
    cand_vals: list[str],
) -> DrugRelation:
    """
    Return the BEST (least severe) relation found across all ci × cand pairs.
    Short-circuits on EXACT.  Empty list → EXACT (absence handled elsewhere).
    """
    if not ci_vals or not cand_vals:
        return DrugRelation.EXACT

    best, best_idx = DrugRelation.DIFFERENT, _RELATION_ORDER.index(DrugRelation.DIFFERENT)
    for ci_v in ci_vals:
        for cand_v in cand_vals:
            rel = drug_relation(ci_v, cand_v)
            idx = _RELATION_ORDER.index(rel)
            if idx < best_idx:
                best, best_idx = rel, idx
            if best is DrugRelation.EXACT:
                return DrugRelation.EXACT
    return best
