"""
Statistical comparator — structured comparison of numeric clinical facts.

Compares statistical_identity fields (sample_size, p_value, hazard_ratio,
odds_ratio, lower/upper CI bounds, median, percentage) between the CI and
a document candidate.

Design
------
* Type-agnostic: the comparator iterates _STAT_KEY_META rather than
  branching on statistical_identity.type.  This means a CI of type
  "confidence_interval" and a candidate of type "hazard_ratio" still
  MATCH if their lower_ci and upper_ci values agree.

* Per-key tolerances: exact for integers (sample_size, confidence_level),
  absolute for small decimals (p_value), relative for ratios and medians.

* Only the most-severe conflict is reported (mirrors temporal.py).
  A MATCH is returned when at least one key matched and none conflicted.
  UNKNOWN is returned when neither side has comparable values.

Adding a new statistical key:
  1. Add an entry to _STAT_KEY_META.
  2. Nothing else needs to change.
"""
from __future__ import annotations
from .base import (
    ComparisonResult, ClinicalContext,
    CMP_MATCH, CMP_CONFLICT, CMP_UNKNOWN,
    _SEV_NONE, _SEV_LOW, _SEV_MEDIUM, _SEV_HIGH,
    _SEV_RANK,
)

# ─── Per-key comparison metadata ──────────────────────────────────────────────
#
# (abs_tol, rel_tol, conflict_weight, conflict_severity)
#
#   abs_tol   — maximum absolute difference allowed for a MATCH  (0.0 = exact)
#   rel_tol   — maximum relative difference (fraction) allowed   (0.0 = exact)
#   weight    — additive penalty when this key conflicts
#   severity  — _SEV_* constant reported in comparator_trace
#
# Both tolerances are checked independently; either satisfying → MATCH.
_STAT_KEY_META: dict[str, tuple[float, float, float, str]] = {
    # key                abs_tol  rel_tol  weight  severity
    "sample_size":      (0.0,    0.0,    -0.28,  _SEV_MEDIUM),  # exact integers
    "confidence_level": (0.0,    0.0,    -0.10,  _SEV_LOW),     # 90 / 95 / 99
    "p_value":          (5e-5,   0.0,    -0.30,  _SEV_HIGH),    # absolute; captures rounding
    "hazard_ratio":     (0.0,    0.015,  -0.35,  _SEV_HIGH),    # 1.5% relative
    "odds_ratio":       (0.0,    0.015,  -0.35,  _SEV_HIGH),
    "lower_ci":         (0.0,    0.015,  -0.25,  _SEV_MEDIUM),
    "upper_ci":         (0.0,    0.015,  -0.25,  _SEV_MEDIUM),
    "median":           (0.0,    0.015,  -0.25,  _SEV_MEDIUM),
    "percentage":       (0.5,    0.0,    -0.22,  _SEV_MEDIUM),  # 0.5 pp absolute
}

# String-valued key: median_unit ("months" vs "days" is a confirmed conflict).
_STAT_UNIT_KEYS: frozenset[str] = frozenset({"median_unit", "ci_unit"})
_STAT_UNIT_META: dict[str, tuple[float, str]] = {
    "median_unit": (-0.15, _SEV_LOW),
    "ci_unit":     (-0.10, _SEV_LOW),
}


def _stat_match(
    ci_val:  float,
    cand_val: float,
    abs_tol: float,
    rel_tol: float,
) -> bool:
    """True when the two numeric values agree within the specified tolerance."""
    if ci_val == cand_val:
        return True
    diff = abs(ci_val - cand_val)
    if abs_tol > 0.0 and diff <= abs_tol:
        return True
    if rel_tol > 0.0:
        ref = max(abs(ci_val), abs(cand_val))
        if ref > 0.0 and diff / ref <= rel_tol:
            return True
    return False


def _compare_statistical(
    ci_ctx: ClinicalContext, cand_ctx: ClinicalContext,
) -> ComparisonResult:
    """
    Generic statistical comparator.

    Iterates all numeric keys in _STAT_KEY_META and string keys in
    _STAT_UNIT_KEYS.  Reports the single most-severe conflict (if any),
    or MATCH when at least one key agreed and none conflicted,
    or UNKNOWN when neither side carries comparable values.
    """
    ci_si   = ci_ctx.statistical_identity
    cand_si = cand_ctx.statistical_identity

    # If either side has no statistical identity, nothing to compare.
    if not ci_si or not cand_si:
        return ComparisonResult("statistical", CMP_UNKNOWN)

    matched_keys: list[str] = []

    # Track worst conflict across all keys.
    worst_sev: str       = _SEV_NONE
    worst_w:   float     = 0.0
    worst_ev:  dict | None = None

    # ── Numeric keys ──────────────────────────────────────────────────────────
    for key, (abs_tol, rel_tol, w, sev) in _STAT_KEY_META.items():
        ci_raw   = ci_si.get(key)
        cand_raw = cand_si.get(key)
        if ci_raw is None:
            continue   # CI doesn't specify this key — nothing to compare
        if cand_raw is None:
            continue   # Candidate doesn't have it — data gap, not contradiction

        try:
            ci_val   = float(ci_raw)
            cand_val = float(cand_raw)
        except (TypeError, ValueError):
            continue

        if _stat_match(ci_val, cand_val, abs_tol, rel_tol):
            matched_keys.append(key)
        else:
            if _SEV_RANK[sev] > _SEV_RANK[worst_sev]:
                worst_sev = sev
                worst_w   = w
                worst_ev  = {
                    "key":       key,
                    "ci":        ci_raw,
                    "candidate": cand_raw,
                }

    # ── String / unit keys ────────────────────────────────────────────────────
    for key, (w, sev) in _STAT_UNIT_META.items():
        ci_unit   = (ci_si.get(key) or "").lower().strip()
        cand_unit = (cand_si.get(key) or "").lower().strip()
        if not ci_unit or not cand_unit:
            continue
        if ci_unit == cand_unit:
            matched_keys.append(key)
        elif _SEV_RANK[sev] > _SEV_RANK[worst_sev]:
            worst_sev = sev
            worst_w   = w
            worst_ev  = {
                "key":       key,
                "ci":        ci_unit,
                "candidate": cand_unit,
            }

    # ── Decision ──────────────────────────────────────────────────────────────
    if worst_ev is not None:
        return ComparisonResult(
            "statistical", CMP_CONFLICT,
            worst_w, worst_sev,
            {"conflict": worst_ev, "matched": matched_keys},
        )
    if matched_keys:
        return ComparisonResult(
            "statistical", CMP_MATCH,
            evidence={"matched": matched_keys},
        )
    return ComparisonResult("statistical", CMP_UNKNOWN)
