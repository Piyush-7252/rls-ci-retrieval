"""Temporal context comparator."""
from __future__ import annotations
from .base import (
    ComparisonResult, ClinicalContext,
    CMP_MATCH, CMP_CONFLICT, CMP_UNKNOWN,
    _SEV_NONE, _SEV_RANK,
    _TEMPORAL_FIELD_SEVERITY,
)


def _compare_temporal(
    ci_ctx: ClinicalContext, cand_ctx: ClinicalContext,
) -> ComparisonResult:
    """
    Temporal context comparison across _TEMPORAL_FIELD_SEVERITY fields.
    Only the single most-severe field conflict is reported per candidate to
    prevent multiple small differences from stacking into a false FATAL.
    Severity and weight are field-specific (cycle != day != protocol_version).
    """
    ci_temp   = ci_ctx.temporal
    cand_temp = cand_ctx.temporal
    if not ci_temp or not cand_temp:
        return ComparisonResult("temporal", CMP_UNKNOWN)

    worst_sev: str      = _SEV_NONE
    worst_w:   float    = 0.0
    worst_ev:  dict | None = None
    for tf, (sev, w) in _TEMPORAL_FIELD_SEVERITY.items():
        ci_v   = str(ci_temp.get(tf)   or "").lower().strip()
        cand_v = str(cand_temp.get(tf) or "").lower().strip()
        if ci_v and cand_v and ci_v != cand_v:
            if _SEV_RANK[sev] > _SEV_RANK[worst_sev]:
                worst_sev = sev
                worst_w   = w
                worst_ev  = {"field": tf, "ci": ci_v, "candidate": cand_v}

    if worst_ev is not None:
        return ComparisonResult("temporal", CMP_CONFLICT, worst_w, worst_sev, worst_ev)
    return ComparisonResult("temporal", CMP_MATCH)
