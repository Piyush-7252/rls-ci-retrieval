"""Trial phase comparator."""
from __future__ import annotations
from .base import (
    ComparisonResult, ClinicalContext,
    CMP_MATCH, CMP_CONFLICT, CMP_UNKNOWN,
    _CONFLICT_METADATA,
    _CMP_PHASE_LABELS,
    _cmp_slot_vals, _cmp_has_conflict,
)


def _compare_phase(
    ci_ctx: ClinicalContext, cand_ctx: ClinicalContext,
) -> ComparisonResult:
    """Trial phase binary comparison."""
    ci_vals   = _cmp_slot_vals(ci_ctx.facts,   "phase", ci_ctx.entities,   _CMP_PHASE_LABELS)
    cand_vals = _cmp_slot_vals(cand_ctx.facts, "phase", cand_ctx.entities, _CMP_PHASE_LABELS)
    if not ci_vals or not cand_vals:
        return ComparisonResult("phase", CMP_UNKNOWN)
    if _cmp_has_conflict(ci_vals, cand_vals):
        _m = _CONFLICT_METADATA["phase"]
        return ComparisonResult("phase", CMP_CONFLICT, _m["weight"], _m["severity"],
                                {"ci": ci_vals[:2], "candidate": cand_vals[:2]})
    return ComparisonResult("phase", CMP_MATCH)
