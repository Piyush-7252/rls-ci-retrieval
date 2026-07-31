"""Study arm comparator."""
from __future__ import annotations
from .base import (
    ComparisonResult, ClinicalContext,
    CMP_MATCH, CMP_CONFLICT, CMP_UNKNOWN,
    _CONFLICT_METADATA,
    _CMP_ARM_LABELS,
    _cmp_slot_vals, _cmp_has_conflict,
)


def _compare_study_arm(
    ci_ctx: ClinicalContext, cand_ctx: ClinicalContext,
) -> ComparisonResult:
    """Study arm binary comparison."""
    ci_vals   = _cmp_slot_vals(ci_ctx.facts,   "study_arm",
                               ci_ctx.entities,   _CMP_ARM_LABELS)
    cand_vals = _cmp_slot_vals(cand_ctx.facts, "study_arm",
                               cand_ctx.entities, _CMP_ARM_LABELS)
    if not ci_vals or not cand_vals:
        return ComparisonResult("study_arm", CMP_UNKNOWN)
    if _cmp_has_conflict(ci_vals, cand_vals):
        _m = _CONFLICT_METADATA["study_arm"]
        return ComparisonResult("study_arm", CMP_CONFLICT, _m["weight"], _m["severity"],
                                {"ci": ci_vals[:2], "candidate": cand_vals[:2]})
    return ComparisonResult("study_arm", CMP_MATCH)
