"""Endpoint comparator — family-normalised binary comparison."""
from __future__ import annotations
from .base import (
    ComparisonResult, ClinicalContext,
    CMP_MATCH, CMP_CONFLICT, CMP_UNKNOWN,
    _SEV_NONE,
    _CONFLICT_METADATA,
    _cmp_ep_vals, _cmp_has_conflict,
)


def _compare_endpoint(
    ci_ctx: ClinicalContext, cand_ctx: ClinicalContext,
) -> ComparisonResult:
    """
    Endpoint comparison with _ep_family() normalisation.
    ORR / "overall response rate" / "overall response (PR or better)" all resolve
    to the same family and do not produce a false CONFLICT.
    """
    ci_vals   = _cmp_ep_vals(ci_ctx.facts,   ci_ctx.entities)
    cand_vals = _cmp_ep_vals(cand_ctx.facts, cand_ctx.entities)
    if not ci_vals or not cand_vals:
        return ComparisonResult("endpoint", CMP_UNKNOWN)
    if _cmp_has_conflict(ci_vals, cand_vals):
        _m = _CONFLICT_METADATA["endpoint"]
        return ComparisonResult("endpoint", CMP_CONFLICT, _m["weight"], _m["severity"],
                                {"ci": ci_vals[:2], "candidate": cand_vals[:2]})
    return ComparisonResult("endpoint", CMP_MATCH, 0.0, _SEV_NONE,
                            {"ci": ci_vals[:2], "candidate": cand_vals[:2]})
