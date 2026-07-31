"""Statement modality comparator."""
from __future__ import annotations
from .base import (
    ComparisonResult, ClinicalContext,
    CMP_MATCH, CMP_CONFLICT, CMP_UNKNOWN,
    _SEV_NONE,
    _CONFLICT_METADATA, _MODALITY_GROUP,
)


def _compare_modality(
    ci_ctx: ClinicalContext, cand_ctx: ClinicalContext,
) -> ComparisonResult:
    """
    Modality group comparison.
    OBJECTIVE vs PROCEDURE/REQUIREMENT — CI asks what to measure;
    candidate describes what patients must do.  Cross-group = CONFLICT.
    GENERAL or unclassified modality returns UNKNOWN
    (conservative: absence of data != contradiction).
    """
    ci_mod   = ci_ctx.modality
    cand_mod = cand_ctx.modality
    if ci_mod == "GENERAL" or cand_mod == "GENERAL":
        return ComparisonResult("modality", CMP_UNKNOWN)
    ci_grp   = _MODALITY_GROUP.get(ci_mod,   "unclassified")
    cand_grp = _MODALITY_GROUP.get(cand_mod, "unclassified")
    if ci_grp == "unclassified" or cand_grp == "unclassified":
        return ComparisonResult("modality", CMP_UNKNOWN)
    if ci_grp != cand_grp:
        _m = _CONFLICT_METADATA["modality"]
        return ComparisonResult("modality", CMP_CONFLICT, _m["weight"], _m["severity"],
                                {"ci_modality": ci_mod,  "candidate_modality": cand_mod,
                                 "ci_group":    ci_grp,  "candidate_group":    cand_grp})
    return ComparisonResult("modality", CMP_MATCH, 0.0, _SEV_NONE,
                            {"ci_group": ci_grp, "candidate_group": cand_grp})
