"""Treatment regimen comparator."""
from __future__ import annotations
from .base import (
    ComparisonResult, ClinicalContext,
    CMP_MATCH, CMP_CONFLICT, CMP_UNKNOWN,
    _CONFLICT_METADATA,
)


def _compare_regimen(
    ci_ctx: ClinicalContext, cand_ctx: ClinicalContext,
) -> ComparisonResult:
    """
    Treatment regimen comparison across three treatment_identity sub-fields:
      line_of_therapy  — first-line vs second-line etc.
      combination flag — monotherapy vs combination backbone
      companion_drugs  — specific combination partners

    Each sub-field conflict is named individually in evidence so the explanation
    layer can say "line of therapy mismatch" rather than an opaque "regimen mismatch".
    primary_drug is covered by _compare_drug and is excluded here.
    """
    ci_treat   = ci_ctx.treatment
    cand_treat = cand_ctx.treatment
    if not ci_treat or not cand_treat:
        return ComparisonResult("regimen", CMP_UNKNOWN)

    sub_conflicts: list[dict] = []

    ci_lot   = (ci_treat.get("line_of_therapy")   or "").lower().strip()
    cand_lot = (cand_treat.get("line_of_therapy")  or "").lower().strip()
    if ci_lot and cand_lot and ci_lot != cand_lot:
        sub_conflicts.append({"field": "line_of_therapy",
                              "ci": ci_lot, "candidate": cand_lot})

    ci_combo   = ci_treat.get("combination")
    cand_combo = cand_treat.get("combination")
    if ci_combo is not None and cand_combo is not None and bool(ci_combo) != bool(cand_combo):
        sub_conflicts.append({"field": "combination",
                              "ci": ci_combo, "candidate": cand_combo})

    ci_comp   = {v.lower().strip() for v in (ci_treat.get("companion_drugs")   or []) if v}
    cand_comp = {v.lower().strip() for v in (cand_treat.get("companion_drugs") or []) if v}
    if ci_comp and cand_comp and not (ci_comp & cand_comp):
        sub_conflicts.append({"field": "companion_drugs",
                              "ci": sorted(ci_comp)[:2], "candidate": sorted(cand_comp)[:2]})

    if sub_conflicts:
        _m = _CONFLICT_METADATA["regimen"]
        return ComparisonResult("regimen", CMP_CONFLICT, _m["weight"], _m["severity"],
                                {"conflicts": sub_conflicts})
    if ci_lot or ci_combo is not None or ci_comp:
        return ComparisonResult("regimen", CMP_MATCH)
    return ComparisonResult("regimen", CMP_UNKNOWN)
