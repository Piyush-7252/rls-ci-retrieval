"""Drug identity comparator."""
from __future__ import annotations
from .base import (
    ComparisonResult, ClinicalContext,
    CMP_MATCH, CMP_RELATED, CMP_SPECIALIZATION, CMP_GENERALIZATION,
    CMP_UNKNOWN, CMP_CONFLICT,
    _SEV_NONE, _SEV_LOW,
    _CONFLICT_METADATA, _DRUG_CONTRA_WEIGHTS,
    _CMP_DRUG_LABELS, _CMP_TREAT_DRUG_SUBTYPES,
    _cmp_slot_vals, _cmp_has_conflict,
)


def _compare_drug(
    ci_ctx: ClinicalContext, cand_ctx: ClinicalContext,
) -> ComparisonResult:
    """
    Drug identity comparison — gradient via aggregator's identity_overlap.drug.relation.
    Falls back to binary string overlap when the relation field is absent.

    EXACT        -> MATCH           (exact same compound)
    COMBINATION  -> RELATED         (one drug is part of the other's regimen)
    SAME_FAMILY  -> SPECIALIZATION  (same mechanism class; candidate is a specific instance)
    RELATED      -> GENERALIZATION  (candidate covers a broader / lateral drug space)
    DIFFERENT    -> CONFLICT        (confirmed distinct drug)
    <no overlap> -> CONFLICT        (binary string fallback)

    Primary-drug guard: when CI has an explicit treatment_identity.primary_drug and
    that drug is absent from the candidate, the outcome is downgraded from MATCH to
    RELATED even if companion drugs overlap.  This prevents pomalidomide / lenalidomide
    companion overlap from masking a missing teclistamab primary drug.
    """
    ci_vals   = _cmp_slot_vals(ci_ctx.facts,   "drug", ci_ctx.entities,   _CMP_DRUG_LABELS)
    cand_vals = _cmp_slot_vals(cand_ctx.facts, "drug", cand_ctx.entities, _CMP_DRUG_LABELS)
    if not ci_vals or not cand_vals:
        return ComparisonResult("drug", CMP_UNKNOWN)

    agg_rel = (cand_ctx.identity_overlap.get("drug") or {}).get("relation")
    if agg_rel and agg_rel in _DRUG_CONTRA_WEIGHTS:
        w = _DRUG_CONTRA_WEIGHTS[agg_rel]
        if w == 0.0:
            # Aggregator computed EXACT match — but verify primary drug is present
            outcome, sev, w = _apply_primary_drug_guard(
                ci_ctx, cand_vals, CMP_MATCH, _SEV_NONE, 0.0
            )
            return ComparisonResult("drug", outcome, w, sev,
                                    {"relation": agg_rel,
                                     "ci": ci_vals[:2], "candidate": cand_vals[:2]})
        _outcome_map: dict[str, tuple[str, str]] = {
            "DIFFERENT":   (CMP_CONFLICT,       _CONFLICT_METADATA["drug"]["severity"]),
            "SAME_FAMILY": (CMP_SPECIALIZATION, _SEV_LOW),
            "RELATED":     (CMP_GENERALIZATION, _SEV_LOW),
        }
        outcome, sev = _outcome_map.get(agg_rel, (CMP_RELATED, _SEV_LOW))
        return ComparisonResult("drug", outcome, w, sev,
                                {"relation": agg_rel,
                                 "ci": ci_vals[:2], "candidate": cand_vals[:2]})

    # Fallback: binary string overlap
    if _cmp_has_conflict(ci_vals, cand_vals):
        _m = _CONFLICT_METADATA["drug"]
        return ComparisonResult("drug", CMP_CONFLICT, _m["weight"], _m["severity"],
                                {"ci": ci_vals[:2], "candidate": cand_vals[:2]})

    # No conflict on drug list — but check primary drug before reporting MATCH
    outcome, sev, w = _apply_primary_drug_guard(
        ci_ctx, cand_vals, CMP_MATCH, _SEV_NONE, 0.0
    )
    return ComparisonResult("drug", outcome, w, sev,
                            {"ci": ci_vals[:2], "candidate": cand_vals[:2]})


def _apply_primary_drug_guard(
    ci_ctx:   ClinicalContext,
    cand_vals: list[str],
    default_outcome: str,
    default_sev:     str,
    default_w:       float,
) -> tuple[str, str, float]:
    """
    Downgrade a drug MATCH to RELATED when the CI specifies a primary drug that
    is absent from the candidate's drug list.

    Companion-drug overlap (e.g. shared pomalidomide / lenalidomide in a
    multi-drug regimen) is real clinical evidence, but it should not mask the
    absence of the primary agent the CI is asking about.

    Returns (outcome, severity, weight) — unchanged when the guard does not fire.
    """
    ci_primary = (ci_ctx.treatment.get("primary_drug") or "").lower().strip()
    if not ci_primary:
        return default_outcome, default_sev, default_w

    # Check whether the primary drug (or a close substring) appears in cand_vals
    primary_present = any(
        ci_primary in cv or cv in ci_primary
        for cv in cand_vals
    )
    if primary_present:
        return default_outcome, default_sev, default_w

    # Primary drug explicitly absent — companion overlap is insufficient for MATCH
    return CMP_RELATED, _SEV_LOW, _DRUG_CONTRA_WEIGHTS.get("COMBINATION", -0.10)
