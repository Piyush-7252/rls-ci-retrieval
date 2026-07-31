"""Negation inversion comparator."""
from __future__ import annotations
from .base import (
    ComparisonResult, ClinicalContext,
    CMP_CONFLICT, CMP_UNKNOWN,
    _SEV_HIGH,
    _CONFLICT_METADATA,
)


def _compare_negation(
    ci_ctx: ClinicalContext, cand_ctx: ClinicalContext,
) -> ComparisonResult:
    """
    Negation inversion detection.
    Fires when CI negates a slot the candidate affirms (or vice versa).
    Compares slot + affirmation status to avoid false contradictions when both
    sides negate the same slot (consistent negation is not a conflict).
    Only the first detected inversion is returned.
    """
    ci_neg   = set(ci_ctx.negated_slots or [])
    cand_neg = set(cand_ctx.negated_slots or [])
    ci_facts   = ci_ctx.facts
    cand_facts = cand_ctx.facts

    for slot in ci_neg:
        if slot not in cand_neg and cand_facts.get(slot):
            return ComparisonResult(
                "negation", CMP_CONFLICT,
                _CONFLICT_METADATA["negation"]["weight"], _SEV_HIGH,
                {"slot": slot, "ci": "negated", "candidate": "affirmed",
                 "candidate_values": (cand_facts.get(slot) or [])[:2]},
            )
    for slot in cand_neg:
        if slot not in ci_neg and ci_facts.get(slot):
            return ComparisonResult(
                "negation", CMP_CONFLICT,
                _CONFLICT_METADATA["negation"]["weight"], _SEV_HIGH,
                {"slot": slot, "ci": "affirmed", "candidate": "negated",
                 "ci_values": (ci_facts.get(slot) or [])[:2]},
            )
    return ComparisonResult("negation", CMP_UNKNOWN)
