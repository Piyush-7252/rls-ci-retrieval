"""
Shared OpenSearch enrichment field builder.
============================================
Single source of truth for all ClinicalObject enrichment fields that must appear
in BOTH the ``ci-objects`` index and the ``semantic-objects`` (document) index.

How to add a new Knowledge Layer field
---------------------------------------
1. Add the extraction logic to ``shared/clinical_fact_extractor.enrich_object()``
   (and propagation to ``propagate_effective_facts`` if it supports inheritance).
2. Add it to ``ENRICHMENT_DEFAULTS`` below.
3. That is all — both the CI index and document index pick it up automatically.

Nothing else changes.  No manual bookkeeping across two Lambda files.
"""

from __future__ import annotations

from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Default values for every enrichment field.
# Used both as a fallback in build_enrichment_fields() and as documentation
# of the full enrichment schema.
# ─────────────────────────────────────────────────────────────────────────────

ENRICHMENT_DEFAULTS: dict[str, Any] = {
    # ── Structural classification ─────────────────────────────────────────────
    "study_context":       "GENERAL",   # CURRENT | HISTORICAL | CITED | GENERAL
    "statement_type":      "GENERAL",   # OBJECTIVE | ENDPOINT | ELIGIBILITY | …
    "object_subtype":      "GENERAL",   # INCLUSION | EXCLUSION | PRIMARY | SECONDARY | …
    "modality":            "GENERAL",   # OBJECTIVE | REQUIREMENT | ELIGIBILITY | …

    # ── Fact slots ────────────────────────────────────────────────────────────
    "facts":               {},          # raw slot→[values] from spaCy+rules
    "own_facts":           {},          # slots this object's own text contributes
    "effective_facts":     {},          # own_facts merged with inherited context
    "inherited_slots":     [],          # slot names whose values came from context
    "slot_provenance":     {},          # slot → "explicit" | "heading" | "paragraph"

    # ── Structural identity ───────────────────────────────────────────────────
    "study_hierarchy":     {},          # {study_part, phase, cohort, arm}
    "clinical_identity":   {},          # {disease, drug, endpoint, population}

    # ── Relations ─────────────────────────────────────────────────────────────
    "clinical_relations":  [],          # [{type, target, direction}]

    # ── Knowledge Layer ───────────────────────────────────────────────────────
    "negated_slots":       [],          # slot names asserted as absent/not-met
    "treatment_identity":  {},          # {primary_drug, companion_drugs, regimen, line_of_therapy}
    "endpoint_identity":   {},          # {endpoint, type, criteria, threshold, assessor, timepoint}
    "population_identity": {},          # {disease, relapsed, refractory, prior_lines, ecog_status}
    "temporal_context":    {},          # {anchor, cycle, day, week, window, timepoint}

    # ── Fingerprint ───────────────────────────────────────────────────────────
    "clinical_signature":  {},          # canonical {intent, drug, endpoint, population, phase, …}

    # ── Numeric / Statistical Identity ────────────────────────────────────────
    # Extracted deterministically by _extract_statistical_identity().
    # Used by the numeric retriever for structured-field matching instead of
    # semantic search for CI types where the number IS the confidential value.
    # Keys (all optional): sample_size, confidence_level, lower_ci, upper_ci,
    #   ci_unit, p_value, percentage, hazard_ratio, odds_ratio, median, median_unit
    "statistical_identity": {},
}


def build_enrichment_fields(obj: dict) -> dict:
    """
    Extract all ClinicalObject enrichment fields from an enriched object dict.

    Call this from BOTH the CI index handler and the document index handler to
    guarantee the same enrichment schema in both OpenSearch indices.

    ``obj`` may be a CI dict (from the NER Lambda) or a semantic-object dict
    (from the Document NER Lambda).  The function is schema-agnostic: it reads
    only the keys listed in ``ENRICHMENT_DEFAULTS`` and falls back to defaults
    for any that are absent.

    Returns a flat dict suitable for direct merge into an OpenSearch document::

        doc = {
            **ci_specific_fields(ci),
            **build_enrichment_fields(ci),
        }
    """
    # Use explicit None check: obj.get(field, default) returns None when the
    # key exists with value None (e.g. after partial enrichment that set a field
    # to None before the full pipeline ran).  Treat None the same as absent.
    return {
        field: (obj[field] if (field in obj and obj[field] is not None) else default)
        for field, default in ENRICHMENT_DEFAULTS.items()
    }
