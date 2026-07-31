"""
clinical_enrichment_pipeline.py
────────────────────────────────
Centralized enrichment pipeline.  Both CIs and document objects flow through
this module and produce an identical ClinicalObject schema.

                    +────────────────────────────+
  Document ──────▶  │  ClinicalEnrichmentPipeline │
  CI ────────────▶  │                             │
                    +────────────────────────────+
                                 │
               ┌─────────────────────────────────┐
               │ facts              (GLiNER slots) │
               │ own_facts          (per-object)   │
               │ effective_facts    (+ inherited)  │
               │ inherited_slots    (audit trail)  │
               │ study_hierarchy    (structured)   │
               │ clinical_identity  (comparable)  │
               │ statement_type     (classified)  │
               │ study_context      (CURRENT/…)   │
               │ clinical_relations (spaCy verbs) │
               │ object_subtype     (filter key)  │
               └─────────────────────────────────┘

Behavioural difference between modes
──────────────────────────────────────
  CI (self-contained)
      A CI is a single, complete clinical statement.  It carries its own
      context — there is no parent heading or preceding paragraph to inherit
      from.  Therefore:

          own_facts == effective_facts == facts
          inherited_slots == []

      study_hierarchy and clinical_identity are still extracted from the CI's
      own text so the reranker can compare structured identities rather than
      raw text on both sides of the lookup.

  Document (multi-object chunk)
      A chunk contains many typed objects in document order:
      headings → paragraphs → sentences → table rows → list items.
      Context flows downward: a heading establishes the drug; the paragraph
      below it inherits that drug even if the paragraph text doesn't name it
      again; individual sentences inherit from their paragraph.

      effective_facts = ancestor_context ∪ own_facts  (own wins on conflict)
      inherited_slots = slots whose values came from an ancestor, not this
                        object's own text.  Used by the reranker to apply a
                        0.75 confidence discount vs 1.0 for explicit facts.

Public API
──────────
  enrich_ci(text, entities, section_category="", heading_path=None)
      → dict  full ClinicalObject enrichment for one CI

  enrich_document_objects(objects)
      → None  mutates objects in-place; applies full chunk-level inheritance
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def enrich_ci(
    text:             str,
    entities:         list[dict],
    section_category: str           = "",
    heading_path:     Optional[list] = None,
    asset:            Optional[dict] = None,
) -> dict:
    """
    Full ClinicalObject enrichment for a single CI.

    A CI is self-contained, so own_facts == effective_facts — no inheritance.
    study_hierarchy and clinical_identity are still produced so the reranker
    compares two ClinicalObjects with identical schemas rather than a CI against
    a document object.

    Parameters
    ----------
    text             : normalized CI text (from normalization stage)
    entities         : NER entities; must carry ``object_start`` / ``object_end``
                       character offsets (CI NER adapts start/end → object_start/end)
    section_category : canonical section label (rarely set for CIs; defaults to "")
    heading_path     : heading breadcrumb list (rarely set for CIs; defaults to [])
    asset            : optional CI asset metadata dict; when population_identity
                       finds no disease in the CI text, the asset's indication /
                       therapeuticArea / disease field is used as a fallback.

    Returns
    -------
    dict with all ClinicalObject fields:
        facts, own_facts, effective_facts, inherited_slots,
        study_hierarchy, clinical_identity,
        statement_type, study_context, clinical_relations, object_subtype
    """
    from shared.clinical_fact_extractor import (
        enrich_object,
        _extract_study_hierarchy,
        _build_clinical_identity,
    )

    heading_path = heading_path or []

    # ── Stage 1 ── base enrichment (facts, statement_type, study_context,
    #               clinical_relations, object_subtype)
    base = enrich_object(text, entities, section_category, heading_path)

    # ── Stage 1b ── population disease from asset (fallback)
    # When the CI text itself doesn't mention the disease (common for objective
    # sections that reference the drug/endpoint but not the indication), inherit
    # from asset.indication / therapeuticArea so population comparisons work.
    if asset and not (base["population_identity"] or {}).get("disease"):
        for _field in ("indication", "therapeuticArea", "disease", "condition"):
            _val = asset.get(_field)
            if _val and isinstance(_val, str):
                base["population_identity"] = {
                    **base["population_identity"],
                    "disease":  _val,
                    "_source": "asset",
                }
                break

    own_facts = base["facts"]

    # ── Stage 2 ── self-contained propagation
    #   CI has no parent context → own == effective, nothing inherited
    effective_facts: dict[str, list] = {k: list(v) for k, v in own_facts.items() if v}

    # Promote population_identity.disease into effective_facts.
    # Disease is inferred from text patterns (e.g. IMWG → MM) not from NER
    # entity labels, so it never flows through the entity→slot pipeline.
    # Promoting it here makes it available to _build_clinical_identity and
    # to comparators that read effective_facts directly.
    _disease = (base.get("population_identity") or {}).get("disease")
    if _disease and "disease" not in effective_facts:
        effective_facts["disease"] = [_disease]
    inherited_slots: list[str]       = []
    # For a CI all slots come directly from its own text — nothing is inherited
    slot_provenance: dict[str, str]  = {slot: "explicit" for slot in effective_facts}

    # ── Stage 2b ── deterministic relation fallback
    # When spaCy was unavailable or produced no relations but the statement type
    # implies a drug→endpoint measurement intent (objective / efficacy sections),
    # emit a single inferred relation so downstream comparators have something
    # to work with.  Confidence 0.50 signals it was not dep-parse derived.
    if not base.get("clinical_relations"):
        _stmt = base.get("statement_type", "")
        if _stmt in {"PRIMARY_OBJECTIVE", "SECONDARY_OBJECTIVE", "EXPLORATORY",
                     "EFFICACY", "PHARMACOKINETICS", "PHARMACODYNAMICS"}:
            _drugs = effective_facts.get("drug",     [])
            _eps   = effective_facts.get("endpoint", [])
            if _drugs and _eps:
                base["clinical_relations"] = [{
                    "drug":       _drugs[0],
                    "endpoint":   _eps[0],
                    "relation":   "measured_by",
                    "verb":       "_inferred",
                    "confidence": 0.50,
                }]

    # ── Stage 3 ── structural decomposition of the CI's own text
    study_hierarchy = _extract_study_hierarchy(heading_path, effective_facts, text)

    # ── Stage 4 ── canonical identity (structured comparison target for reranker)
    clinical_identity = _build_clinical_identity(
        effective_facts, study_hierarchy, section_category,
        base.get("treatment_identity"),
        base.get("endpoint_identity"),
    )

    return {
        **base,
        "own_facts":         own_facts,
        "effective_facts":   effective_facts,
        "inherited_slots":   inherited_slots,
        "slot_provenance":   slot_provenance,
        "study_hierarchy":   study_hierarchy,
        "clinical_identity": clinical_identity,
    }


def enrich_document_objects(objects: list) -> None:
    """
    Apply full multi-object context inheritance to a document chunk's objects.

    Wrapper over ``propagate_effective_facts`` that makes the pipeline API
    symmetric: ``enrich_ci`` and ``enrich_document_objects`` are the two entry
    points into the same enrichment contract — one per source type.

    Must be called AFTER ``enrich_object`` has populated ``facts`` on every
    object in the chunk.  Mutates objects in-place.

    Inheritance model (document order within the chunk):
        heading   → partial context reset; propagates own facts downward
        paragraph → inherits context; contributes own facts upward
        list      → same as paragraph
        table     → same as paragraph
        table_row → inherits context; does NOT contribute upward (too granular)
        sentence  → inherits context; does NOT contribute upward
    """
    from shared.clinical_fact_extractor import propagate_effective_facts
    propagate_effective_facts(objects)
