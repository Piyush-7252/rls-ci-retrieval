"""
Clinical comparator package.

Each comparator is a pure function  (ClinicalContext, ClinicalContext) → ComparisonResult.
Adding a new clinical dimension requires:
  1. Creating a new file in this package (e.g. dose.py) with a _compare_dose function.
  2. Importing it here and appending it to _COMPARATORS.
  Nothing else in the pipeline needs to change.

Public API:
  _COMPARATORS  — ordered tuple of all active comparator functions
  Everything from .base is re-exported for convenience.
"""
from .base import (
    _SEV_NONE, _SEV_LOW, _SEV_MEDIUM, _SEV_HIGH, _SEV_FATAL,
    _SEV_RANK, _RANK_SEV,
    CMP_MATCH, CMP_RELATED, CMP_SPECIALIZATION, CMP_GENERALIZATION,
    CMP_UNKNOWN, CMP_CONFLICT,
    ComparisonResult,
    ClinicalContext, _build_ci_context, _build_cand_context,
    _CONFLICT_METADATA, _CONFLICT_WEIGHTS,
    _DRUG_CONTRA_WEIGHTS, _MODALITY_GROUP, _TEMPORAL_FIELD_SEVERITY,
    _ENDPOINT_FAMILY, _ep_family,
)
from .drug       import _compare_drug
from .endpoint   import _compare_endpoint
from .population import _compare_population
from .phase      import _compare_phase
from .biomarker  import _compare_biomarker
from .study_arm  import _compare_study_arm
from .modality   import _compare_modality
from .regimen    import _compare_regimen
from .temporal   import _compare_temporal
from .negation   import _compare_negation
from .statistical import _compare_statistical

# Ordered registry — the orchestrator in lambda_function.py iterates this tuple.
# Append here to activate a new comparator; remove to disable one.
_COMPARATORS: tuple = (
    _compare_drug,
    _compare_endpoint,
    _compare_population,
    _compare_phase,
    _compare_biomarker,
    _compare_study_arm,
    _compare_modality,
    _compare_regimen,
    _compare_temporal,
    _compare_negation,
    _compare_statistical,
)

__all__ = [
    # Severity
    "_SEV_NONE", "_SEV_LOW", "_SEV_MEDIUM", "_SEV_HIGH", "_SEV_FATAL",
    "_SEV_RANK", "_RANK_SEV",
    # Outcomes
    "CMP_MATCH", "CMP_RELATED", "CMP_SPECIALIZATION", "CMP_GENERALIZATION",
    "CMP_UNKNOWN", "CMP_CONFLICT",
    # Types
    "ComparisonResult", "ClinicalContext",
    "_build_ci_context", "_build_cand_context",
    # Metadata
    "_CONFLICT_METADATA", "_CONFLICT_WEIGHTS",
    "_DRUG_CONTRA_WEIGHTS", "_MODALITY_GROUP", "_TEMPORAL_FIELD_SEVERITY",
    "_ENDPOINT_FAMILY", "_ep_family",
    # Comparators
    "_compare_drug", "_compare_endpoint", "_compare_population",
    "_compare_phase", "_compare_biomarker", "_compare_study_arm",
    "_compare_modality", "_compare_regimen", "_compare_temporal", "_compare_negation",
    "_compare_statistical",
    "_COMPARATORS",
]
