"""Explicit OpenSearch mapping for the ci-objects index.

This is the single source of truth for the CI OpenSearch schema.
The sparse embedding is stored as JSON and is intentionally NOT mapped
as an arbitrary object, because sparse-vector token names would otherwise
become OpenSearch fields and cause mapping explosion.
"""

from __future__ import annotations


def _knn_field(dim: int = 1024) -> dict:
    return {
        "type": "knn_vector",
        "dimension": dim,
        "method": {
            "name": "hnsw",
            "engine": "faiss",
            "space_type": "innerproduct",
            "parameters": {
                "ef_construction": 128,
                "m": 16,
            },
        },
    }


CI_OBJECTS_MAPPING = {
    "settings": {
        "number_of_shards": 5,
        "number_of_replicas": 1,
        "index.knn": True,
        "refresh_interval": "30s",
        # Guardrail only; the schema below is deliberately bounded.
        "index.mapping.total_fields.limit": 2000,
    },
    "mappings": {
        "dynamic_templates": [
            {
                "strings_as_keyword": {
                    "match_mapping_type": "string",
                    "mapping": {"type": "keyword"},
                }
            }
        ],
        "properties": {
            # CI identity
            "ci_id": {"type": "keyword"},
            "known_ci": {"type": "keyword"},
            "category": {"type": "keyword"},
            "status": {"type": "keyword"},
            "assets": {"type": "object", "dynamic": False},

            # NLP
            "normalized_text": {
                "type": "text",
                "analyzer": "english",
            },
            "tokens": {"type": "keyword"},
            "entities": {
                "type": "nested",
                "properties": {
                    "text": {"type": "keyword"},
                    "label": {"type": "keyword"},
                    "sub_type": {"type": "keyword"},
                    "score": {"type": "float"},
                    "object_start": {"type": "integer"},
                    "object_end": {"type": "integer"},
                    "document_start": {"type": "integer"},
                    "document_end": {"type": "integer"},
                },
            },
            "ner_model": {"type": "keyword"},

            # Ontology
            "ontology_expansions": {
                "type": "object",
                "enabled": False,
            },
            "ontology_synonyms": {
                "type": "keyword",
                "index": False,
            },
            "regex_patterns": {
                "type": "keyword",
                "index": False,
            },

            # Embeddings
            "dense_vector": _knn_field(1024),
            "sparse_vector_json": {
                "type": "keyword",
                "index": False,
            },
            "embedding_model": {"type": "keyword"},

            # Shared ClinicalObject enrichment.
            "study_context": {"type": "keyword"},
            "statement_type": {"type": "keyword"},
            "object_subtype": {"type": "keyword"},
            "modality": {"type": "keyword"},
            "inherited_slots": {"type": "keyword"},
            "negated_slots": {"type": "keyword"},

            # Preserve arbitrary enrichment JSON without dynamically
            # generating fields from arbitrary keys.
            "facts": {"type": "object", "dynamic": False},
            "own_facts": {"type": "object", "dynamic": False},
            "effective_facts": {"type": "object", "dynamic": False},
            "slot_provenance": {"type": "object", "dynamic": False},
            "study_hierarchy": {"type": "object", "dynamic": False},
            "clinical_identity": {"type": "object", "dynamic": False},
            "clinical_relations": {"type": "object", "dynamic": False},
            "treatment_identity": {"type": "object", "dynamic": False},
            "endpoint_identity": {"type": "object", "dynamic": False},
            "population_identity": {"type": "object", "dynamic": False},
            "temporal_context": {"type": "object", "dynamic": False},
            "clinical_signature": {"type": "object", "dynamic": False},
            "statistical_identity": {"type": "object", "dynamic": False},

            # Tenant / project
            "tenant_id": {"type": "keyword"},
            "tenant_name": {"type": "keyword"},
            "tenant_schema": {"type": "keyword"},
        },
    },
}
