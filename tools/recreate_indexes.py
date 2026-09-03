"""
Recreate ci-objects, document-chunks and semantic-objects OpenSearch indexes from scratch.

This script:
  1. Reads the current document counts per index so you know what will be lost.
  2. Asks for confirmation (skip with --yes).
  3. Deletes document-chunks and semantic-objects.
  4. Recreates both with the correct, fully explicit field mappings.
  5. Prints the dispatch commands you need to run to re-index every document.

Usage:
    export OPENSEARCH_ENDPOINT=search-rls-dev-rhitzxwnctmuyq2l4kny5kwelu.eu-west-1.es.amazonaws.com
    python tools/recreate_indexes.py [--ci-only] [--yes] [--region us-east-1]

Flags:
    --yes       Skip the confirmation prompt.
    --region    AWS region (default: eu-west-1).
    --chunks-index    Name of the chunks index (default: document-chunks).
    --objects-index   Name of the objects index (default: semantic-objects).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ci_objects_mapping import CI_OBJECTS_MAPPING

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

# ── AWS / OpenSearch connection ───────────────────────────────────────────────

def _get_os(endpoint: str, region: str):
    import boto3
    from opensearchpy import OpenSearch, RequestsHttpConnection
    from requests_aws4auth import AWS4Auth

    frozen  = boto3.Session().get_credentials().get_frozen_credentials()
    awsauth = AWS4Auth(
        frozen.access_key, frozen.secret_key, region, "es",
        session_token=frozen.token,
    )
    return OpenSearch(
        hosts=[{"host": endpoint, "port": 443}],
        http_auth=awsauth, use_ssl=True, verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=60,
    )


# ── Mappings (single source of truth — mirrors the create_*_index.py scripts) ─

def _chunks_mapping(n_shards: int = 5) -> dict:
    return {
        "settings": {
            "number_of_shards":   n_shards,
            "number_of_replicas": 0,
            "index.knn":          True,
            "refresh_interval":   "30s",
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
                # Identity
                "document_id":          {"type": "keyword"},
                "tenant_id":            {"type": "keyword"},
                "tenant_schema":        {"type": "keyword"},
                "tenant_name":          {"type": "keyword"},
                "project_id":           {"type": "keyword"},
                "chunk_id":             {"type": "keyword"},
                "chunk_idx":            {"type": "integer"},
                "parent_chunk_idx":     {"type": "integer"},
                "prev_chunk_idx":       {"type": "integer"},
                "next_chunk_idx":       {"type": "integer"},
                "page_start":           {"type": "integer"},
                "page_end":             {"type": "integer"},
                # Full-text
                "raw_text": {
                    "type": "text", "analyzer": "english",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 512}},
                },
                "normalized_text":      {"type": "text", "analyzer": "english"},
                "tokens":               {"type": "keyword"},
                # NER entities
                "entities": {
                    "type": "nested",
                    "properties": {
                        "text":           {"type": "keyword"},
                        "label":          {"type": "keyword"},
                        "sub_type":       {"type": "keyword"},
                        "score":          {"type": "float"},
                        "object_start":   {"type": "integer"},
                        "object_end":     {"type": "integer"},
                        "document_start": {"type": "integer"},
                        "document_end":   {"type": "integer"},
                    }
                },
                # Ontology
                "ontology_expansions":  {"type": "object", "enabled": False},
                "ontology_synonyms":    {"type": "keyword", "index": False},
                # Vectors — dense_vector is knn-searched; heading is stored only
                "dense_vector":         _knn_field(1024),
                "heading_dense_vector": {"type": "float", "index": False},
                "sparse_vector_json":   {"type": "keyword", "index": False},
                # Metadata
                "embedding_model":      {"type": "keyword"},
                "geometry":             {
                    "type": "object",
                    "properties": {
                        "geometry_source":    {"type": "keyword"},
                        "geometry_precision": {"type": "keyword"},
                        "is_authoritative":   {"type": "boolean"},
                        "rects":              {"type": "float", "index": False},
                        "bbox":               {"type": "float", "index": False},
                        "object_type":        {"type": "keyword"},
                        "source_object_id":   {"type": "keyword"},
                        "page":               {"type": "integer"},

                        "page_distribution": {
                            "type": "nested",
                            "properties": {
                                "page": {
                                    "type": "integer"
                                },

                                # Exact portion of the semantic candidate
                                # belonging to this page. This is the source
                                # text used by the text-search highlight mode.
                                "text": {
                                    "type": "text",
                                    "analyzer": "english",
                                    "fields": {
                                        "keyword": {
                                            "type": "keyword",
                                            "ignore_above": 512
                                        }
                                    }
                                },

                                "rects": {
                                    "type": "float",
                                    "index": False
                                },
                                "paragraph_bbox": {
                                    "type": "float",
                                    "index": False,
                                },
                                "bbox": {
                                    "type": "float",
                                    "index": False
                                },
                                "geometry_source": {
                                    "type": "keyword"
                                },
                                "geometry_precision": {
                                    "type": "keyword"
                                },
                                "is_authoritative": {
                                    "type": "boolean"
                                },
                                "source_object_id": {
                                    "type": "keyword"
                                },
                                "source_span_ids": {
                                    "type": "keyword"
                                },

                                # Native Apryse spans contributing to the
                                # semantic sentence/object on this page.
                                # For containing geometry we intentionally keep
                                # the complete native span text + rect even when
                                # only part of the span belongs to the sentence.
                                "contributing_spans": {
                                    "type": "nested",
                                    "properties": {
                                        "text": {
                                            "type": "text",
                                            "analyzer": "english",
                                            "fields": {
                                                "keyword": {
                                                    "type": "keyword",
                                                    "ignore_above": 512
                                                }
                                            }
                                        },
                                        "rect": {
                                            "type": "float",
                                            "index": False
                                        },
                                        "span_id": {
                                            "type": "keyword"
                                        },
                                        "page": {
                                            "type": "integer"
                                        },
                                        "source_object_id": {
                                            "type": "keyword"
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
            }
        },
    }


def _knn_field(dim: int) -> dict:
    return {
        "type":      "knn_vector",
        "dimension": dim,
        "method": {
            "name":       "hnsw",
            "engine":     "faiss",
            "space_type": "innerproduct",
            "parameters": {"ef_construction": 128, "m": 16},
        },
    }


def _objects_mapping() -> dict:
    return {
        "settings": {
            "number_of_shards":   5,
            "number_of_replicas": 0,
            "index.knn":          True,
            "refresh_interval":   "30s",
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
                # Identity
                "object_id":         {"type": "keyword"},
                "document_id":       {"type": "keyword"},
                "tenant_id":         {"type": "keyword"},
                "tenant_schema":     {"type": "keyword"},
                "tenant_name":       {"type": "keyword"},
                "project_id":        {"type": "keyword"},
                "parent_chunk_id":   {"type": "keyword"},
                "position":          {"type": "integer"},
                "global_position":   {"type": "integer"},
                "type":              {"type": "keyword"},
                # Full-text
                "text": {
                    "type": "text", "analyzer": "english",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 512}},
                },
                "normalized_text":   {"type": "text", "analyzer": "english"},
                # Section context
                "section":           {"type": "keyword"},
                "section_number":    {"type": "keyword"},
                "section_depth":     {"type": "integer"},
                "section_level":     {"type": "integer"},
                "section_category":  {"type": "keyword"},
                "heading_path":      {"type": "keyword"},
                "semantic_path":     {"type": "keyword"},
                "section_confidence": {"type": "float"},
                "document_position": {"type": "integer"},
                "chunk_idx":         {"type": "integer"},
                "parent_chunk_idx":  {"type": "integer"},
                "prev_chunk_idx":    {"type": "integer"},
                "next_chunk_idx":    {"type": "integer"},
                "category":          {"type": "keyword"},
                "boost_weight":      {"type": "float"},
                "indexable":         {"type": "boolean"},
                "parent_heading":    {"type": "keyword"},
                "prev_object_pos":   {"type": "integer"},
                "next_object_pos":   {"type": "integer"},
                # Vectors — both fields are knn-searched in vector_retriever
                "dense_vector":         _knn_field(1024),
                "heading_dense_vector": _knn_field(1024),
                # NER entities
                "entities": {
                    "type": "nested",
                    "properties": {
                        "text":           {"type": "keyword"},
                        "label":          {"type": "keyword"},
                        "sub_type":       {"type": "keyword"},
                        "score":          {"type": "float"},
                        "object_start":   {"type": "integer"},
                        "object_end":     {"type": "integer"},
                        "document_start": {"type": "integer"},
                        "document_end":   {"type": "integer"},
                    }
                },
                # ClinicalObject enrichment fields
                "study_context":       {"type": "keyword"},
                "statement_type":      {"type": "keyword"},
                "object_subtype":      {"type": "keyword"},
                "modality":            {"type": "keyword"},
                "inherited_slots":     {"type": "keyword"},
                "negated_slots":       {"type": "keyword"},
                "facts":               {"type": "object"},
                "own_facts":           {"type": "object"},
                "effective_facts":     {"type": "object"},
                "slot_provenance":     {"type": "object"},
                "study_hierarchy":     {"type": "object"},
                "clinical_identity":   {"type": "object"},
                "clinical_relations":  {"type": "object"},
                "treatment_identity":  {"type": "object"},
                "endpoint_identity":   {"type": "object"},
                "population_identity": {"type": "object"},
                "temporal_context":    {"type": "object"},
                "clinical_signature":  {"type": "object"},
                "statistical_identity": {"type": "object"},
                # Canonical candidate geometry — produced upstream during
                # extraction/chunk construction. Indexing stores it verbatim;
                # it does not infer or reinterpret geometry.
                #
                # Geometry supports three downstream highlight modes:
                #
                #   1. paragraph:
                #        geometry.page_distribution[].bbox
                #
                #   2. span:
                #        geometry.page_distribution[].contributing_spans[].rect
                #
                #   3. text_search:
                #        geometry.page_distribution[].text
                #        (search the text on each page independently; do not
                #         use stored geometry for the actual highlight)
                #
                # page_distribution is nested because page-local text and
                # contributing spans must remain associated with their page.
                "geometry": {
                    "type": "object",
                    "properties": {
                        "geometry_source":    {"type": "keyword"},
                        "geometry_precision": {"type": "keyword"},
                        "is_authoritative":   {"type": "boolean"},
                        "rects":              {"type": "float", "index": False},
                        "paragraph_bbox":     {"type": "float", "index": False},
                        "bbox":               {"type": "float", "index": False},
                        "object_type":        {"type": "keyword"},
                        "source_object_id":   {"type": "keyword"},
                        "page":               {"type": "integer"},

                        "page_distribution": {
                            "type": "nested",
                            "properties": {
                                "page": {
                                    "type": "integer"
                                },

                                # Exact portion of the semantic candidate
                                # belonging to this page. This is the source
                                # text used by the text-search highlight mode.
                                "text": {
                                    "type": "text",
                                    "analyzer": "english",
                                    "fields": {
                                        "keyword": {
                                            "type": "keyword",
                                            "ignore_above": 512
                                        }
                                    }
                                },

                                "rects": {
                                    "type": "float",
                                    "index": False
                                },
                                "bbox": {
                                    "type": "float",
                                    "index": False
                                },
                                "geometry_source": {
                                    "type": "keyword"
                                },
                                "geometry_precision": {
                                    "type": "keyword"
                                },
                                "is_authoritative": {
                                    "type": "boolean"
                                },
                                "source_object_id": {
                                    "type": "keyword"
                                },
                                "source_span_ids": {
                                    "type": "keyword"
                                },

                                # Native Apryse spans contributing to the
                                # semantic sentence/object on this page.
                                # For containing geometry we intentionally keep
                                # the complete native span text + rect even when
                                # only part of the span belongs to the sentence.
                                "contributing_spans": {
                                    "type": "nested",
                                    "properties": {
                                        "text": {
                                            "type": "text",
                                            "analyzer": "english",
                                            "fields": {
                                                "keyword": {
                                                    "type": "keyword",
                                                    "ignore_above": 512
                                                }
                                            }
                                        },
                                        "rect": {
                                            "type": "float",
                                            "index": False
                                        },
                                        "span_id": {
                                            "type": "keyword"
                                        },
                                        "page": {
                                            "type": "integer"
                                        },
                                        "source_object_id": {
                                            "type": "keyword"
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                # Sentence-specific fields
                "parent_object_id":    {"type": "keyword"},
                "prev_sentence_id":    {"type": "keyword"},
                "next_sentence_id":    {"type": "keyword"},
                "paragraph_text":      {"type": "text", "analyzer": "english"},
                "prev_sentence_text":  {"type": "text", "analyzer": "english"},
                "next_sentence_text":  {"type": "text", "analyzer": "english"},
                # Display-only / embedding units. Geometry is deliberately absent.
                "page":                {"type": "integer"},
                "bbox":                {"type": "float", "index": False},
                "list_id":             {"type": "keyword"},
                "list_level":          {"type": "integer"},
                "list_label":          {"type": "keyword"},
                "list_number_format":  {"type": "keyword"},
                "table_id":            {"type": "keyword"},
                "cell_id":             {"type": "keyword"},
                "table_role":          {"type": "keyword"},
                "row_index":           {"type": "integer"},
                "row_start":           {"type": "integer"},
                "col_start":           {"type": "integer"},
                "row_span":            {"type": "integer"},
                "col_span":            {"type": "integer"},
                "display_spans": {
                    "type": "nested",
                    "properties": {
                        "type":      {"type": "keyword"},
                        "text":      {"type": "text", "analyzer": "english"},
                        "geometry": {
                            "type": "object",
                            "properties": {
                                "geometry_source": {"type": "keyword"},
                                "geometry_precision": {"type": "keyword"},
                                "is_authoritative": {"type": "boolean"},
                                "rects": {"type": "float", "index": False},
                                "bbox": {"type": "float", "index": False},
                                "object_type": {"type": "keyword"},
                                "source_object_id": {"type": "keyword"},
                                "page": {"type": "integer"},
                                "page_distribution": {
                                    "type": "nested",
                                    "properties": {
                                        "page": {"type": "integer"},
                                        "text": {"type": "text", "analyzer": "english"},
                                        "rects": {"type": "float", "index": False},
                                        "bbox": {"type": "float", "index": False},
                                        "geometry_source": {"type": "keyword"},
                                        "geometry_precision": {"type": "keyword"},
                                        "is_authoritative": {"type": "boolean"},
                                        "source_object_id": {"type": "keyword"},
                                        "source_span_ids": {"type": "keyword"},
                                        "contributing_spans": {
                                            "type": "nested",
                                            "properties": {
                                                "text": {"type": "text", "analyzer": "english"},
                                                "rect": {"type": "float", "index": False},
                                                "span_id": {"type": "keyword"},
                                                "page": {"type": "integer"},
                                                "source_object_id": {"type": "keyword"},
                                            },
                                        },
                                    },
                                },
                            },
                        },
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": 1024,
                            "index": True,
                            "space_type": "cosinesimil",
                            "method": {"name": "hnsw", "engine": "faiss"}
                        }
                    }
                },
            }
        },
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _doc_counts_per_document(client, index: str) -> dict[str, int]:
    """Return {document_id: chunk_count} for every document in the index.

    Falls back to document_id.keyword when document_id is a text field
    (the auto-inferred mapping bug we are here to fix).
    """
    if not client.indices.exists(index=index):
        return {}

    def _agg(field: str) -> dict[str, int]:
        body = {
            "size": 0,
            "aggs": {
                "by_document": {
                    "terms": {"field": field, "size": 500}
                }
            }
        }
        resp = client.search(index=index, body=body)
        return {
            b["key"]: b["doc_count"]
            for b in resp["aggregations"]["by_document"]["buckets"]
        }

    try:
        return _agg("document_id")
    except Exception:
        # document_id is a text field — use .keyword sub-field instead
        return _agg("document_id.keyword")


def main(
    endpoint: str,
    region: str,
    chunks_index: str,
    objects_index: str,
    ci_index: str,
    yes: bool,
    chunks_shards: int,
    ci_only: bool,
) -> None:
    client = _get_os(endpoint, region)

    if ci_only:
        print(f"\\n{'─' * 70}")
        print(f"  CI-only OpenSearch recreation: {endpoint}")
        print(f"{'─' * 70}\\n")

        ci_exists = client.indices.exists(index=ci_index)
        ci_count = client.count(index=ci_index)["count"] if ci_exists else 0
        print(f"Existing {ci_index} documents: {ci_count:,}")

        if not yes:
            answer = input(
                f"Delete and recreate '{ci_index}' ONLY? "
                f"'{chunks_index}' and '{objects_index}' will NOT be touched. [yes/N] "
            ).strip().lower()
            if answer != "yes":
                print("Aborted.")
                sys.exit(0)

        if ci_exists:
            print(f"  Deleting {ci_index} …", end=" ", flush=True)
            client.indices.delete(index=ci_index)
            print("deleted")
        else:
            print(f"  {ci_index} does not exist — skipping delete")

        print(f"  Creating {ci_index} …", end=" ", flush=True)
        resp = client.indices.create(index=ci_index, body=CI_OBJECTS_MAPPING)
        print(f"ok  (acknowledged={resp.get('acknowledged')})")

        ci_mapping = client.indices.get_mapping(index=ci_index)[ci_index]["mappings"]
        ci_props = ci_mapping.get("properties", {})
        ci_sparse = ci_props.get("sparse_vector_json", {})
        ci_dense = ci_props.get("dense_vector", {})
        ci_templates = ci_mapping.get("dynamic_templates", [])

        ci_ok = (
            ci_props.get("ci_id", {}).get("type") == "keyword"
            and ci_dense.get("type") == "knn_vector"
            and ci_dense.get("dimension") == 1024
            and ci_sparse == {"type": "keyword", "index": False}
            and "sparse_vector" not in ci_props
            and any("strings_as_keyword" in t for t in ci_templates)
        )

        print(
            f"  [{'✓' if ci_ok else '✗'}] {ci_index}  "
            f"ci_id={ci_props.get('ci_id', {}).get('type', 'MISSING')}  "
            f"dense_vector={ci_dense.get('type', 'MISSING')}  "
            f"sparse_vector_json="
            f"{'keyword/index:false' if ci_sparse == {'type': 'keyword', 'index': False} else 'MISSING/WRONG'}"
        )

        if not ci_ok:
            raise RuntimeError("ci-objects mapping verification failed")

        print(f"\\nDone. ONLY '{ci_index}' was deleted and recreated.")
        print(f"'{chunks_index}' and '{objects_index}' were not modified.")
        return

    # ── 1. Show what is currently in both indexes ─────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  OpenSearch: {endpoint}")
    print(f"{'─' * 70}\n")

    chunks_counts  = _doc_counts_per_document(client, chunks_index)
    objects_counts = _doc_counts_per_document(client, objects_index)
    ci_count = client.count(index=ci_index)["count"] if client.indices.exists(index=ci_index) else 0

    # Merge all document_ids seen in either document index.
    all_doc_ids = sorted(set(chunks_counts) | set(objects_counts))

    if all_doc_ids:
        print(f"{'Document ID':<70}  {'chunks':>8}  {'objects':>9}")
        print(f"{'─' * 70}  {'─' * 8}  {'─' * 9}")
        for doc_id in all_doc_ids:
            c = chunks_counts.get(doc_id, 0)
            o = objects_counts.get(doc_id, 0)
            print(f"{doc_id:<70}  {c:>8,}  {o:>9,}")
        print(f"{'─' * 70}  {'─' * 8}  {'─' * 9}")
        print(
            f"{'TOTAL':<70}  "
            f"{sum(chunks_counts.values()):>8,}  "
            f"{sum(objects_counts.values()):>9,}"
        )
    else:
        print("Both document indexes are empty — nothing to lose.")

    print(f"Existing {ci_index} documents: {ci_count:,}")
    print()

    # ── 2. Confirm ────────────────────────────────────────────────────────────
    if not yes:
        answer = input(
            f"Delete and recreate '{ci_index}', '{chunks_index}' and '{objects_index}'? "
            f"All data above will be PERMANENTLY deleted. [yes/N] "
        ).strip().lower()
        if answer != "yes":
            print("Aborted.")
            sys.exit(0)

    # ── 3. Delete indexes ─────────────────────────────────────────────────────
    for idx in (ci_index, chunks_index, objects_index):
        if client.indices.exists(index=idx):
            print(f"  Deleting {idx} …", end=" ", flush=True)
            client.indices.delete(index=idx)
            print("deleted")
        else:
            print(f"  {idx} does not exist — skipping delete")

    # ── 4. Recreate with explicit mappings ────────────────────────────────────
    print(f"\n  Creating {ci_index} …", end=" ", flush=True)
    resp = client.indices.create(index=ci_index, body=CI_OBJECTS_MAPPING)
    print(f"ok  (acknowledged={resp.get('acknowledged')})")

    print(f"\n  Creating {chunks_index} …", end=" ", flush=True)
    resp = client.indices.create(index=chunks_index, body=_chunks_mapping(chunks_shards))
    print(f"ok  (acknowledged={resp.get('acknowledged')})")

    print(f"  Creating {objects_index} …", end=" ", flush=True)
    resp = client.indices.create(index=objects_index, body=_objects_mapping())
    print(f"ok  (acknowledged={resp.get('acknowledged')})")

    # ── 5. Verify new mappings ────────────────────────────────────────────────
    print()

    ci_mapping = client.indices.get_mapping(index=ci_index)[ci_index]["mappings"]
    ci_props = ci_mapping.get("properties", {})
    ci_sparse = ci_props.get("sparse_vector_json", {})
    ci_dense = ci_props.get("dense_vector", {})
    ci_templates = ci_mapping.get("dynamic_templates", [])

    ci_ok = (
        ci_props.get("ci_id", {}).get("type") == "keyword"
        and ci_dense.get("type") == "knn_vector"
        and ci_dense.get("dimension") == 1024
        and ci_sparse == {"type": "keyword", "index": False}
        and not any(
            key.startswith("sparse_vector") and key != "sparse_vector_json"
            for key in ci_props
        )
        and any("strings_as_keyword" in t for t in ci_templates)
    )

    print(
        f"  [{'✓' if ci_ok else '✗'}] {ci_index}  "
        f"ci_id={ci_props.get('ci_id', {}).get('type', 'MISSING')}  "
        f"dense_vector={ci_dense.get('type', 'MISSING')}  "
        f"sparse_vector_json="
        f"{'keyword/index:false' if ci_sparse == {'type': 'keyword', 'index': False} else 'MISSING/WRONG'}"
    )

    for idx in (chunks_index, objects_index):
        mapping = client.indices.get_mapping(index=idx)
        doc_id_type = mapping[idx]["mappings"]["properties"]["document_id"]["type"]
        templates = mapping[idx]["mappings"].get("dynamic_templates", [])
        has_template = any(
            "strings_as_keyword" in t for t in templates
        )
        props = mapping[idx]["mappings"].get("properties", {})
        geometry = props.get("geometry", {})
        geometry_props = (
            geometry.get("properties", {})
            if geometry.get("type") == "object"
            else {}
        )
        required_geometry = {
            "geometry_source",
            "geometry_precision",
            "is_authoritative",
            "rects",
            "bbox",
            "object_type",
            "page_distribution",
        }

        page_distribution = geometry_props.get("page_distribution", {})
        page_distribution_props = (
            page_distribution.get("properties", {})
            if page_distribution.get("type") == "nested"
            else {}
        )

        required_page_distribution = {
            "page",
            "text",
            "rects",
            "bbox",
            "paragraph_bbox",
            "geometry_source",
            "geometry_precision",
            "is_authoritative",
            "source_object_id",
            "contributing_spans",
        }

        contributing_spans = page_distribution_props.get("contributing_spans", {})
        contributing_span_props = (
            contributing_spans.get("properties", {})
            if contributing_spans.get("type") == "nested"
            else {}
        )

        required_contributing_spans = {
            "text",
            "rect",
            "span_id",
        }

        has_geometry = (
            required_geometry.issubset(geometry_props)
            and page_distribution.get("type") == "nested"
            and required_page_distribution.issubset(page_distribution_props)
            and contributing_spans.get("type") == "nested"
            and required_contributing_spans.issubset(contributing_span_props)
        )

        status = (
            "✓"
            if doc_id_type == "keyword"
            and has_template
            and (idx != objects_index or has_geometry)
            else "✗"
        )
        geometry_status = (
            "present"
            if idx != objects_index or has_geometry
            else "MISSING"
        )
        print(
            f"  [{status}] {idx}  document_id={doc_id_type}  "
            f"dynamic_template={'keyword' if has_template else 'MISSING'}  "
            f"geometry={geometry_status}"
        )

    # ── 6. Print re-dispatch commands ─────────────────────────────────────────
    if all_doc_ids:
        print(f"\n{'─' * 70}")
        print("  Re-dispatch commands (run these to re-index all documents):")
        print(f"{'─' * 70}\n")
        queue_url = "https://sqs.eu-west-1.amazonaws.com/064051750322/rls-ci-retrieval-document-chunk-worker-queue"
        for doc_id in all_doc_ids:
            print(
                f"python3.12 tools/dispatch_chunks_to_sqs.py \\\n"
                f"  --document-id '{doc_id}' \\\n"
                f"  --cache-path '.cache/{doc_id}/full_tables.json' \\\n"
                f"  --chunks-cache-dir '.cache' \\\n"
                f"  --queue-url '{queue_url}' \\\n"
                f"  --region {region}\n"
            )

    print("\nDone. Indexes are empty and ready for fresh indexing.")
    print("Canonical candidate geometry mapping:")
    print("  geometry.{geometry_source,geometry_precision,is_authoritative,rects,bbox,object_type}")
    print("  geometry.page_distribution[]")
    print("  geometry.page_distribution[].text")
    print("  geometry.page_distribution[].contributing_spans[]")
    print("Geometry is produced upstream by extraction/chunk construction;")
    print("the indexer should only copy those fields and must not infer geometry.")
    print("Highlight modes:")
    print("  paragraph  -> page_distribution[].bbox")
    print("  span       -> page_distribution[].contributing_spans[].rect")
    print("  text_search -> page_distribution[].text, searched per page")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Drop and recreate OpenSearch indexes; use --ci-only for CI only."
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the confirmation prompt",
    )
    parser.add_argument(
        "--region", default="us-east-1",
        help="AWS region (default: us-east-1)",
    )
    parser.add_argument(
        "--chunks-index", default="document-chunks",
        help="Name of the chunks index (default: document-chunks)",
    )
    parser.add_argument(
        "--objects-index", default="semantic-objects",
        help="Name of the objects index (default: semantic-objects)",
    )
    parser.add_argument(
        "--ci-index", default="ci-objects",
        help="Name of the CI index (default: ci-objects)",
    )
    parser.add_argument(
        "--ci-only",
        action="store_true",
        help="Delete and recreate only ci-objects; never touch document indexes",
    )
    parser.add_argument(
        "--chunks-shards", type=int, default=5,
        help="Number of shards for document-chunks (default: 5)",
    )
    args = parser.parse_args()

    endpoint = os.environ.get(
        "OPENSEARCH_ENDPOINT",
        "search-rls-qa-u7jwn3q2hr3hxp7y2ydab34tfq.us-east-1.es.amazonaws.com",
    )
    if not endpoint:
        print("ERROR: OPENSEARCH_ENDPOINT env var not set", file=sys.stderr)
        sys.exit(1)

    main(
        endpoint      = endpoint,
        region        = args.region,
        chunks_index  = args.chunks_index,
        objects_index = args.objects_index,
        ci_index      = args.ci_index,
        yes           = args.yes,
        chunks_shards = args.chunks_shards,
        ci_only       = args.ci_only,
    )
