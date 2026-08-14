"""
Recreate document-chunks and semantic-objects OpenSearch indexes from scratch.

This script:
  1. Reads the current document counts per index so you know what will be lost.
  2. Asks for confirmation (skip with --yes).
  3. Deletes document-chunks and semantic-objects.
  4. Recreates both with the correct, fully explicit field mappings.
  5. Prints the dispatch commands you need to run to re-index every document.

Usage:
    export OPENSEARCH_ENDPOINT=search-rls-dev-rhitzxwnctmuyq2l4kny5kwelu.eu-west-1.es.amazonaws.com
    python tools/recreate_indexes.py [--yes] [--region eu-west-1]

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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
                # Sentence-specific fields
                "parent_object_id":    {"type": "keyword"},
                "char_start":          {"type": "integer"},
                "char_end":            {"type": "integer"},
                "prev_sentence_id":    {"type": "keyword"},
                "next_sentence_id":    {"type": "keyword"},
                "paragraph_text":      {"type": "text", "analyzer": "english"},
                "prev_sentence_text":  {"type": "text", "analyzer": "english"},
                "next_sentence_text":  {"type": "text", "analyzer": "english"},
                # Display-only (never embedded)
                "page":  {"type": "integer"},
                "bbox":  {"type": "float", "index": False},
                "display_spans": {
                    "type": "nested",
                    "properties": {
                        "type":  {"type": "keyword"},
                        "text":  {"type": "text", "analyzer": "english"},
                        "start": {"type": "integer"},
                        "end":   {"type": "integer"},
                        "bbox":  {"type": "float", "index": False},
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


def _total_count(client, index: str) -> int:
    if not client.indices.exists(index=index):
        return 0
    return client.count(index=index)["count"]


def main(
    endpoint: str,
    region: str,
    chunks_index: str,
    objects_index: str,
    yes: bool,
    chunks_shards: int,
) -> None:
    client = _get_os(endpoint, region)

    # ── 1. Show what is currently in both indexes ─────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  OpenSearch: {endpoint}")
    print(f"{'─' * 70}\n")

    chunks_counts  = _doc_counts_per_document(client, chunks_index)
    objects_counts = _doc_counts_per_document(client, objects_index)

    # Merge all document_ids seen in either index
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
        print("Both indexes are empty — nothing to lose.")

    print()

    # ── 2. Confirm ────────────────────────────────────────────────────────────
    if not yes:
        answer = input(
            f"Delete and recreate '{chunks_index}' and '{objects_index}'? "
            f"All data above will be PERMANENTLY deleted. [yes/N] "
        ).strip().lower()
        if answer != "yes":
            print("Aborted.")
            sys.exit(0)

    # ── 3. Delete indexes ─────────────────────────────────────────────────────
    for idx in (chunks_index, objects_index):
        if client.indices.exists(index=idx):
            print(f"  Deleting {idx} …", end=" ", flush=True)
            client.indices.delete(index=idx)
            print("deleted")
        else:
            print(f"  {idx} does not exist — skipping delete")

    # ── 4. Recreate with explicit mappings ────────────────────────────────────
    print(f"\n  Creating {chunks_index} …", end=" ", flush=True)
    resp = client.indices.create(index=chunks_index, body=_chunks_mapping(chunks_shards))
    print(f"ok  (acknowledged={resp.get('acknowledged')})")

    print(f"  Creating {objects_index} …", end=" ", flush=True)
    resp = client.indices.create(index=objects_index, body=_objects_mapping())
    print(f"ok  (acknowledged={resp.get('acknowledged')})")

    # ── 5. Verify new mappings ────────────────────────────────────────────────
    print()
    for idx in (chunks_index, objects_index):
        mapping   = client.indices.get_mapping(index=idx)
        doc_id_type = mapping[idx]["mappings"]["properties"]["document_id"]["type"]
        templates   = mapping[idx]["mappings"].get("dynamic_templates", [])
        has_template = any(
            "strings_as_keyword" in t for t in templates
        )
        status = "✓" if doc_id_type == "keyword" and has_template else "✗"
        print(
            f"  [{status}] {idx}  document_id={doc_id_type}  "
            f"dynamic_template={'keyword' if has_template else 'MISSING'}"
        )

    # ── 6. Print re-dispatch commands ─────────────────────────────────────────
    if all_doc_ids:
        print(f"\n{'─' * 70}")
        print("  Re-dispatch commands (run these to re-index all documents):")
        print(f"{'─' * 70}\n")
        queue_url = "https://sqs.eu-west-1.amazonaws.com/064051750322/rls-ci-chunk-queue"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Drop and recreate document-chunks and semantic-objects indexes."
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the confirmation prompt",
    )
    parser.add_argument(
        "--region", default="eu-west-1",
        help="AWS region (default: eu-west-1)",
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
        "--chunks-shards", type=int, default=5,
        help="Number of shards for document-chunks (default: 5)",
    )
    args = parser.parse_args()

    endpoint = os.environ.get(
        "OPENSEARCH_ENDPOINT",
        "search-rls-dev-rhitzxwnctmuyq2l4kny5kwelu.eu-west-1.es.amazonaws.com",
    )
    if not endpoint:
        print("ERROR: OPENSEARCH_ENDPOINT env var not set", file=sys.stderr)
        sys.exit(1)

    main(
        endpoint      = endpoint,
        region        = args.region,
        chunks_index  = args.chunks_index,
        objects_index = args.objects_index,
        yes           = args.yes,
        chunks_shards = args.chunks_shards,
    )
