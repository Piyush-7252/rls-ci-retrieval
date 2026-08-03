"""
Create (or recreate) the semantic-objects OpenSearch index.

Usage:
    python tests/create_semantic_objects_index.py [--recreate]

--recreate  deletes the existing index before creating a fresh one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# ── AWS / OpenSearch connection ───────────────────────────────────────────────
OPENSEARCH_ENDPOINT    = os.environ.get("OPENSEARCH_ENDPOINT", "")
SEMANTIC_OBJECTS_INDEX = os.environ.get("SEMANTIC_OBJECTS_INDEX", "semantic-objects")
AWS_REGION             = os.environ.get("AWS_REGION", "eu-west-1")


def _get_os():
    import boto3
    from opensearchpy import OpenSearch, RequestsHttpConnection
    from requests_aws4auth import AWS4Auth

    frozen  = boto3.Session().get_credentials().get_frozen_credentials()
    awsauth = AWS4Auth(
        frozen.access_key, frozen.secret_key, AWS_REGION, "es",
        session_token=frozen.token,
    )
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_ENDPOINT, "port": 443}],
        http_auth=awsauth, use_ssl=True, verify_certs=True,
        connection_class=RequestsHttpConnection,
    )


# ── Index mapping ─────────────────────────────────────────────────────────────

MAPPING = {
    "settings": {
        "number_of_shards":   1,
        "number_of_replicas": 0,          # 0 for single-node dev; bump to 1 for multi-node
        "knn":                False,      # set to True when using k-NN plugin
    },
    "mappings": {
        "properties": {
            # ── RETRIEVAL — searched, ranked, embedded ─────────────────────
            "object_id":         {"type": "keyword"},
            "document_id":       {"type": "keyword"},
            "parent_chunk_id":   {"type": "keyword"},
            "position":          {"type": "integer"},   # chunk-local (for object_id)
            "global_position":   {"type": "integer"},   # document-global (context expansion)
            "type":              {"type": "keyword"},
            "text": {
                "type":     "text",
                "analyzer": "english",
                "fields": {
                    "keyword": {"type": "keyword", "ignore_above": 512},
                }
            },
            # Lowercased, punctuation-stripped copy — used by Exact/Fuzzy/Token scorers
            "normalized_text":   {"type": "text", "analyzer": "english"},
            # Section heading at time of object (for reranker context)
            "section":           {"type": "keyword"},
            "section_number":    {"type": "keyword"},
            "section_depth":     {"type": "integer"},
            "section_level":     {"type": "integer"},
            "category":          {"type": "keyword"},
            "boost_weight":      {"type": "float"},
            "indexable":         {"type": "boolean"},
            "parent_heading":    {"type": "keyword"},
            "prev_object_pos":   {"type": "integer"},
            "next_object_pos":   {"type": "integer"},
            # Stored as float array; change to "knn_vector" when k-NN plugin is ready
            "dense_vector": {
                "type":  "float",
                "index": False,
            },
            "entities": {
                "type": "nested",
                "properties": {
                    "text":           {"type": "keyword"},
                    "label":          {"type": "keyword"},
                    "sub_type":       {"type": "keyword"},
                    "score":          {"type": "float"},
                    # object_start / object_end → index into obj["text"] directly
                    "object_start":   {"type": "integer"},
                    "object_end":     {"type": "integer"},
                    # document_start / document_end → traceability to source buffer
                    "document_start": {"type": "integer"},
                    "document_end":   {"type": "integer"},
                }
            },
            # ── DISPLAY — UI rendering / PDF annotation only, never embedded
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


def main(recreate: bool = False) -> None:
    if not OPENSEARCH_ENDPOINT:
        print("ERROR: OPENSEARCH_ENDPOINT env var not set", file=sys.stderr)
        sys.exit(1)

    os_client = _get_os()
    idx       = SEMANTIC_OBJECTS_INDEX

    if recreate and os_client.indices.exists(index=idx):
        print(f"Deleting existing index: {idx}")
        os_client.indices.delete(index=idx)

    if os_client.indices.exists(index=idx):
        print(f"Index already exists (use --recreate to rebuild): {idx}")
        return

    print(f"Creating index: {idx}")
    resp = os_client.indices.create(index=idx, body=MAPPING)
    print(json.dumps(resp, indent=2))
    print(f"\nDone — {idx} is ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true",
                        help="Delete and recreate the index (all data will be lost)")
    args = parser.parse_args()
    main(recreate=args.recreate)
