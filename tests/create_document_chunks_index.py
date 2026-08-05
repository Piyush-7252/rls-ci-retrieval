"""
Create (or recreate) the document-chunks OpenSearch index.

Usage:
    python tests/create_document_chunks_index.py [--recreate]

--recreate  deletes the existing index before creating a fresh one.

All string ID fields are explicitly mapped as ``keyword``.
A ``dynamic_templates`` rule maps any *other* new string field added in the
future to ``keyword`` by default — preventing the accidental ``text`` mapping
that broke ``term(document_id=...)`` filtering when the index was first
auto-created by OpenSearch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# ── AWS / OpenSearch connection ───────────────────────────────────────────────
OPENSEARCH_ENDPOINT = os.environ.get("OPENSEARCH_ENDPOINT", "")
OPENSEARCH_INDEX    = os.environ.get("OPENSEARCH_INDEX", "document-chunks")
AWS_REGION          = os.environ.get("AWS_REGION", "eu-west-1")


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
        "number_of_shards":   5,      # 5 shards matches the live index
        "number_of_replicas": 0,      # 0 for single-node dev; bump to 1 for multi-node
        "refresh_interval":   "30s",  # reduce refresh overhead during bulk indexing
    },
    "mappings": {
        # ── Safety net: any NEW string field added to _build_chunk_doc in the
        # future defaults to keyword, not text.  This prevents the auto-inference
        # bug that caused document_id to be mapped as text the first time.
        "dynamic_templates": [
            {
                "strings_as_keyword": {
                    "match_mapping_type": "string",
                    "mapping": {"type": "keyword"},
                }
            }
        ],
        "properties": {
            # ── Identity ──────────────────────────────────────────────────────
            "document_id":          {"type": "keyword"},
            "chunk_id":             {"type": "keyword"},
            "chunk_idx":            {"type": "integer"},
            "parent_chunk_idx":     {"type": "integer"},
            "prev_chunk_idx":       {"type": "integer"},
            "next_chunk_idx":       {"type": "integer"},
            "page_start":           {"type": "integer"},
            "page_end":             {"type": "integer"},

            # ── Full-text retrieval ────────────────────────────────────────────
            "raw_text": {
                "type":     "text",
                "analyzer": "english",
                # .keyword sub-field for exact-match or aggregation if needed
                "fields": {"keyword": {"type": "keyword", "ignore_above": 512}},
            },
            "normalized_text": {
                "type":     "text",
                "analyzer": "english",
            },
            # tokens is a list of strings — each token stored as a keyword term
            "tokens":               {"type": "keyword"},

            # ── NER entities (nested for per-entity queries) ───────────────────
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

            # ── Ontology ──────────────────────────────────────────────────────
            # ontology_expansions is a list of dicts:
            #   [{expanded: [...], original: "IMWG", type: "abbreviation"}, …]
            # No retriever queries this field directly (ontology_retriever uses
            # normalized_text instead), so store it but don't index it.
            "ontology_expansions":  {"type": "object", "enabled": False},
            # ontology_synonyms is a raw JSON string — not searched, just stored.
            "ontology_synonyms":    {"type": "keyword", "index": False},

            # ── Vectors ───────────────────────────────────────────────────────
            # Stored as float arrays.  index=False means no inverted-index or
            # HNSW graph is built for these; they are returned in _source only.
            # Change to "knn_vector" when the k-NN plugin is enabled.
            "dense_vector":         {"type": "float", "index": False},
            "heading_dense_vector": {"type": "float", "index": False},
            # sparse_vector_json is a raw JSON string — not searched, just stored.
            "sparse_vector_json":   {"type": "keyword", "index": False},

            # ── Embedding metadata ─────────────────────────────────────────────
            "embedding_model":      {"type": "keyword"},
        }
    },
}


def main(recreate: bool = False) -> None:
    if not OPENSEARCH_ENDPOINT:
        print("ERROR: OPENSEARCH_ENDPOINT env var not set", file=sys.stderr)
        sys.exit(1)

    os_client = _get_os()
    idx       = OPENSEARCH_INDEX

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
    parser = argparse.ArgumentParser(
        description="Create (or recreate) the document-chunks OpenSearch index."
    )
    parser.add_argument(
        "--recreate", action="store_true",
        help="Delete and recreate the index (all existing data will be lost)",
    )
    args = parser.parse_args()
    main(recreate=args.recreate)
