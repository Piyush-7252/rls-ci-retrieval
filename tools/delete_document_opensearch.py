#!/usr/bin/env python3
import os
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

REGION = "eu-west-1"
OPENSEARCH_ENDPOINT = os.getenv(
    "OPENSEARCH_ENDPOINT",
    "search-rls-dev-rhitzxwnctmuyq2l4kny5kwelu.eu-west-1.es.amazonaws.com",
)
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "document-chunks")
SEMANTIC_OBJECTS_INDEX = os.getenv("SEMANTIC_OBJECTS_INDEX", "semantic-objects")

DOCUMENT_ID = (
    "20260507091044838_oc04wpn_10993_REDACTED_Protocol-Amendment-1-FD-64407564MMY3002-218114_1245209-(2)"
)

def get_client():
    credentials = boto3.Session().get_credentials()
    if credentials is None:
        raise RuntimeError("AWS credentials not found")
    creds = credentials.get_frozen_credentials()
    auth = AWS4Auth(
        creds.access_key, creds.secret_key, REGION, "es",
        session_token=creds.token,
    )
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_ENDPOINT, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=120,
    )

def count_matching(client, index):
    return client.count(
        index=index,
        body={"query": {"term": {"document_id": DOCUMENT_ID}}},
    )["count"]

def delete_matching(client, index):
    return client.delete_by_query(
        index=index,
        body={"query": {"term": {"document_id": DOCUMENT_ID}}},
        conflicts="proceed",
        refresh=True,
        wait_for_completion=True,
        slices="auto",
    )

def main():
    client = get_client()
    indexes = [OPENSEARCH_INDEX, SEMANTIC_OBJECTS_INDEX]

    print("=" * 80)
    print("DELETE DOCUMENT DATA FROM OPENSEARCH")
    print("=" * 80)
    print(f"Endpoint       : {OPENSEARCH_ENDPOINT}")
    print(f"Document index : {OPENSEARCH_INDEX}")
    print(f"Semantic index : {SEMANTIC_OBJECTS_INDEX}")
    print(f"Document ID    : {DOCUMENT_ID}")
    print("=" * 80)

    before = {}
    for index in indexes:
        before[index] = count_matching(client, index)
        print(f"[BEFORE] {index}: {before[index]}")

    print("\nDeleting matching documents...\n")

    for index in indexes:
        if before[index] == 0:
            print(f"[SKIP] {index}: nothing to delete")
            continue

        try:
            result = delete_matching(client, index)
            print(f"[DONE] {index}")
            print(f"       deleted          : {result.get('deleted', 0)}")
            print(f"       version_conflicts: {result.get('version_conflicts', 0)}")
            failures = result.get("failures", [])
            if failures:
                print(f"       failures         : {len(failures)}")
                for failure in failures[:5]:
                    print(f"       {failure}")
        except Exception as exc:
            print(f"[ERROR] {index}: {exc}")

    print("\nVerifying...\n")
    all_empty = True
    for index in indexes:
        try:
            after = count_matching(client, index)
            print(f"[AFTER] {index}: {after}")
            if after != 0:
                all_empty = False
        except Exception as exc:
            all_empty = False
            print(f"[ERROR] verification {index}: {exc}")

    print()
    print("=" * 80)
    print(
        "✓ COMPLETE: no matching documents remain"
        if all_empty
        else "⚠ WARNING: matching documents still remain"
    )
    print("=" * 80)

if __name__ == "__main__":
    main()
