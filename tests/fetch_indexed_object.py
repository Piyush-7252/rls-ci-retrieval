#!/usr/bin/env python3

"""
Fetch an indexed object from OpenSearch and dump the COMPLETE _source.

Read-only:
- Does NOT modify OpenSearch
- Does NOT update anything
- Does NOT re-index anything

Usage:

    python tests/fetch_indexed_object.py \
        --object-id "YOUR_OBJECT_ID"

Or for the N = 8 object:

    python tests/fetch_indexed_object.py \
        --object-id "20260726062234599_4xs0l7p_10993_REDACTED_Protocol-Amendment-1-FD-64407564MMY3002-218114_1245209_chunk_0127_obj_0000_s0"
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from pathlib import Path


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_ENDPOINT = (
    "search-rls-dev-rhitzxwnctmuyq2l4kny5kwelu.eu-west-1.es.amazonaws.com"
)

DEFAULT_INDEX = "document-chunks"
DEFAULT_REGION = "eu-west-1"


# ---------------------------------------------------------------------
# OpenSearch connection
# ---------------------------------------------------------------------

def get_opensearch(endpoint: str, region: str):

    try:
        import boto3
        from opensearchpy import OpenSearch, RequestsHttpConnection
        from requests_aws4auth import AWS4Auth
    except ImportError as e:
        print("❌ Missing dependency:")
        print(f"   {e}")
        print()
        print("Install with:")
        print("   pip install boto3 opensearch-py requests-aws4auth")
        sys.exit(1)

    print("🔐 Loading AWS credentials...")

    session = boto3.Session()
    credentials = session.get_credentials()

    if credentials is None:
        print("❌ No AWS credentials found.")
        print("   Make sure your AWS profile/environment is configured.")
        sys.exit(1)

    frozen = credentials.get_frozen_credentials()

    awsauth = AWS4Auth(
        frozen.access_key,
        frozen.secret_key,
        region,
        "es",
        session_token=frozen.token,
    )

    client = OpenSearch(
        hosts=[
            {
                "host": endpoint,
                "port": 443,
            }
        ],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=60,
    )

    return client


# ---------------------------------------------------------------------
# Pretty recursive field printer
# ---------------------------------------------------------------------

def print_json(value, indent=0):

    prefix = " " * indent

    if isinstance(value, dict):

        for key, val in value.items():

            if isinstance(val, (dict, list)):

                print(f"{prefix}{key}:")

                if isinstance(val, dict):
                    print_json(val, indent + 2)
                else:
                    print_json(val, indent + 2)

            else:
                print(f"{prefix}{key}: {repr(val)}")

    elif isinstance(value, list):

        for i, item in enumerate(value):

            print(f"{prefix}[{i}]")

            if isinstance(item, (dict, list)):
                print_json(item, indent + 2)
            else:
                print(f"{prefix}  {repr(item)}")

    else:
        print(f"{prefix}{repr(value)}")


# ---------------------------------------------------------------------
# Fetch exact object
# ---------------------------------------------------------------------

def fetch_by_id(client, index, object_id):

    print()
    print("=" * 100)
    print("FETCHING INDEXED OBJECT")
    print("=" * 100)

    print(f"Index:      {index}")
    print(f"Object ID:  {object_id}")
    print()

    # First try direct GET.
    try:

        response = client.get(
            index=index,
            id=object_id,
        )

        if response.get("found"):

            print("✅ Found object using direct GET")
            return response

    except Exception as e:

        print(f"⚠️ Direct GET failed: {e}")

    # -----------------------------------------------------------------
    # Fallback: search _source.object_id
    # -----------------------------------------------------------------

    print()
    print("🔎 Trying search by object_id...")

    query = {
        "size": 10,
        "query": {
            "term": {
                "object_id": object_id
            }
        }
    }

    try:

        response = client.search(
            index=index,
            body=query,
        )

        hits = response.get("hits", {}).get("hits", [])

        if not hits:

            print("❌ No object found.")
            return None

        print(f"✅ Found {len(hits)} matching object(s)")

        return hits[0]

    except Exception as e:

        print(f"❌ Search failed: {e}")
        return None


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Fetch complete indexed object from OpenSearch"
    )

    parser.add_argument(
        "--object-id",
        help="Exact semantic-objects object_id",
        default="e69f37be-9be8-41b4-baca-4e6fefa22ea4__29__20260507091044838_oc04wpn_10993_REDACTED_Protocol-Amendment-1-FD-64407564MMY3002-218114_1245209-(2)_chunk_0008"
    )

    parser.add_argument(
        "--index",
        default=DEFAULT_INDEX,
        help="OpenSearch index",
    )

    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help="AWS region",
    )

    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="OpenSearch endpoint",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output file",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Connect
    # -----------------------------------------------------------------

    print("📡 Connecting to OpenSearch...")

    client = get_opensearch(
        args.endpoint,
        args.region,
    )

    print("✅ Connected")

    # -----------------------------------------------------------------
    # Fetch
    # -----------------------------------------------------------------

    response = fetch_by_id(
        client,
        args.index,
        args.object_id,
    )

    if response is None:
        sys.exit(1)

    # -----------------------------------------------------------------
    # Extract complete source
    # -----------------------------------------------------------------

    source = response.get("_source")

    if source is None:

        print("❌ Response contains no _source")

        print()
        print("FULL RESPONSE:")
        print(json.dumps(response, indent=2, ensure_ascii=False))

        sys.exit(1)

    # -----------------------------------------------------------------
    # Print metadata
    # -----------------------------------------------------------------

    print()
    print("=" * 100)
    print("OPENSEARCH METADATA")
    print("=" * 100)

    print(f"_index:       {response.get('_index')}")
    print(f"_id:          {response.get('_id')}")
    print(f"_version:     {response.get('_version')}")
    print(f"_seq_no:      {response.get('_seq_no')}")
    print(f"_primary_term: {response.get('_primary_term')}")

    # -----------------------------------------------------------------
    # Print COMPLETE source
    # -----------------------------------------------------------------

    print()
    print("=" * 100)
    print("COMPLETE _SOURCE")
    print("=" * 100)

    print_json(source)

    # -----------------------------------------------------------------
    # Important coordinate fields
    # -----------------------------------------------------------------

    coordinate_fields = [
        "object_id",
        "parent_chunk_id",
        "type",
        "text",
        "normalized_text",
        "page",
        "bbox",
        "position",
        "global_position",
        "document_position",
        "char_start",
        "char_end",
        "page_char_start",
        "page_char_end",
        "display_spans",
        "paragraph_text",
        "prev_sentence_text",
        "next_sentence_text",
    ]

    print()
    print("=" * 100)
    print("COORDINATE / POSITION FIELDS")
    print("=" * 100)

    for field in coordinate_fields:

        if field in source:

            print()
            print(f"### {field}")

            print(
                json.dumps(
                    source[field],
                    indent=2,
                    ensure_ascii=False,
                )
            )

        else:

            print()
            print(f"### {field}")
            print("NOT PRESENT")

    # -----------------------------------------------------------------
    # Save complete source
    # -----------------------------------------------------------------

    if args.output:

        output_path = Path(args.output)

    else:

        safe_id = args.object_id.replace("/", "_")

        output_path = Path(
            f"indexed_object_{safe_id}.json"
        )

    output_payload = {
        "opensearch_metadata": {
            "_index": response.get("_index"),
            "_id": response.get("_id"),
            "_version": response.get("_version"),
            "_seq_no": response.get("_seq_no"),
            "_primary_term": response.get("_primary_term"),
        },
        "_source": source,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output_payload,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 100)
    print("SAVED")
    print("=" * 100)

    print(f"📄 {output_path.resolve()}")


if __name__ == "__main__":
    main()