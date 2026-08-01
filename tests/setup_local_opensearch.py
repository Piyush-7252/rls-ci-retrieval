"""
Create (or recreate) all OpenSearch indices on a local Docker instance.

Usage:
    python tests/setup_local_opensearch.py [--recreate]

Requires Docker OpenSearch running:
    docker compose up -d

--recreate  deletes existing indices before creating fresh ones.
"""

from __future__ import annotations

import argparse
import sys

from opensearchpy import OpenSearch

LOCAL_HOST = "localhost"
LOCAL_PORT = 9200


def _get_os() -> OpenSearch:
    return OpenSearch(
        hosts=[{"host": LOCAL_HOST, "port": LOCAL_PORT}],
        use_ssl=False,
        verify_certs=False,
    )


# Fully dynamic — no field mappings, let OpenSearch infer everything.
# This matches AWS behaviour where document-chunks was auto-created.
_SETTINGS = {"number_of_shards": 1, "number_of_replicas": 0}

INDICES = ["document-chunks", "semantic-objects", "ci-objects"]


def main(recreate: bool = False) -> None:
    try:
        client = _get_os()
        info = client.info()
        print(f"Connected to OpenSearch {info['version']['number']} at {LOCAL_HOST}:{LOCAL_PORT}\n")
    except Exception as exc:
        print(f"ERROR: Could not connect to local OpenSearch at {LOCAL_HOST}:{LOCAL_PORT}")
        print(f"  {exc}")
        print("\n  Start it with:  docker compose up -d")
        sys.exit(1)

    for name in INDICES:
        if recreate and client.indices.exists(index=name):
            print(f"  Deleting: {name}")
            client.indices.delete(index=name)

        if client.indices.exists(index=name):
            print(f"  Already exists (--recreate to rebuild): {name}")
            continue

        resp = client.indices.create(index=name, body={"settings": _SETTINGS})
        print(f"  Created: {name}  →  {resp}")

    print("\nAll indices ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create local OpenSearch indices")
    parser.add_argument("--recreate", action="store_true",
                        help="Delete and recreate all indices (data will be lost)")
    args = parser.parse_args()
    main(recreate=args.recreate)

