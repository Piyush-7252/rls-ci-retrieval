# tools/dump_ci_objects.py

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth


REGION = "eu-west-1"
INDEX = "ci-objects"

OUTPUT_DIR = Path("localfiles/opensearch_dumps")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENDPOINT = os.environ["OPENSEARCH_ENDPOINT"]


def get_client() -> OpenSearch:
    creds = boto3.Session().get_credentials()

    if creds is None:
        raise RuntimeError("No AWS credentials available")

    creds = creds.get_frozen_credentials()

    auth = AWS4Auth(
        creds.access_key,
        creds.secret_key,
        REGION,
        "es",
        session_token=creds.token,
    )

    return OpenSearch(
        hosts=[{"host": ENDPOINT, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=120,
        max_retries=3,
        retry_on_timeout=True,
    )


def fetch_all(client: OpenSearch) -> list[dict]:
    print(f"Fetching ALL documents from {INDEX} ...")

    docs: list[dict] = []

    body = {
        "size": 1000,
        "query": {
            "match_all": {}
        },
        "sort": [
            {
                "_doc": "asc"
            }
        ],
    }

    resp = client.search(
        index=INDEX,
        body=body,
        scroll="5m",
    )

    scroll_id = resp.get("_scroll_id")
    hits = resp.get("hits", {}).get("hits", [])

    while hits:
        for hit in hits:
            docs.append({
                "_id": hit.get("_id"),
                "_index": hit.get("_index"),
                "_source": hit.get("_source", {}),
            })

        print(f"  fetched: {len(docs)}", flush=True)

        resp = client.scroll(
            scroll_id=scroll_id,
            scroll="5m",
        )

        scroll_id = resp.get("_scroll_id")
        hits = resp.get("hits", {}).get("hits", [])

    if scroll_id:
        try:
            client.clear_scroll(scroll_id=scroll_id)
        except Exception:
            pass

    return docs


def validate(docs: list[dict]) -> dict:
    print("\n=== VALIDATION ===")

    total = len(docs)

    missing_tenant_id = []
    missing_tenant_name = []
    missing_tenant_schema = []
    missing_project_id = []
    missing_ci_id = []
    missing_known_ci = []

    bad_global_id = []
    duplicate_logical_ids: dict[str, list[str]] = {}

    tenant_project_counts: dict[str, int] = {}

    for item in docs:
        os_id = str(item.get("_id", ""))
        src = item.get("_source", {})

        tenant_id = src.get("tenant_id")
        tenant_name = src.get("tenant_name")
        tenant_schema = src.get("tenant_schema")
        project_id = src.get("project_id")
        ci_id = src.get("ci_id")
        known_ci = src.get("known_ci")

        if not tenant_id:
            missing_tenant_id.append(os_id)

        if not tenant_name:
            missing_tenant_name.append(os_id)

        if not tenant_schema:
            missing_tenant_schema.append(os_id)

        if not project_id:
            missing_project_id.append(os_id)

        if ci_id is None:
            missing_ci_id.append(os_id)

        if not known_ci:
            missing_known_ci.append(os_id)

        # Expected global CI OpenSearch ID:
        #
        # tenant__project__ci_<ci_id>
        if tenant_id is not None and project_id is not None and ci_id is not None:
            expected = (
                f"{tenant_id}__"
                f"{project_id}__"
                f"ci_{ci_id}"
            )

            if os_id != expected:
                bad_global_id.append({
                    "opensearch_id": os_id,
                    "expected": expected,
                    "tenant_id": tenant_id,
                    "project_id": project_id,
                    "ci_id": ci_id,
                })

        logical_key = str(ci_id)

        duplicate_logical_ids.setdefault(logical_key, []).append(os_id)

        scope_key = f"{tenant_id}__{project_id}"
        tenant_project_counts[scope_key] = (
            tenant_project_counts.get(scope_key, 0) + 1
        )

    duplicates = {
        ci_id: ids
        for ci_id, ids in duplicate_logical_ids.items()
        if len(ids) > 1
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "index": INDEX,
        "endpoint": ENDPOINT,

        "total_documents": total,

        "missing": {
            "tenant_id": len(missing_tenant_id),
            "tenant_name": len(missing_tenant_name),
            "tenant_schema": len(missing_tenant_schema),
            "project_id": len(missing_project_id),
            "ci_id": len(missing_ci_id),
            "known_ci": len(missing_known_ci),
        },

        "bad_global_id_count": len(bad_global_id),

        "duplicate_logical_ci_id_count": len(duplicates),

        "tenant_project_counts": tenant_project_counts,

        "bad_global_ids": bad_global_id,

        "duplicate_logical_ci_ids": duplicates,
    }

    print(f"Total documents             : {total}")
    print(f"Missing tenant_id           : {len(missing_tenant_id)}")
    print(f"Missing tenant_name         : {len(missing_tenant_name)}")
    print(f"Missing tenant_schema       : {len(missing_tenant_schema)}")
    print(f"Missing project_id          : {len(missing_project_id)}")
    print(f"Missing ci_id               : {len(missing_ci_id)}")
    print(f"Missing known_ci             : {len(missing_known_ci)}")
    print(f"Bad global OpenSearch IDs   : {len(bad_global_id)}")
    print(f"Duplicate logical CI IDs    : {len(duplicates)}")

    print("\nTenant/project distribution:")

    for scope, count in sorted(tenant_project_counts.items()):
        print(f"  {scope}: {count}")

    return report


def main() -> None:
    client = get_client()

    print(f"OpenSearch endpoint: {ENDPOINT}")
    print(f"Index: {INDEX}")

    count = client.count(index=INDEX)["count"]
    print(f"OpenSearch count: {count}")

    docs = fetch_all(client)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dump_path = OUTPUT_DIR / f"ci_objects_{timestamp}.json"
    report_path = OUTPUT_DIR / f"ci_objects_validation_{timestamp}.json"

    report = validate(docs)

    dump = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "index": INDEX,
        "count": len(docs),
        "documents": docs,
    }

    dump_path.write_text(
        json.dumps(dump, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== DONE ===")
    print(f"Full dump : {dump_path}")
    print(f"Validation: {report_path}")


if __name__ == "__main__":
    main()