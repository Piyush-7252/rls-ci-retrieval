import os
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

REGION = "eu-west-1"
INDEX = "ci-objects"

endpoint = os.environ["OPENSEARCH_ENDPOINT"]

creds = boto3.Session().get_credentials().get_frozen_credentials()
auth = AWS4Auth(
    creds.access_key,
    creds.secret_key,
    REGION,
    "es",
    session_token=creds.token,
)

os_client = OpenSearch(
    hosts=[{"host": endpoint, "port": 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=120,
)

print(f"Deleting ALL documents from: {INDEX}")

before = os_client.count(index=INDEX)["count"]
print(f"Documents before: {before}")

if before:
    response = os_client.delete_by_query(
        index=INDEX,
        body={
            "query": {
                "match_all": {}
            }
        },
        conflicts="proceed",
        refresh=True,
        wait_for_completion=True,
    )

    print(f"Deleted: {response.get('deleted', 0)}")
    print(f"Version conflicts: {response.get('version_conflicts', 0)}")

after = os_client.count(index=INDEX)["count"]
print(f"Documents after: {after}")

if after == 0:
    print("✓ ci-objects is empty")
else:
    print(f"⚠ Still contains {after} documents")