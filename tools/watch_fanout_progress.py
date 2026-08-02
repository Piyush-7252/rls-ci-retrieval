from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth


def _build_os_client(region: str, endpoint: str) -> OpenSearch:
    frozen = boto3.Session(region_name=region).get_credentials().get_frozen_credentials()
    awsauth = AWS4Auth(
        frozen.access_key,
        frozen.secret_key,
        region,
        "es",
        session_token=frozen.token,
    )
    return OpenSearch(
        hosts=[{"host": endpoint, "port": 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )


def _count_doc_chunks(client: OpenSearch, index: str, document_id: str) -> int:
    resp = client.count(
        index=index,
        body={"query": {"term": {"document_id": document_id}}},
    )
    return int(resp.get("count", 0))


def _count_semantic_objects(client: OpenSearch, index: str, document_id: str) -> int:
    resp = client.count(
        index=index,
        body={"query": {"term": {"document_id": document_id}}},
    )
    return int(resp.get("count", 0))


def _queue_stats(region: str, queue_url: str) -> tuple[int, int, int]:
    sqs = boto3.client("sqs", region_name=region)
    attrs = sqs.get_queue_attributes(
        QueueUrl=queue_url,
        AttributeNames=[
            "ApproximateNumberOfMessages",
            "ApproximateNumberOfMessagesNotVisible",
            "ApproximateNumberOfMessagesDelayed",
        ],
    )["Attributes"]

    visible = int(attrs.get("ApproximateNumberOfMessages", "0"))
    inflight = int(attrs.get("ApproximateNumberOfMessagesNotVisible", "0"))
    delayed = int(attrs.get("ApproximateNumberOfMessagesDelayed", "0"))
    return visible, inflight, delayed


def _fmt_eta(seconds: float) -> str:
    if seconds <= 0:
        return "done"
    mins, sec = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    return f"{hrs:02d}:{mins:02d}:{sec:02d}"


def _print_snapshot(
    client: OpenSearch,
    region: str,
    queue_url: str,
    chunk_index: str,
    semantic_index: str,
    document_id: str,
    expected_chunks: int,
    start_epoch: float,
) -> None:
    now = time.time()
    elapsed = max(now - start_epoch, 1.0)

    done_chunks = _count_doc_chunks(client, chunk_index, document_id)
    done_objects = _count_semantic_objects(client, semantic_index, document_id)

    visible, inflight, delayed = (0, 0, 0)
    if queue_url:
        visible, inflight, delayed = _queue_stats(region, queue_url)

    pct = (done_chunks / expected_chunks * 100.0) if expected_chunks > 0 else 0.0
    rate = done_chunks / elapsed
    remaining = max(expected_chunks - done_chunks, 0)
    eta = _fmt_eta(remaining / rate) if rate > 0 else "unknown"

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(
        f"[{ts}] chunks={done_chunks}/{expected_chunks} ({pct:.2f}%) "
        f"rate={rate:.2f} chunk/s eta={eta} objects={done_objects} "
        f"queue_visible={visible} queue_inflight={inflight} queue_delayed={delayed}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Track fan-out run progress from OpenSearch + SQS")
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--expected-chunks", type=int, required=True)
    parser.add_argument("--region", default="eu-west-1")
    parser.add_argument("--opensearch-endpoint", required=True)
    parser.add_argument("--chunk-index", default="document-chunks")
    parser.add_argument("--semantic-index", default="semantic-objects")
    parser.add_argument("--queue-url", default="")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()

    client = _build_os_client(args.region, args.opensearch_endpoint)
    start_epoch = time.time()

    _print_snapshot(
        client=client,
        region=args.region,
        queue_url=args.queue_url,
        chunk_index=args.chunk_index,
        semantic_index=args.semantic_index,
        document_id=args.document_id,
        expected_chunks=args.expected_chunks,
        start_epoch=start_epoch,
    )

    if not args.watch:
        return

    while True:
        time.sleep(max(args.interval, 5))
        _print_snapshot(
            client=client,
            region=args.region,
            queue_url=args.queue_url,
            chunk_index=args.chunk_index,
            semantic_index=args.semantic_index,
            document_id=args.document_id,
            expected_chunks=args.expected_chunks,
            start_epoch=start_epoch,
        )


if __name__ == "__main__":
    main()
