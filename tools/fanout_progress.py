from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth


def _build_os_client(region: str, endpoint: str) -> OpenSearch:
    creds = boto3.Session(region_name=region).get_credentials()
    if creds is None:
        raise RuntimeError("AWS credentials not found")
    frozen = creds.get_frozen_credentials()
    auth = AWS4Auth(
        frozen.access_key,
        frozen.secret_key,
        region,
        "es",
        session_token=frozen.token,
    )
    return OpenSearch(
        hosts=[{"host": endpoint, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )


def _fmt_duration(seconds: float) -> str:
    if seconds < 0:
        return "unknown"
    s = int(seconds)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m}m {s}s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Track fan-out indexing progress")
    parser.add_argument("--queue-url", required=True)
    parser.add_argument("--os-endpoint", required=True)
    parser.add_argument("--region", default="eu-west-1")
    parser.add_argument("--document-id", default="")
    parser.add_argument("--total-chunks", type=int, default=0)
    parser.add_argument("--start-epoch", type=float, default=0.0,
                        help="Unix epoch when dispatch started (optional) for ETA")
    parser.add_argument("--watch", type=int, default=0,
                        help="Refresh every N seconds until queue drains (0 = one shot)")
    args = parser.parse_args()

    session = boto3.Session(region_name=args.region)
    sqs = session.client("sqs")
    os_client = _build_os_client(args.region, args.os_endpoint)

    def snapshot() -> tuple[int, int, int, int]:
        attrs = sqs.get_queue_attributes(
            QueueUrl=args.queue_url,
            AttributeNames=[
                "ApproximateNumberOfMessages",
                "ApproximateNumberOfMessagesNotVisible",
                "ApproximateNumberOfMessagesDelayed",
            ],
        )["Attributes"]
        visible = int(attrs.get("ApproximateNumberOfMessages", "0"))
        in_flight = int(attrs.get("ApproximateNumberOfMessagesNotVisible", "0"))
        delayed = int(attrs.get("ApproximateNumberOfMessagesDelayed", "0"))

        if args.document_id:
            query = {"query": {"term": {"document_id": args.document_id}}}
            done = int(os_client.count(index="document-chunks", body=query)["count"])
        else:
            done = int(os_client.count(index="document-chunks")["count"])

        return visible, in_flight, delayed, done

    start_ts = args.start_epoch if args.start_epoch > 0 else time.time()
    start_done = None

    while True:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        visible, in_flight, delayed, done = snapshot()
        if start_done is None:
            start_done = done

        elapsed = max(time.time() - start_ts, 1.0)
        processed = max(done - start_done, 0)
        rate = processed / elapsed

        print(f"[{now}]")
        print(f"queue_visible={visible} queue_inflight={in_flight} queue_delayed={delayed}")
        print(f"indexed_document_chunks={done}")

        if args.total_chunks > 0:
            pct = (done / args.total_chunks) * 100.0
            remaining = max(args.total_chunks - done, 0)
            eta = (remaining / rate) if rate > 0 else -1
            print(f"progress={done}/{args.total_chunks} ({pct:.2f}%)")
            print(f"rate={rate:.2f} chunks/s eta={_fmt_duration(eta)}")

        print("-")

        if args.watch <= 0:
            break
        if visible == 0 and in_flight == 0 and delayed == 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
