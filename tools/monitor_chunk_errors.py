"""
Continuously tails CloudWatch logs for rls-ci-chunk-worker and captures all
errors with timestamp, chunk_id, and full error context.

Usage
-----
    python tools/monitor_chunk_errors.py
    python tools/monitor_chunk_errors.py --output localfiles/inspection/errors_100c.jsonl
    python tools/monitor_chunk_errors.py --interval 20  # poll every 20s (default 30s)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3

REGION    = "eu-west-1"
LOG_GROUP = "/aws/lambda/rls-ci-chunk-worker"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def ts_label(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def tail_errors(output_path: Path, interval: int) -> None:
    logs   = boto3.client("logs", region_name=REGION)
    out_fh = output_path.open("a", encoding="utf-8")

    # start from "now"
    start_ms = int(time.time() * 1000)
    seen_event_ids: set[str] = set()

    error_counts: dict[str, int] = {}   # error_type → count
    total_errors = 0

    print(f"[{now_utc()}] Monitoring {LOG_GROUP}", flush=True)
    print(f"  Writing errors → {output_path}", flush=True)
    print(f"  Poll interval : {interval}s", flush=True)
    print(f"  Ctrl-C to stop\n", flush=True)

    try:
        while True:
            now_ms = int(time.time() * 1000)

            try:
                paginator = logs.get_paginator("filter_log_events")
                pages = paginator.paginate(
                    logGroupName=LOG_GROUP,
                    startTime=start_ms,
                    endTime=now_ms,
                    filterPattern='?ERROR ?error ?failed ?Traceback ?"exceeded"',
                )

                new_events: list[dict] = []
                for page in pages:
                    for evt in page.get("events", []):
                        eid = evt.get("eventId", evt["timestamp"])
                        if eid not in seen_event_ids:
                            seen_event_ids.add(eid)
                            new_events.append(evt)

            except Exception as exc:
                print(f"[{now_utc()}] WARNING: CloudWatch poll error: {exc}", flush=True)
                time.sleep(interval)
                continue

            if new_events:
                for evt in new_events:
                    msg      = evt["message"].rstrip()
                    ts_label_ = ts_label(evt["timestamp"])
                    stream   = evt.get("logStreamName", "?")

                    # Parse chunk_id from log line if present
                    chunk_id = "unknown"
                    for part in msg.split():
                        if part.startswith("chunk_id="):
                            chunk_id = part[len("chunk_id="):]
                        elif "chunk_id=" in part:
                            chunk_id = part.split("chunk_id=", 1)[1].split()[0]

                    # Classify error type
                    error_type = "UNKNOWN"
                    lower = msg.lower()
                    if "timeout" in lower or "connectiontimeout" in lower:
                        error_type = "TIMEOUT"
                    elif "throttling" in lower or "throughputexceeded" in lower or "too many requests" in lower:
                        error_type = "THROTTLE"
                    elif "fields" in lower and "exceeded" in lower:
                        error_type = "FIELD_LIMIT"
                    elif "429" in msg:
                        error_type = "HTTP_429"
                    elif "503" in msg:
                        error_type = "HTTP_503"
                    elif "memory" in lower or "oom" in lower:
                        error_type = "OOM"
                    elif "traceback" in lower:
                        error_type = "EXCEPTION"
                    elif "error" in lower or "failed" in lower:
                        error_type = "ERROR"

                    error_counts[error_type] = error_counts.get(error_type, 0) + 1
                    total_errors += 1

                    record = {
                        "timestamp":  ts_label_,
                        "epoch_ms":   evt["timestamp"],
                        "stream":     stream,
                        "chunk_id":   chunk_id,
                        "error_type": error_type,
                        "message":    msg,
                    }
                    out_fh.write(json.dumps(record) + "\n")
                    out_fh.flush()

                    print(f"  [{ts_label_}] {error_type:12s}  chunk={chunk_id}", flush=True)
                    # Print full message for non-trivial lines
                    if error_type not in ("EXCEPTION",):
                        print(f"    {msg[:200]}", flush=True)

                print(f"\n  [{now_utc()}] Cumulative errors: {total_errors}  breakdown: {error_counts}\n", flush=True)

            start_ms = now_ms
            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n[{now_utc()}] Stopped.", flush=True)
        print(f"  Total errors captured: {total_errors}", flush=True)
        print(f"  Breakdown: {json.dumps(error_counts, indent=2)}", flush=True)
        print(f"  Output: {output_path}", flush=True)
        out_fh.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output",   default="localfiles/inspection/chunk_errors_100c.jsonl",
                   help="JSONL file to write errors to")
    p.add_argument("--interval", type=int, default=30,
                   help="CloudWatch poll interval in seconds")
    args = p.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tail_errors(out, args.interval)


if __name__ == "__main__":
    main()
