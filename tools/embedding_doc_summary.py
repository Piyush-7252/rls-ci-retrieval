"""
embedding_doc_summary.py
------------------------
Reads [ChunkSummary] lines from CloudWatch (rls-ci-chunk-worker) and prints
a document-level ingestion dashboard.

Usage
-----
    # Last 6 hours (default)
    python tools/embedding_doc_summary.py

    # Custom time window
    python tools/embedding_doc_summary.py --hours 12

    # Specific document
    python tools/embedding_doc_summary.py --doc 10993-co-jnj-64407564

    # Save raw summaries to JSONL
    python tools/embedding_doc_summary.py --output /tmp/embed_summary.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import boto3

REGION    = "eu-west-1"
LOG_GROUP = "/aws/lambda/rls-ci-retrieval-document-chunk-worker"


# ── CloudWatch fetcher ────────────────────────────────────────────────────────

def _fetch_chunk_summaries(hours: float, doc_filter: str | None) -> list[dict]:
    """Pull all [ChunkSummary] log lines from the last `hours` hours."""
    logs     = boto3.client("logs", region_name=REGION)
    start_ms = int((time.time() - hours * 3600) * 1000)
    end_ms   = int(time.time() * 1000)

    pattern = '"[ChunkSummary]"'
    if doc_filter:
        pattern += f' "{doc_filter}"'

    paginator = logs.get_paginator("filter_log_events")
    pages = paginator.paginate(
        logGroupName=LOG_GROUP,
        startTime=start_ms,
        endTime=end_ms,
        filterPattern=pattern,
    )

    summaries: list[dict] = []
    for page in pages:
        for evt in page.get("events", []):
            parsed = _parse_chunk_summary(evt["message"])
            if parsed:
                parsed["_log_ts"] = evt["timestamp"]
                summaries.append(parsed)

    return summaries


# ── Parser ────────────────────────────────────────────────────────────────────

_KV_RE = re.compile(r'(\w+)=([\S]+)')

def _parse_chunk_summary(line: str) -> dict | None:
    """Parse a [ChunkSummary] log line into a dict."""
    if "[ChunkSummary]" not in line:
        return None
    kv: dict[str, str] = {}
    for key, val in _KV_RE.findall(line):
        kv[key] = val
    if "chunk" not in kv or "doc" not in kv:
        return None

    def _int(k: str) -> int:
        try:
            return int(kv.get(k, 0))
        except ValueError:
            return 0

    def _float(k: str) -> float:
        try:
            return float(kv.get(k, "0").rstrip("s"))
        except ValueError:
            return 0.0

    return {
        "chunk":               kv.get("chunk", ""),
        "doc":                 kv.get("doc", ""),
        "objects":             _int("objects"),
        "sentences":           _int("sentences"),
        "embed_requests":      _int("embed_requests"),
        "bedrock_calls":       _int("bedrock_calls"),
        "bedrock_success":     _int("bedrock_success"),
        "bedrock_throttles":   _int("bedrock_throttles"),
        "extra_attempts":      _int("bedrock_extra_attempts"),
        "total_backoff_ms":    _int("total_backoff_ms"),
        "embedding_time_s":    _float("embedding_time"),
        "cold_start":          kv.get("cold_start", "False").lower() == "true",
        # v2 fields (present only in new logs — default to 0 / -1 for old logs)
        "bedrock_lat_avg_ms":  _int("bedrock_lat_avg_ms"),
        "bedrock_lat_p95_ms":  _int("bedrock_lat_p95_ms"),
        "bedrock_lat_max_ms":  _int("bedrock_lat_max_ms"),
        "peak_inflight":       _int("peak_inflight"),
        "max_workers":         _int("max_workers"),
        "memory_limit_mb":     _int("memory_limit_mb"),
        "queue_wait_ms":       int(kv["queue_wait_ms"]) if kv.get("queue_wait_ms") else -1,
    }


# ── Aggregator ────────────────────────────────────────────────────────────────

def _aggregate(summaries: list[dict]) -> dict[str, dict]:
    """Aggregate chunk-level summaries into per-document totals."""
    docs: dict[str, dict] = defaultdict(lambda: {
        "chunks":            0,
        "embed_requests":    0,
        "bedrock_calls":     0,
        "bedrock_success":   0,
        "throttle_events":   0,
        "extra_attempts":    0,
        "total_backoff_ms":  0,
        "total_embed_s":     0.0,
        "max_chunk_s":       0.0,
        "chunk_times":       [],   # per-chunk embedding_time_s
        "objects_per_chunk": [],   # per-chunk object count
        "throttled_chunks":  0,    # chunks with ≥1 throttle event
        "slowest_chunks":    [],   # [(embedding_time_s, chunk_id), ...] — keep top-10
        "cold_start_chunks": 0,   # chunks that were cold starts
        "min_ts_ms":         float("inf"),  # earliest CloudWatch event timestamp (ms)
        "max_ts_ms":         0,             # latest  CloudWatch event timestamp (ms)
        # v2 per-chunk lists
        "embed_requests_per_chunk": [],
        "bedrock_lat_avgs":         [],
        "bedrock_lat_p95s":         [],
        "bedrock_lat_maxs":         [],
        "peak_inflight_list":       [],
        "queue_wait_ms_list":       [],
        "memory_limit_mb":          0,
        "max_workers":              0,
    })

    for s in summaries:
        d = docs[s["doc"]]
        d["chunks"]           += 1
        d["embed_requests"]   += s["embed_requests"]
        d["bedrock_calls"]    += s["bedrock_calls"]
        d["bedrock_success"]  += s["bedrock_success"]
        d["throttle_events"]  += s["bedrock_throttles"]
        d["extra_attempts"]   += s["extra_attempts"]
        d["total_backoff_ms"] += s["total_backoff_ms"]
        d["total_embed_s"]    += s["embedding_time_s"]
        d["max_chunk_s"]       = max(d["max_chunk_s"], s["embedding_time_s"])
        d["chunk_times"].append(s["embedding_time_s"])
        d["objects_per_chunk"].append(s["objects"])
        if s["bedrock_throttles"] > 0:
            d["throttled_chunks"] += 1
        if s["cold_start"]:
            d["cold_start_chunks"] += 1
        import heapq
        heapq.nlargest  # ensure available
        d["slowest_chunks"].append((s["embedding_time_s"], s["chunk"]))
        d["slowest_chunks"] = sorted(d["slowest_chunks"], reverse=True)[:10]
        if s.get("_log_ts"):
            d["min_ts_ms"] = min(d["min_ts_ms"], s["_log_ts"])
            d["max_ts_ms"] = max(d["max_ts_ms"], s["_log_ts"])
        # v2 fields
        d["embed_requests_per_chunk"].append(s["embed_requests"])
        if s.get("bedrock_lat_avg_ms", 0) > 0:
            d["bedrock_lat_avgs"].append(s["bedrock_lat_avg_ms"])
            d["bedrock_lat_p95s"].append(s["bedrock_lat_p95_ms"])
            d["bedrock_lat_maxs"].append(s["bedrock_lat_max_ms"])
        if s.get("peak_inflight", 0) > 0:
            d["peak_inflight_list"].append(s["peak_inflight"])
        if s.get("queue_wait_ms", -1) >= 0:
            d["queue_wait_ms_list"].append(s["queue_wait_ms"])
        d["memory_limit_mb"] = max(d["memory_limit_mb"], s.get("memory_limit_mb", 0))
        d["max_workers"]     = max(d["max_workers"],     s.get("max_workers",     0))

    return dict(docs)


# ── Formatter ─────────────────────────────────────────────────────────────────

def _fmt_ms(ms: int) -> str:
    """Format milliseconds as human-readable duration."""
    if ms < 1_000:
        return f"{ms}ms"
    s = ms / 1000
    if s < 60:
        return f"{s:.1f}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m {int(s)}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m"


def _pct(values: list[float], p: float) -> float:
    """Return the p-th percentile (0-100) of a sorted-or-unsorted list."""
    if not values:
        return 0.0
    sv = sorted(values)
    idx = (p / 100) * (len(sv) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sv) - 1)
    return sv[lo] + (sv[hi] - sv[lo]) * (idx - lo)


def _histogram(values: list[float], buckets: list[tuple[float, float, str]], width: int = 30) -> str:
    """ASCII bar chart. buckets = [(lo, hi, label), ...] — last hi is ignored (open-ended)."""
    counts = [0] * len(buckets)
    for v in values:
        for i, (lo, hi, _) in enumerate(buckets):
            if v >= lo and (i == len(buckets) - 1 or v < hi):
                counts[i] += 1
                break
    total    = max(sum(counts), 1)
    max_cnt  = max(counts) if counts else 1
    label_w  = max(len(b[2]) for b in buckets)
    lines    = []
    for (lo, hi, label), cnt in zip(buckets, counts):
        bar = "█" * int(width * cnt / max_cnt)
        pct = 100 * cnt / total
        lines.append(f"  {label:<{label_w}}  {bar:<{width}}  {cnt:>5,}  ({pct:4.1f}%)")
    return "\n".join(lines)


def _print_doc_summary(doc_id: str, d: dict, configured_concurrency: int = 10) -> None:
    calls         = max(d["bedrock_calls"], 1)
    chunks        = max(d["chunks"], 1)
    throttle_pct  = 100 * d["throttle_events"] / calls
    avg_attempts  = 1 + d["extra_attempts"] / calls
    avg_chunk_s   = d["total_embed_s"] / chunks
    backoff_share       = 100 * d["total_backoff_ms"] / max(d["total_embed_s"] * 1000, 1)
    worker_throughput   = d["embed_requests"] / max(d["total_embed_s"], 1)
    wall_clock_s        = (d.get("max_ts_ms", 0) - d.get("min_ts_ms", float("inf"))) / 1000.0
    if d.get("min_ts_ms", float("inf")) >= float("inf") or wall_clock_s <= 0:
        wall_clock_s = d["total_embed_s"]   # fallback: no timestamps available
    pipeline_throughput = d["embed_requests"] / max(wall_clock_s, 1)
    parallelism         = d["total_embed_s"] / max(wall_clock_s, 1)
    utilization_pct     = 100 * parallelism / max(configured_concurrency, 1)

    ct  = d["chunk_times"]
    p50 = _pct(ct, 50)
    p95 = _pct(ct, 95)
    p99 = _pct(ct, 99)

    success_rate  = 100 * d["bedrock_success"] / calls
    throttled_chunk_pct = 100 * d["throttled_chunks"] / chunks

    oc      = d["objects_per_chunk"]
    obj_avg = sum(oc) / max(len(oc), 1)
    obj_p95 = _pct([float(x) for x in oc], 95)
    obj_p99 = _pct([float(x) for x in oc], 99)
    obj_max = max(oc) if oc else 0

    W = 62
    print(f"\n{'═'*W}")
    print(f"  Document : {doc_id}")
    print(f"{'─'*W}")
    print(f"  Chunks processed      : {d['chunks']:>10,}")
    print(f"  Embedding requests    : {d['embed_requests']:>10,}")
    print(f"  Bedrock calls         : {d['bedrock_calls']:>10,}")
    print(f"  Bedrock success       : {d['bedrock_success']:>10,}   (success rate {success_rate:.2f}%)")
    print(f"  Throughput  worker    : {worker_throughput:>9.2f}  embed/s  (per aggregate compute)")
    print(f"  Throughput  pipeline  : {pipeline_throughput:>9.2f}  embed/s  (per wall-clock)")
    print(f"{'─'*W}")
    print(f"  Throttle events       : {d['throttle_events']:>10,}   ({throttle_pct:.1f}% of calls)")
    print(f"  Throttled chunks      : {d['throttled_chunks']:>10,}   ({throttled_chunk_pct:.1f}% of chunks)")
    print(f"  Cold-start chunks     : {d['cold_start_chunks']:>10,}")
    print(f"  Extra attempts        : {d['extra_attempts']:>10,}")
    print(f"  Avg attempts/call     : {avg_attempts:>10.2f}")
    print(f"  Total backoff         : {_fmt_ms(d['total_backoff_ms']):>10}   (backoff share {backoff_share:.1f}% of embed time)")
    print(f"{'─'*W}")
    print(f"  Wall-clock ingestion  : {_fmt_ms(int(wall_clock_s*1000)):>10}")
    print(f"  Aggregate Lambda time : {_fmt_ms(int(d['total_embed_s']*1000)):>10}")
    print(f"  Effective parallelism : {parallelism:>9.1f}×")
    print(f"  Configured concurrency: {configured_concurrency:>9,}")
    print(f"  Utilization           : {utilization_pct:>9.1f}%")
    lat_avgs = d.get("bedrock_lat_avgs", [])
    if lat_avgs:
        lat_p95s = d.get("bedrock_lat_p95s", [])
        lat_maxs = d.get("bedrock_lat_maxs", [])
        overall_lat_avg = int(sum(lat_avgs) / len(lat_avgs))
        overall_lat_p95 = int(sum(lat_p95s) / max(len(lat_p95s), 1))
        overall_lat_max = max(lat_maxs) if lat_maxs else 0
        print(f"{'─'*W}")
        print(f"  Bedrock latency  avg  : {overall_lat_avg:>7}ms")
        print(f"  Bedrock latency  P95  : {overall_lat_p95:>7}ms   (avg of per-chunk P95s)")
        print(f"  Bedrock latency  max  : {overall_lat_max:>7}ms   (single call max)")
        peak_list = d.get("peak_inflight_list", [])
        if peak_list:
            pif_avg = sum(peak_list) / len(peak_list)
            pif_max = max(peak_list)
            mw = d.get("max_workers", 0)
            mw_str = f"   (EMBEDDING_MAX_WORKERS={mw})" if mw else ""
            print(f"  Peak in-flight   avg  : {pif_avg:>7.1f}   max: {pif_max}{mw_str}")
    print(f"{'─'*W}")
    print(f"  Chunk duration  avg   : {avg_chunk_s:>9.2f}s")
    print(f"  Chunk duration  P50   : {p50:>9.2f}s")
    print(f"  Chunk duration  P95   : {p95:>9.2f}s")
    print(f"  Chunk duration  P99   : {p99:>9.2f}s")
    print(f"  Chunk duration  max   : {d['max_chunk_s']:>9.2f}s")
    print(f"{'─'*W}")
    er = d.get("embed_requests_per_chunk", [])
    if er:
        er_avg = sum(er) / len(er)
        er_p95 = _pct([float(x) for x in er], 95)
        er_max = max(er)
        print(f"  Embed req/chunk avg   : {er_avg:>9.1f}")
        print(f"  Embed req/chunk P95   : {er_p95:>9.1f}")
        print(f"  Embed req/chunk max   : {er_max:>9,}")
    print(f"  Objects/chunk   avg   : {obj_avg:>9.1f}")
    print(f"  Objects/chunk   P95   : {obj_p95:>9.1f}")
    print(f"  Objects/chunk   P99   : {obj_p99:>9.1f}")
    print(f"  Objects/chunk   max   : {obj_max:>9,}")
    if ct:
        buckets = [
            (0,   2,  " 0-2 s "),
            (2,   5,  " 2-5 s "),
            (5,  10,  " 5-10s "),
            (10, 20,  "10-20s "),
            (20, 1e9, "  20s+ "),
        ]
        print(f"{'─'*W}")
        print(f"  Chunk duration distribution:")
        print(_histogram(ct, buckets, width=28))
    slowest = sorted(d.get("slowest_chunks", []), reverse=True)[:5]
    if slowest:
        print(f"{'─'*W}")
        print(f"  Top slowest chunks:")
        for t, cid in slowest:
            print(f"    {cid:<40}  {t:.1f}s")
    qw = d.get("queue_wait_ms_list", [])
    if qw:
        print(f"{'─'*W}")
        print(f"  Queue wait      P50   : {_fmt_ms(int(_pct(qw, 50))):>10}")
        print(f"  Queue wait      P95   : {_fmt_ms(int(_pct(qw, 95))):>10}")
        print(f"  Queue wait      max   : {_fmt_ms(int(max(qw))):>10}")
    if d.get("memory_limit_mb", 0):
        print(f"{'─'*W}")
        print(f"  Memory limit          : {d['memory_limit_mb']:>7}MB   (configured Lambda limit)")
    print(f"{'═'*W}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Document-level embedding ingestion dashboard")
    p.add_argument("--hours",  type=float, default=6.0,
                   help="How many hours back to look in CloudWatch (default: 6)")
    p.add_argument("--doc",    default=None,
                   help="Filter to a specific document ID substring")
    p.add_argument("--output", default=None,
                   help="Save raw chunk summaries to this JSONL file")
    p.add_argument("--concurrency", type=int, default=10,
                   help="Configured Lambda concurrency limit for utilization %% (default: 10)")
    args = p.parse_args()

    print(f"Fetching [ChunkSummary] logs from last {args.hours:.0f}h …", flush=True)
    summaries = _fetch_chunk_summaries(args.hours, args.doc)

    if not summaries:
        print("No [ChunkSummary] lines found. "
              "Make sure the embedding Lambda has been deployed and chunks have been processed.")
        sys.exit(0)

    print(f"Found {len(summaries):,} chunk summaries.", flush=True)

    if args.output:
        out = Path(args.output)
        out.write_text("\n".join(json.dumps(s) for s in summaries) + "\n", encoding="utf-8")
        print(f"Raw summaries saved → {out}")

    docs = _aggregate(summaries)
    for doc_id in sorted(docs):
        _print_doc_summary(doc_id, docs[doc_id], configured_concurrency=args.concurrency)

    # Cross-document totals if more than one doc
    if len(docs) > 1:
        total = {
            "chunks":           sum(d["chunks"]           for d in docs.values()),
            "embed_requests":    sum(d["embed_requests"]   for d in docs.values()),
            "bedrock_calls":     sum(d["bedrock_calls"]    for d in docs.values()),
            "bedrock_success":   sum(d["bedrock_success"]  for d in docs.values()),
            "throttle_events":   sum(d["throttle_events"]  for d in docs.values()),
            "extra_attempts":    sum(d["extra_attempts"]   for d in docs.values()),
            "total_backoff_ms":  sum(d["total_backoff_ms"] for d in docs.values()),
            "total_embed_s":     sum(d["total_embed_s"]    for d in docs.values()),
            "throttled_chunks":  sum(d["throttled_chunks"] for d in docs.values()),
            "cold_start_chunks": sum(d.get("cold_start_chunks", 0) for d in docs.values()),
            "slowest_chunks":    sorted(
                [item for d in docs.values() for item in d["slowest_chunks"]],
                reverse=True,
            )[:10],
            "max_chunk_s":      max(d["max_chunk_s"]      for d in docs.values()),
            "chunk_times":      [t for d in docs.values() for t in d["chunk_times"]],
            "objects_per_chunk":[o for d in docs.values() for o in d["objects_per_chunk"]],
            "min_ts_ms":        min((d.get("min_ts_ms", float("inf")) for d in docs.values()), default=float("inf")),
            "max_ts_ms":        max((d.get("max_ts_ms", 0)            for d in docs.values()), default=0),
            "embed_requests_per_chunk": [e for d in docs.values() for e in d.get("embed_requests_per_chunk", [])],
            "bedrock_lat_avgs": [x for d in docs.values() for x in d.get("bedrock_lat_avgs", [])],
            "bedrock_lat_p95s": [x for d in docs.values() for x in d.get("bedrock_lat_p95s", [])],
            "bedrock_lat_maxs": [x for d in docs.values() for x in d.get("bedrock_lat_maxs", [])],
            "peak_inflight_list": [x for d in docs.values() for x in d.get("peak_inflight_list", [])],
            "queue_wait_ms_list": [x for d in docs.values() for x in d.get("queue_wait_ms_list", [])],
            "memory_limit_mb":  max((d.get("memory_limit_mb", 0) for d in docs.values()), default=0),
            "max_workers":      max((d.get("max_workers",     0) for d in docs.values()), default=0),
        }
        print()
        _print_doc_summary(f"ALL DOCUMENTS ({len(docs)} docs)", total, configured_concurrency=args.concurrency)


if __name__ == "__main__":
    main()
