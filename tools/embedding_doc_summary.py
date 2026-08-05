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
LOG_GROUP = "/aws/lambda/rls-ci-chunk-worker"


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


def _print_doc_summary(doc_id: str, d: dict) -> None:
    calls         = max(d["bedrock_calls"], 1)
    chunks        = max(d["chunks"], 1)
    throttle_pct  = 100 * d["throttle_events"] / calls
    avg_attempts  = 1 + d["extra_attempts"] / calls
    avg_chunk_s   = d["total_embed_s"] / chunks
    backoff_share = 100 * d["total_backoff_ms"] / max(d["total_embed_s"] * 1000, 1)
    throughput    = d["embed_requests"] / max(d["total_embed_s"], 1)

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
    print(f"  Throughput            : {throughput:>9.2f}  embed/s")
    print(f"{'─'*W}")
    print(f"  Throttle events       : {d['throttle_events']:>10,}   ({throttle_pct:.1f}% of calls)")
    print(f"  Throttled chunks      : {d['throttled_chunks']:>10,}   ({throttled_chunk_pct:.1f}% of chunks)")
    print(f"  Cold-start chunks     : {d['cold_start_chunks']:>10,}")
    print(f"  Extra attempts        : {d['extra_attempts']:>10,}")
    print(f"  Avg attempts/call     : {avg_attempts:>10.2f}")
    print(f"  Total backoff         : {_fmt_ms(d['total_backoff_ms']):>10}   (backoff share {backoff_share:.1f}% of embed time)")
    print(f"{'─'*W}")
    print(f"  Total embed time      : {_fmt_ms(int(d['total_embed_s']*1000)):>10}")
    print(f"  Chunk duration  avg   : {avg_chunk_s:>9.2f}s")
    print(f"  Chunk duration  P50   : {p50:>9.2f}s")
    print(f"  Chunk duration  P95   : {p95:>9.2f}s")
    print(f"  Chunk duration  P99   : {p99:>9.2f}s")
    print(f"  Chunk duration  max   : {d['max_chunk_s']:>9.2f}s")
    print(f"{'─'*W}")
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
        _print_doc_summary(doc_id, docs[doc_id])

    # Cross-document totals if more than one doc
    if len(docs) > 1:
        total = {
            "chunks":           sum(d["chunks"]           for d in docs.values()),
            "embed_requests":   sum(d["embed_requests"]   for d in docs.values()),
            "bedrock_calls":    sum(d["bedrock_calls"]    for d in docs.values()),
            "bedrock_success":  sum(d["bedrock_success"]  for d in docs.values()),
            "throttle_events":  sum(d["throttle_events"]  for d in docs.values()),
            "extra_attempts":   sum(d["extra_attempts"]   for d in docs.values()),
            "total_backoff_ms": sum(d["total_backoff_ms"] for d in docs.values()),
            "total_embed_s":    sum(d["total_embed_s"]    for d in docs.values()),
            "throttled_chunks":  sum(d["throttled_chunks"] for d in docs.values()),
            "slowest_chunks":    sorted(
                [item for d in docs.values() for item in d["slowest_chunks"]],
                reverse=True,
            )[:10],
            "max_chunk_s":      max(d["max_chunk_s"]      for d in docs.values()),
            "chunk_times":      [t for d in docs.values() for t in d["chunk_times"]],
            "objects_per_chunk":[o for d in docs.values() for o in d["objects_per_chunk"]],
        }
        print()
        _print_doc_summary(f"ALL DOCUMENTS ({len(docs)} docs)", total)


if __name__ == "__main__":
    main()
