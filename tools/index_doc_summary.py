"""
index_doc_summary.py
--------------------
Reads [IndexSummary] lines from CloudWatch (rls-ci-chunk-worker) and prints
a document-level indexing dashboard.

Usage
-----
    # Last 6 hours (default)
    python tools/index_doc_summary.py

    # Custom time window
    python tools/index_doc_summary.py --hours 12

    # Specific document
    python tools/index_doc_summary.py --doc 10993-co-jnj-64407564

    # Save raw summaries to JSONL
    python tools/index_doc_summary.py --output /tmp/index_summary.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import boto3

REGION    = "eu-west-1"
LOG_GROUP = "/aws/lambda/rls-ci-retrieval-document-chunk-worker"


# ── CloudWatch fetcher ────────────────────────────────────────────────────────

def _fetch_index_summaries(hours: float, doc_filter: str | None) -> list[dict]:
    """Pull all [IndexSummary] log lines from the last `hours` hours."""
    logs     = boto3.client("logs", region_name=REGION)
    start_ms = int((time.time() - hours * 3600) * 1000)
    end_ms   = int(time.time() * 1000)

    pattern = '"[IndexSummary]"'
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
            parsed = _parse_index_summary(evt["message"])
            if parsed:
                parsed["_log_ts"] = evt["timestamp"]
                summaries.append(parsed)

    return summaries


# ── Parser ────────────────────────────────────────────────────────────────────

_KV_RE = re.compile(r'(\w+)=([\S]+)')


def _parse_index_summary(line: str) -> dict | None:
    """Parse an [IndexSummary] log line into a dict."""
    if "[IndexSummary]" not in line:
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
        "bulk_docs":           _int("bulk_docs"),
        "objects":             _int("objects"),
        "sentences":           _int("sentences"),
        "bulk_size_mb":        _float("bulk_size_mb"),
        "bulk_latency_s":      _float("bulk_latency"),
        "docs_per_sec":        _float("docs_per_sec"),
        "obj_per_sec":         _float("obj_per_sec"),
        "sent_per_sec":        _float("sent_per_sec"),
        "avg_obj_chars":       _int("avg_obj_chars"),
        "max_obj_chars":       _int("max_obj_chars"),
        "http_retries":        _int("http_retries"),
        "partial_doc_retries": _int("partial_doc_retries"),
        "failed":              _int("failed"),
    }


# ── Aggregator ────────────────────────────────────────────────────────────────

def _aggregate(summaries: list[dict]) -> dict[str, dict]:
    """Aggregate chunk-level summaries into per-document totals."""
    docs: dict[str, dict] = defaultdict(lambda: {
        "chunks":              0,
        "bulk_docs":           0,
        "objects":             0,
        "sentences":           0,
        "total_size_mb":       0.0,
        "total_latency_s":     0.0,
        "max_latency_s":       0.0,
        "http_retries":        0,
        "partial_doc_retries": 0,
        "failed":              0,
        # per-chunk lists for percentiles
        "latency_per_chunk":   [],   # bulk_latency_s
        "size_per_chunk":      [],   # bulk_size_mb
        "docs_per_sec_list":   [],
        "obj_per_sec_list":    [],
        "sent_per_sec_list":   [],
        "avg_obj_chars_list":  [],
        "max_obj_chars_list":  [],
        # retry exposure
        "retried_chunks":      0,    # chunks with http_retries > 0
        "partial_chunks":      0,    # chunks with partial_doc_retries > 0
        # slowest chunks [(latency_s, chunk_id)]
        "slowest_chunks":      [],
        # largest payloads [(size_mb, chunk_id)]
        "largest_chunks":      [],
        "min_ts_ms":           float("inf"),  # earliest CloudWatch event timestamp (ms)
        "max_ts_ms":           0,             # latest  CloudWatch event timestamp (ms)
    })

    for s in summaries:
        d = docs[s["doc"]]
        d["chunks"]              += 1
        d["bulk_docs"]           += s["bulk_docs"]
        d["objects"]             += s["objects"]
        d["sentences"]           += s["sentences"]
        d["total_size_mb"]       += s["bulk_size_mb"]
        d["total_latency_s"]     += s["bulk_latency_s"]
        d["max_latency_s"]        = max(d["max_latency_s"], s["bulk_latency_s"])
        d["http_retries"]        += s["http_retries"]
        d["partial_doc_retries"] += s["partial_doc_retries"]
        d["failed"]              += s["failed"]
        d["latency_per_chunk"].append(s["bulk_latency_s"])
        d["size_per_chunk"].append(s["bulk_size_mb"])
        d["docs_per_sec_list"].append(s["docs_per_sec"])
        d["obj_per_sec_list"].append(s["obj_per_sec"])
        d["sent_per_sec_list"].append(s["sent_per_sec"])
        d["avg_obj_chars_list"].append(s["avg_obj_chars"])
        d["max_obj_chars_list"].append(s["max_obj_chars"])
        if s["http_retries"] > 0:
            d["retried_chunks"] += 1
        if s["partial_doc_retries"] > 0:
            d["partial_chunks"] += 1
        d["slowest_chunks"].append((s["bulk_latency_s"], s["chunk"]))
        d["slowest_chunks"] = sorted(d["slowest_chunks"], reverse=True)[:10]
        d["largest_chunks"].append((s["bulk_size_mb"], s["chunk"]))
        d["largest_chunks"] = sorted(d["largest_chunks"], reverse=True)[:10]
        if s.get("_log_ts"):
            d["min_ts_ms"] = min(d["min_ts_ms"], s["_log_ts"])
            d["max_ts_ms"] = max(d["max_ts_ms"], s["_log_ts"])

    return dict(docs)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct(values: list[float], p: float) -> float:
    """Return the p-th percentile (0-100) via linear interpolation."""
    if not values:
        return 0.0
    sv  = sorted(values)
    idx = (p / 100) * (len(sv) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sv) - 1)
    return sv[lo] + (sv[hi] - sv[lo]) * (idx - lo)


def _fmt_s(s: float) -> str:
    """Format seconds as human-readable duration."""
    if s < 60:
        return f"{s:.1f}s"
    m, rem = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m {int(rem)}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m"


def _histogram(values: list[float], buckets: list[tuple[float, float, str]], width: int = 28) -> str:
    """ASCII bar chart. buckets = [(lo, hi, label), ...]."""
    counts = [0] * len(buckets)
    for v in values:
        for i, (lo, hi, _) in enumerate(buckets):
            if v >= lo and (i == len(buckets) - 1 or v < hi):
                counts[i] += 1
                break
    total   = max(sum(counts), 1)
    max_cnt = max(counts) if counts else 1
    label_w = max(len(b[2]) for b in buckets)
    lines   = []
    for (lo, hi, label), cnt in zip(buckets, counts):
        bar = "█" * int(width * cnt / max_cnt)
        pct = 100 * cnt / total
        lines.append(f"  {label:<{label_w}}  {bar:<{width}}  {cnt:>5,}  ({pct:4.1f}%)")
    return "\n".join(lines)


# ── Formatter ─────────────────────────────────────────────────────────────────

def _print_doc_summary(doc_id: str, d: dict) -> None:
    chunks        = max(d["chunks"], 1)
    bulk_docs     = max(d["bulk_docs"], 1)

    # Throughput — median (P50) per-chunk throughput avoids outliers skewing the mean
    avg_docs_sec  = _pct(d["docs_per_sec_list"],  50)
    avg_obj_sec   = _pct(d["obj_per_sec_list"],   50)
    avg_sent_sec  = _pct(d["sent_per_sec_list"],  50)

    avg_latency   = d["total_latency_s"] / chunks
    p50_latency   = _pct(d["latency_per_chunk"], 50)
    p95_latency   = _pct(d["latency_per_chunk"], 95)
    p99_latency   = _pct(d["latency_per_chunk"], 99)

    wall_clock_s  = (d.get("max_ts_ms", 0) - d.get("min_ts_ms", float("inf"))) / 1000.0
    if d.get("min_ts_ms", float("inf")) >= float("inf") or wall_clock_s <= 0:
        wall_clock_s = d["total_latency_s"]   # fallback: no timestamps available
    parallelism   = d["total_latency_s"] / max(wall_clock_s, 1)

    avg_size_mb   = d["total_size_mb"] / chunks
    p95_size_mb   = _pct(d["size_per_chunk"], 95)
    max_size_mb   = max(d["size_per_chunk"]) if d["size_per_chunk"] else 0.0

    avg_obj_chars = _pct(d["avg_obj_chars_list"], 50)
    max_obj_chars = max(d["max_obj_chars_list"]) if d["max_obj_chars_list"] else 0

    http_retry_pct    = 100 * d["retried_chunks"]  / chunks
    partial_retry_pct = 100 * d["partial_chunks"]  / chunks

    W = 64
    print(f"\n{'═'*W}")
    print(f"  Document : {doc_id}")
    print(f"{'─'*W}")
    print(f"  Chunks processed      : {d['chunks']:>10,}")
    print(f"  Total docs indexed    : {d['bulk_docs']:>10,}  (chunk + objects + sentences)")
    print(f"  Objects indexed       : {d['objects']:>10,}")
    print(f"  Sentences indexed     : {d['sentences']:>10,}")
    print(f"{'─'*W}")
    print(f"  Total payload         : {d['total_size_mb']:>9.2f} MB")
    print(f"  Avg payload / chunk   : {avg_size_mb:>9.2f} MB")
    print(f"  P95 payload / chunk   : {p95_size_mb:>9.2f} MB")
    print(f"  Max payload / chunk   : {max_size_mb:>9.2f} MB")
    print(f"{'─'*W}")
    print(f"  Wall-clock ingestion  : {_fmt_s(wall_clock_s):>10}")
    print(f"  Aggregate worker time : {d['total_latency_s']:>9.1f}  s")
    print(f"  Bulk share of wall-clk: {100 * d['total_latency_s'] / max(wall_clock_s, 1):>9.1f}  %")
    print(f"  Bulk latency  avg     : {avg_latency:>9.3f}  s")
    print(f"  Bulk latency  P50     : {p50_latency:>9.3f}  s")
    print(f"  Bulk latency  P95     : {p95_latency:>9.3f}  s")
    print(f"  Bulk latency  P99     : {p99_latency:>9.3f}  s")
    print(f"  Bulk latency  max     : {d['max_latency_s']:>9.3f}  s")
    print(f"{'─'*W}")
    print(f"  Throughput  docs/s    P50 : {avg_docs_sec:>8.1f}")
    print(f"  Throughput  objs/s    P50 : {avg_obj_sec:>8.1f}")
    print(f"  Throughput  sents/s   P50 : {avg_sent_sec:>8.1f}")
    print(f"  Avg obj text chars    P50 : {avg_obj_chars:>8.0f}")
    print(f"  Max obj text chars        : {max_obj_chars:>8,}")
    print(f"{'─'*W}")
    print(f"  HTTP retries          : {d['http_retries']:>10,}  ({http_retry_pct:.1f}% of chunks affected)")
    print(f"  Partial doc retries   : {d['partial_doc_retries']:>10,}  ({partial_retry_pct:.1f}% of chunks affected)")
    print(f"  Still-failed docs     : {d['failed']:>10,}")

    lt = d["latency_per_chunk"]
    if lt:
        buckets = [
            (0,    0.5,  "  0-0.5s"),
            (0.5,  1.0,  "0.5-1.0s"),
            (1.0,  2.0,  "  1-2 s "),
            (2.0,  5.0,  "  2-5 s "),
            (5.0,  1e9,  "    5s+ "),
        ]
        print(f"{'─'*W}")
        print(f"  Bulk latency distribution:")
        print(_histogram(lt, buckets))

    slowest = sorted(d.get("slowest_chunks", []), reverse=True)[:5]
    if slowest:
        print(f"{'─'*W}")
        print(f"  Top 5 slowest chunks:")
        for t, cid in slowest:
            print(f"    {cid:<44}  {t:.3f}s")

    largest = sorted(d.get("largest_chunks", []), reverse=True)[:5]
    if largest:
        print(f"{'─'*W}")
        print(f"  Top 5 largest payloads:")
        for sz, cid in largest:
            print(f"    {cid:<44}  {sz:.2f} MB")

    print(f"{'═'*W}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Document-level index ingestion dashboard")
    p.add_argument("--hours",  type=float, default=6.0,
                   help="How many hours back to look in CloudWatch (default: 6)")
    p.add_argument("--doc",    default=None,
                   help="Filter to a specific document ID substring")
    p.add_argument("--output", default=None,
                   help="Save raw chunk summaries to this JSONL file")
    args = p.parse_args()

    print(f"Fetching [IndexSummary] logs from last {args.hours:.0f}h …", flush=True)
    summaries = _fetch_index_summaries(args.hours, args.doc)

    if not summaries:
        print("No [IndexSummary] lines found. "
              "Make sure the index Lambda has been deployed and chunks have been processed.")
        sys.exit(0)

    print(f"Found {len(summaries):,} index summaries.", flush=True)

    if args.output:
        out = Path(args.output)
        out.write_text("\n".join(json.dumps(s) for s in summaries) + "\n", encoding="utf-8")
        print(f"Raw summaries saved → {out}")

    docs = _aggregate(summaries)
    for doc_id in sorted(docs):
        _print_doc_summary(doc_id, docs[doc_id])

    # Cross-document totals if more than one doc
    if len(docs) > 1:
        total: dict = {
            "chunks":              sum(d["chunks"]              for d in docs.values()),
            "bulk_docs":           sum(d["bulk_docs"]           for d in docs.values()),
            "objects":             sum(d["objects"]             for d in docs.values()),
            "sentences":           sum(d["sentences"]           for d in docs.values()),
            "total_size_mb":       sum(d["total_size_mb"]       for d in docs.values()),
            "total_latency_s":     sum(d["total_latency_s"]     for d in docs.values()),
            "max_latency_s":       max(d["max_latency_s"]       for d in docs.values()),
            "http_retries":        sum(d["http_retries"]        for d in docs.values()),
            "partial_doc_retries": sum(d["partial_doc_retries"] for d in docs.values()),
            "failed":              sum(d["failed"]              for d in docs.values()),
            "retried_chunks":      sum(d["retried_chunks"]      for d in docs.values()),
            "partial_chunks":      sum(d["partial_chunks"]      for d in docs.values()),
            "latency_per_chunk":   [v for d in docs.values() for v in d["latency_per_chunk"]],
            "size_per_chunk":      [v for d in docs.values() for v in d["size_per_chunk"]],
            "docs_per_sec_list":   [v for d in docs.values() for v in d["docs_per_sec_list"]],
            "obj_per_sec_list":    [v for d in docs.values() for v in d["obj_per_sec_list"]],
            "sent_per_sec_list":   [v for d in docs.values() for v in d["sent_per_sec_list"]],
            "avg_obj_chars_list":  [v for d in docs.values() for v in d["avg_obj_chars_list"]],
            "max_obj_chars_list":  [v for d in docs.values() for v in d["max_obj_chars_list"]],
            "slowest_chunks":      sorted(
                (item for d in docs.values() for item in d["slowest_chunks"]),
                reverse=True,
            )[:10],
            "largest_chunks":      sorted(
                (item for d in docs.values() for item in d["largest_chunks"]),
                reverse=True,
            )[:10],
        }
        _print_doc_summary("ALL DOCUMENTS", total)


if __name__ == "__main__":
    main()
