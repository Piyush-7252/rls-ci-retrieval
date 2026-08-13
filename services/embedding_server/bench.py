"""
Benchmark — BGE-M3 throughput on M4 / MPS
==========================================
Measures texts/sec, tokens/sec, and latency percentiles for batch sizes
16, 32, 64, 128, 256 against a running embedding server.

Usage
------
  # Start server first (in another terminal):
  python services/embedding_server/server.py

  # Run benchmark:
  python services/embedding_server/bench.py

  # Against a remote server:
  SERVER_URL=http://10.0.1.50:8080 python services/embedding_server/bench.py

  # Override batch sizes to test:
  BATCH_SIZES=32,64,128 python services/embedding_server/bench.py

  # Use a real document's chunks as input:
  INPUT_FILE=localfiles/ci/132.json python services/embedding_server/bench.py
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from pathlib import Path

import aiohttp

SERVER_URL   = os.environ.get("SERVER_URL",   "http://localhost:8080")
BATCH_SIZES  = [int(x) for x in os.environ.get("BATCH_SIZES", "16,32,64,128,256").split(",")]
WARMUP_CALLS = int(os.environ.get("WARMUP_CALLS", "3"))
BENCH_CALLS  = int(os.environ.get("BENCH_CALLS",  "20"))
CONCURRENCY  = int(os.environ.get("CONCURRENCY",  "10"))  # simulate 10 Lambda workers
INPUT_FILE   = os.environ.get("INPUT_FILE",   "")

# ─── sample texts ─────────────────────────────────────────────────────────────

_DEFAULT_TEXTS = [
    "The primary endpoint of the study was overall survival at 24 months.",
    "Patients with moderate renal impairment (eGFR 30-59 mL/min/1.73m²) received a reduced dose.",
    "Adverse events of grade 3 or higher were observed in 23% of the treatment arm.",
    "Randomisation was performed using a stratified permuted-block design.",
    "The investigational medicinal product was administered as a 30-minute IV infusion.",
    "Concomitant use of strong CYP3A4 inhibitors was prohibited throughout the treatment period.",
    "Protocol amendment 2 revised the inclusion criteria to lower the minimum age to 18 years.",
    "Secondary endpoints included progression-free survival, objective response rate, and quality of life.",
    "Patients were stratified by ECOG performance status (0-1 vs 2) and prior therapy.",
    "The pharmacokinetic profile showed dose-proportional exposure with a half-life of 14 hours.",
    "No clinically meaningful drug-drug interactions were identified with standard-of-care medications.",
    "The study was conducted across 47 sites in 12 countries in accordance with ICH E6 GCP.",
    "An independent Data Safety Monitoring Board reviewed unblinded safety data every 6 months.",
    "Efficacy was assessed by an independent radiological review committee using RECIST v1.1.",
    "The Kaplan-Meier median overall survival was 18.3 months (95% CI: 15.1-21.4).",
    "Complete response was observed in 12 patients (8%) in the experimental arm.",
]


def _load_texts_from_file(path: str) -> list[str]:
    """Extract text strings from a CI or chunk JSON file."""
    data = json.loads(Path(path).read_text())
    texts: list[str] = []
    if isinstance(data, list):
        for item in data:
            t = item.get("knownCI") or item.get("text") or item.get("normalized_text", "")
            if t:
                texts.append(str(t)[:500])
    elif isinstance(data, dict):
        t = data.get("text") or data.get("normalized_text", "")
        if t:
            texts.append(str(t)[:500])
    return texts or _DEFAULT_TEXTS


def _build_text_batch(size: int, base_texts: list[str]) -> list[str]:
    """Repeat base texts to reach target batch size."""
    result = []
    while len(result) < size:
        result.extend(base_texts)
    return result[:size]


# ─── benchmark runner ─────────────────────────────────────────────────────────

async def _single_call(session: aiohttp.ClientSession, texts: list[str]) -> dict:
    t0 = time.perf_counter()
    async with session.post(
        f"{SERVER_URL}/embed",
        json={
            "texts":      texts,
            "input_type": "search_document",
            "truncate":   True,
        },
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
    total_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "total_ms":         total_ms,
        "gpu_queue_wait_ms":    data.get("gpu_queue_wait_ms", -1),
        "gpu_batch_size":       data.get("gpu_batch_size", len(texts)),
        "gpu_inference_ms":     data.get("gpu_inference_ms", -1),
        "gpu_total_request_ms": data.get("gpu_total_request_ms", -1),
        "n_texts":          len(texts),
        "dimensions":       data.get("dimensions", 0),
    }


async def _bench_batch_size(
    session: aiohttp.ClientSession,
    batch_size: int,
    base_texts: list[str],
) -> None:
    texts = _build_text_batch(batch_size, base_texts)

    # Warm up
    for _ in range(WARMUP_CALLS):
        await _single_call(session, texts)

    # Benchmark: CONCURRENCY concurrent workers each making BENCH_CALLS requests
    async def _worker() -> list[dict]:
        results = []
        for _ in range(BENCH_CALLS):
            r = await _single_call(session, texts)
            results.append(r)
        return results

    t_wall = time.perf_counter()
    all_results_nested = await asyncio.gather(*[_worker() for _ in range(CONCURRENCY)])
    wall_s = time.perf_counter() - t_wall

    results = [r for worker in all_results_nested for r in worker]
    total_calls  = len(results)
    total_texts  = sum(r["n_texts"] for r in results)

    def _pct(key: str, p: float) -> int:
        samples = sorted(r[key] for r in results if r[key] >= 0)
        if not samples:
            return -1
        idx = int(p / 100 * (len(samples) - 1))
        return samples[idx]

    texts_per_sec = int(total_texts / wall_s)

    # Estimate tokens (assume avg ~75 tokens per text for clinical text)
    avg_tokens_per_text = 75
    tokens_per_sec = texts_per_sec * avg_tokens_per_text

    print(f"\n{'─'*62}")
    print(f"  Batch size (per Lambda call): {batch_size:>6}")
    print(f"  Concurrency (Lambda workers): {CONCURRENCY:>6}")
    print(f"  Total calls:                  {total_calls:>6}")
    print(f"  Wall clock:                   {wall_s:>6.1f}s")
    print(f"{'─'*62}")
    print(f"  Throughput (texts/sec):     {texts_per_sec:>8,}")
    print(f"  Throughput (tokens/sec):    {tokens_per_sec:>8,}  (est. @{avg_tokens_per_text} tok/text)")
    print(f"{'─'*62}")
    print(f"  GPU batch size actual:      {_pct('gpu_batch_size', 50):>8}")
    print(f"  GPU inference ms  P50:      {_pct('gpu_inference_ms', 50):>8}")
    print(f"  GPU inference ms  P95:      {_pct('gpu_inference_ms', 95):>8}")
    print(f"  GPU inference ms  P99:      {_pct('gpu_inference_ms', 99):>8}")
    print(f"  GPU queue wait ms P50:      {_pct('gpu_queue_wait_ms', 50):>8}")
    print(f"  GPU queue wait ms P95:      {_pct('gpu_queue_wait_ms', 95):>8}")
    print(f"  Total request ms  P50:      {_pct('total_ms', 50):>8}")
    print(f"  Total request ms  P95:      {_pct('total_ms', 95):>8}")
    print(f"  Total request ms  P99:      {_pct('total_ms', 99):>8}")
    print(f"{'─'*62}")


async def _main() -> None:
    base_texts = _load_texts_from_file(INPUT_FILE) if INPUT_FILE else _DEFAULT_TEXTS
    print(f"\nRLS Embedding Benchmark — {SERVER_URL}")
    print(f"Model loaded from server /metrics")

    async with aiohttp.ClientSession() as session:
        # Check server health
        try:
            async with session.get(f"{SERVER_URL}/health") as r:
                info = await r.json()
            print(f"Device: {info.get('device')}   Model: {info.get('model')}")
        except Exception as exc:
            print(f"ERROR: Server not reachable at {SERVER_URL}: {exc}")
            return

        for bs in BATCH_SIZES:
            await _bench_batch_size(session, bs, base_texts)

    print("\nBenchmark complete.")
    print(f"\nTo estimate GPU requirements:")
    print(f"  required_GPUs = (target_docs_per_hour × texts_per_doc) / (texts_per_sec × 3600)")


if __name__ == "__main__":
    asyncio.run(_main())
