"""
GPU Embedding Server — BGE-M3 with dynamic batching
====================================================
FastAPI server compatible with the embedding_worker Stage-2 API contract.

Dynamic batching
-----------------
Incoming requests are placed on an asyncio queue.  A background loop
collects requests up to MAX_BATCH_SIZE texts OR until BATCH_TIMEOUT_MS
elapses, whichever comes first, then runs a single model.encode() call
and dispatches results back to all waiting callers.

This means 100 concurrent Stage-2 Lambdas each sending ~16 texts get
combined into batches of ~1,600 texts on the GPU — far more efficient
than one forward-pass per HTTP request.

API
----
  POST /embed    — embed texts (Stage-2 contract)
  GET  /health   — readiness probe
  GET  /metrics  — aggregated throughput / latency stats

Env vars
---------
  MODEL_NAME          BAAI/bge-m3 (default)
  MAX_BATCH_SIZE      256 (default) — max total texts in one GPU forward-pass
  BATCH_TIMEOUT_MS    20  (default) — max wait to fill a batch
  DEVICE              auto|cpu|mps|cuda (default: auto)
  PORT                8080 (default)
  HOST                0.0.0.0 (default)

Start
------
  pip install -r services/embedding_server/requirements.txt
  python services/embedding_server/server.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import statistics
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── config ───────────────────────────────────────────────────────────────────

MODEL_NAME       = os.environ.get("MODEL_NAME",       "BAAI/bge-m3")
MAX_BATCH_SIZE   = int(os.environ.get("MAX_BATCH_SIZE",   "256"))
BATCH_TIMEOUT_MS = int(os.environ.get("BATCH_TIMEOUT_MS", "20"))
_DEVICE_ENV      = os.environ.get("DEVICE", "auto")
PORT             = int(os.environ.get("PORT", "8080"))
HOST             = os.environ.get("HOST", "0.0.0.0")

# ─── device selection ─────────────────────────────────────────────────────────

def _select_device() -> str:
    if _DEVICE_ENV != "auto":
        return _DEVICE_ENV
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = _select_device()

# ─── model (lazy, loaded once at startup) ─────────────────────────────────────

_model = None


def _get_model():
    global _model
    if _model is not None:
        return _model
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError:
        raise ImportError("Run: pip install FlagEmbedding")
    logger.info("Loading %s on %s …", MODEL_NAME, DEVICE)
    t0 = time.monotonic()
    # use_fp16=True is faster on both MPS and CUDA; ignored on CPU
    _model = BGEM3FlagModel(MODEL_NAME, use_fp16=(DEVICE != "cpu"), device=DEVICE)
    logger.info("Model ready in %.1fs  device=%s", time.monotonic() - t0, DEVICE)
    return _model


# ─── telemetry ────────────────────────────────────────────────────────────────

class _Stats:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.total_requests   = 0
        self.total_texts      = 0
        self.total_batches    = 0
        self.inference_ms_samples: list[int] = []
        self.queue_wait_ms_samples: list[int] = []
        self.batch_size_samples:    list[int] = []

    async def record(self, *, n_requests: int, n_texts: int,
                     inference_ms: int, queue_wait_ms_list: list[int]) -> None:
        async with self._lock:
            self.total_requests += n_requests
            self.total_texts    += n_texts
            self.total_batches  += 1
            self.inference_ms_samples.append(inference_ms)
            self.batch_size_samples.append(n_texts)
            self.queue_wait_ms_samples.extend(queue_wait_ms_list)
            # Keep last 10,000 samples to bound memory
            if len(self.inference_ms_samples) > 10_000:
                self.inference_ms_samples = self.inference_ms_samples[-10_000:]
            if len(self.queue_wait_ms_samples) > 10_000:
                self.queue_wait_ms_samples = self.queue_wait_ms_samples[-10_000:]
            if len(self.batch_size_samples) > 10_000:
                self.batch_size_samples = self.batch_size_samples[-10_000:]

    def snapshot(self) -> dict:
        def _pct(samples: list[int], p: float) -> int:
            if not samples:
                return -1
            s = sorted(samples)
            idx = int(p / 100 * (len(s) - 1))
            return s[idx]

        return {
            "total_requests":        self.total_requests,
            "total_texts":           self.total_texts,
            "total_batches":         self.total_batches,
            "inference_ms_avg":      int(statistics.mean(self.inference_ms_samples))   if self.inference_ms_samples   else -1,
            "inference_ms_p50":      _pct(self.inference_ms_samples,   50),
            "inference_ms_p95":      _pct(self.inference_ms_samples,   95),
            "inference_ms_p99":      _pct(self.inference_ms_samples,   99),
            "queue_wait_ms_avg":     int(statistics.mean(self.queue_wait_ms_samples))  if self.queue_wait_ms_samples  else -1,
            "queue_wait_ms_p50":     _pct(self.queue_wait_ms_samples,  50),
            "queue_wait_ms_p95":     _pct(self.queue_wait_ms_samples,  95),
            "queue_wait_ms_p99":     _pct(self.queue_wait_ms_samples,  99),
            "batch_size_avg":        round(statistics.mean(self.batch_size_samples), 1) if self.batch_size_samples else -1,
            "batch_size_max":        max(self.batch_size_samples) if self.batch_size_samples else -1,
            "device":                DEVICE,
            "model":                 MODEL_NAME,
            "max_batch_size":        MAX_BATCH_SIZE,
            "batch_timeout_ms":      BATCH_TIMEOUT_MS,
        }


_stats = _Stats()

# ─── request dataclass ────────────────────────────────────────────────────────

@dataclass
class _PendingRequest:
    request_id:  str
    document_id: str
    chunk_id:    str
    texts:       list[str]
    input_type:  str
    enqueue_ns:  int = field(default_factory=lambda: time.perf_counter_ns())
    future:      asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())


# ─── dynamic batcher ─────────────────────────────────────────────────────────

_queue: asyncio.Queue[_PendingRequest] = asyncio.Queue()


async def _batcher_loop() -> None:
    """Background task: drain queue → single GPU forward-pass → resolve futures."""
    logger.info("Batcher started  max_batch=%d  timeout_ms=%d  device=%s",
                MAX_BATCH_SIZE, BATCH_TIMEOUT_MS, DEVICE)
    model = _get_model()
    timeout_s = BATCH_TIMEOUT_MS / 1000

    while True:
        # Block until at least one request arrives
        first = await _queue.get()
        batch: list[_PendingRequest] = [first]
        deadline = time.perf_counter() + timeout_s

        # Collect more until MAX_BATCH_SIZE texts or timeout
        total_texts = len(first.texts)
        while total_texts < MAX_BATCH_SIZE:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                req = await asyncio.wait_for(_queue.get(), timeout=remaining)
                if total_texts + len(req.texts) > MAX_BATCH_SIZE:
                    # Would overflow — put back and stop collecting
                    await _queue.put(req)
                    break
                batch.append(req)
                total_texts += len(req.texts)
            except asyncio.TimeoutError:
                break

        all_texts = [t for req in batch for t in req.texts]
        n_texts   = len(all_texts)
        t_infer   = time.perf_counter()

        try:
            # Run on GPU — use_fp16 handled at model init
            output = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: model.encode(
                    all_texts,
                    batch_size=n_texts,
                    max_length=512,
                    return_dense=True,
                    return_sparse=False,   # sparse computed per-chunk in embedding_worker
                    return_colbert_vecs=False,
                ),
            )
            infer_ms = int((time.perf_counter() - t_infer) * 1000)
            dense    = output["dense_vecs"]  # numpy array (n_texts, dims)

            # Distribute results back to each waiting request
            offset = 0
            queue_waits: list[int] = []
            for req in batch:
                n = len(req.texts)
                q_wait_ms = int((t_infer - req.enqueue_ns / 1e9) * 1000)
                queue_waits.append(q_wait_ms)
                req.future.set_result({
                    "embeddings":          dense[offset: offset + n].tolist(),
                    "gpu_queue_wait_ms":   q_wait_ms,
                    "gpu_batch_size":      n_texts,
                    "gpu_inference_ms":    infer_ms,
                    "gpu_total_request_ms": q_wait_ms + infer_ms,
                })
                offset += n

            await _stats.record(
                n_requests=len(batch),
                n_texts=n_texts,
                inference_ms=infer_ms,
                queue_wait_ms_list=queue_waits,
            )
            logger.debug("batch n_req=%d n_texts=%d infer_ms=%d device=%s",
                         len(batch), n_texts, infer_ms, DEVICE)

        except Exception as exc:
            logger.exception("Batch inference failed: %s", exc)
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(exc)


# ─── API ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="RLS Embedding Server", version="1.0.0")


class EmbedBody(BaseModel):
    request_id:  str = ""
    document_id: str = ""
    chunk_id:    str = ""
    texts:       list[str]
    input_type:  str = "search_document"
    truncate:    bool = True


class EmbedResponse(BaseModel):
    embeddings:            list[list[float]]
    model:                 str
    dimensions:            int
    gpu_queue_wait_ms:     int
    gpu_batch_size:        int
    gpu_inference_ms:      int
    gpu_total_request_ms:  int


@app.on_event("startup")
async def _startup() -> None:
    _get_model()  # warm up model before accepting traffic
    asyncio.create_task(_batcher_loop())


@app.post("/embed", response_model=EmbedResponse)
async def embed(body: EmbedBody) -> EmbedResponse:
    if not body.texts:
        raise HTTPException(status_code=422, detail="texts must not be empty")

    t_total = time.perf_counter()
    req = _PendingRequest(
        request_id  = body.request_id or str(uuid.uuid4()),
        document_id = body.document_id,
        chunk_id    = body.chunk_id,
        texts       = [t[:25_000] for t in body.texts] if body.truncate else body.texts,
        input_type  = body.input_type,
        enqueue_ns  = time.perf_counter_ns(),
        future      = asyncio.get_event_loop().create_future(),
    )
    await _queue.put(req)
    result: dict = await req.future

    total_ms = int((time.perf_counter() - t_total) * 1000)
    vecs     = result["embeddings"]
    dims     = len(vecs[0]) if vecs else 0

    return EmbedResponse(
        embeddings           = vecs,
        model                = MODEL_NAME,
        dimensions           = dims,
        gpu_queue_wait_ms    = result["gpu_queue_wait_ms"],
        gpu_batch_size       = result["gpu_batch_size"],
        gpu_inference_ms     = result["gpu_inference_ms"],
        gpu_total_request_ms = total_ms,
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}


@app.get("/metrics")
async def metrics() -> dict:
    return _stats.snapshot()


# ─── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
