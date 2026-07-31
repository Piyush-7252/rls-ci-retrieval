"""
Document Pipeline — Stage 1: Orchestrator
==========================================
Triggered by : S3 PutObject event (PDF upload) or API Gateway
Fan-out to   : Extraction Lambda (one async invocation per page range)

Input event
-----------
{
    "document_id": str,          # unique ID for this document
    "s3_bucket":   str,          # S3 bucket containing the PDF
    "s3_key":      str,          # S3 object key
    "total_pages": int,          # total pages in the PDF
    "chunk_size":  int           # pages per chunk (optional, default = CHUNK_SIZE env)
}

Output
------
{
    "document_id":       str,
    "chunks_dispatched": int,
    "chunk_ids":         list[str]
}
"""

from __future__ import annotations

import json
import logging
import math
import os
import uuid
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EXTRACTION_LAMBDA_ARN = os.environ.get("EXTRACTION_LAMBDA_ARN", "")
CHUNK_SIZE            = int(os.environ.get("CHUNK_SIZE", "20"))

# ─── lazy AWS client ──────────────────────────────────────────────────────────
_aws: dict = {}

def _get(service: str):
    if service not in _aws:
        import boto3
        _aws[service] = boto3.client(service)
    return _aws[service]


# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    document_id = event.get("document_id") or str(uuid.uuid4())
    logger.info("[Orchestrator] start document_id=%s", document_id)

    try:
        result = _process(event, document_id)
    except Exception as exc:
        logger.error("[Orchestrator] failed document_id=%s error=%s", document_id, exc)
        raise

    logger.info(
        "[Orchestrator] done document_id=%s chunks_dispatched=%d",
        document_id,
        result["chunks_dispatched"],
    )
    return result


def _process(event: dict, document_id: str) -> dict:
    total_pages = int(event["total_pages"])
    chunk_size  = int(event.get("chunk_size", CHUNK_SIZE))
    s3_bucket   = event["s3_bucket"]
    s3_key      = event["s3_key"]

    chunks = _build_page_ranges(document_id, s3_bucket, s3_key, total_pages, chunk_size)

    for chunk in chunks:
        _get("lambda").invoke(
            FunctionName   = EXTRACTION_LAMBDA_ARN,
            InvocationType = "Event",          # async — fire and forget
            Payload        = json.dumps(chunk).encode(),
        )
        logger.info(
            "[Orchestrator] dispatched chunk_id=%s pages=%d-%d",
            chunk["chunk_id"],
            chunk["page_start"],
            chunk["page_end"],
        )

    return {
        "document_id":       document_id,
        "chunks_dispatched": len(chunks),
        "chunk_ids":         [c["chunk_id"] for c in chunks],
    }


def _build_page_ranges(
    document_id: str,
    s3_bucket:   str,
    s3_key:      str,
    total_pages: int,
    chunk_size:  int,
) -> list[dict]:
    chunks = []
    for start in range(1, total_pages + 1, chunk_size):
        end      = min(start + chunk_size - 1, total_pages)
        chunk_id = f"{document_id}_chunk_{len(chunks):04d}"
        chunks.append({
            "document_id": document_id,
            "chunk_id":    chunk_id,
            "s3_bucket":   s3_bucket,
            "s3_key":      s3_key,
            "page_start":  start,
            "page_end":    end,
        })
    return chunks
