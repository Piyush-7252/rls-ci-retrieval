#!/usr/bin/env python3
"""
test_orchestrator_local.py — Test orchestrator Lambda on AWS with event payload.

Usage:
  python3 tests/test_orchestrator_local.py \
    --document-id "20260726062234599_4xs0l7p_10993_REDACTED_Protocol-Amendment-1-FD-64407564MMY3002-218114_1245209" \
    --ci-file "localfiles/ci/ahmedCis.json" \
    --max-cis 5 \
    --batch-size 5 \
    --function-name "rls-ci-retrieval-search-orchestrator" \
    --aws-region "eu-west-1" \
    --output "localfiles/orchestrator_test_response.json"
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

# Add root to path
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_cis(ci_file: str, max_cis: int | None = None) -> list[dict]:
    """Load CIs from JSON file."""
    ci_path = ROOT / ci_file
    logger.info(f"Loading CIs from {ci_path}")
    
    with open(ci_path) as f:
        cis = json.load(f)
    
    if not isinstance(cis, list):
        raise ValueError(f"Expected list of CIs, got {type(cis)}")
    
    if max_cis:
        cis = cis[:max_cis]
    
    logger.info(f"Loaded {len(cis)} CIs")
    return cis


def load_document_context(document_id: str) -> dict:
    """Load document context from document_assets.json if available."""
    doc_assets_path = ROOT / "localfiles" / "doc_cache" / f"{document_id}.json"
    
    if doc_assets_path.exists():
        logger.info(f"Loading document context from {doc_assets_path}")
        with open(doc_assets_path) as f:
            return json.load(f)
    
    logger.warning(f"Document context not found at {doc_assets_path}, using empty context")
    return {
        "document_id": document_id,
        "pages": 0,
        "chunks_indexed": 0,
    }


def build_event(
    document_id: str,
    cis: list[dict],
    document_context: dict | None = None,
    batch_size: int = 5,
    skip_rerank: bool = False,
    skip_verify: bool = False,
) -> dict:
    """Build orchestrator Lambda event."""
    return {
        "search_id": f"local-test-{datetime.now():%Y%m%d_%H%M%S}",
        "document_id": document_id,
        "cis": cis,
        "document_context": document_context or {},
        "batch_size": batch_size,
        "skip_rerank": skip_rerank,
        "skip_verify": skip_verify,
        "tenant": {
            "name": "default",
            "id": "default-tenant",
        }
    }


def invoke_orchestrator_lambda(event: dict, function_name: str = "rls-ci-retrieval-search-orchestrator") -> dict:
    """Invoke orchestrator Lambda on AWS."""
    import boto3
    
    logger.info(f"Invoking orchestrator Lambda: {function_name}")
    logger.info(f"  CIs: {len(event['cis'])}")
    logger.info(f"  Search ID: {event['search_id']}")
    logger.info(f"  Document: {event['document_id'][:60]}...")
    
    # Create Lambda client
    lambda_client = boto3.client("lambda", region_name=os.environ.get("AWS_REGION", "eu-west-1"))
    
    t0 = time.perf_counter()
    try:
        # Invoke Lambda synchronously
        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType="RequestResponse",  # Synchronous (wait for response)
            Payload=json.dumps(event),
        )
        elapsed = time.perf_counter() - t0
        
        # Parse response
        if response["StatusCode"] != 200:
            raise RuntimeError(f"Lambda returned status {response['StatusCode']}")
        
        payload = json.loads(response["Payload"].read())
        
        # Check for Lambda errors
        if "FunctionError" in response:
            logger.error(f"❌ Lambda invocation error: {payload}")
            raise RuntimeError(f"Lambda error: {payload}")
        
        logger.info(f"✅ Lambda invocation completed in {elapsed:.1f}s")
        logger.info(f"   Status: {payload.get('status')}")
        logger.info(f"   CIs: {payload.get('completed_cis')}/{payload.get('expected_cis')} completed")
        logger.info(f"   Failures: {payload.get('failed_cis')}")
        
        return payload
    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.error(f"❌ Lambda invocation failed after {elapsed:.1f}s: {e}", exc_info=True)
        raise


def save_response(response: dict, output_file: str) -> None:
    """Save response JSON to file."""
    output_path = ROOT / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(response, f, indent=2)
    
    logger.info(f"✅ Response saved to {output_path}")
    logger.info(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test orchestrator Lambda locally")
    parser.add_argument("--document-id", required=True, help="Document ID")
    parser.add_argument("--ci-file", required=True, help="Path to CI JSON file")
    parser.add_argument("--max-cis", type=int, default=None, help="Max CIs to test (default: all)")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size (default: 5)")
    parser.add_argument("--skip-rerank", action="store_true", help="Skip reranking")
    parser.add_argument("--skip-verify", action="store_true", help="Skip verification")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--function-name", default="rls-ci-retrieval-search-orchestrator", help="Lambda function name")
    parser.add_argument("--aws-region", default=os.environ.get("AWS_REGION", "eu-west-1"), help="AWS region")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Orchestrator Lambda Test (AWS)")
    logger.info("=" * 80)
    logger.info(f"Function: {args.function_name}")
    logger.info(f"Region: {args.aws_region}")
    logger.info(f"Document ID: {args.document_id[:70]}...")
    logger.info(f"CI file: {args.ci_file}")
    logger.info(f"Max CIs: {args.max_cis or 'all'}")
    logger.info(f"Batch size: {args.batch_size}")
    print()
    
    # Set AWS region
    os.environ["AWS_REGION"] = args.aws_region
    
    # Load CIs and context
    cis = load_cis(args.ci_file, args.max_cis)
    doc_context = load_document_context(args.document_id)
    
    # Build event
    event = build_event(
        document_id=args.document_id,
        cis=cis,
        document_context=doc_context,
        batch_size=args.batch_size,
        skip_rerank=args.skip_rerank,
        skip_verify=args.skip_verify,
    )
    
    # Invoke
    t_start = time.perf_counter()
    response = invoke_orchestrator_lambda(event, function_name=args.function_name)
    total_time = time.perf_counter() - t_start
    
    # Save response
    save_response(response, args.output)
    
    print()
    logger.info("=" * 80)
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
