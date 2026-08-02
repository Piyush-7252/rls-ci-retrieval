#!/usr/bin/env bash
set -euo pipefail

# Required inputs
DOCUMENT_ID="${DOCUMENT_ID:-}"
S3_BUCKET="${S3_BUCKET:-}"
S3_KEY="${S3_KEY:-}"
FULL_TABLES_KEY="${FULL_TABLES_KEY:-}"
ROLE_ARN="${ROLE_ARN:-}"

# Optional inputs
AWS_REGION="${AWS_REGION:-eu-west-1}"
FUNCTION_NAME="${FUNCTION_NAME:-rls-ci-chunk-worker}"
QUEUE_NAME="${QUEUE_NAME:-rls-ci-chunk-queue}"
DLQ_NAME="${DLQ_NAME:-rls-ci-chunk-dlq}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-50}"
EMBEDDING_MAX_WORKERS="${EMBEDDING_MAX_WORKERS:-8}"
TIMEOUT="${TIMEOUT:-900}"
MEMORY_SIZE="${MEMORY_SIZE:-10240}"
OS_ENDPOINT="${OS_ENDPOINT:-search-rls-dev-rhitzxwnctmuyq2l4kny5kwelu.eu-west-1.es.amazonaws.com}"
TOTAL_CHUNKS="${TOTAL_CHUNKS:-27666}"
LIMIT="${LIMIT:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "$DOCUMENT_ID" || -z "$S3_BUCKET" || -z "$S3_KEY" || -z "$FULL_TABLES_KEY" ]]; then
  echo "ERROR: Set DOCUMENT_ID, S3_BUCKET, S3_KEY, FULL_TABLES_KEY"
  exit 1
fi

if [[ -z "$ROLE_ARN" ]]; then
  echo "ERROR: Set ROLE_ARN for Lambda deployment"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/4] Deploy chunk worker image Lambda"
AWS_REGION="$AWS_REGION" \
FUNCTION_NAME="$FUNCTION_NAME" \
ROLE_ARN="$ROLE_ARN" \
TIMEOUT="$TIMEOUT" \
MEMORY_SIZE="$MEMORY_SIZE" \
OPENSEARCH_ENDPOINT="$OS_ENDPOINT" \
EMBEDDING_MAX_WORKERS="$EMBEDDING_MAX_WORKERS" \
tools/deploy_chunk_worker.sh

echo "[2/4] Create queue + DLQ + event source mapping"
SETUP_OUT="$(AWS_REGION="$AWS_REGION" FUNCTION_NAME="$FUNCTION_NAME" QUEUE_NAME="$QUEUE_NAME" DLQ_NAME="$DLQ_NAME" MAX_CONCURRENCY="$MAX_CONCURRENCY" tools/setup_fanout_queue.sh)"
echo "$SETUP_OUT"
QUEUE_URL="$(echo "$SETUP_OUT" | awk -F= '/^QUEUE_URL=/{print $2}')"

if [[ -z "$QUEUE_URL" ]]; then
  echo "ERROR: failed to resolve QUEUE_URL"
  exit 1
fi

echo "[3/4] Dispatch chunks from full_tables.json"
START_EPOCH="$(date +%s)"
DISPATCH_ARGS=(
  --document-id "$DOCUMENT_ID"
  --s3-bucket "$S3_BUCKET"
  --s3-key "$S3_KEY"
  --full-tables-key "$FULL_TABLES_KEY"
  --queue-url "$QUEUE_URL"
  --region "$AWS_REGION"
)

if [[ "$LIMIT" != "0" ]]; then
  DISPATCH_ARGS+=(--limit "$LIMIT")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  python3 tools/dispatch_chunks_to_sqs.py "${DISPATCH_ARGS[@]}" --dry-run
  exit 0
fi

python3 tools/dispatch_chunks_to_sqs.py "${DISPATCH_ARGS[@]}"

echo "[4/4] Live progress snapshot"
python3 tools/fanout_progress.py \
  --queue-url "$QUEUE_URL" \
  --os-endpoint "$OS_ENDPOINT" \
  --region "$AWS_REGION" \
  --document-id "$DOCUMENT_ID" \
  --total-chunks "$TOTAL_CHUNKS" \
  --start-epoch "$START_EPOCH"

echo "Done. For live watch:"
echo "python3 tools/fanout_progress.py --queue-url \"$QUEUE_URL\" --os-endpoint \"$OS_ENDPOINT\" --region \"$AWS_REGION\" --document-id \"$DOCUMENT_ID\" --total-chunks \"$TOTAL_CHUNKS\" --start-epoch \"$START_EPOCH\" --watch 30"
