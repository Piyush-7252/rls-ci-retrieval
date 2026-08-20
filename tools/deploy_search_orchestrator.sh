#!/usr/bin/env bash
# Deploy the rls-search-orchestrator Lambda (search fan-out orchestrator).
#
# Required env vars:
#   ROLE_ARN   — IAM role ARN for the Lambda function
#
# Optional overrides:
#   FUNCTION_NAME, ECR_REPO, AWS_REGION, TIMEOUT, MEMORY_SIZE,
#   OPENSEARCH_ENDPOINT, OPENSEARCH_CI_INDEX, WORKER_LAMBDA_ARN
#
# WORKER_LAMBDA_ARN is resolved automatically from rls-search-worker if not set.
#
# Example:
#   ROLE_ARN=arn:aws:iam::064051750322:role/service-role/rls-llm-cim-annotation-role-wgwvt366 \
#   tools/deploy_search_orchestrator.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

AWS_REGION="${AWS_REGION:-eu-west-1}"
FUNCTION_NAME="${FUNCTION_NAME:-rls-ci-retrieval-search-orchestrator}"
ECR_REPO="${ECR_REPO:-rls-ci-retrieval-search-orchestrator}"
IMAGE_TAG="${IMAGE_TAG:-$(date +%Y%m%d-%H%M%S)}"
LATEST_TAG="${LATEST_TAG:-latest}"
ROLE_ARN="${ROLE_ARN:-}"
TIMEOUT="${TIMEOUT:-900}"
MEMORY_SIZE="${MEMORY_SIZE:-10240}"

CI_LOOKUP_WORKERS="${CI_LOOKUP_WORKERS:-10}"
MAX_WORKERS="${MAX_WORKERS:-3}"
OPENSEARCH_ENDPOINT="${OPENSEARCH_ENDPOINT:-search-rls-dev-rhitzxwnctmuyq2l4kny5kwelu.eu-west-1.es.amazonaws.com}"
OPENSEARCH_CI_INDEX="${OPENSEARCH_CI_INDEX:-ci-objects}"
OPENSEARCH_MAXSIZE="${OPENSEARCH_MAXSIZE:-128}"
RESULTS_BUCKET="${RESULTS_BUCKET:-rls-file-bucket-eu}"
RESULTS_PREFIX="${RESULTS_PREFIX:-rls-ci-retrieval-search-results}"

# Resolve WORKER_LAMBDA_ARN automatically if not provided.
WORKER_LAMBDA_NAME="${WORKER_LAMBDA_NAME:-rls-ci-retrieval-search-worker}"
if [[ -z "${WORKER_LAMBDA_ARN:-}" ]]; then
  echo "WORKER_LAMBDA_ARN not set — resolving from function '${WORKER_LAMBDA_NAME}'..."
  WORKER_LAMBDA_ARN="$(aws lambda get-function-configuration \
    --function-name "$WORKER_LAMBDA_NAME" \
    --region "$AWS_REGION" \
    --query FunctionArn --output text 2>/dev/null || echo '')"
  if [[ -z "$WORKER_LAMBDA_ARN" ]]; then
    echo "ERROR: could not resolve ARN for '${WORKER_LAMBDA_NAME}'."
    echo "Deploy rls-search-worker first, or set WORKER_LAMBDA_ARN explicitly."
    exit 1
  fi
  echo "  → ${WORKER_LAMBDA_ARN}"
fi

if [[ -z "$ROLE_ARN" ]]; then
  echo "ERROR: ROLE_ARN is required"
  echo "Example: ROLE_ARN=arn:aws:iam::<acct>:role/... tools/deploy_search_orchestrator.sh"
  exit 1
fi

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"
IMAGE_URI="${ECR_URI}:${IMAGE_TAG}"
LATEST_IMAGE_URI="${ECR_URI}:${LATEST_TAG}"

aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION" >/dev/null

aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

wait_for_lambda_update() {
  aws lambda wait function-updated-v2 \
    --function-name "$FUNCTION_NAME" \
    --region "$AWS_REGION"
}

retry_lambda_update() {
  local attempt=1
  local max_attempts=6
  while true; do
    if "$@"; then
      return 0
    fi
    local exit_code=$?
    if [[ "$attempt" -ge "$max_attempts" ]]; then
      echo "ERROR: command failed after ${attempt} attempts: $*"
      return "$exit_code"
    fi
    echo "Lambda is busy (attempt ${attempt}/${max_attempts}). Waiting and retrying..."
    wait_for_lambda_update || true
    attempt=$((attempt + 1))
  done
}

echo "Building and pushing image: ${IMAGE_URI}"
if ! docker buildx inspect >/dev/null 2>&1; then
  docker buildx create --use >/dev/null
fi

CACHE_IMAGE_URI="${ECR_URI}:buildcache"

docker buildx build \
  --progress=plain \
  --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  --push \
  --cache-from "type=registry,ref=${CACHE_IMAGE_URI}" \
  --cache-to   "type=registry,ref=${CACHE_IMAGE_URI},mode=max,image-manifest=true,oci-mediatypes=true" \
  -f "$ROOT_DIR/lambdas/orchestrator/search-orchestrator/Dockerfile" \
  -t "$IMAGE_URI" \
  -t "$LATEST_IMAGE_URI" \
  "$ROOT_DIR"

if aws lambda get-function --function-name "$FUNCTION_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
  retry_lambda_update aws lambda update-function-code \
    --function-name "$FUNCTION_NAME" \
    --image-uri "$IMAGE_URI" \
    --region "$AWS_REGION" >/dev/null
else
  aws lambda create-function \
    --function-name "$FUNCTION_NAME" \
    --package-type Image \
    --code ImageUri="$IMAGE_URI" \
    --role "$ROLE_ARN" \
    --timeout "$TIMEOUT" \
    --memory-size "$MEMORY_SIZE" \
    --region "$AWS_REGION" >/dev/null
fi

wait_for_lambda_update

retry_lambda_update aws lambda update-function-configuration \
  --function-name "$FUNCTION_NAME" \
  --timeout "$TIMEOUT" \
  --memory-size "$MEMORY_SIZE" \
  --environment "Variables={CI_LOOKUP_WORKERS=${CI_LOOKUP_WORKERS},MAX_WORKERS=${MAX_WORKERS},OPENSEARCH_ENDPOINT=${OPENSEARCH_ENDPOINT},OPENSEARCH_CI_INDEX=${OPENSEARCH_CI_INDEX},OPENSEARCH_MAXSIZE=${OPENSEARCH_MAXSIZE},WORKER_LAMBDA_ARN=${WORKER_LAMBDA_ARN},RESULTS_BUCKET=${RESULTS_BUCKET},RESULTS_PREFIX=${RESULTS_PREFIX}}" \
  --region "$AWS_REGION" >/dev/null

wait_for_lambda_update

echo ""
echo "Deployed ${FUNCTION_NAME}"
echo "Image:         ${IMAGE_URI}"
echo "Worker ARN:    ${WORKER_LAMBDA_ARN}"
echo "Results S3:    s3://${RESULTS_BUCKET}/${RESULTS_PREFIX}/"
