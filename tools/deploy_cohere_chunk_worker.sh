#!/usr/bin/env bash
# Deploys rls-ci-chunk-worker-cohere — same image as Titan worker,
# EMBEDDING_PROVIDER=cohere selects lambdas/embedding_cohere at runtime.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

AWS_REGION="${AWS_REGION:-eu-west-1}"
FUNCTION_NAME="${FUNCTION_NAME:-rls-ci-chunk-worker-cohere}"
ECR_REPO="${ECR_REPO:-rls-ci-chunk-worker}"
IMAGE_TAG="${IMAGE_TAG:-$(date +%Y%m%d-%H%M%S)}"
LATEST_TAG="${LATEST_TAG:-latest-cohere}"
ROLE_ARN="${ROLE_ARN:-}"
TIMEOUT="${TIMEOUT:-900}"
MEMORY_SIZE="${MEMORY_SIZE:-10240}"

# ── Cohere-specific env vars ──────────────────────────────────────────────────
EMBEDDING_PROVIDER="${EMBEDDING_PROVIDER:-cohere}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-cohere.embed-v4:0}"
EMBEDDING_DEBUG="${EMBEDDING_DEBUG:-true}"
COHERE_BATCH_SIZE="${COHERE_BATCH_SIZE:-96}"

# ── Infrastructure ─────────────────────────────────────────────────────────────
OPENSEARCH_ENDPOINT="${OPENSEARCH_ENDPOINT:-search-rls-dev-rhitzxwnctmuyq2l4kny5kwelu.eu-west-1.es.amazonaws.com}"
OPENSEARCH_INDEX="${OPENSEARCH_INDEX:-document-chunks}"
SEMANTIC_OBJECTS_INDEX="${SEMANTIC_OBJECTS_INDEX:-semantic-objects-cohere}"
OPENSEARCH_CI_INDEX="${OPENSEARCH_CI_INDEX:-ci-objects}"
NER_MODEL="${NER_MODEL:-gliner}"
HF_TOKEN="${HF_TOKEN:-}"

if [[ -z "$ROLE_ARN" ]]; then
  echo "ERROR: ROLE_ARN is required"
  echo "Example: ROLE_ARN=arn:aws:iam::<acct>:role/rls-ci-worker-role tools/deploy_cohere_chunk_worker.sh"
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

CACHE_IMAGE_URI="${ECR_URI}:buildcache-cohere"

docker buildx build \
  --progress=plain \
  --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  --push \
  --cache-from "type=registry,ref=${CACHE_IMAGE_URI}" \
  --cache-to   "type=registry,ref=${CACHE_IMAGE_URI},mode=max,image-manifest=true,oci-mediatypes=true" \
  -f "$ROOT_DIR/lambdas/chunk_worker/Dockerfile" \
  -t "$IMAGE_URI" \
  -t "$LATEST_IMAGE_URI" \
  "$ROOT_DIR"

ENV_VARS="OPENSEARCH_ENDPOINT=${OPENSEARCH_ENDPOINT}"
ENV_VARS+=",OPENSEARCH_INDEX=${OPENSEARCH_INDEX}"
ENV_VARS+=",SEMANTIC_OBJECTS_INDEX=${SEMANTIC_OBJECTS_INDEX}"
ENV_VARS+=",OPENSEARCH_CI_INDEX=${OPENSEARCH_CI_INDEX}"
ENV_VARS+=",NER_MODEL=${NER_MODEL}"
ENV_VARS+=",EMBEDDING_PROVIDER=${EMBEDDING_PROVIDER}"
ENV_VARS+=",EMBEDDING_MODEL=${EMBEDDING_MODEL}"
ENV_VARS+=",EMBEDDING_DEBUG=${EMBEDDING_DEBUG}"
ENV_VARS+=",COHERE_BATCH_SIZE=${COHERE_BATCH_SIZE}"
ENV_VARS+=",HF_HOME=/tmp/hf_cache"
[[ -n "$HF_TOKEN" ]] && ENV_VARS+=",HF_TOKEN=${HF_TOKEN}"

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
  --environment "Variables={${ENV_VARS}}" \
  --region "$AWS_REGION" >/dev/null

wait_for_lambda_update

echo "Deployed ${FUNCTION_NAME}"
echo "Image: ${IMAGE_URI}"
echo
echo "EMBEDDING_PROVIDER : ${EMBEDDING_PROVIDER}"
echo "EMBEDDING_MODEL    : ${EMBEDDING_MODEL}"
echo "EMBEDDING_DEBUG    : ${EMBEDDING_DEBUG}"
echo "SEMANTIC_OBJECTS_INDEX : ${SEMANTIC_OBJECTS_INDEX}"
echo
echo "Next: connect a Cohere SQS queue to this Lambda"
echo "aws lambda create-event-source-mapping \\"
echo "  --function-name ${FUNCTION_NAME} \\"
echo "  --event-source-arn <COHERE_SQS_QUEUE_ARN> \\"
echo "  --batch-size 1 \\"
echo "  --function-response-types ReportBatchItemFailures \\"
echo "  --scaling-config MaximumConcurrency=10 \\"
echo "  --region ${AWS_REGION}"
