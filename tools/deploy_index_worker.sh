#!/usr/bin/env bash
# Deploy rls-index-worker (Stage 3: S3 embedded chunk → OpenSearch).
#
# Required:
#   ROLE_ARN              IAM role ARN
#   OPENSEARCH_ENDPOINT   OpenSearch domain endpoint
#
# Example:
#   ROLE_ARN=arn:aws:iam::064051750322:role/rls-ci-worker-role \
#   tools/deploy_index_worker.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

AWS_REGION="${AWS_REGION:-eu-west-1}"
FUNCTION_NAME="${FUNCTION_NAME:-rls-index-worker}"
ECR_REPO="${ECR_REPO:-rls-index-worker}"
IMAGE_TAG="${IMAGE_TAG:-$(date +%Y%m%d-%H%M%S)}"
LATEST_TAG="${LATEST_TAG:-latest}"
ROLE_ARN="${ROLE_ARN:-}"
TIMEOUT="${TIMEOUT:-120}"
MEMORY_SIZE="${MEMORY_SIZE:-512}"

OPENSEARCH_ENDPOINT="${OPENSEARCH_ENDPOINT:-search-rls-dev-rhitzxwnctmuyq2l4kny5kwelu.eu-west-1.es.amazonaws.com}"
OPENSEARCH_INDEX="${OPENSEARCH_INDEX:-document-chunks}"
SEMANTIC_OBJECTS_INDEX="${SEMANTIC_OBJECTS_INDEX:-semantic-objects}"
OPENSEARCH_CI_INDEX="${OPENSEARCH_CI_INDEX:-ci-objects}"
NOTIFY_SERVER_URL="${NOTIFY_SERVER_URL:-}"
NOTIFY_SERVER_TIMEOUT="${NOTIFY_SERVER_TIMEOUT:-5}"

if [[ -z "$ROLE_ARN" ]]; then echo "ERROR: ROLE_ARN is required"; exit 1; fi

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"
IMAGE_URI="${ECR_URI}:${IMAGE_TAG}"
LATEST_IMAGE_URI="${ECR_URI}:${LATEST_TAG}"

aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION" >/dev/null

aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

wait_for_lambda_update() {
  aws lambda wait function-updated-v2 --function-name "$FUNCTION_NAME" --region "$AWS_REGION"
}

retry_lambda_update() {
  local attempt=1; local max=6
  while true; do
    if "$@"; then return 0; fi
    local ec=$?
    [[ "$attempt" -ge "$max" ]] && { echo "ERROR: failed after ${attempt} attempts"; return "$ec"; }
    echo "Lambda busy (attempt ${attempt}/${max}). Retrying…"
    wait_for_lambda_update || true
    attempt=$((attempt + 1))
  done
}

echo "Building and pushing: ${IMAGE_URI}"
[[ -z "$(docker buildx inspect 2>/dev/null)" ]] && docker buildx create --use >/dev/null

CACHE_URI="${ECR_URI}:buildcache"
docker buildx build \
  --progress=plain --platform linux/amd64 --provenance=false --sbom=false --push \
  --cache-from "type=registry,ref=${CACHE_URI}" \
  --cache-to   "type=registry,ref=${CACHE_URI},mode=max,image-manifest=true,oci-mediatypes=true" \
  -f "$ROOT_DIR/lambdas/index_worker/Dockerfile" \
  -t "$IMAGE_URI" -t "$LATEST_IMAGE_URI" \
  "$ROOT_DIR"

ENV_VARS="Variables={OPENSEARCH_ENDPOINT=${OPENSEARCH_ENDPOINT},OPENSEARCH_INDEX=${OPENSEARCH_INDEX},SEMANTIC_OBJECTS_INDEX=${SEMANTIC_OBJECTS_INDEX},OPENSEARCH_CI_INDEX=${OPENSEARCH_CI_INDEX},NOTIFY_SERVER_URL=${NOTIFY_SERVER_URL},NOTIFY_SERVER_TIMEOUT=${NOTIFY_SERVER_TIMEOUT}}"

if aws lambda get-function --function-name "$FUNCTION_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
  retry_lambda_update aws lambda update-function-code \
    --function-name "$FUNCTION_NAME" --image-uri "$IMAGE_URI" --region "$AWS_REGION" >/dev/null
else
  aws lambda create-function \
    --function-name "$FUNCTION_NAME" --package-type Image \
    --code ImageUri="$IMAGE_URI" --role "$ROLE_ARN" \
    --timeout "$TIMEOUT" --memory-size "$MEMORY_SIZE" --region "$AWS_REGION" >/dev/null
fi

wait_for_lambda_update
retry_lambda_update aws lambda update-function-configuration \
  --function-name "$FUNCTION_NAME" --timeout "$TIMEOUT" --memory-size "$MEMORY_SIZE" \
  --environment "$ENV_VARS" --region "$AWS_REGION" >/dev/null
wait_for_lambda_update

echo ""
echo "Deployed ${FUNCTION_NAME}"
echo "Image:   ${IMAGE_URI}"
echo ""
echo "Next: connect the index queue"
echo "aws lambda create-event-source-mapping \\"
echo "  --function-name ${FUNCTION_NAME} \\"
echo "  --event-source-arn <INDEX_QUEUE_ARN> \\"
echo "  --batch-size 1 --region ${AWS_REGION}"
