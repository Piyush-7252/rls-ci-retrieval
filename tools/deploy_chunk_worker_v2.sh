#!/usr/bin/env bash
# Deploy rls-ci-chunk-worker-v2 (GPU embedding API + S3 artifact cache + notify_server).
# Does NOT touch the existing rls-ci-chunk-worker Lambda.
#
# Required env vars:
#   ROLE_ARN            IAM role ARN for the Lambda
#   EMBEDDING_API_URL   URL of the GPU embedding service (e.g. http://host:8080/embed)
#   ARTIFACT_BUCKET     S3 bucket for enriched-chunk cache
#
# Optional env vars (all have defaults):
#   FUNCTION_NAME, ECR_REPO, AWS_REGION, NOTIFY_SERVER_URL, EMBEDDING_MODEL,
#   ENRICHMENT_VERSION, EMBEDDING_API_KEY, EMBEDDING_API_TIMEOUT,
#   NOTIFY_SERVER_TIMEOUT, OPENSEARCH_*, NER_MODEL, HF_TOKEN
#
# Example:
#   ROLE_ARN=arn:aws:iam::064051750322:role/rls-ci-worker-role \
#   EMBEDDING_API_URL=http://10.0.1.50:8080/embed \
#   ARTIFACT_BUCKET=rls-chunk-artifacts \
#   NOTIFY_SERVER_URL=http://10.0.1.50:9000 \
#   tools/deploy_chunk_worker_v2.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

AWS_REGION="${AWS_REGION:-eu-west-1}"
FUNCTION_NAME="${FUNCTION_NAME:-rls-ci-chunk-worker-v2}"
ECR_REPO="${ECR_REPO:-rls-ci-chunk-worker-v2}"
IMAGE_TAG="${IMAGE_TAG:-$(date +%Y%m%d-%H%M%S)}"
LATEST_TAG="${LATEST_TAG:-latest}"
ROLE_ARN="${ROLE_ARN:-}"
TIMEOUT="${TIMEOUT:-900}"
MEMORY_SIZE="${MEMORY_SIZE:-10240}"

# Embedding API
EMBEDDING_API_URL="${EMBEDDING_API_URL:-}"
EMBEDDING_API_KEY="${EMBEDDING_API_KEY:-}"
EMBEDDING_API_TIMEOUT="${EMBEDDING_API_TIMEOUT:-120}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-gpu-embed}"

# Artifact cache
ARTIFACT_BUCKET="${ARTIFACT_BUCKET:-}"
ENRICHMENT_VERSION="${ENRICHMENT_VERSION:-1}"

# Notify server
NOTIFY_SERVER_URL="${NOTIFY_SERVER_URL:-}"
NOTIFY_SERVER_TIMEOUT="${NOTIFY_SERVER_TIMEOUT:-5}"

# OpenSearch (same as v1 defaults)
OPENSEARCH_ENDPOINT="${OPENSEARCH_ENDPOINT:-search-rls-dev-rhitzxwnctmuyq2l4kny5kwelu.eu-west-1.es.amazonaws.com}"
OPENSEARCH_INDEX="${OPENSEARCH_INDEX:-document-chunks}"
SEMANTIC_OBJECTS_INDEX="${SEMANTIC_OBJECTS_INDEX:-semantic-objects}"
OPENSEARCH_CI_INDEX="${OPENSEARCH_CI_INDEX:-ci-objects}"
NER_MODEL="${NER_MODEL:-gliner}"
HF_TOKEN="${HF_TOKEN:-}"

if [[ -z "$ROLE_ARN" ]]; then
  echo "ERROR: ROLE_ARN is required"
  echo "Example: ROLE_ARN=arn:aws:iam::<acct>:role/rls-ci-worker-role tools/deploy_chunk_worker_v2.sh"
  exit 1
fi
if [[ -z "$EMBEDDING_API_URL" ]]; then
  echo "ERROR: EMBEDDING_API_URL is required (e.g. http://host:8080/embed)"
  exit 1
fi
if [[ -z "$ARTIFACT_BUCKET" ]]; then
  echo "ERROR: ARTIFACT_BUCKET is required (S3 bucket for enrichment cache)"
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
  -f "$ROOT_DIR/lambdas/chunk_worker_v2/Dockerfile" \
  -t "$IMAGE_URI" \
  -t "$LATEST_IMAGE_URI" \
  "$ROOT_DIR"

# Build env vars string (no spaces around = to stay shell-safe)
ENV_VARS="Variables={"
ENV_VARS+="OPENSEARCH_ENDPOINT=${OPENSEARCH_ENDPOINT},"
ENV_VARS+="OPENSEARCH_INDEX=${OPENSEARCH_INDEX},"
ENV_VARS+="SEMANTIC_OBJECTS_INDEX=${SEMANTIC_OBJECTS_INDEX},"
ENV_VARS+="OPENSEARCH_CI_INDEX=${OPENSEARCH_CI_INDEX},"
ENV_VARS+="NER_MODEL=${NER_MODEL},"
ENV_VARS+="EMBEDDING_API_URL=${EMBEDDING_API_URL},"
ENV_VARS+="EMBEDDING_API_TIMEOUT=${EMBEDDING_API_TIMEOUT},"
ENV_VARS+="EMBEDDING_MODEL=${EMBEDDING_MODEL},"
ENV_VARS+="ARTIFACT_BUCKET=${ARTIFACT_BUCKET},"
ENV_VARS+="ENRICHMENT_VERSION=${ENRICHMENT_VERSION},"
ENV_VARS+="NOTIFY_SERVER_URL=${NOTIFY_SERVER_URL},"
ENV_VARS+="NOTIFY_SERVER_TIMEOUT=${NOTIFY_SERVER_TIMEOUT},"
ENV_VARS+="HF_HOME=/tmp/hf_cache"
if [[ -n "$EMBEDDING_API_KEY" ]]; then
  ENV_VARS+=",EMBEDDING_API_KEY=${EMBEDDING_API_KEY}"
fi
if [[ -n "$HF_TOKEN" ]]; then
  ENV_VARS+=",HF_TOKEN=${HF_TOKEN}"
fi
ENV_VARS+="}"

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
  --environment "$ENV_VARS" \
  --region "$AWS_REGION" >/dev/null

wait_for_lambda_update

echo ""
echo "Deployed ${FUNCTION_NAME}"
echo "Image:   ${IMAGE_URI}"
echo ""
echo "Next: connect an SQS queue to this Lambda"
echo "aws lambda create-event-source-mapping \\"
echo "  --function-name ${FUNCTION_NAME} \\"
echo "  --event-source-arn <SQS_QUEUE_ARN> \\"
echo "  --batch-size 1 \\"
echo "  --region ${AWS_REGION}"
