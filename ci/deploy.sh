#!/bin/bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Comprehensive Lambda Deployment Script
# Usage: ./ci/deploy.sh <environment> <image_tag>
# Example: ./ci/deploy.sh dev abc123def456
# ─────────────────────────────────────────────────────────────────────────────

ENVIRONMENT=${1:-dev}
IMAGE_TAG=${2:-latest}
AWS_REGION=${AWS_DEFAULT_REGION:-eu-west-1}

echo "🚀 Deploying to $ENVIRONMENT with image tag $IMAGE_TAG in region $AWS_REGION"

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Wait for Lambda to be idle
# ─────────────────────────────────────────────────────────────────────────────
wait_lambda() {
    local function_name=$1
    echo "⏳ Waiting for $function_name to be idle..."
    aws lambda wait function-updated-v2 --function-name "$function_name" --region "$AWS_REGION" 2>/dev/null || true
    sleep 2
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Retry Lambda updates (they can fail if function is still updating)
# ─────────────────────────────────────────────────────────────────────────────
retry_update() {
    local function_name=$1
    shift
    
    for attempt in 1 2 3 4 5; do
        if "$@"; then
            echo "✅ Updated $function_name (attempt $attempt/5)"
            return 0
        fi
        if [ $attempt -lt 5 ]; then
            echo "⚠️  Attempt $attempt/5 failed, waiting before retry..."
            wait_lambda "$function_name"
        fi
    done
    
    echo "❌ Failed to update $function_name after 5 attempts"
    return 1
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Deploy or update a Lambda function
# ─────────────────────────────────────────────────────────────────────────────
deploy_lambda() {
    local function_name=$1
    local image_uri=$2
    local timeout=${3:-900}
    local memory=${4:-10240}
    local env_vars=${5:-""}
    
    echo ""
    echo "📦 Deploying: $function_name"
    echo "   Image: $image_uri"
    echo "   Timeout: ${timeout}s, Memory: ${memory}MB"
    
    # Check if function exists
    if aws lambda get-function --function-name "$function_name" --region "$AWS_REGION" >/dev/null 2>&1; then
        echo "   ↻ Updating existing function..."
        retry_update "$function_name" \
            aws lambda update-function-code \
                --function-name "$function_name" \
                --image-uri "$image_uri" \
                --region "$AWS_REGION" > /dev/null
    else
        echo "   ✨ Creating new function..."
        aws lambda create-function \
            --function-name "$function_name" \
            --package-type Image \
            --code "ImageUri=$image_uri" \
            --role "$LAMBDA_ROLE_ARN" \
            --timeout "$timeout" \
            --memory-size "$memory" \
            --region "$AWS_REGION" \
            --environment "Variables=$env_vars" > /dev/null
    fi
    
    wait_lambda "$function_name"
    
    # Update configuration
    if [ ! -z "$env_vars" ]; then
        retry_update "$function_name" \
            aws lambda update-function-configuration \
                --function-name "$function_name" \
                --timeout "$timeout" \
                --memory-size "$memory" \
                --environment "Variables=$env_vars" \
                --region "$AWS_REGION" > /dev/null
    fi
    
    wait_lambda "$function_name"
    echo "✅ $function_name deployed successfully"
}

# ─────────────────────────────────────────────────────────────────────────────
# Get account ID and ECR registry
# ─────────────────────────────────────────────────────────────────────────────
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$AWS_REGION")
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "🔐 AWS Account: $ACCOUNT_ID"
echo "📍 ECR Registry: $ECR_REGISTRY"

# ─────────────────────────────────────────────────────────────────────────────
# Load environment-specific configuration
# ─────────────────────────────────────────────────────────────────────────────
case "$ENVIRONMENT" in
    dev)
        OPENSEARCH_ENDPOINT="${OPENSEARCH_ENDPOINT_DEV:-search-rls-dev-rhitzxwnctmuyq2l4kny5kwelu.eu-west-1.es.amazonaws.com}"
        SEMANTIC_OBJECTS_INDEX="${SEMANTIC_OBJECTS_INDEX_DEV:-semantic-objects}"
        DOCUMENT_CHUNKS_INDEX="${DOCUMENT_CHUNKS_INDEX_DEV:-document-chunks}"
        CI_OBJECTS_INDEX="${CI_OBJECTS_INDEX_DEV:-ci-objects}"
        OPENSEARCH_MAXSIZE="${OPENSEARCH_MAXSIZE_DEV:-256}"
        ;;
    qa)
        OPENSEARCH_ENDPOINT="${OPENSEARCH_ENDPOINT_QA:-search-rls-qa-xyz.eu-west-1.es.amazonaws.com}"
        SEMANTIC_OBJECTS_INDEX="${SEMANTIC_OBJECTS_INDEX_QA:-semantic-objects}"
        DOCUMENT_CHUNKS_INDEX="${DOCUMENT_CHUNKS_INDEX_QA:-document-chunks}"
        CI_OBJECTS_INDEX="${CI_OBJECTS_INDEX_QA:-ci-objects}"
        OPENSEARCH_MAXSIZE="${OPENSEARCH_MAXSIZE_QA:-512}"
        ;;
    cqa)
        OPENSEARCH_ENDPOINT="${OPENSEARCH_ENDPOINT_CQA:-search-rls-cqa-xyz.eu-west-1.es.amazonaws.com}"
        SEMANTIC_OBJECTS_INDEX="${SEMANTIC_OBJECTS_INDEX_CQA:-semantic-objects}"
        DOCUMENT_CHUNKS_INDEX="${DOCUMENT_CHUNKS_INDEX_CQA:-document-chunks}"
        CI_OBJECTS_INDEX="${CI_OBJECTS_INDEX_CQA:-ci-objects}"
        OPENSEARCH_MAXSIZE="${OPENSEARCH_MAXSIZE_CQA:-512}"
        ;;
    prod)
        OPENSEARCH_ENDPOINT="${OPENSEARCH_ENDPOINT_PROD:-search-rls-prod-xyz.eu-west-1.es.amazonaws.com}"
        SEMANTIC_OBJECTS_INDEX="${SEMANTIC_OBJECTS_INDEX_PROD:-semantic-objects}"
        DOCUMENT_CHUNKS_INDEX="${DOCUMENT_CHUNKS_INDEX_PROD:-document-chunks}"
        CI_OBJECTS_INDEX="${CI_OBJECTS_INDEX_PROD:-ci-objects}"
        OPENSEARCH_MAXSIZE="${OPENSEARCH_MAXSIZE_PROD:-1024}"
        ;;
    *)
        echo "❌ Invalid environment: $ENVIRONMENT"
        echo "   Valid options: dev, qa, cqa, prod"
        exit 1
        ;;
esac

echo ""
echo "⚙️  Environment Config:"
echo "   OpenSearch: $OPENSEARCH_ENDPOINT"
echo "   Indexes: $SEMANTIC_OBJECTS_INDEX / $DOCUMENT_CHUNKS_INDEX / $CI_OBJECTS_INDEX"

# ─────────────────────────────────────────────────────────────────────────────
# Define Lambda deployments
# ─────────────────────────────────────────────────────────────────────────────

# Common environment variables for all Lambdas
COMMON_ENV="{OPENSEARCH_ENDPOINT=$OPENSEARCH_ENDPOINT,SEMANTIC_OBJECTS_INDEX=$SEMANTIC_OBJECTS_INDEX,DOCUMENT_CHUNKS_INDEX=$DOCUMENT_CHUNKS_INDEX,CI_OBJECTS_INDEX=$CI_OBJECTS_INDEX,OPENSEARCH_MAXSIZE=$OPENSEARCH_MAXSIZE,HF_HUB_OFFLINE=1,HF_HOME=/var/task/models}"

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: Document Processing Workers
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "═════════════════════════════════════════════════════════════════════════════"
echo "STAGE 1: Document Processing Workers"
echo "═════════════════════════════════════════════════════════════════════════════"

# Document Chunk Worker
deploy_lambda \
    "rls-ci-retrieval-document-chunk-worker" \
    "$ECR_REGISTRY/document-chunk-worker:$IMAGE_TAG" \
    "900" "10240" \
    "$COMMON_ENV,NER_MODEL=gliner,EMBEDDING_MODEL=amazon.titan-embed-text-v2:0,EMBEDDING_MAX_WORKERS=8"

# CI Worker
deploy_lambda \
    "rls-ci-retrieval-ci-worker" \
    "$ECR_REGISTRY/ci-worker:$IMAGE_TAG" \
    "900" "3072" \
    "$COMMON_ENV"

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: Search Workers
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "═════════════════════════════════════════════════════════════════════════════"
echo "STAGE 2: Search Workers"
echo "═════════════════════════════════════════════════════════════════════════════"

# Search Orchestrator
deploy_lambda \
    "rls-ci-retrieval-search-orchestrator" \
    "$ECR_REGISTRY/search-orchestrator:$IMAGE_TAG" \
    "900" "3072" \
    "$COMMON_ENV"

# Search Worker
deploy_lambda \
    "rls-ci-retrieval-search-worker" \
    "$ECR_REGISTRY/search-worker:$IMAGE_TAG" \
    "900" "3072" \
    "$COMMON_ENV"

# ─────────────────────────────────────────────────────────────────────────────
# Optional: PDF Processing (only if images exist)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "═════════════════════════════════════════════════════════════════════════════"
echo "OPTIONAL: PDF Processing Images (if they exist in your workflow)"
echo "═════════════════════════════════════════════════════════════════════════════"


echo ""
echo "🎉 All Lambda functions deployed successfully to $ENVIRONMENT!"
echo ""
echo "Summary:"
echo "  Environment: $ENVIRONMENT"
echo "  Image Tag: $IMAGE_TAG"
echo "  Region: $AWS_REGION"
echo "  Account: $ACCOUNT_ID"
