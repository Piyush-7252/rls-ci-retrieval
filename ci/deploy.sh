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
# NOTE: Environment variables are NOT managed by this script.
#       They must be set manually in AWS Console or Lambda config.
#       Only updates: Docker image, timeout, memory, and lifecycle.
# ─────────────────────────────────────────────────────────────────────────────
deploy_lambda() {
    local function_name=$1
    local image_uri=$2
    local timeout=${3:-900}
    local memory=${4:-10240}
    
    echo ""
    echo "📦 Deploying: $function_name"
    echo "   Image: $image_uri"
    echo "   Timeout: ${timeout}s, Memory: ${memory}MB"
    echo "   ℹ️  Environment variables: NOT managed by this script"
    
    # Check if function exists
    if aws lambda get-function --function-name "$function_name" --region "$AWS_REGION" >/dev/null 2>&1; then
        echo "   ↻ Updating function code..."
        retry_update "$function_name" \
            aws lambda update-function-code \
                --function-name "$function_name" \
                --image-uri "$image_uri" \
                --region "$AWS_REGION" > /dev/null
        
        wait_lambda "$function_name"
        
        echo "   ↻ Updating function configuration (timeout, memory)..."
        retry_update "$function_name" \
            aws lambda update-function-configuration \
                --function-name "$function_name" \
                --timeout "$timeout" \
                --memory-size "$memory" \
                --region "$AWS_REGION" > /dev/null
    else
        echo "   ✨ Creating new function..."
        echo "   ⚠️  You must MANUALLY set environment variables after function creation!"
        aws lambda create-function \
            --function-name "$function_name" \
            --package-type Image \
            --code "ImageUri=$image_uri" \
            --role "$LAMBDA_ROLE_ARN" \
            --timeout "$timeout" \
            --memory-size "$memory" \
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
# Validate environment
# ─────────────────────────────────────────────────────────────────────────────
case "$ENVIRONMENT" in
    dev|qa|cqa|prod)
        # Valid environment
        ;;
    *)
        echo "❌ Invalid environment: $ENVIRONMENT"
        echo "   Valid options: dev, qa, cqa, prod"
        exit 1
        ;;
esac

echo ""
echo "⚙️  Deployment Info:"
echo "   Environment: $ENVIRONMENT"
echo "   Region: $AWS_REGION"
echo "   Account: $ACCOUNT_ID"
echo "   ℹ️  Environment variables must be set manually in AWS Lambda Console"

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
    "$ECR_REGISTRY/rls-ci-retrieval-document-chunk-worker:$IMAGE_TAG" \
    "900" "10240"

# CI Worker
deploy_lambda \
    "rls-ci-retrieval-ci-worker" \
    "$ECR_REGISTRY/rls-ci-retrieval-ci-worker:$IMAGE_TAG" \
    "900" "3072"

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
    "$ECR_REGISTRY/rls-ci-retrieval-search-orchestrator:$IMAGE_TAG" \
    "900" "3072"

# Search Worker
deploy_lambda \
    "rls-ci-retrieval-search-worker" \
    "$ECR_REGISTRY/rls-ci-retrieval-search-worker:$IMAGE_TAG" \
    "900" "3072"

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
echo ""
echo "⚠️  IMPORTANT: Set Lambda environment variables manually in AWS Console:"
echo "   - OPENSEARCH_ENDPOINT"
echo "   - SEMANTIC_OBJECTS_INDEX"
echo "   - DOCUMENT_CHUNKS_INDEX"
echo "   - CI_OBJECTS_INDEX"
echo "   - OPENSEARCH_MAXSIZE"
echo "   - NER_MODEL (for document-chunk-worker)"
echo "   - EMBEDDING_MODEL (for document-chunk-worker)"
echo "   - EMBEDDING_MAX_WORKERS (for document-chunk-worker)"
