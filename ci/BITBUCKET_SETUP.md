# Bitbucket Pipelines CI/CD Setup

This document describes the complete Bitbucket Pipelines configuration for building and deploying all services to AWS Lambda across multiple environments (DEV, QA, CQA, PROD).

## Overview

The pipeline automates:
1. **Build Phase**: Docker image creation for all services
2. **Push Phase**: Images pushed to ECR in all target environments
3. **Deploy Phase**: Manual deployments to individual environments with automatic Lambda updates

## Pipeline Structure

```
┌─────────────────────────────────────────────────────────┐
│ Main Branch: Auto-trigger on push to main               │
├─────────────────────────────────────────────────────────┤
│ STEP 1: Build & Push All Images to All ECRs             │
│         ├── document-chunk-worker (Lambda)              │
│         ├── ci-worker (Lambda)                          │
│         ├── search-worker (Lambda)                      │
│         └── search-orchestrator (Lambda)                │
│                                                         │
│         → Push to: DEV ECR, QA ECR, CQA ECR, PROD ECR   │
├─────────────────────────────────────────────────────────┤
│ STEP 2: Deploy to DEV (Manual Trigger)                  │
│ STEP 3: Deploy to QA (Manual Trigger)                   │
│ STEP 4: Deploy to CQA (Manual Trigger)                  │
│ STEP 5: Deploy to PROD (Manual Trigger)                 │
└─────────────────────────────────────────────────────────┘
```

## Required Environment Variables in Bitbucket

Configure these in **Bitbucket Repository Settings → Pipelines → Environment variables**:

### AWS Account Information

```
AWS_DEV_ACCOUNT_ID              = 111111111111
AWS_QA_ACCOUNT_ID               = 222222222222
AWS_PROD_ACCOUNT_ID             = 333333333333

AWS_DEFAULT_REGION_DEV          = eu-west-1
AWS_DEFAULT_REGION_QA           = eu-west-1
AWS_DEFAULT_REGION_CQA          = eu-west-2
AWS_DEFAULT_REGION_PROD         = eu-west-1

AWS_ROLE_ARN_DEV                = arn:aws:iam::111111111111:role/bitbucket-oidc-role
AWS_ROLE_ARN_QA                 = arn:aws:iam::222222222222:role/bitbucket-oidc-role
AWS_ROLE_ARN_PROD               = arn:aws:iam::333333333333:role/bitbucket-oidc-role
```

### OpenSearch Endpoints

```
# DEV Environment
OPENSEARCH_ENDPOINT_DEV         = search-rls-dev-rhitzxwnctmuyq2l4kny5kwelu.eu-west-1.es.amazonaws.com
SEMANTIC_OBJECTS_INDEX_DEV      = semantic-objects
DOCUMENT_CHUNKS_INDEX_DEV       = document-chunks
CI_OBJECTS_INDEX_DEV            = ci-objects
OPENSEARCH_MAXSIZE_DEV          = 256

# QA Environment
OPENSEARCH_ENDPOINT_QA          = search-rls-qa-xxx.eu-west-1.es.amazonaws.com
SEMANTIC_OBJECTS_INDEX_QA       = semantic-objects
DOCUMENT_CHUNKS_INDEX_QA        = document-chunks
CI_OBJECTS_INDEX_QA             = ci-objects
OPENSEARCH_MAXSIZE_QA           = 512

# CQA Environment
OPENSEARCH_ENDPOINT_CQA         = search-rls-cqa-xxx.eu-west-2.es.amazonaws.com
SEMANTIC_OBJECTS_INDEX_CQA      = semantic-objects
DOCUMENT_CHUNKS_INDEX_CQA       = document-chunks
CI_OBJECTS_INDEX_CQA            = ci-objects
OPENSEARCH_MAXSIZE_CQA          = 512

# PROD Environment
OPENSEARCH_ENDPOINT_PROD        = search-rls-prod-xxx.eu-west-1.es.amazonaws.com
SEMANTIC_OBJECTS_INDEX_PROD     = semantic-objects
DOCUMENT_CHUNKS_INDEX_PROD      = document-chunks
CI_OBJECTS_INDEX_PROD           = ci-objects
OPENSEARCH_MAXSIZE_PROD         = 1024
```

### Lambda Role ARNs

```
LAMBDA_ROLE_ARN_DEV             = arn:aws:iam::111111111111:role/lambda-execution-role
LAMBDA_ROLE_ARN_QA              = arn:aws:iam::222222222222:role/lambda-execution-role
LAMBDA_ROLE_ARN_PROD            = arn:aws:iam::333333333333:role/lambda-execution-role
```

## Workflow Usage

### 1. **Automatic Build & Push (Trigger: push to main)**

When code is pushed to the main branch:

1. All Docker images are built with the commit SHA as the tag
2. Images are pushed to ECR in all 4 environments (DEV, QA, CQA, PROD)
3. Pipeline waits for manual deployment triggers

**Status Check:**
```bash
# Check if images were pushed to DEV ECR
aws ecr describe-images \
  --repository-name document-chunk-worker \
  --region eu-west-1 \
  --query 'imageDetails[*].[imageTags,imagePushedAt]'
```

### 2. **Manual Deployments**

After successful build & push, manually trigger deployments:

#### In Bitbucket UI:
1. Go to **Pipelines** tab
2. Click **Run pipeline**
3. Select deployment (DEV, QA, CQA, or PROD)
4. Click **Run**

#### Via Bitbucket API:
```bash
# Example: Deploy to DEV
curl -X POST \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer <BITBUCKET_TOKEN>" \
  https://api.bitbucket.org/2.0/repositories/<WORKSPACE>/<REPO>/pipelines/ \
  -d '{
    "target": {
      "ref_type": "branch",
      "type": "pipeline_ref_target",
      "ref_name": "main"
    },
    "variables": [
      {
        "key": "DEPLOY_ENV",
        "value": "dev"
      }
    ]
  }'
```

### 3. **Deployment Process**

Each deployment step:

1. **Authenticates** to the target AWS account using OIDC (OpenID Connect)
2. **Runs** `ci/deploy.sh <environment> <image_tag>`
3. **Deploys** all Lambda functions in the proper order
4. **Waits** for each Lambda to be idle before proceeding to the next
5. **Updates** Lambda configuration (timeout, memory, environment variables)

## Lambda Functions Deployed

### Stage 1: Document Processing
- `rls-ci-retrieval-document-chunk-worker` - Main chunking + extraction
- `rls-ci-retrieval-ci-worker` - Confidential information processing

### Stage 2: Search
- `rls-ci-retrieval-search-orchestrator` - Search orchestration
- `rls-ci-retrieval-search-worker` - Search execution

## Dockerfile Requirements

Ensure these Dockerfiles exist in your repository:

```
├── Dockerfile.extraction           
├── lambdas/
│   ├── worker/document-chunk-worker/Dockerfile
│   ├── worker/ci-worker/Dockerfile
│   ├── worker/search-worker/Dockerfile
│   └── orchestrator/search-orchestrator/Dockerfile
```

## Troubleshooting

### Build Fails

**Problem**: Docker build times out

**Solution**: Increase timeout in `bitbucket-pipelines.yml`:
```yaml
definitions:
  services:
    docker:
      memory: 8192  # Increase from 5120
```

### ECR Login Fails

**Problem**: `no basic auth credentials` error

**Solution**: Verify AWS account IDs and OIDC role ARNs:
```bash
# Test OIDC authentication
aws sts get-caller-identity --region eu-west-1
```

### Lambda Update Hangs

**Problem**: Lambda function stuck in "Updating"

**Solution**: The deploy script retries up to 5 times. If still stuck:
```bash
# Manual check
aws lambda get-function --function-name rls-ci-retrieval-search-worker --region eu-west-1

# Manual update
aws lambda update-function-code \
  --function-name rls-ci-retrieval-search-worker \
  --image-uri <ACCOUNT_ID>.dkr.ecr.eu-west-1.amazonaws.com/search-worker:<TAG> \
  --region eu-west-1
```

## Custom Pipelines

The configuration includes custom pipeline triggers for manual builds:

```bash
# Rebuild and push without deploying
# Go to Pipelines → Custom → build-and-push

# Deploy only (without rebuild)
# Go to Pipelines → Custom → deploy-dev|qa|cqa|prod
```

## Environment Variable Precedence

Variables can be set at multiple levels (highest to lowest priority):

1. Bitbucket step-level variables (inside step definitions)
2. Bitbucket repository variables
3. Default values in `ci/deploy.sh`
4. Environment-specific defaults

## Security Considerations

✅ **Using OIDC** instead of static AWS credentials
- No need to store AWS Access Keys in Bitbucket
- Automatic credential rotation
- Audit trail in AWS CloudTrail

✅ **Principle of Least Privilege**
- Each AWS account role has minimal required permissions
- Separate roles for DEV, QA, CQA, PROD
- Lambda functions have specific execution roles

## Monitoring Deployments

### View Pipeline Execution

In Bitbucket UI:
- **Pipelines** tab → Click on commit SHA
- View logs for each step
- Check environment variables (sanitized)

### Check Lambda Updates

```bash
# View Lambda updates
aws lambda list-function-url-configs \
  --function-name rls-ci-retrieval-search-worker \
  --region eu-west-1

# Check recent image updates
aws ecr describe-images \
  --repository-name search-worker \
  --region eu-west-1 \
  --query 'imageDetails[0:5].[imageTags,imagePushedAt]' \
  --output table
```

## Next Steps

1. **Set up OIDC in each AWS account**: [AWS OpenID Connect setup](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create_for_idp_oidc.html)
2. **Configure repository variables** in Bitbucket
3. **Create Dockerfiles** for all services if not already present
4. **Test build pipeline** with a test push
5. **Manual deploy** to DEV first
6. **Verify Lambda updates** and monitor logs
7. **Rollout** to QA → CQA → PROD

## Support

For issues or questions:
- Check Bitbucket Pipelines logs
- Verify AWS IAM permissions
- Ensure ECR repositories exist
- Confirm OpenSearch endpoints are reachable
