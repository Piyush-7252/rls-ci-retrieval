# CI/CD Pipeline Documentation

Complete documentation for building, testing, and deploying RLS services using Bitbucket Pipelines to AWS Lambda across DEV, QA, CQA, and PROD environments.

## 📖 Documentation Index

- **[BITBUCKET_SETUP.md](./BITBUCKET_SETUP.md)** - Complete setup guide for Bitbucket Pipelines
  - Environment variables
  - OIDC configuration
  - Troubleshooting guide

- **[DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)** - Step-by-step deployment instructions
  - Quick start checklist
  - Manual deployment procedures
  - Common tasks and troubleshooting
  - Rollback procedures

- **[.env.example](./.env.example)** - Environment variables template
  - Copy and customize for your AWS accounts
  - Set in Bitbucket Repository Settings

## 🏗️ Architecture Overview

```
GitHub / Local Machine
    ↓
    git push origin main
    ↓
Bitbucket Repository
    ↓
    ┌─────────────────────────────────────────┐
    │ Bitbucket Pipelines (bitbucket-pipelines.yml) │
    └─────────────────────────────────────────┘
    ↓
    ┌──────────────────────────────────────────────┐
    │ STEP 1: Build All Docker Images              │
    │ (Multi-platform builds with caching)         │
    ├──────────────────────────────────────────────┤
    │                                              │
    │ • document-chunk-worker (Lambda)             │
    │ • ci-worker (Lambda)                         │
    │ • search-worker (Lambda)                     │
    │ • search-orchestrator (Lambda)               │
    └──────────────────────────────────────────────┘
    ↓
    ┌──────────────────────────────────────────────┐
    │ STEP 2: Push to ECR in All Environments      │
    ├─────────────────┬──────────────┬──────────────┤
    │ ECR (DEV)       │ ECR (QA)     │ ECR (PROD)   │
    │ Account 1       │ Account 2    │ Account 3    │
    │ Region: eu-w1   │ Region: eu-w1│ Region: eu-w1│
    └────────────────┬──────────────┬──────────────┘
                     │
                     ↓
    ┌──────────────────────────────────────────────┐
    │ STEP 3: Manual Deployment Triggers           │
    ├──────────────┬──────────────┬──────────────┤
    │ Deploy: DEV  │ Deploy: QA   │ Deploy: PROD │
    │ (Manual)     │ (Manual)     │ (Manual)     │
    └──────────────┴──────────────┴──────────────┘
    ↓
    ┌──────────────────────────────────────────────┐
    │ AWS Lambda Functions (Updated via Deploy)    │
    ├──────────────────────────────────────────────┤
    │ • document-chunk-worker (9GB RAM, 900s)      │
    │ • ci-worker (3GB RAM, 900s)                  │
    │ • search-orchestrator (3GB RAM, 900s)        │
    │ • search-worker (3GB RAM, 900s)              │
    └──────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Initial Setup (One-time)

Follow [BITBUCKET_SETUP.md](./BITBUCKET_SETUP.md) section "Required Environment Variables":

```bash
# In Bitbucket: Repository Settings → Pipelines → Environment variables
# Add all variables from ci/.env.example
```

### 2. Push Code and Build

```bash
git add .
git commit -m "feat: update services"
git push origin main
# Automatically triggers: Build & Push All Images step
```

### 3. Deploy to Environment

In Bitbucket UI:
1. Go to **Pipelines** tab
2. Find the commit SHA
3. Click **Run pipeline**
4. Select environment (DEV, QA, CQA, or PROD)
5. Click **Run**

## 📁 File Structure

```
ci/
├── deploy.sh                  # Lambda deployment script
├── bitbucket-pipelines.yml    # Pipeline configuration
├── BITBUCKET_SETUP.md         # Setup documentation
├── DEPLOYMENT_GUIDE.md        # Deployment procedures
├── CI_CD_README.md            # This file
└── .env.example               # Environment variables template
```

## 🔑 Key Components

### `bitbucket-pipelines.yml`
- Defines build and deployment steps
- Handles multi-environment deployments
- Uses OIDC for AWS authentication
- Builds 6 Docker images from multiple Dockerfiles

### `ci/deploy.sh`
- Orchestrates Lambda function updates
- Handles function creation and updates
- Retries on Lambda update failures
- Sets environment-specific configurations

### Environment Variables
- Stored in Bitbucket Repository Settings
- Never committed to source code
- Per-environment configuration (DEV/QA/CQA/PROD)
- Includes AWS account IDs, regions, and OpenSearch endpoints

## 🔄 Pipeline Stages

### Stage 1: Build (Automatic on main push)

```yaml
Step: "Build & Push All Images to All ECRs"
├── Build Docker images with commit SHA
├── Push to DEV ECR
├── Push to QA ECR
├── Push to CQA ECR
└── Push to PROD ECR
```

**Duration**: ~20-25 minutes

### Stage 2: Deploy (Manual trigger)

```yaml
Step: "Deploy to <ENVIRONMENT>"
├── Authenticate to AWS via OIDC
├── Update/create Lambda functions
├── Set environment variables
├── Update Lambda configuration
└── Verify deployment
```

**Duration**: ~3-5 minutes per environment

## 📊 Services Deployed

| Function Name | Repository | Docker Image | Memory | Timeout | Purpose |
|---|---|---|---|---|---|
| `document-chunk-worker` | `lambdas/worker/...` | `document-chunk-worker` | 10GB | 900s | PDF chunking, NER, embedding |
| `ci-worker` | `lambdas/worker/...` | `ci-worker` | 3GB | 900s | Confidential info processing |
| `search-orchestrator` | `lambdas/orchestrator/...` | `search-orchestrator` | 3GB | 900s | Search coordination |
| `search-worker` | `lambdas/worker/...` | `search-worker` | 3GB | 900s | Search execution |

## 🔐 Security Features

✅ **OIDC Authentication**
- No static AWS credentials stored in Bitbucket
- Automatic credential rotation
- Audit trail in AWS CloudTrail

✅ **Per-Environment Isolation**
- Separate AWS accounts per environment
- Separate OpenSearch endpoints
- Separate Lambda roles

✅ **Least Privilege Access**
- OIDC roles have minimal required permissions
- Lambda roles scoped to specific resources
- No over-provisioned credentials

## 🛠️ Local Development Alternative

To deploy locally without Bitbucket:

```bash
# 1. Build images locally
docker build -f lambdas/worker/search-worker/Dockerfile -t search-worker:local .

# 2. Tag and push to ECR
aws ecr get-login-password --region eu-west-1 | \
  docker login --username AWS --password-stdin 111111111111.dkr.ecr.eu-west-1.amazonaws.com

docker tag search-worker:local \
  111111111111.dkr.ecr.eu-west-1.amazonaws.com/search-worker:local

docker push 111111111111.dkr.ecr.eu-west-1.amazonaws.com/search-worker:local

# 3. Update Lambda manually
aws lambda update-function-code \
  --function-name rls-ci-retrieval-search-worker \
  --image-uri 111111111111.dkr.ecr.eu-west-1.amazonaws.com/search-worker:local \
  --region eu-west-1
```

## 📈 Monitoring

### Pipeline Status

**Bitbucket UI**: Pipelines tab → Click commit SHA

### Lambda Deployment Status

```bash
# Check function status
aws lambda get-function \
  --function-name rls-ci-retrieval-search-worker \
  --region eu-west-1

# View recent logs
aws logs tail /aws/lambda/rls-ci-retrieval-search-worker --follow --max-items 20

# Check function metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Invocations \
  --dimensions Name=FunctionName,Value=rls-ci-retrieval-search-worker \
  --statistics Sum \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-02T00:00:00Z \
  --period 3600
```

## ❌ Troubleshooting

### Bitbucket Logs

Check pipeline logs in **Pipelines** tab:

```bash
# Look for common errors:
# - "no basic auth credentials" → ECR auth failed
# - "ResourceAlreadyExists" → Repository exists
# - "Lambda busy" → Function updating, will retry
```

### AWS Verification

```bash
# Test OIDC token
aws sts get-caller-identity --region eu-west-1

# Check ECR images
aws ecr describe-images --repository-name search-worker --region eu-west-1

# Check Lambda function
aws lambda get-function --function-name rls-ci-retrieval-search-worker --region eu-west-1
```

For detailed troubleshooting, see [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md#troubleshooting).

## 🔄 Common Workflows

### Scenario 1: Deploy Latest Code to DEV Only

```bash
git push origin main
# Wait for build to complete in Bitbucket

# In Bitbucket UI:
# Pipelines → Click commit SHA → Run pipeline → Select "Deploy to DEV"
```

### Scenario 2: Hotfix Deployment to PROD

```bash
git checkout -b hotfix/search-worker-fix
# Make changes
git push origin hotfix/search-worker-fix
git push origin hotfix/search-worker-fix:main

# Or merge PR:
git checkout main && git pull
git merge hotfix/search-worker-fix
git push origin main

# Then deploy via Bitbucket UI
```

### Scenario 3: Rollback Previous Version

```bash
# Get previous image
PREV_SHA=$(git rev-parse HEAD~1)

# Check if image exists in ECR
aws ecr describe-images \
  --repository-name search-worker \
  --image-ids imageTag=$PREV_SHA \
  --region eu-west-1

# Manually update Lambda if found
aws lambda update-function-code \
  --function-name rls-ci-retrieval-search-worker \
  --image-uri 111111111111.dkr.ecr.eu-west-1.amazonaws.com/search-worker:$PREV_SHA \
  --region eu-west-1
```

## 📚 References

- [Bitbucket Pipelines Documentation](https://support.atlassian.com/bitbucket-cloud/docs/get-started-with-bitbucket-pipelines/)
- [AWS OIDC for Bitbucket](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create_for_idp_oidc.html)
- [AWS Lambda Deployments](https://docs.aws.amazon.com/lambda/latest/dg/deploying-lambda-apps.html)
- [ECR Repositories](https://docs.aws.amazon.com/AmazonECR/latest/userguide/)

## ✅ Checklist for First Deployment

- [ ] Set up OIDC in AWS accounts (DEV, QA, PROD)
- [ ] Create ECR repositories in all accounts
- [ ] Configure environment variables in Bitbucket
- [ ] Create Dockerfile for all services
- [ ] Test code changes locally
- [ ] Push to main branch
- [ ] Monitor build step in Bitbucket
- [ ] Manually trigger deployment to DEV
- [ ] Verify Lambda functions were updated
- [ ] Check Lambda logs for errors
- [ ] Deploy to QA after successful DEV
- [ ] Deploy to PROD after QA verification

## 🎯 Next Steps

1. Complete [BITBUCKET_SETUP.md](./BITBUCKET_SETUP.md) setup
2. Run [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) first deployment
3. Monitor in Bitbucket Pipelines UI
4. Check AWS Lambda function logs
5. Set up CloudWatch alarms (optional)

---

**Last Updated**: 2026-09-01  
**Version**: 1.0  
**Maintained By**: DevOps Team
