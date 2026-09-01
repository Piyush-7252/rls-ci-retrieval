# Bitbucket Deployment Quick Reference

## 🚀 Quick Start Checklist

- [ ] Set up OIDC in each AWS account (DEV, QA, CQA, PROD)
- [ ] Create OIDC roles in each AWS account
- [ ] Create ECR repositories in each account
- [ ] Configure environment variables in Bitbucket (see `.env.example`)
- [ ] Push code to main branch to trigger build
- [ ] Verify images were built and pushed to ECR
- [ ] Manually trigger deployments to each environment

## 📋 Step-by-Step Deployment Process

### Step 1: Initial Setup (One-time)

```bash
# 1. Set up OIDC in AWS (per account)
# See: https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create_for_idp_oidc.html

# 2. Create OIDC provider in each AWS account
# Provider URL: https://api.bitbucket.org
# Audience: ari:cloud:bitbucket::workspace/<YOUR_WORKSPACE>

# 3. Create IAM role with trust policy
cat > trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::ACCOUNT_ID:oidc-provider/api.bitbucket.org"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "api.bitbucket.org:aud": "ari:cloud:bitbucket::workspace/YOUR_WORKSPACE"
        }
      }
    }
  ]
}
EOF

# 4. Attach policies for ECR and Lambda access
aws iam attach-role-policy \
  --role-name bitbucket-oidc-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser

aws iam attach-role-policy \
  --role-name bitbucket-oidc-role \
  --policy-arn arn:aws:iam::aws:policy/AWSLambda_FullAccess

# 5. Set environment variables in Bitbucket
# See ci/.env.example
```

### Step 2: Code Push and Build

```bash
# Push code to main branch
git add .
git commit -m "feat: update Lambda deployments"
git push origin main

# This automatically triggers:
# 1. Build all Docker images
# 2. Push to ECR in all 4 environments (DEV, QA, CQA, PROD)
# 3. Wait for manual deployment trigger
```

### Step 3: Monitor Build

In **Bitbucket → Pipelines**:
1. Click the commit SHA
2. Watch "Build & Push All Images to All ECRs" step
3. Verify all images were pushed successfully
4. Check for errors in the logs

### Step 4: Deploy to Environment

In **Bitbucket → Pipelines**:

1. **Click "Run pipeline"** for the same commit
2. **Select deployment**: Deploy to DEV | QA | CQA | PROD
3. **Click "Run"**

Or manually trigger via Bitbucket UI:
- Pipelines → Custom pipelines → `deploy-dev` / `deploy-qa` / etc.

### Step 5: Verify Deployment

```bash
# Check Lambda function was updated
aws lambda get-function \
  --function-name rls-ci-retrieval-search-worker \
  --region eu-west-1 \
  --query 'Configuration.[FunctionArn,LastModified,CodeSha256]'

# Check Lambda logs
aws logs tail /aws/lambda/rls-ci-retrieval-search-worker --follow

# Test Lambda invocation (optional)
aws lambda invoke \
  --function-name rls-ci-retrieval-search-worker \
  --payload '{"test": true}' \
  response.json

cat response.json
```

## 📊 Pipeline Execution Times

| Step | Duration | Notes |
|------|----------|-------|
| Build all images | ~15-20 min | Docker builds + compilation |
| Push to 4 ECRs | ~5 min | Parallel pushes to DEV, QA, CQA, PROD |
| Deploy to DEV | ~3-5 min | All Lambda functions updated |
| Deploy to QA | ~3-5 min | All Lambda functions updated |
| Deploy to CQA | ~3-5 min | All Lambda functions updated |
| Deploy to PROD | ~3-5 min | All Lambda functions updated |

## 🔧 Common Tasks

### Rebuild and Push (Skip Deploy)

```
Bitbucket → Pipelines → Custom pipelines → build-and-push
```

### Deploy Without Rebuild

```
Bitbucket → Pipelines → Custom pipelines → deploy-dev
```

### Deploy Only One Lambda (Manual)

```bash
# Get ECR image URI
IMAGE_URI=$(aws ecr describe-images \
  --repository-name search-worker \
  --region eu-west-1 \
  --query 'imageDetails[0].imageUri' \
  --output text)

# Update Lambda
aws lambda update-function-code \
  --function-name rls-ci-retrieval-search-worker \
  --image-uri $IMAGE_URI \
  --region eu-west-1
```

### Check Image Status in ECR

```bash
# List all images in repository
aws ecr describe-images \
  --repository-name search-worker \
  --region eu-west-1 \
  --query 'imageDetails[*].[imageTags,imagePushedAt,imageSizeInBytes]' \
  --output table

# Get image digest for specific tag
aws ecr describe-images \
  --repository-name search-worker \
  --image-ids imageTag=abc123def456 \
  --region eu-west-1 \
  --query 'imageDetails[0].[imageDigest,imageSizeInBytes]'
```

## ⚠️ Troubleshooting

### Issue: Build timeout

**Solution**: Increase Docker memory in `bitbucket-pipelines.yml`:
```yaml
definitions:
  services:
    docker:
      memory: 8192  # Increase from 5120
```

### Issue: ECR login fails

**Cause**: Invalid AWS credentials or OIDC configuration

**Solution**:
```bash
# Test OIDC authentication
aws sts get-caller-identity --region eu-west-1

# If fails, verify:
# 1. OIDC provider URL (https://api.bitbucket.org)
# 2. Role trust policy includes correct workspace ARN
# 3. Environment variables are set correctly
```

### Issue: Lambda update hangs

**Cause**: Lambda is still updating from previous deployment

**Solution**: The deploy script retries 5 times automatically. If still stuck:
```bash
# Check Lambda status
aws lambda get-function-concurrency \
  --function-name rls-ci-retrieval-search-worker \
  --region eu-west-1

# Wait a few minutes and retry manually
aws lambda update-function-code \
  --function-name rls-ci-retrieval-search-worker \
  --image-uri <NEW_IMAGE_URI> \
  --region eu-west-1
```

### Issue: Images not pushed to all ECRs

**Cause**: Docker buildx or authentication failure

**Solution**:
```bash
# Check ECR repositories exist in all accounts
for account in 111111111111 222222222222 333333333333; do
  aws ecr describe-repositories \
    --region eu-west-1 \
    --account-id $account \
    --query 'repositories[*].repositoryName'
done

# Check Bitbucket pipeline logs for authentication errors
```

## 📚 Advanced Features

### Conditional Deployments (Optional)

To deploy only specific services based on changes:

```yaml
# Add path-based filtering in bitbucket-pipelines.yml
definitions:
  caches:
    pip: ~/.cache/pip
  
steps:
  - step: &deploy-search-only
      name: "Deploy Search Services Only"
      trigger: manual
      script:
        - bash ./ci/deploy.sh $ENVIRONMENT $IMAGE_TAG search-only
```

### Deployment Notifications (Optional)

Send deployment status to Slack:

```bash
# In ci/deploy.sh
notify_slack() {
  local status=$1
  local environment=$2
  
  curl -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"Deployment to $environment: $status\"}" \
    $SLACK_WEBHOOK_URL
}
```

### Rollback Procedure

```bash
# Get previous image digest
PREVIOUS_IMAGE=$(aws ecr describe-images \
  --repository-name search-worker \
  --region eu-west-1 \
  --query 'imageDetails[1].imageDigest' \
  --output text)

# Rollback Lambda
aws lambda update-function-code \
  --function-name rls-ci-retrieval-search-worker \
  --image-uri 111111111111.dkr.ecr.eu-west-1.amazonaws.com/search-worker@$PREVIOUS_IMAGE \
  --region eu-west-1
```

## 🎯 Best Practices

1. **Always test in DEV first** before deploying to QA/PROD
2. **Monitor Lambda logs** after deployment
3. **Keep image tags organized** (commit SHA is ideal)
4. **Review pipeline logs** for any warnings
5. **Document manual overrides** if needed
6. **Set up CloudWatch alarms** for Lambda errors

## 📞 Support Resources

- **Bitbucket Pipelines Docs**: https://support.atlassian.com/bitbucket-cloud/docs/get-started-with-bitbucket-pipelines/
- **AWS OIDC Setup**: https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create_for_idp_oidc.html
- **Lambda Troubleshooting**: https://docs.aws.amazon.com/lambda/latest/dg/troubleshooting.html
- **ECR Guide**: https://docs.aws.amazon.com/AmazonECR/latest/userguide/
