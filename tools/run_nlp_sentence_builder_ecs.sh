#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="${AWS_REGION:-eu-west-1}"
ECS_CLUSTER="${ECS_CLUSTER:-}"
TASK_DEFINITION="${TASK_DEFINITION:-NLP_SENTENCE_BUILDER}"
SUBNETS="${SUBNETS:-}"
SECURITY_GROUPS="${SECURITY_GROUPS:-}"
ASSIGN_PUBLIC_IP="${ASSIGN_PUBLIC_IP:-DISABLED}"

: "${ECS_CLUSTER:?ECS_CLUSTER is required}"
: "${SUBNETS:?SUBNETS is required (comma-separated)}"
: "${SECURITY_GROUPS:?SECURITY_GROUPS is required (comma-separated)}"
: "${INPUT_BUCKET:?INPUT_BUCKET is required}"
: "${FULL_TABLES_KEY:?FULL_TABLES_KEY is required}"
: "${QUEUE_URL:?QUEUE_URL is required}"
: "${DOCUMENT_ID:?DOCUMENT_ID is required}"
: "${TENANT_ID:?TENANT_ID is required}"
: "${TENANT_NAME:?TENANT_NAME is required}"
: "${TENANT_SCHEMA:?TENANT_SCHEMA is required}"
: "${PROJECT_ID:?PROJECT_ID is required}"

ENV_VARS=(
  "AWS_REGION=${AWS_REGION}"
  "INPUT_BUCKET=${INPUT_BUCKET}"
  "FULL_TABLES_KEY=${FULL_TABLES_KEY}"
  "SOURCE_S3_KEY=${SOURCE_S3_KEY:-}"
  "QUEUE_URL=${QUEUE_URL}"
  "PAYLOAD_BUCKET=${PAYLOAD_BUCKET:-$INPUT_BUCKET}"
  "PAYLOAD_PREFIX=${PAYLOAD_PREFIX:-nlp-sentence-builder-payloads}"
  "DOCUMENT_ID=${DOCUMENT_ID}"
  "FILE_ID=${FILE_ID:-}"
  "TENANT_ID=${TENANT_ID}"
  "TENANT_NAME=${TENANT_NAME}"
  "TENANT_SCHEMA=${TENANT_SCHEMA}"
  "PROJECT_ID=${PROJECT_ID}"
  "CALLBACK_URL=${CALLBACK_URL:-}"
  "CALLBACK_TIMEOUT_SECONDS=${CALLBACK_TIMEOUT_SECONDS:-30}"
)

export ENV_VARS_JSON="$(printf '%s\n' "${ENV_VARS[@]}")"
OVERRIDES_JSON="$(python3 - <<'PY'
import json, os
env=[]
for x in os.environ.get("ENV_VARS_JSON", "").splitlines():
    if "=" in x:
        k,v=x.split("=",1)
        env.append({"name":k,"value":v})
print(json.dumps({"containerOverrides":[{"name":os.environ.get("CONTAINER_NAME","nlp-sentence-builder"),"environment":env}]}))
PY
)"

aws ecs run-task \
  --cluster "$ECS_CLUSTER" \
  --task-definition "$TASK_DEFINITION" \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SECURITY_GROUPS],assignPublicIp=$ASSIGN_PUBLIC_IP}" \
  --overrides "$OVERRIDES_JSON" \
  --region "$AWS_REGION"
