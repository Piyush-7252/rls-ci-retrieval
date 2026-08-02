#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="${AWS_REGION:-eu-west-1}"
FUNCTION_NAME="${FUNCTION_NAME:-rls-ci-chunk-worker}"
QUEUE_NAME="${QUEUE_NAME:-rls-ci-chunk-queue}"
DLQ_NAME="${DLQ_NAME:-rls-ci-chunk-dlq}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"

get_or_create_queue_url() {
  local name="$1"
  local attrs_json="${2:-}"
  local url

  if url="$(aws sqs get-queue-url --queue-name "$name" --region "$AWS_REGION" --query QueueUrl --output text 2>/dev/null)"; then
    echo "$url"
    return
  fi

  if [[ -n "$attrs_json" ]]; then
    url="$(aws sqs create-queue --queue-name "$name" --region "$AWS_REGION" --attributes "$attrs_json" --query QueueUrl --output text)"
  else
    url="$(aws sqs create-queue --queue-name "$name" --region "$AWS_REGION" --query QueueUrl --output text)"
  fi
  echo "$url"
}

DLQ_URL="$(get_or_create_queue_url "$DLQ_NAME")"
DLQ_ARN="$(aws sqs get-queue-attributes --queue-url "$DLQ_URL" --attribute-names QueueArn --region "$AWS_REGION" --query Attributes.QueueArn --output text)"

QUEUE_URL="$(get_or_create_queue_url "$QUEUE_NAME")"
QUEUE_ATTRS="$(DLQ_ARN="$DLQ_ARN" python3 - <<'PY'
import json
import os

print(json.dumps({
  "VisibilityTimeout": "900",
  "ReceiveMessageWaitTimeSeconds": "20",
  "RedrivePolicy": json.dumps({
    "deadLetterTargetArn": os.environ["DLQ_ARN"],
    "maxReceiveCount": "5"
  })
}))
PY
)"
aws sqs set-queue-attributes \
  --queue-url "$QUEUE_URL" \
  --attributes "$QUEUE_ATTRS" \
  --region "$AWS_REGION" >/dev/null
QUEUE_ARN="$(aws sqs get-queue-attributes --queue-url "$QUEUE_URL" --attribute-names QueueArn --region "$AWS_REGION" --query Attributes.QueueArn --output text)"

EXISTING_UUID="$(aws lambda list-event-source-mappings \
  --function-name "$FUNCTION_NAME" \
  --event-source-arn "$QUEUE_ARN" \
  --region "$AWS_REGION" \
  --query 'EventSourceMappings[0].UUID' --output text 2>/dev/null || true)"

if [[ -z "$EXISTING_UUID" || "$EXISTING_UUID" == "None" ]]; then
  aws lambda create-event-source-mapping \
    --function-name "$FUNCTION_NAME" \
    --event-source-arn "$QUEUE_ARN" \
    --batch-size "$BATCH_SIZE" \
    --function-response-types ReportBatchItemFailures \
    --scaling-config "MaximumConcurrency=${MAX_CONCURRENCY}" \
    --region "$AWS_REGION" >/dev/null
  echo "Created event source mapping: ${QUEUE_ARN} -> ${FUNCTION_NAME}"
else
  aws lambda update-event-source-mapping \
    --uuid "$EXISTING_UUID" \
    --batch-size "$BATCH_SIZE" \
    --scaling-config "MaximumConcurrency=${MAX_CONCURRENCY}" \
    --region "$AWS_REGION" >/dev/null
  echo "Updated event source mapping: ${EXISTING_UUID}"
fi

echo "QUEUE_URL=${QUEUE_URL}"
echo "QUEUE_ARN=${QUEUE_ARN}"
echo "DLQ_URL=${DLQ_URL}"
echo "DLQ_ARN=${DLQ_ARN}"
