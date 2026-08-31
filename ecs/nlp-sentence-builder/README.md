# NLP_SENTENCE_BUILDER ECS task

This is the document-level preprocessing producer for the indexing pipeline.

It:

1. Downloads `full_tables.json` from S3.
2. Reconstructs the global Apryse document structure.
3. Runs `shared.apryse_parser.parse_pages()`.
4. Runs `shared.section_chunker.build_section_chunks()`.
5. Runs `shared.sentence_builder._build_objects()` for every section.
6. Sends one self-contained document chunk per SQS message.
7. Offloads oversized SQS payloads to S3.
8. Calls the backend indexing-status callback at `PROCESSING`, `DISPATCHED`, and `FAILED`.

The task definition family is `rls-ci-retrieval-nlp-sentence-builder`.

## Deploy

```bash
TASK_EXECUTION_ROLE_ARN=arn:aws:iam::<account>:role/<ecs-execution-role> \
TASK_ROLE_ARN=arn:aws:iam::<account>:role/<ecs-task-role> \
tools/deploy_nlp_sentence_builder_ecs.sh
```

The deployment script builds an amd64 ECR image and registers a new Fargate task-definition revision. It does **not** create the ECS cluster or IAM roles because those are environment-specific and should be reused/configured by infrastructure.

## Run one document

```bash
ECS_CLUSTER=<cluster> \
SUBNETS=<subnet-1,subnet-2> \
SECURITY_GROUPS=<sg-id> \
INPUT_BUCKET=<bucket> \
FULL_TABLES_KEY=<tenant>/<project>/extraction/<file>/full_tables.json \
SOURCE_S3_KEY=<tenant>/<project>/documents/<file>.pdf \
QUEUE_URL=<queue-url> \
DOCUMENT_ID=<document-id> \
FILE_ID=<file-id> \
TENANT_ID=<tenant-id> \
TENANT_NAME=<tenant-name> \
TENANT_SCHEMA=<tenant-schema> \
PROJECT_ID=<project-id> \
CALLBACK_URL=https://backend.example.com \
tools/run_nlp_sentence_builder_ecs.sh
```

For production, the backend should call `ecs:RunTask` itself rather than invoking the helper script. The same environment variables become ECS container overrides.
