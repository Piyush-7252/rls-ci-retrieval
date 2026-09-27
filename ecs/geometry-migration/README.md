# GEOMETRY_MIGRATION ECS task

One-time (resumable) migration of already-indexed geometry (`document-chunks`
+ `semantic-objects`) to S3, driven by `app.py`. See its module docstring for
the full algorithm and manifest state machine (`PENDING -> PROCESSING ->
SUCCESS / FAILED`). Same architecture as `nlp-sentence-builder`: a plain
`CMD ["python", "/app/app.py"]`, no wrapper/entrypoint script — every setting
is read from env vars (see below), so a task can be driven entirely by the
ECS task definition / `run-task` overrides with no command-line args.

The image bakes in nothing manifest-related — `app.py` handles the manifest
itself, entirely via `MANIFEST_S3_BUCKET`/`MANIFEST_S3_KEY`, so it works on
Fargate's ephemeral filesystem:

1. On startup, if `MANIFEST_S3_BUCKET`/`MANIFEST_S3_KEY` are set, it downloads
   that object to `MANIFEST_FILE` and resumes from it (whatever `PENDING` /
   `SUCCESS` / `FAILED` state it holds). If the key doesn't exist yet, it
   falls back to discovering documents fresh from OpenSearch.
2. After every document (and once more at the end), it saves `MANIFEST_FILE`
   locally **and** re-uploads it to the same S3 key — so a crash, OOM kill, or
   the task simply being stopped mid-run still leaves the true latest state
   in S3, and rerunning the task with the same bucket/key picks up from there.

To seed a run from an existing local snapshot (e.g. `devgeometry/manifest.json`
or `qageomtry/manifest.json`), upload it to the target key yourself before the
first task run — e.g. `aws s3 cp devgeometry/manifest.json
s3://rls-file-bucket-eu/extractions/rls-ci-retrieval-geometry-migration/manifest.json`.
Without that, the first run just re-discovers from OpenSearch, which is safe
(deterministic S3 output keys mean reprocessing overwrites, not duplicates)
but wastes time re-walking documents already migrated.

The task definition family is `rls-ci-retrieval-geometry-migration` (container
name matches, same convention as `rls-ci-retrieval-nlp-sentence-builder`).

## Environment variables

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `AWS_REGION` | yes | `eu-west-1` | Region for OpenSearch/S3 |
| `OPENSEARCH_ENDPOINT` | yes | dev domain | OpenSearch domain host (no scheme) |
| `BUCKET` | yes | — | S3 bucket the extraction outputs live in |
| `MANIFEST_FILE` | no | `geometry_migration_manifest.json` | Local path inside the container the manifest is read/written at |
| `MANIFEST_S3_BUCKET` / `MANIFEST_S3_KEY` | no | — | If both set, the manifest is downloaded from here at startup and re-uploaded here after every save |
| `DOCUMENT_WORKERS` | no | `8` | Concurrent documents |
| `OPENSEARCH_CONCURRENCY` | no | `16` | Global cap on concurrent OpenSearch scroll streams |
| `SKIP_TENANT` | no | — | Comma-separated `tenantName`s to exclude (e.g. `QA AUTH TEST`) |
| `RETRY_FAILED` | no | `false` | Only (re)process `FAILED` manifest entries |
| `RETRY_ERROR_CONTAINS` | no | — | With `RETRY_FAILED`, further narrow to errors containing this substring (e.g. `scroll failed`) |
| `OVERWRITE` | no | `false` | Reprocess documents even if already `SUCCESS` |
| `DRY_RUN` | no | `false` | Don't write to S3, just log |
| `LIMIT` | no | `0` | Only process the first N documents (0 = no limit) |
| `ONLY_DOCUMENT_ID` | no | — | Process a single manifest entry by its composite id |

`AWS_PROFILE` is intentionally left unset in ECS — the task's IAM role already
provides credentials; that flag/env var only matters for local runs against a
named `~/.aws` profile.

## Deploying

Like `nlp-sentence-builder`, [.github/workflows/deploy-geometry-migration.yml](../../.github/workflows/deploy-geometry-migration.yml)
only builds/pushes the image and registers a new task-definition revision with
that image — it never runs the task itself. This means **the task definition
must already exist** (register it once yourself, e.g. via the AWS console or
a one-off `aws ecs register-task-definition`, with the container name
`rls-ci-retrieval-geometry-migration`, the env vars below, and the IAM roles
described under "Run as an ECS task"). After that, every push to `main`
touching `ecs/geometry-migration/**` (or a manual `workflow_dispatch`) rolls a
new image into a new revision of that same task definition; you then trigger
an actual run with `tools/run_geometry_migration_ecs.sh`.

## Build & run locally

```bash
docker build -f ecs/geometry-migration/Dockerfile -t geometry-migration .
docker run --rm \
  -e AWS_REGION=us-east-1 \
  -e OPENSEARCH_ENDPOINT=search-rls-qa-u7jwn3q2hr3hxp7y2ydab34tfq.us-east-1.es.amazonaws.com \
  -e BUCKET=rls-file-bucket-qa \
  -e MANIFEST_S3_BUCKET=rls-file-bucket-qa \
  -e MANIFEST_S3_KEY=extractions/rls-ci-retrieval-geometry-migration/manifest.json \
  -e SKIP_TENANT="QA AUTH TEST" \
  -v ~/.aws:/root/.aws:ro \
  geometry-migration
```

## Run as an ECS task

```bash
ECS_CLUSTER=<cluster> \
SUBNETS=<subnet-1,subnet-2> \
SECURITY_GROUPS=<sg-id> \
BUCKET=rls-file-bucket-qa \
OPENSEARCH_ENDPOINT=search-rls-qa-u7jwn3q2hr3hxp7y2ydab34tfq.us-east-1.es.amazonaws.com \
AWS_REGION=us-east-1 \
MANIFEST_S3_BUCKET=rls-file-bucket-qa \
MANIFEST_S3_KEY=extractions/rls-ci-retrieval-geometry-migration/manifest.json \
SKIP_TENANT="QA AUTH TEST" \
DOCUMENT_WORKERS=16 \
OPENSEARCH_CONCURRENCY=32 \
tools/run_geometry_migration_ecs.sh
```

The task's IAM role must have `es:ESHttpGet`/`es:ESHttpPost` on the target
OpenSearch domain and `s3:GetObject`/`s3:PutObject` on both the extraction
bucket (`BUCKET`) and the manifest bucket (`MANIFEST_S3_BUCKET`, if different),
scoped to the right account for the environment being migrated
(dev vs QA use different AWS accounts — see the migration's manifest for the
account each tenant's data lives in).
