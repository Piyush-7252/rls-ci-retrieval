"""
Internal backend notification helpers shared by Lambdas.

Callbacks are intentionally non-fatal: a callback outage must not turn a
successfully indexed CI into an SQS failure. The Lambda logs callback failures
so they remain observable.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

from shared.secret_manager import get_tenant_api_key

logger = logging.getLogger(__name__)

CALLBACK_URL = os.environ.get("CALLBACK_URL", "").rstrip("/")
_CALLBACK_TIMEOUT = float(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "30"))


def notify_server(
    path: str,
    *,
    tenant: dict[str, Any],
    body: dict[str, Any],
) -> bool:
    """
    PATCH an internal backend endpoint.

    Returns True when the server accepts the callback, False when the callback
    is unavailable or fails. Never raises.
    """
    if not CALLBACK_URL:
        logger.info("[ServerNotify] CALLBACK_URL not configured; skipping path=%s", path)
        return False

    tenant_schema = str(
        tenant.get("schema")
        or tenant.get("tenant_schema")
        or ""
    )
    if not tenant_schema:
        logger.warning(
            "[ServerNotify] missing tenant schema; skipping path=%s body=%s",
            path, body,
        )
        return False

    url = f"{CALLBACK_URL}/{path.lstrip('/')}"
    payload = json.dumps(body).encode("utf-8")

    try:
        aws_key = get_tenant_api_key(tenant_schema)
    except Exception as exc:
        logger.warning(
            "[ServerNotify] could not fetch API key tenant_schema=%s error=%s",
            tenant_schema, exc,
        )
        return False

    request = urllib.request.Request(
        url,
        data=payload,
        method="PATCH",
        headers={
            "Content-Type": "application/json",
            "x-tenant": tenant_schema,
            "x-awskey": aws_key,
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=_CALLBACK_TIMEOUT) as response:
            logger.info(
                "[ServerNotify] OK status=%s tenant_schema=%s path=%s",
                response.status, tenant_schema, path,
            )
            return 200 <= response.status < 300
    except urllib.error.HTTPError as exc:
        logger.warning(
            "[ServerNotify] HTTP failure status=%s tenant_schema=%s path=%s body=%s",
            exc.code, tenant_schema, path, body,
        )
    except Exception as exc:
        logger.warning(
            "[ServerNotify] callback failed tenant_schema=%s path=%s error=%s",
            tenant_schema, path, exc,
        )

    return False


def notify_ci_status(
    *,
    ci: dict[str, Any],
    status: str,
    attempt_id: str | None = None,
    error: str | None = None,
) -> bool:
    """Notify the backend of one CI indexing lifecycle transition."""
    ci_id = ci.get("id")
    if ci_id is None:
        logger.warning("[ServerNotify] CI status callback missing ci id")
        return False

    body: dict[str, Any] = {
        "attemptId": attempt_id or "",
    }
    if error:
        body["error"] = error

    status_path = {
        "PROCESSING": f"/api/internal/ci/{ci_id}/processing",
        "INDEXED": f"/api/internal/ci/{ci_id}/indexed",
        "DELETED": f"/api/internal/ci/{ci_id}/deleted",
        "FAILED": f"/api/internal/ci/{ci_id}/failed",
    }.get(status)

    if not status_path:
        raise ValueError(f"Unsupported CI callback status: {status}")

    return notify_server(
        status_path,
        tenant=ci,
        body=body,
    )


def notify_cim_annotation_job_status(
    *,
    job_id: int | str,
    tenant: dict[str, Any],
    status: str,
) -> bool:
    """Notify the backend of a CIM annotation job lifecycle transition.

    The backend exposes:
      PUT /api/cim-annotation-job/{id}/status/on-going
      PUT /api/cim-annotation-job/{id}/status/completed
      PUT /api/cim-annotation-job/{id}/status/failed

    These callbacks are intentionally non-fatal.
    """
    path = {
        "ON_GOING": f"/api/cim-annotation-job/{job_id}/status/on-going",
        "COMPLETED": f"/api/cim-annotation-job/{job_id}/status/completed",
        "FAILED": f"/api/cim-annotation-job/{job_id}/status/failed",
    }.get(status)

    if not path:
        raise ValueError(f"Unsupported CIM annotation job status: {status}")

    # The backend status endpoints do not require a body, but sending the
    # lifecycle status makes the callback useful for request tracing.
    return _notify_server_method(
        path,
        tenant=tenant,
        method="PUT",
        body={"status": status},
    )


def _notify_server_method(
    path: str,
    *,
    tenant: dict[str, Any],
    method: str,
    body: dict[str, Any] | None = None,
) -> bool:
    """Generic non-fatal authenticated internal-server request."""
    if not CALLBACK_URL:
        logger.info(
            "[ServerNotify] CALLBACK_URL not configured; skipping method=%s path=%s",
            method, path,
        )
        return False

    tenant_schema = str(
        tenant.get("schema")
        or tenant.get("tenant_schema")
        or ""
    )
    if not tenant_schema:
        logger.warning(
            "[ServerNotify] missing tenant schema; skipping method=%s path=%s",
            method, path,
        )
        return False

    try:
        aws_key = get_tenant_api_key(tenant_schema)
    except Exception as exc:
        logger.warning(
            "[ServerNotify] could not fetch API key tenant_schema=%s error=%s",
            tenant_schema, exc,
        )
        return False

    payload = json.dumps(body or {}).encode("utf-8")
    request = urllib.request.Request(
        f"{CALLBACK_URL}/{path.lstrip('/')}",
        data=payload,
        method=method,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-tenant": tenant_schema,
            "x-awskey": aws_key,
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=_CALLBACK_TIMEOUT) as response:
            logger.info(
                "[ServerNotify] OK method=%s status=%s tenant_schema=%s path=%s",
                method, response.status, tenant_schema, path,
            )
            return 200 <= response.status < 300
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        logger.warning(
            "[ServerNotify] HTTP failure method=%s status=%s tenant_schema=%s "
            "path=%s detail=%s",
            method, exc.code, tenant_schema, path, detail,
        )
    except Exception as exc:
        logger.warning(
            "[ServerNotify] callback failed method=%s tenant_schema=%s "
            "path=%s error=%s",
            method, tenant_schema, path, exc,
        )

    return False

def get_cim_annotation_job_cis(
    job_id: int | str,
    *,
    tenant: dict[str, Any],
) -> list[dict[str, Any]]:
    """Fetch all CIs belonging to one CIM annotation job.

    Uses the same tenant-scoped internal authentication as server callbacks,
    but performs a GET against the backend CI endpoint.  The backend is the
    source of truth for which CIs belong to the document/job; OpenSearch is
    used later by the search orchestrator only to load enriched CI data.

    Raises on configuration, HTTP, transport, or malformed-response errors so
    the caller can fail the search rather than silently searching the wrong CI
    set.
    """
    if job_id is None or str(job_id) == "":
        raise ValueError("job_id is required")

    if not CALLBACK_URL:
        raise RuntimeError("CALLBACK_URL is required to fetch CIM annotation job CIs")

    tenant_schema = str(tenant.get("tenant_schema") or ""
    )
    if not tenant_schema:
        raise ValueError("tenant schema is required to fetch CIM annotation job CIs")

    try:
        aws_key = get_tenant_api_key(tenant_schema)
    except Exception as exc:
        logger.error(
            "[ServerNotify] could not fetch API key tenant_schema=%s error=%s",
            tenant_schema, exc,
        )
        raise

    url = f"{CALLBACK_URL}/api/cim-annotation-job/{job_id}/cis"
    request = urllib.request.Request(
        url,
        method="GET",
        headers={
            "Accept": "application/json",
            "x-tenant": tenant_schema,
            "x-awskey": aws_key,
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=_CALLBACK_TIMEOUT) as response:
            raw = response.read()
            if not (200 <= response.status < 300):
                raise RuntimeError(
                    f"CI endpoint returned HTTP {response.status}"
                )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        logger.error(
            "[ServerNotify] CI fetch HTTP failure status=%s tenant_schema=%s "
            "job_id=%s detail=%s",
            exc.code, tenant_schema, job_id, detail,
        )
        raise
    except Exception as exc:
        logger.error(
            "[ServerNotify] CI fetch failed tenant_schema=%s job_id=%s error=%s",
            tenant_schema, job_id, exc,
        )
        raise

    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(
            f"CIM annotation CI endpoint returned invalid JSON for job {job_id}"
        ) from exc

    # Support either the current bare-array response or a future envelope.
    if isinstance(payload, list):
        cis = payload
    elif isinstance(payload, dict) and isinstance(payload.get("cis"), list):
        cis = payload["cis"]
    else:
        raise ValueError(
            f"CIM annotation CI endpoint returned unexpected payload for job {job_id}"
        )

    if not all(isinstance(ci, dict) for ci in cis):
        raise ValueError(
            f"CIM annotation CI endpoint returned a non-object CI for job {job_id}"
        )

    logger.info(
        "[ServerNotify] fetched CIM annotation CIs tenant_schema=%s job_id=%s count=%d",
        tenant_schema, job_id, len(cis),
    )
    return cis



def notify_document_indexing_status(
    *,
    job_id: int | str,
    tenant: dict[str, Any],
    status: str,
    attempt_id: str | None = None,
    expected_chunks: int | None = None,
    dispatched_chunks: int | None = None,
    failed_dispatch_chunks: int | None = None,
    error: str | None = None,
) -> bool:
    """Notify backend with the complete document indexing/dispatch state.

    This callback is best-effort. The backend is the source of truth, while
    the ECS/Lambda workers report the latest attempt statistics.
    """
    body: dict[str, Any] = {"status": status}

    if attempt_id is not None:
        body["attemptId"] = str(attempt_id)
    if expected_chunks is not None:
        body["expectedChunks"] = int(expected_chunks)
    if dispatched_chunks is not None:
        body["dispatchedChunks"] = int(dispatched_chunks)
    if failed_dispatch_chunks is not None:
        body["failedDispatchChunks"] = int(failed_dispatch_chunks)
    if error:
        body["error"] = str(error)

    logger.info(
        "[ServerNotify] document status job_id=%s status=%s attempt_id=%s "
        "expected=%s dispatched=%s failed_dispatch=%s indexed=%s failed=%s",
        job_id,
        status,
        attempt_id,
        expected_chunks,
        dispatched_chunks,
        failed_dispatch_chunks
    )

    return notify_server(
        f"/api/internal/documents/{job_id}/indexing-status",
        tenant=tenant,
        body=body,
    )

