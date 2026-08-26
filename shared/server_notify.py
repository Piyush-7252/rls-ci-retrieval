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
