"""
AWS Secrets Manager helpers shared by Lambdas.

Tenant secrets are keyed by tenant schema. The full JSON secret is cached
briefly in the warm Lambda container to avoid a Secrets Manager call for
every callback.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import boto3

_REGION = os.environ.get("AWS_REGION", "eu-west-1")
_SECRET_CACHE_TTL = float(os.environ.get("TENANT_SECRET_CACHE_TTL", "300"))

_secret_cache: dict[str, tuple[dict[str, Any], float]] = {}
_sm = None


def _get_sm():
    global _sm
    if _sm is None:
        _sm = boto3.client("secretsmanager", region_name=_REGION)
    return _sm


def get_tenant_secret(tenant_schema: str) -> dict[str, Any]:
    """Return the full tenant secret JSON object."""
    if not tenant_schema:
        raise ValueError("tenant_schema is required")

    now = time.time()
    cached = _secret_cache.get(tenant_schema)
    if cached and now - cached[1] < _SECRET_CACHE_TTL:
        return cached[0]

    secret_name = tenant_schema
    resp = _get_sm().get_secret_value(SecretId=secret_name)
    secret_string = resp.get("SecretString")
    if not secret_string:
        raise ValueError(f"Secret {secret_name!r} has no SecretString")

    secret = json.loads(secret_string)
    if not isinstance(secret, dict):
        raise ValueError(f"Secret {secret_name!r} must contain a JSON object")

    _secret_cache[tenant_schema] = (secret, now)
    return secret


def get_tenant_api_key(tenant_schema: str) -> str:
    """Return the API key used for internal server callbacks."""
    secret = get_tenant_secret(tenant_schema)
    api_key = secret.get("password")
    if not api_key:
        raise KeyError(
            f"Tenant secret {tenant_schema!r} does not contain 'password'"
        )
    return str(api_key)
