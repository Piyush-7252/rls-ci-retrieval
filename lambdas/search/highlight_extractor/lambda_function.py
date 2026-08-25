"""Legacy compatibility wrapper.

Geometry is resolved upstream and stored on the candidate semantic object.
This module never reconstructs geometry and never emits page/character offsets.
"""
from __future__ import annotations
from typing import Any


def _process(req: dict) -> dict:
    if req.get("_failed") or req.get("_early_exit"):
        return req
    for candidate in req.get("verified_candidates", []):
        obj = candidate.get("matched_object") or candidate.get("indexed_object") or {}
        geometry = obj.get("geometry") if isinstance(obj, dict) else None
        if isinstance(geometry, dict):
            candidate["geometry"] = geometry
    return req


def handler(event: dict, context: Any = None) -> dict:
    return _process(event)
