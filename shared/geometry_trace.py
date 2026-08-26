"""
Geometry trace helpers used only for debugging geometry preservation.

The tracer is intentionally read-only: it never repairs, reconstructs, or
changes geometry.  It records the first point at which an object that had
usable geometry becomes geometry-empty.
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

logger = logging.getLogger("geometry_trace")

_TRACE_ENABLED = os.environ.get("GEOMETRY_TRACE", "1").lower() not in {"0", "false", "off", "no"}
_FAIL_ON_LOSS = os.environ.get("GEOMETRY_TRACE_FAIL_ON_LOSS", "0").lower() in {"1", "true", "yes", "on"}

def _usable(g: Any) -> bool:
    if not isinstance(g, dict):
        return False
    if g.get("rects"):
        return True
    if g.get("bbox"):
        return True
    if g.get("page_distribution"):
        for p in g.get("page_distribution") or []:
            if isinstance(p, dict) and (p.get("rects") or p.get("bbox")):
                return True
    # Native Apryse objects sometimes carry only a rect/bbox at the wrapper.
    return False

def geometry_state(obj: Any) -> str:
    if not isinstance(obj, dict):
        return "NOT_OBJECT"
    if "geometry" not in obj:
        return "GEOMETRY_KEY_MISSING"
    g = obj.get("geometry")
    if g is None:
        return "GEOMETRY_NULL"
    if not isinstance(g, dict):
        return "GEOMETRY_INVALID"
    if _usable(g):
        return "GEOMETRY_OK"
    return "GEOMETRY_EMPTY"

def _key(obj: dict, index: int = 0) -> str:
    oid = obj.get("object_id") or obj.get("source_object_id")
    if oid:
        return str(oid)
    text = str(obj.get("text", ""))
    raw = f"{obj.get('page', obj.get('page_number', 0))}|{obj.get('type','')}|{text}"
    return "anon-" + hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()[:12] + f"-{index}"

def _objects(value: Any):
    if isinstance(value, dict):
        ext = value.get("extraction")
        if isinstance(ext, dict) and isinstance(ext.get("objects"), list):
            return ext["objects"]
        if isinstance(value.get("objects"), list):
            return value["objects"]
        if isinstance(value.get("paragraph_objects"), list):
            return value["paragraph_objects"]
        if isinstance(value.get("pages"), list):
            out = []
            for p in value["pages"]:
                if isinstance(p, dict):
                    out.extend(p.get("paragraph_objects", []) or [])
            return out
    return []

def snapshot(value: Any) -> dict[str, str]:
    return {_key(o, i): geometry_state(o) for i, o in enumerate(_objects(value))}

def _describe(obj: dict, index: int) -> str:
    return (
        f"object={obj.get('object_id') or '<pre-id>'} "
        f"type={obj.get('type')} page={obj.get('page', obj.get('page_number'))} "
        f"bbox={bool(obj.get('bbox'))} "
        f"text={str(obj.get('text','')).replace(chr(10),' ')[:100]!r}"
    )

def trace(stage: str, value: Any, previous: dict[str, str] | None = None,
          *, logger_name: str | None = None) -> dict[str, str]:
    """
    Log a complete geometry checkpoint and immediately log every
    present -> missing/empty transition compared with the previous checkpoint.
    """
    if not _TRACE_ENABLED:
        return snapshot(value)
    log = logging.getLogger(logger_name or "geometry_trace")
    objs = _objects(value)
    states = snapshot(value)
    counts = {}
    for state in states.values():
        counts[state] = counts.get(state, 0) + 1
    log.info(
        "[GEOM-CHECKPOINT] stage=%s objects=%d ok=%d empty=%d key_missing=%d null=%d invalid=%d",
        stage, len(objs), counts.get("GEOMETRY_OK", 0),
        counts.get("GEOMETRY_EMPTY", 0), counts.get("GEOMETRY_KEY_MISSING", 0),
        counts.get("GEOMETRY_NULL", 0), counts.get("GEOMETRY_INVALID", 0),
    )
    if previous:
        prev_by_key = previous
        for i, obj in enumerate(objs):
            k = _key(obj, i)
            before = prev_by_key.get(k)
            after = states.get(k)
            if before == "GEOMETRY_OK" and after != "GEOMETRY_OK":
                log.error(
                    "[GEOMETRY-LOSS] stage=%s previous=%s current=%s %s",
                    stage, before, after, _describe(obj, i),
                )
                if _FAIL_ON_LOSS:
                    raise RuntimeError(
                        f"Geometry lost at {stage}: {obj.get('object_id') or obj.get('text','')[:80]}"
                    )
    return states

def trace_raw_apryse(stage: str, doc_structure: dict) -> dict:
    """Trace native Apryse pages/elements before parser transformation."""
    if not _TRACE_ENABLED:
        return {}
    pages = doc_structure.get("pages", []) if isinstance(doc_structure, dict) else []
    elements = 0
    with_rect = 0
    with_bbox = 0
    with_contents = 0
    for page in pages:
        for el in page.get("elements", []) or []:
            elements += 1
            if el.get("rect"):
                with_rect += 1
            if el.get("bbox"):
                with_bbox += 1
            if el.get("contents"):
                with_contents += 1
    logger.info(
        "[GEOM-RAW-APRYSE] stage=%s pages=%d elements=%d rect=%d bbox=%d contents=%d",
        stage, len(pages), elements, with_rect, with_bbox, with_contents,
    )
    return {
        "pages": len(pages), "elements": elements,
        "rect": with_rect, "bbox": with_bbox, "contents": with_contents,
    }

def trace_index_docs(stage: str, docs: list[dict], previous: dict[str, str] | None = None) -> dict[str, str]:
    """Checkpoint docs immediately before JSON serialization / OpenSearch write."""
    return trace(stage, {"objects": docs}, previous)
