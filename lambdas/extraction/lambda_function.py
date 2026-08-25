"""
Document Pipeline — Stage 2: Extraction
=========================================
Uses Apryse SDK ``DataExtractionModule`` with ``e_DocStructure`` mode.
Triggered by : Orchestrator Lambda (async)
Fan-out to   : Normalize Lambda (async)

Input chunk
-----------
{
    "document_id": str,
    "chunk_id":    str,
    "s3_bucket":   str,
    "s3_key":      str,
    "page_start":  int,   # 1-based, matches Apryse pageNumber
    "page_end":    int,
}

Appends
-------
"extraction": {
    "raw_text": str,       # all text from this chunk joined for downstream stages
    "pages":    list[Page] # one entry per page; schema from shared.apryse_parser.parse_page
}

Page schema  (produced by shared/apryse_parser.py)
----------------------------------------------------
{
    "page_number":  int,
    "width":        float,
    "height":       float,
    "headings":     list[{"level": int, "text": str, "rect": list}],
    "paragraphs":   list[str],
    "headers":      list[str],
    "footers":      list[str],
    "tables":       list[{"rect": list, "rows": list[list[str]]}],
    "lists":        list[{"items": list[str]}],
    "raw_text":     str,
    "doc_structure": dict,   # raw Apryse page object — preserved verbatim
}
"""

from __future__ import annotations

import copy
import json
import logging
import os
import tempfile
from typing import Any

from shared.apryse_parser import parse_pages
from shared.geometry import SentenceSpan

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

NORMALIZE_LAMBDA_ARN = os.environ.get("NORMALIZE_LAMBDA_ARN", "")
APRYSE_LICENSE_KEY   = os.environ.get("APRYSE_LICENSE_KEY", "")
# Path to the Apryse StructuredOutput native module (deployed as a Lambda layer)
APRYSE_RESOURCE_PATH = os.environ.get("APRYSE_RESOURCE_PATH", "/opt/apryse")

# ─── lazy AWS clients ─────────────────────────────────────────────────────────
_aws: dict = {}

def _get(service: str):
    if service not in _aws:
        import boto3
        _aws[service] = boto3.client(service)
    return _aws[service]


# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    chunk_id = event.get("chunk_id", "unknown")
    logger.info(
        "[Stage 2 Extraction] start chunk_id=%s pages=%s-%s",
        chunk_id,
        event.get("page_start"),
        event.get("page_end"),
    )

    try:
        result = _process(event)
    except Exception as exc:
        logger.error("[Stage 2 Extraction] failed chunk_id=%s error=%s", chunk_id, exc)
        raise

    logger.info(
        "[Stage 2 Extraction] done chunk_id=%s pages_extracted=%d",
        chunk_id,
        len(result["extraction"]["pages"]),
    )

    _get("lambda").invoke(
        FunctionName   = NORMALIZE_LAMBDA_ARN,
        InvocationType = "Event",
        Payload        = json.dumps(result).encode(),
    )
    return result


def _process(chunk: dict) -> dict:
    chunk_id      = chunk["chunk_id"]
    pdf_bytes     = _fetch_pdf_bytes(chunk["s3_bucket"], chunk["s3_key"])
    doc_structure = _run_apryse_extraction(pdf_bytes)
    pages         = parse_pages(doc_structure, chunk["page_start"], chunk["page_end"])
    raw_text      = "\n\n".join(p["raw_text"] for p in pages)
    objects       = _build_objects(chunk_id, pages)

    return {
        **chunk,
        "extraction": {
            "raw_text": raw_text,
            "pages":    pages,
            "objects":  objects,   # semantic objects — one embedding each
        },
    }


def _fetch_pdf_bytes(bucket: str, key: str) -> bytes:
    response = _get("s3").get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


# ─────────────────────────────────────────────────────────────────────────────
# Sentence builder
# ─────────────────────────────────────────────────────────────────────────────

import re as _re

_MIN_SPAN_CHARS = 10   # discard display spans shorter than this

# ── Object category heuristics ────────────────────────────────────────────────
# Matches administrative metadata lines (Status:, Date:, Prepared by:, …)
_CATEGORY_METADATA_RE = _re.compile(
    r'^(status|date|prepared by|edms (number|doc)?|version|effective date)[:;]',
    _re.IGNORECASE,
)
# Matches legal / confidentiality language anywhere in the text
_CATEGORY_LEGAL_RE = _re.compile(
    r'confidential|trade secret|proprietary|foia|protective order',
    _re.IGNORECASE,
)
_nlp = None            # spaCy model — loaded once on first use


def _get_nlp():
    """Lazy-load spaCy. Falls back to regex if spaCy is unavailable."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            # Prefer SciSpaCy biomedical model if available, else standard sm
            for model in ("en_core_sci_sm", "en_core_web_sm"):
                try:
                    _nlp = spacy.load(model, disable=["ner", "tagger", "lemmatizer", "attribute_ruler"])
                    logger.info("[Extraction] spaCy model loaded: %s", model)
                    break
                except OSError:
                    continue
        except ImportError:
            pass
    return _nlp


_SENT_SPLIT_RE = _re.compile(r'(?<=[.!?])\s+')  # regex fallback only

_NORM_STRIP_RE  = _re.compile(r'[^a-z0-9\s\-]')  # keep hyphens for drug names


def _normalize_text(text: str) -> str:
    """Lowercase, strip non-alphanumeric (keep hyphens), collapse whitespace."""
    t = _NORM_STRIP_RE.sub(' ', text.lower())
    return ' '.join(t.split())


def _is_page_boilerplate(bbox: list, page_height: float) -> bool:
    """
    True if the element sits in the footer zone (bottom ~12 % of the page).

    Apryse uses ``originTop`` coordinates so bbox[1] is the top-left Y
    (small = near top of page, large = near bottom of page).

    Note: Apryse-tagged ``header`` elements are already excluded in parse_page
    before reaching _build_objects, so only misclassified footer paragraphs
    (e.g. running confidentiality notices, page-number lines) need this filter.
    """
    if len(bbox) < 4:
        return False
    return bbox[1] > page_height * 0.88


def _normalize_geometry_text(text: str) -> str:
    """Normalize only whitespace for source-span geometry matching."""
    return " ".join(str(text or "").split())


def _extract_rects_for_text(
    spans: list[dict],
    target_text: str,
    start_span_index: int = 0,
) -> tuple[list[list[float]], int, list[dict]]:
    """Resolve a sentence to native Apryse spans and page-local text/geometry.

    There are deliberately NO character-level PDF rectangles here.

    A sentence is matched against the ordered native Apryse span text. Every
    native span touched by the sentence becomes a contributing span. Its full
    native rect is preserved, even when the sentence only uses part of that
    span. That is the contract for ``containing`` geometry.

    ``page_distribution`` is the common downstream contract for all three
    highlight modes:

      - paragraph mode: use ``paragraph_bbox`` supplied by the parent object.
      - span mode: use ``contributing_spans[].rect``.
      - text-search mode: use the exact ``text`` for that page and let Apryse
        perform the actual PDF text search.

    The ``text`` in each page group is the exact sentence portion belonging to
    that page; the contributing span text is the complete native span text.
    """
    source_spans = spans or []
    target = _normalize_geometry_text(target_text)

    if not target or not source_spans:
        return [], start_span_index, []

    start_span_index = max(0, min(start_span_index, len(source_spans)))

    pieces: list[str] = []
    ranges: list[tuple[int, int, int]] = []
    cursor = 0

    for idx in range(start_span_index, len(source_spans)):
        raw_text = str(source_spans[idx].get("text", "") or "")
        normalized = _normalize_geometry_text(raw_text)
        if not normalized:
            continue
        if pieces:
            cursor += 1
        span_start = cursor
        cursor += len(normalized)
        pieces.append(normalized)
        ranges.append((span_start, cursor, idx))

    if not pieces:
        return [], start_span_index, []

    source = " ".join(pieces)
    match_start = source.find(target)
    if match_start < 0:
        match_start = source.casefold().find(target.casefold())
    if match_start < 0:
        return [], start_span_index, []

    match_end = match_start + len(target)
    matched_indices: list[int] = []
    for span_start, span_end, idx in ranges:
        if span_end <= match_start:
            continue
        if span_start >= match_end:
            break
        matched_indices.append(idx)

    if not matched_indices:
        return [], start_span_index, []

    first_span_start = next(item[0] for item in ranges if item[2] == matched_indices[0])
    last_span_end = next(item[1] for item in ranges if item[2] == matched_indices[-1])

    exact_ownership = (
        match_start == first_span_start
        and match_end == last_span_end
    )

    # Map source span index -> normalized source range.
    range_by_idx = {idx: (span_start, span_end) for span_start, span_end, idx in ranges}

    rects: list[list[float]] = []
    groups: dict[int, dict] = {}

    for idx in matched_indices:
        span = source_spans[idx]
        rect = span.get("rect", [])
        if not (isinstance(rect, (list, tuple)) and len(rect) >= 4):
            continue

        rect_list = [float(v) for v in rect[:4]]
        rects.append(rect_list)

        page = int(span.get("page") or 0)
        group = groups.setdefault(page, {
            "page": page,
            "text_parts": [],
            "bbox": [],
            "rects": [],
            "contributing_spans": [],
        })
        group["rects"].append(rect_list)

        # Full native span text is retained for SPAN highlight mode.
        group["contributing_spans"].append({
            "text": str(span.get("text", "") or ""),
            "rect": rect_list,
            "page": page,
            "span_index": span.get("span_index", idx),
        })

        # Exact sentence text belonging to this source span/page is retained
        # separately for TEXT_SEARCH mode. This can be a partial substring of
        # the native span text; no partial rectangle is ever invented.
        span_start, span_end = range_by_idx[idx]
        overlap_start = max(match_start, span_start)
        overlap_end = min(match_end, span_end)
        normalized_span_text = _normalize_geometry_text(span.get("text", ""))
        rel_start = max(0, overlap_start - span_start)
        rel_end = max(rel_start, overlap_end - span_start)
        segment_text = normalized_span_text[rel_start:rel_end].strip()
        if segment_text:
            group["text_parts"].append(segment_text)

    if not rects:
        return [], start_span_index, []

    # Deduplicate rectangles without losing page ownership.
    deduped: list[list[float]] = []
    seen: set[tuple] = set()
    for rect in rects:
        key = tuple(round(float(v), 6) for v in rect[:4])
        if key not in seen:
            seen.add(key)
            deduped.append(rect)

    distribution: list[dict] = []
    for page in sorted(groups):
        group = groups[page]
        rect_group = group["rects"]
        if not rect_group:
            continue

        distribution.append({
            "page": page,
            # Exact sentence portion for TEXT_SEARCH mode.
            "text": " ".join(group["text_parts"]).strip(),
            # Span geometry for SPAN mode.
            "rects": rect_group,
            "bbox": _union_bbox_many(rect_group),
            "contributing_spans": group["contributing_spans"],
            "geometry_source": "apryse_span",
            "geometry_precision": "exact" if exact_ownership else "containing",
            "is_authoritative": bool(exact_ownership),
        })

    # Exact ownership can advance the cursor safely. Containing geometry may
    # share a native source span with an adjacent sentence, so leave the cursor
    # unchanged and let the next text match locate its own touched spans.
    next_cursor = matched_indices[-1] + 1 if exact_ownership else start_span_index
    return deduped, next_cursor, distribution

def _extract_sentence_rects(
    spans: list[dict],
    sentence_text: str,
    start_span_index: int = 0,
) -> tuple[list[list[float]], int]:
    """Backward-compatible wrapper for text-based geometry resolution."""
    rects, next_index, _distribution = _extract_rects_for_text(
        spans,
        sentence_text,
        start_span_index,
    )
    return rects, next_index


def _make_display_spans_with_geometry(
    text: str,
    kind: str,
    bbox: list | None = None,
    page: int = 0,
    object_id: str | None = None,
    apryse_spans: list[dict] | None = None,
    object_position_in_page: int = 0,
    parent_page_distribution: list[dict] | None = None,
) -> list[dict]:
    """
    Produce display_spans while preserving Apryse-native geometry.

    IMPORTANT GEOMETRY RULE
    ------------------------
    Character offsets are NOT used to select geometry.

    For paragraph sentences we:

        spaCy sentence text
            -> text match against ordered Apryse source spans
            -> native rects of the matched source spans

    This removes the old dependency on synthetic ``start``/``end`` values
    generated by joining Apryse spans with assumed spaces.

    Character positions may still be emitted in the legacy display-span fields
    for compatibility with downstream consumers, but they are not involved in
    geometry resolution.
    """
    _bbox = bbox or []
    _apryse_spans = apryse_spans or []

    def _make_span(
        span_type: str,
        span_text: str,
        rects: list[list[float]],
        geom_source: str,
        char_start: int = 0,
        char_end: int | None = None,
        page_distribution: list[dict] | None = None,
    ) -> dict:
        if char_end is None:
            char_end = char_start + len(span_text)

        span_obj = SentenceSpan(
            text=span_text,
            page=page,
            char_start=char_start,
            char_end=char_end,
            rects=rects,
            source_object_id=object_id,
            span_type=span_type,
            geometry_source=geom_source,
            page_distribution=list(page_distribution or []),
        )

        # The source is always the native Apryse span. Precision is a property
        # of the text-to-span ownership match: exact when the sentence consumes
        # whole source spans, containing when it touches a partial span.
        if geom_source == "apryse_span" and page_distribution:
            span_obj.geometry_precision = (
                "exact"
                if all(
                    d.get("geometry_precision") == "exact"
                    for d in page_distribution
                )
                else "containing"
            )
            span_obj.is_authoritative = span_obj.geometry_precision == "exact"

        # Geometry is created here, at chunk/extraction time, as the
        # single source of truth. Downstream/indexing must only copy this
        # already-resolved geometry; it must never reinterpret display_spans.
        # Every sentence page group keeps its own sentence/span geometry, but
        # paragraph highlighting must use the ORIGINAL Apryse paragraph bbox.
        # For sentences that cross pages, resolve the parent paragraph bbox by
        # page from parent_page_distribution. Never derive it by unioning the
        # sentence/contributing-span rects: those only cover the sentence.
        parent_by_page: dict[int, list[float]] = {}
        for parent_entry in (parent_page_distribution or []):
            if not isinstance(parent_entry, dict):
                continue
            try:
                parent_page = int(parent_entry.get("page") or 0)
            except (TypeError, ValueError):
                continue
            parent_bbox = parent_entry.get("paragraph_bbox") or parent_entry.get("bbox") or []
            if parent_page and isinstance(parent_bbox, (list, tuple)) and len(parent_bbox) >= 4:
                parent_by_page[parent_page] = [float(v) for v in parent_bbox[:4]]

        sentence_page_distribution = []
        for entry in (page_distribution or []):
            if not isinstance(entry, dict):
                continue
            entry_copy = dict(entry)
            try:
                entry_page = int(entry_copy.get("page") or 0)
            except (TypeError, ValueError):
                entry_page = 0

            paragraph_bbox = parent_by_page.get(entry_page)
            if paragraph_bbox:
                entry_copy["paragraph_bbox"] = list(paragraph_bbox)

            sentence_page_distribution.append(entry_copy)

        geometry_dict = {
            "geometry_source": span_obj.geometry_source,
            "geometry_precision": span_obj.geometry_precision,
            "is_authoritative": span_obj.is_authoritative,
            "object_type": span_type,
            "source_object_id": span_obj.source_object_id,
            # page_distribution is the single authoritative geometry source.
            # Each page entry carries its own sentence rects/contributing
            # spans and, when applicable, the parent paragraph bbox.
            "page_distribution": sentence_page_distribution,
        }

        return {
            "type": span_type,
            "text": span_text,
            "bbox": _bbox,
            "geometry": geometry_dict,
            "_sentence_span": span_obj,
        }

    def _resolve(
        target_text: str,
        cursor: int,
    ) -> tuple[list[list[float]], str, int, list[dict]]:
        rects, next_cursor, page_distribution = _extract_rects_for_text(
            _apryse_spans,
            target_text,
            cursor,
        )
        if rects:
            source = "apryse_span" 
            return rects, source, next_cursor, page_distribution
        # No exact source-span geometry. For a sentence this is a bounded
        # containing-object fallback and remains non-authoritative.
        if _bbox:
            page_distribution = [{
                "page": page,
                "bbox": list(_bbox),
                "rects": [list(_bbox)],
                "geometry_source": "object_bbox",
                "geometry_precision": "object",
                "is_authoritative": False,
            }]
            return [_bbox], "object_bbox", cursor, page_distribution
        return [], "none", cursor, []

    if kind == "list":
        spans: list[dict] = []
        cursor = 0
        geometry_cursor = 0

        for line in text.split("\n"):
            stripped = line.strip()
            if len(stripped) < _MIN_SPAN_CHARS:
                continue

            start = text.find(stripped, cursor)
            if start == -1:
                start = cursor
            end = start + len(stripped)

            item_rects, geom_source, geometry_cursor, item_distribution = _resolve(
                stripped,
                geometry_cursor,
            )

            spans.append(
                _make_span(
                    "list_item",
                    stripped,
                    item_rects,
                    geom_source,
                    start,
                    end,
                    item_distribution,
                )
            )
            cursor = end

        if spans:
            return spans

        list_rects, geom_source, _, list_distribution = _resolve(text, 0)
        return [
            _make_span(
                "list_item",
                text,
                list_rects,
                geom_source,
                0,
                None,
                list_distribution,
            )
        ]

    if kind != "paragraph":
        obj_rects, geom_source, _, obj_distribution = _resolve(text, 0)
        return [
            _make_span(
                kind,
                text,
                obj_rects,
                geom_source,
                0,
                None,
                obj_distribution,
            )
        ]

    # Paragraph: split into sentences and resolve geometry by sentence TEXT.
    nlp = _get_nlp()
    spans: list[dict] = []
    geometry_cursor = 0

    if nlp is not None:
        doc = nlp(text)
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            if len(sentence_text) < _MIN_SPAN_CHARS:
                continue

            sent_rects, geom_source, next_geometry_cursor, sent_distribution = _resolve(
                sentence_text,
                geometry_cursor,
            )

            # Only advance the source geometry cursor when we actually matched
            # source spans. This prevents one failed sentence from poisoning all
            # following sentence lookups.
            if sent_rects and geom_source == "apryse_span":
                geometry_cursor = next_geometry_cursor

            spans.append(
                _make_span(
                    "sentence",
                    sent.text,
                    sent_rects,
                    geom_source,
                    sent.start_char,
                    sent.end_char,
                    sent_distribution,
                )
            )
    else:
        cursor = 0
        for part in _SENT_SPLIT_RE.split(text):
            part = part.strip()
            if len(part) < _MIN_SPAN_CHARS:
                continue

            start = text.find(part, cursor)
            if start == -1:
                start = cursor
            end = start + len(part)

            part_rects, geom_source, next_geometry_cursor, part_distribution = _resolve(
                part,
                geometry_cursor,
            )
            if part_rects and geom_source == "apryse_span":
                geometry_cursor = next_geometry_cursor

            spans.append(
                _make_span(
                    "sentence",
                    part,
                    part_rects,
                    geom_source,
                    start,
                    end,
                    part_distribution,
                )
            )
            cursor = end

    if spans:
        return spans

    para_rects, geom_source, _, para_distribution = _resolve(text, 0)
    return [
        _make_span(
            "sentence",
            text,
            para_rects,
            geom_source,
            0,
            None,
            para_distribution,
        )
    ]


def _strip_display_span_geometry(spans: list[dict]) -> list[dict]:
    """Build lightweight display/embedding units.

    Sentence entries retain their canonical geometry so the sentence is
    self-contained as it travels downstream. Other display spans remain
    text-only. Embeddings are added later by the embedding stage.
    """
    result: list[dict] = []
    for span in (spans or []):
        if not isinstance(span, dict) or not span.get("text"):
            continue
        item = {
            "type": span.get("type"),
            "text": span.get("text", ""),
        }
        if span.get("type") == "sentence":
            geometry = span.get("geometry")
            if isinstance(geometry, dict):
                item["geometry"] = copy.deepcopy(geometry)
        result.append(item)
    return result


def _make_display_spans(*args, **kwargs) -> list[dict]:
    """Public display-span representation used by embedding/indexing."""
    return _strip_display_span_geometry(
        _make_display_spans_with_geometry(*args, **kwargs)
    )


def _union_bbox(a: list | None, b: list | None) -> list:
    if not a:
        return b or []
    if not b:
        return a or []
    if len(a) != 4 or len(b) != 4:
        return a
    return [
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    ]


def _offset_display_spans(spans: list[dict], offset: int) -> list[dict]:
    shifted = []
    for span in spans or []:
        new_span = copy.deepcopy(span)
        if isinstance(new_span.get("start"), int):
            new_span["start"] += offset
        if isinstance(new_span.get("end"), int):
            new_span["end"] += offset

        sent = new_span.get("_sentence_span")
        if isinstance(sent, SentenceSpan):
            sent.char_start += offset
            sent.char_end += offset
        elif isinstance(sent, dict):
            if isinstance(sent.get("char_start"), int):
                sent["char_start"] += offset
            if isinstance(sent.get("char_end"), int):
                sent["char_end"] += offset

        shifted.append(new_span)
    return shifted


def _page_distribution_for_object(
    page: int,
    bbox: list,
    text: str = "",
    source_spans: list[dict] | None = None,
) -> list[dict]:
    """Create page-local geometry/text for one Apryse semantic object."""
    if not bbox or len(bbox) < 4:
        return []
    rect = [float(v) for v in bbox[:4]]
    return [{
        "page": int(page or 0),
        "text": str(text or ""),
        "paragraph_bbox": rect,
        "bbox": rect,
        "rects": [rect],
        "contributing_spans": list(source_spans or []),
        "geometry_source": "object_bbox",
        "geometry_precision": "object",
        "is_authoritative": True,
    }]


def _merge_page_distributions(
    a: list[dict] | None,
    b: list[dict] | None,
) -> list[dict]:
    """Merge page-local geometry without ever creating a cross-page rectangle."""
    groups: dict[int, dict] = {}

    for entry in list(a or []) + list(b or []):
        if not isinstance(entry, dict):
            continue
        page = int(entry.get("page") or 0)
        if page <= 0:
            continue
        g = groups.setdefault(page, {
            "page": page,
            "text_parts": [],
            "paragraph_bbox": [],
            "bbox": [],
            "rects": [],
            "contributing_spans": [],
            "geometry_source": entry.get("geometry_source", "object_bbox"),
            "geometry_precision": entry.get("geometry_precision", "object"),
            "is_authoritative": bool(entry.get("is_authoritative", True)),
        })
        entry_text = str(entry.get("text", "") or "").strip()
        if entry_text:
            g["text_parts"].append(entry_text)
        for sp in entry.get("contributing_spans") or []:
            if isinstance(sp, dict):
                g["contributing_spans"].append(sp)
        for r in entry.get("rects") or []:
            if isinstance(r, (list, tuple)) and len(r) >= 4:
                rr = [float(v) for v in r[:4]]
                key = tuple(round(v, 6) for v in rr)
                if not any(tuple(round(float(v), 6) for v in x[:4]) == key for x in g["rects"]):
                    g["rects"].append(rr)
        eb = entry.get("bbox") or entry.get("paragraph_bbox") or []
        if isinstance(eb, (list, tuple)) and len(eb) >= 4:
            g["bbox"] = _union_bbox(g["bbox"], list(eb[:4]))
        pb = entry.get("paragraph_bbox") or eb
        if isinstance(pb, (list, tuple)) and len(pb) >= 4:
            g["paragraph_bbox"] = _union_bbox(g["paragraph_bbox"], list(pb[:4]))

    for g in groups.values():
        if not g["bbox"] and g["rects"]:
            g["bbox"] = _union_bbox_many(g["rects"])
        if not g["paragraph_bbox"]:
            g["paragraph_bbox"] = list(g["bbox"])
        g["text"] = " ".join(g.pop("text_parts", [])).strip()
    return [groups[p] for p in sorted(groups)]


def _union_bbox_many(rects: list[list[float]]) -> list:
    if not rects:
        return []
    x1 = min(r[0] for r in rects)
    y1 = min(r[1] for r in rects)
    x2 = max(r[2] for r in rects)
    y2 = max(r[3] for r in rects)
    return [x1, y1, x2, y2]


def _build_objects(chunk_id: str, pages: list[dict], global_offset: int = 0) -> list[dict]:
    """
    Build semantic retrieval candidates directly from parser objects.

    Canonical geometry:
      paragraph/heading/list_item/table_header/table_row/table_cell -> object's own bbox.
    table_header/table_row use native <tr> geometry; table_cell uses native <td>/<th>.
    display_spans remain legacy/UI provenance only and are never inspected by
    downstream geometry selection.
    """
    raw: list[dict] = []
    current_section_number: str | None = None

    for page in pages:
        page_num = page.get("page_number", 0)
        page_height = page.get("height", 792.0)

        for layout_obj in page.get("paragraph_objects", []):
            text = layout_obj.get("text", "").strip()
            rect = layout_obj.get("rect", [])
            kind = layout_obj.get("type", "paragraph")
            level = layout_obj.get("level")

            if kind == "section_marker":
                current_section_number = layout_obj.get("text", "") or None
                continue
            if not text:
                continue

            bbox = list(rect[:4]) if isinstance(rect, (list, tuple)) and len(rect) >= 4 else []
            source_spans = layout_obj.get("source_spans") or layout_obj.get("spans", [])
            # CANONICAL GEOMETRY — created here, once.
            # Indexing/search only carry this object forward.
            page_distribution = _page_distribution_for_object(
                page_num, bbox, text=text, source_spans=source_spans
            )
            geometry = {
                "geometry_source": "object_bbox" if bbox else "none",
                "geometry_precision": "object" if bbox else "none",
                # The candidate itself is the complete semantic object.
                # Therefore its full Apryse object bbox is authoritative.
                "is_authoritative": bool(bbox),
                "rects": [bbox] if bbox else [],
                "bbox": bbox,
                "object_type": kind,
                "source_object_id": None,
                "page": page_num,
                "page_start": page_num,
                "page_end": page_num,
                "page_distribution": page_distribution,
            }
            resolved_spans = _make_display_spans_with_geometry(
                text, kind, bbox=bbox, page=page_num,
                apryse_spans=source_spans,
                object_position_in_page=0,
                parent_page_distribution=page_distribution,
            )
            lightweight_spans = _strip_display_span_geometry(resolved_spans)
            raw.append({
                "type": kind,
                "text": text,
                "normalized_text": _normalize_text(text),
                "page": page_num,
                "bbox": bbox,
                "geometry": geometry,
                "_page_distribution": page_distribution,
                "searchable": not _is_page_boilerplate(bbox, page_height),
                "display_spans": lightweight_spans,
                "source_spans": source_spans,
                "embedding": [],
                "entities": [],
                "_heading_level": level,
                "_section_number": current_section_number,
                "row_start": layout_obj.get("row_start"),
                "col_start": layout_obj.get("col_start"),
                "row_span": layout_obj.get("row_span"),
                "col_span": layout_obj.get("col_span"),
                "table_key": layout_obj.get("_table_key"),
                "table_id": layout_obj.get("table_id", layout_obj.get("_table_key")),
                "cell_id": layout_obj.get("cell_id"),
                "row_index": layout_obj.get("row_index", layout_obj.get("row_start")),
                "table_role": layout_obj.get("table_role"),
                "list_id": layout_obj.get("list_id"),
                "list_level": layout_obj.get("list_level", layout_obj.get("level")),
                "list_label": layout_obj.get("list_label"),
                "list_number_format": layout_obj.get("list_number_format"),
            })

    seen_key: set[tuple] = set()
    deduped: list[dict] = []
    for obj in raw:
        if obj["type"] == "table_cell":
            rect_key = tuple(round(float(v), 4) for v in obj.get("bbox", [])[:4])
            key = (obj["normalized_text"], obj["page"], obj["type"],
                   rect_key, obj.get("row_start"), obj.get("col_start"))
        else:
            key = (obj["normalized_text"], obj["page"], obj["type"])
        if key in seen_key:
            continue
        seen_key.add(key)
        deduped.append(obj)
    raw = deduped

    # Only ordinary paragraphs may be fragment-merged.
    _TERMINAL = {".", "?", "!", ":", ";"}

    def _is_page_continuation(prev: dict, cur: dict) -> bool:
        """Detect a paragraph that Apryse split at a PDF page boundary.

        The strongest signal is an unfinished sentence at the bottom of one
        page followed by text at the top of the next. A short-fragment rule is
        retained for the historical same-page case.
        """
        if prev.get("page") == cur.get("page"):
            return False
        prev_text = (prev.get("text") or "").rstrip()
        cur_text = (cur.get("text") or "").lstrip()
        if not prev_text or not cur_text or prev_text[-1] in _TERMINAL:
            return False

        first_word = _re.match(r"[A-Za-z]+", cur_text)
        first = first_word.group(0).lower() if first_word else ""
        continuation_words = {
            "and", "or", "but", "which", "that", "this", "these", "those",
            "with", "as", "to", "of", "for", "from", "in", "on", "by",
            "through", "respectively", "including", "whereas", "while",
            "who", "whom", "whose", "because", "although", "than",
        }

        # Page geometry is an additional guard against joining unrelated
        # paragraphs that merely lack terminal punctuation.
        prev_bbox = prev.get("bbox") or []
        cur_bbox = cur.get("bbox") or []
        prev_bottom = float(prev_bbox[3]) if len(prev_bbox) >= 4 else 792.0
        cur_top = float(cur_bbox[1]) if len(cur_bbox) >= 2 else 0.0

        near_page_edges = prev_bottom >= 0.72 * 792.0 and cur_top <= 0.30 * 792.0
        return near_page_edges and (
            first in continuation_words
            or cur_text[:1].islower()
            or prev_text.endswith((",", "-", "—", "–"))
        )

    merged: list[dict] = []
    for obj in raw:
        if (merged and obj["type"] == "paragraph" and
                merged[-1]["type"] == "paragraph" and
                obj["searchable"] and merged[-1]["searchable"]):
            prev = merged[-1]
            same_page_fragment = (
                prev["page"] == obj["page"]
                and prev["text"]
                and prev["text"][-1] not in _TERMINAL
                and len(obj["text"].split()) < 8
            )
            cross_page_fragment = _is_page_continuation(prev, obj)
            if same_page_fragment or cross_page_fragment:
                fragment_offset = len(prev["text"].rstrip()) + 1
                prev["text"] = prev["text"].rstrip() + " " + obj["text"].lstrip()
                prev["normalized_text"] = _normalize_text(prev["text"])
                prev["_page_distribution"] = _merge_page_distributions(
                    prev.get("_page_distribution"),
                    obj.get("_page_distribution"),
                )
                prev["page_start"] = min(
                    int(prev.get("page_start") or prev.get("page") or 0),
                    int(obj.get("page_start") or obj.get("page") or 0),
                )
                prev["page_end"] = max(
                    int(prev.get("page_end") or prev.get("page") or 0),
                    int(obj.get("page_end") or obj.get("page") or 0),
                )
                prev["bbox"] = _union_bbox(prev.get("bbox", []), obj.get("bbox", []))
                pd = prev.get("_page_distribution") or []
                pd_rects = [r for entry in pd for r in (entry.get("rects") or [])]
                prev["geometry"] = {
                    "geometry_source": "object_bbox" if pd_rects else "none",
                    "geometry_precision": "object" if pd_rects else "none",
                    "is_authoritative": bool(pd_rects),
                    # Page-local bboxes are intentionally used as the object's
                    # highlight rects. Never create one rectangle spanning pages.
                    "rects": pd_rects,
                    "bbox": prev.get("bbox", []),
                    "object_type": "paragraph",
                    "source_object_id": None,
                    "page": prev.get("page"),
                    "page_start": prev.get("page_start"),
                    "page_end": prev.get("page_end"),
                    "page_distribution": pd,
                }
                prev["source_spans"].extend(obj.get("source_spans", []))

                # Rebuild sentence geometry from the combined native Apryse
                # span stream. display_spans remain lightweight text-only data.
                rebuilt = _make_display_spans_with_geometry(
                    prev["text"],
                    "paragraph",
                    bbox=prev.get("bbox", []),
                    page=prev.get("page", 0),
                    apryse_spans=prev.get("source_spans", []),
                    object_position_in_page=0,
                    parent_page_distribution=prev.get("_page_distribution") or [],
                )
                prev["display_spans"] = _strip_display_span_geometry(rebuilt)
                continue
        merged.append(obj)
    raw = merged

    objects: list[dict] = []
    current_section = None
    current_section_level = None

    for pos, obj in enumerate(raw):
        kind = obj["type"]
        heading_level = obj.pop("_heading_level", None)
        section_num = obj.pop("_section_number", None)
        parent_heading = current_section

        if kind == "heading":
            current_section = obj["text"]
            current_section_level = heading_level or 1

        sec_depth = len(section_num.split(".")) if section_num else None
        text_sample = obj["text"]
        if not obj["searchable"]:
            category = "structural"
        elif _CATEGORY_METADATA_RE.match(text_sample):
            category = "document_metadata"
        elif _CATEGORY_LEGAL_RE.search(text_sample):
            category = "legal"
        elif obj["page"] == 1 and kind == "heading":
            category = "title"
        else:
            category = "clinical"

        indexable = category not in ("structural", "document_metadata", "legal")
        if not indexable:
            boost = 0.0
        elif kind == "heading":
            lvl = heading_level or 2
            boost = round(max(1.5, 3.0 - 0.5 * (lvl - 1)), 1)
        else:
            boost = 1.0

        stored_type = "metadata" if category == "document_metadata" and kind == "paragraph" else kind
        gpos = global_offset + pos

        obj.pop("_page_distribution", None)
        objects.append({
            **obj,
            "object_id": f"{chunk_id}_obj_{pos:04d}",
            "position": pos,
            "global_position": gpos,
            "prev_object_pos": gpos - 1 if gpos > 0 else None,
            "next_object_pos": gpos + 1,
            "type": stored_type,
            "category": category,
            "indexable": indexable,
            "boost_weight": boost,
            "parent_heading": parent_heading,
            "section_number": section_num,
            "section_depth": sec_depth,
            "section": current_section,
            "section_level": current_section_level,
            "page_start": obj.get("page_start", obj.get("page")),
            "page_end": obj.get("page_end", obj.get("page")),
        })


    return objects

def _run_apryse_extraction(pdf_bytes: bytes) -> dict:
    """
    Write the PDF to /tmp, run Apryse e_DocStructure extraction,
    return the parsed JSON dict.

    Apryse is installed as a Lambda Layer (apryse_sdk).
    """
    from apryse_sdk import PDFNet, DataExtractionModule, DataExtractionOptions

    PDFNet.Initialize(APRYSE_LICENSE_KEY)
    PDFNet.AddResourceSearchPath(APRYSE_RESOURCE_PATH)

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        tmp.write(pdf_bytes)
        tmp.close()

        options  = DataExtractionOptions()
        raw_json = DataExtractionModule.ExtractData(
            tmp.name,
            DataExtractionModule.e_DocStructure,
            options,
        )
        return json.loads(raw_json)
    finally:
        os.unlink(tmp.name)
