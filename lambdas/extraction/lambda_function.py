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

import json
import logging
import os
import tempfile
from typing import Any

from shared.apryse_parser import parse_pages

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


def _make_display_spans(text: str, kind: str, bbox: list | None = None) -> list[dict]:
    """
    Produce display_spans for a semantic object.

    - Paragraphs  → spaCy sentences, each with char start/end within the
                    full paragraph text.  Used by the UI to highlight the
                    exact matched sentence without re-splitting at query time.
    - All others  → single span covering the whole text (table row, heading,
                    list item, form field, bullet, signature are atomic display
                    units with their own bbox).

    Returns list of:
        {"type": str, "text": str, "start": int, "end": int, "bbox": list}
    """
    _bbox = bbox or []

    if kind == "list":
        # Each newline-delimited item becomes a list_item span, so the whole
        # list is embedded as one unit but each bullet can be highlighted.
        spans: list[dict] = []
        cursor = 0
        for line in text.split("\n"):
            stripped = line.strip()
            if len(stripped) >= _MIN_SPAN_CHARS:
                start = text.find(stripped, cursor)
                if start == -1:
                    start = cursor
                end = start + len(stripped)
                spans.append({"type": "list_item", "text": stripped,
                               "start": start, "end": end, "bbox": _bbox})
                cursor = end
        return spans or [{"type": "list_item", "text": text,
                          "start": 0, "end": len(text), "bbox": _bbox}]

    if kind != "paragraph":
        return [{"type": kind, "text": text, "start": 0, "end": len(text), "bbox": _bbox}]

    nlp = _get_nlp()
    if nlp is not None:
        doc = nlp(text)
        spans = [
            {"type": "sentence",
             "text": sent.text,
             "start": sent.start_char,
             "end":   sent.end_char,
             "bbox":  _bbox}   # paragraph block bbox shared across its sentences
            for sent in doc.sents
            if len(sent.text.strip()) >= _MIN_SPAN_CHARS
        ]
    else:
        # regex fallback — no char offsets from split(), recompute via search
        spans = []
        cursor = 0
        for part in _SENT_SPLIT_RE.split(text):
            part = part.strip()
            if len(part) >= _MIN_SPAN_CHARS:
                start = text.find(part, cursor)
                end   = start + len(part)
                spans.append({"type": "sentence", "text": part,
                               "start": start, "end": end, "bbox": _bbox})
                cursor = end

    return spans or [{"type": "sentence", "text": text, "start": 0, "end": len(text), "bbox": _bbox}]


def _build_objects(chunk_id: str, pages: list[dict], global_offset: int = 0) -> list[dict]:
    """
    Build a flat list of semantic objects from Apryse paragraph_objects.

    global_offset  — first global_position to assign (0-based across the whole
                     document).  Pass the running count from the caller so that
                     context expansion can query
                     WHERE global_position BETWEEN N-3 AND N+3
                     without worrying about chunk boundaries.

    Each object is the *embedding unit* — one vector per object.
    Paragraph text is NOT split for embedding; only display_spans are split.
    Lists are one object (whole list embedded together); items are display_spans.

    Schema
    ------
    {
      "object_id":       str,   # e.g. "chunk_0001_obj_0042"
      "position":        int,   # 0-based index within chunk (local, for object_id)
      "global_position": int,   # 0-based index within document (for context expansion)
      "type":            str,   # paragraph | heading | table_row | list
      "text":            str,   # full text — the embedding unit
      "normalized_text": str,   # lowercase, stripped — used by Exact/Fuzzy/Token scorers
      "searchable":      bool,    # False for page headers/footers — excluded from index
      "indexable":       bool,    # False for legal/metadata/structural — excluded from vector search
      "category":        str,     # "clinical"|"title"|"document_metadata"|"legal"|"structural"
      "boost_weight":    float,   # retrieval boost: H1=3.0, H2=2.5, …, paragraph=1.0; legal/meta=0.0
      "parent_heading":  str|None,  # text of heading that directly contains this object
      "section":         str|None,  # text of most-recent heading (for reranker context)
      "section_number":  str|None,  # e.g. "5.2.2.8.1" — from numbered markers
      "section_depth":   int|None,  # dot-count + 1 (e.g. "5.2.2.8.1" → 5)
      "section_level":   int|None,  # heading depth (1 = top-level)
      "prev_object_pos": int|None,  # global_position of preceding object (None for first)
      "next_object_pos": int,       # global_position of following object
      "page":            int,
      "bbox":            [x1,y1,x2,y2],
      "display_spans":   list[dict],  # [{type, text, start, end, bbox}] — UX only
      "embedding":       [],          # filled by Embedding Lambda
      "entities":        [],          # filled by NER Lambda; offsets are object-relative
    }
    """
    raw: list[dict] = []
    current_section_number: str | None = None   # updated by section_marker pseudo-objects
    current_section_from_ref: str | None = None  # title extracted from hybrid section refs

    for page in pages:
        page_num    = page.get("page_number", 0)
        page_height = page.get("height", 792.0)
        for layout_obj in page.get("paragraph_objects", []):
            text  = layout_obj.get("text", "").strip()
            rect  = layout_obj.get("rect", [])
            kind  = layout_obj.get("type", "paragraph")
            level = layout_obj.get("level")   # heading depth, None for non-headings

            # Section markers update tracking but are NOT semantic objects
            if kind == "section_marker":
                current_section_number = layout_obj.get("text", "") or None
                title = layout_obj.get("section_title")
                if title:
                    current_section_from_ref = title
                continue

            if not text:
                continue

            raw.append({
                "type":            kind,
                "text":            text,
                "normalized_text": _normalize_text(text),
                "page":            page_num,
                "bbox":            rect,
                "searchable":      not _is_page_boilerplate(rect, page_height),
                "display_spans":   _make_display_spans(text, kind, bbox=rect),
                "embedding":       [],
                "entities":        [],
                "_heading_level":  level,                    # ephemeral
                "_section_number": current_section_number,  # ephemeral
            })

    # ── Deduplicate: drop exact-text duplicates on the same page ─────────────
    # Apryse occasionally extracts the same text region twice (e.g. cover page
    # elements that appear in both the visual and structural layer).
    seen_key: set[tuple] = set()
    deduped: list[dict] = []
    for obj in raw:
        key = (obj["normalized_text"], obj["page"])
        if key in seen_key:
            continue
        seen_key.add(key)
        deduped.append(obj)
    raw = deduped

    # ── Fragment merge: stitch orphaned tail fragments to the preceding paragraph
    # A "fragment" is a non-heading paragraph with < 8 words that does NOT end
    # with terminal punctuation — typically a sentence that wrapped across a
    # page-region boundary ("requirements." appearing alone on the next page).
    _TERMINAL = {".", "?", "!", ":", ";"}
    merged: list[dict] = []
    for obj in raw:
        if (
            merged
            and obj["type"] == "paragraph"
            and merged[-1]["type"] == "paragraph"
            and obj["searchable"]
            and merged[-1]["searchable"]
        ):
            prev = merged[-1]
            prev_words = prev["text"].split()
            # Merge if the previous paragraph ended without terminal punctuation
            # and the current object is a short orphaned fragment.
            if (
                prev["text"]
                and prev["text"][-1] not in _TERMINAL
                and len(obj["text"].split()) < 8
            ):
                merged_text = prev["text"].rstrip() + " " + obj["text"].lstrip()
                prev["text"]            = merged_text
                prev["normalized_text"] = _normalize_text(merged_text)
                prev["display_spans"]   = _make_display_spans(merged_text, "paragraph", bbox=prev["bbox"])
                continue   # swallow this fragment into the previous object
        merged.append(obj)
    raw = merged

    # Assign positions, section metadata, and final object_ids
    objects: list[dict] = []
    current_section       = None
    current_section_level = None

    for pos, obj in enumerate(raw):
        kind          = obj["type"]
        heading_level = obj.pop("_heading_level", None)   # remove before ** expansion
        section_num   = obj.pop("_section_number", None)  # remove before ** expansion

        # Snapshot the current heading BEFORE updating — this is the true parent.
        parent_heading = current_section

        if kind == "heading":
            current_section       = obj["text"]
            current_section_level = heading_level or 1

        sec_depth = len(section_num.split(".")) if section_num else None

        # ── Category — coarse content classification for query-time filtering ──
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

        # ── Indexable — whether this object participates in vector search ──────
        # Structural boilerplate, legal disclaimers, and admin metadata are
        # excluded from embeddings; they add noise without answering questions.
        indexable = category not in ("structural", "document_metadata", "legal")

        # ── Boost weight — governs retrieval scoring priority ─────────────────
        if not indexable:
            boost = 0.0
        elif kind == "heading":
            lvl = heading_level or 2
            boost = round(max(1.5, 3.0 - 0.5 * (lvl - 1)), 1)
        else:
            boost = 1.0

        # ── Reclassify admin paragraphs with a distinct type ──────────────────
        # "Status: Approved", "Date:", "EDMS number:" etc. become type=metadata
        # so downstream filtering never mistakes them for clinical paragraphs.
        stored_type = "metadata" if (category == "document_metadata" and kind == "paragraph") else kind

        gpos = global_offset + pos

        objects.append({
            **obj,                          # raw fields (type, text, bbox, …)
            "object_id":       f"{chunk_id}_obj_{pos:04d}",
            "position":        pos,         # chunk-local (used in object_id)
            "global_position": gpos,        # document-global (context expansion)
            "prev_object_pos": gpos - 1 if gpos > 0 else None,
            "next_object_pos": gpos + 1,
            "type":            stored_type, # overrides raw type for metadata objects
            "category":        category,
            "indexable":       indexable,
            "boost_weight":    boost,
            "parent_heading":  parent_heading,
            "section_number":  section_num,
            "section_depth":   sec_depth,
            "section":         current_section,
            "section_level":   current_section_level,
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
