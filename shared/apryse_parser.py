"""
Apryse DocStructure Parser
===========================
Pure-Python helpers that turn an Apryse ``e_DocStructure`` JSON result into
the structured page dicts used by the rest of the pipeline.

This module has zero external dependencies — it only uses the stdlib.

Apryse e_DocStructure schema (as observed from SDK output)
------------------------------------------------------------
{
  "properties": { "producer": "StructuredOutput", "coordinateSystem": "originTop", "schemaVersion": "1.0" },
  "pages": [
    {
      "properties": { "pageNumber": int, "rotation": int, "width": float, "height": float },
      "elements": [ <Element>, ... ]
    }
  ]
}

Element types
-------------
paragraph  : { type, rect, textStyle, contents: [span, ...] }
heading    : { type, rect, level, textStyle, contents: [span, ...] }
header     : { type, rect, contents: [textbox, ...] }
footer     : { type, rect, contents: [textbox, ...] }
table      : { type, rect, columnWidths, trs: [tr, ...] }
  tr       : { type, rect, tds: [td, ...] }
  td/th    : { type, rect, rowSpan, colSpan, rowStart, colStart, contents: [paragraph, ...] }
list       : { type, rect, labelFormat, level, listId, numberFormat, startValue, items: [listItem, ...] }
  listItem : { type, rect, label: {rect, text, textStyle}, body: {rect, contents: [paragraph, ...]}, contents: [] }
span       : { type, rect, text, style: {bold, weight, italic, underline, pointSize, fontFace} }
image      : { type, rect }
graphic    : { type, rect }
group      : { type, rect, contents: [...] }
textbox    : { type, rect, contents: [paragraph, ...] }
toc        : { type, rect, items: [...] }
"""

from __future__ import annotations
import re as _re

# Matches section numbers like "5", "5.2", "5.2.2.8", "5.2.2.8.1."
_SECTION_NUM_RE = _re.compile(r'^\d+(\.[\d]+)*\.?$')


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def parse_pages(doc_structure: dict, page_start: int, page_end: int) -> list[dict]:
    """
    Parse all pages in *page_start*–*page_end* (inclusive) from an Apryse
    ``e_DocStructure`` result.

    Args:
        doc_structure: parsed JSON from ``DataExtractionModule.ExtractData``
        page_start:    first page number (1-based, as Apryse uses)
        page_end:      last  page number (inclusive)

    Returns:
        List of structured page dicts (one per page in the range).
    """
    result: list[dict] = []
    for page in doc_structure.get("pages", []):
        pn = page.get("properties", {}).get("pageNumber", 0)
        if page_start <= pn <= page_end:
            result.append(parse_page(page))
    return result


def parse_page(page: dict) -> dict:
    """
    Parse a single Apryse page object into a structured dict.

    Output schema
    -------------
    {
        "page_number": int,
        "width":       float,
        "height":      float,
        "headings":    list[{"level": int, "text": str, "rect": list}],
        "paragraphs":  list[str],
        "headers":     list[str],
        "footers":     list[str],
        "tables":      list[{"rect": list, "rows": list[list[str]]}],
        "lists":       list[{"items": list[str]}],
        "raw_text":    str,          # all text joined for downstream stages
        "doc_structure": dict,       # raw Apryse page preserved verbatim
    }
    """
    props    = page.get("properties", {})
    elements = page.get("elements", [])

    headings:   list[dict]       = []
    paragraphs: list[str]        = []
    headers:    list[str]        = []
    footers:    list[str]        = []
    tables:     list[dict]       = []
    lists:      list[dict]       = []
    paragraph_objects: list[dict] = []   # (text, rect, type) for sentence extraction

    for el in elements:
        el_type = el.get("type", "")

        if el_type == "paragraph":
            text = _spans_text(el.get("contents", []))
            if text.strip():
                if _is_bold_heading(el, text):
                    # Apryse missed this heading — reclassify by bold+short heuristic
                    headings.append({
                        "level": 2,
                        "text":  text.strip(),
                        "rect":  el.get("rect", []),
                    })
                    paragraph_objects.append({
                        "text":  text.strip(),
                        "rect":  el.get("rect", []),
                        "type":  "heading",
                        "level": 2,
                        "spans": _extract_span_geometry(el.get("contents", [])),
                    })
                else:
                    paragraphs.append(text.strip())
                    paragraph_objects.append({
                        "text": text.strip(),
                        "rect": el.get("rect", []),
                        "type": "paragraph",
                        "spans": _extract_span_geometry(el.get("contents", [])),
                    })

        elif el_type == "heading":
            text = _spans_text(el.get("contents", []))
            if text.strip():
                headings.append({
                    "level": el.get("level", 1),
                    "text":  text.strip(),
                    "rect":  el.get("rect", []),
                })
                paragraph_objects.append({
                    "text":  text.strip(),
                    "rect":  el.get("rect", []),
                    "type":  "heading",
                    "level": el.get("level", 1),   # heading depth; used for section tracking
                    "spans": _extract_span_geometry(el.get("contents", [])),
                })

        elif el_type == "header":
            text = _recursive_text(el)
            if text.strip():
                headers.append(text.strip())

        elif el_type == "footer":
            text = _recursive_text(el)
            if text.strip():
                footers.append(text.strip())

        elif el_type == "table":
            # Extract rows WITH geometry to preserve cell-level spans.
            rows_with_spans = _extract_table_rows_with_spans(el)
            rows = _extract_table_rows(el)
            tables.append({"rect": el.get("rect", []), "rows": rows})

            # Preserve the native table -> row -> cell structure.  Content
            # nested inside a cell (paragraph/heading/listItem) is emitted as
            # searchable content too; it must never disappear merely because
            # the immediate parent is a table_cell.
            table_id = str(id(el))
            for tr_idx, tr in enumerate(el.get("trs", [])):
                tr_rect = tr.get("rect", [])
                if not (isinstance(tr_rect, (list, tuple)) and len(tr_rect) >= 4):
                    tr_rect = _union_rects([
                        td.get("rect", [])
                        for td in tr.get("tds", [])
                        if isinstance(td.get("rect"), (list, tuple))
                    ])

                row_cells = []
                row_parts = []
                row_span_rects = []

                for td_idx, td in enumerate(tr.get("tds", [])):
                    cell_rect = td.get("rect", [])
                    cell_id = f"{table_id}_r{tr_idx}_c{td_idx}"
                    cell_content = _extract_table_cell_content(
                        td.get("contents", []),
                        table_id=table_id,
                        row_index=tr_idx,
                        row_start=td.get("rowStart", tr_idx),
                        col_start=td.get("colStart", td_idx),
                        cell_id=cell_id,
                        parent_cell_rect=cell_rect,
                    )
                    cell_text = cell_content["text"]
                    cell_spans = cell_content["spans"]

                    if cell_text:
                        paragraph_objects.append({
                            "text": cell_text,
                            "rect": list(cell_rect) if isinstance(cell_rect, (list, tuple)) else [],
                            "type": "table_cell",
                            "spans": cell_spans,
                            "table_role": "header" if td.get("type") == "th" else "body",
                            "row_start": td.get("rowStart", tr_idx),
                            "row_index": tr_idx,
                            "col_start": td.get("colStart", td_idx),
                            "row_span": td.get("rowSpan", 1),
                            "col_span": td.get("colSpan", 1),
                            "_table_key": table_id,
                            "table_id": table_id,
                            "cell_id": cell_id,
                        })
                        row_parts.append(cell_text)
                        row_cells.append(td)
                        row_span_rects.extend(cell_spans)

                    # Nested searchable content: list_item / heading / paragraph
                    # objects retain the same table relationship and their own
                    # native geometry.  They are NOT synthesized from cell text.
                    paragraph_objects.extend(cell_content["objects"])

                row_text = " | ".join(x for x in row_parts if x).strip()
                if row_text:
                    row_is_header = any(
                        isinstance(td, dict) and td.get("type") == "th"
                        for td in row_cells
                    )
                    paragraph_objects.append({
                        "text": row_text,
                        "rect": list(tr_rect) if isinstance(tr_rect, (list, tuple)) else [],
                        "type": "table_header" if row_is_header else "table_row",
                        "spans": row_span_rects,
                        "row_start": tr_idx,
                        "row_index": tr_idx,
                        "_table_key": table_id,
                        "table_id": table_id,
                        "table_role": "header" if row_is_header else "body",
                    })

        elif el_type == "list":
            # Lists are structural containers. Each native listItem is emitted
            # as its own retrieval/highlightable object. Parent list metadata is
            # copied onto the item; geometry remains the native item/body span
            # geometry and is resolved once in the extraction layer.
            items_with_spans = _extract_list_items_with_spans(el)
            for item_index, item_data in enumerate(items_with_spans):
                item_text = (item_data.get("text") or "").strip()
                if not item_text:
                    continue
                paragraph_objects.append({
                    "text": item_text,
                    "rect": list(item_data.get("rect") or el.get("rect", [])),
                    "type": item_data.get("content_type", "list_item"),
                    "spans": list(item_data.get("spans") or []),
                    "list_id": item_data.get("list_id", el.get("listId")),
                    "list_level": item_data.get("list_level", el.get("level")),
                    "list_label": item_data.get("list_label", item_data.get("label")),
                    "list_number_format": item_data.get("list_number_format", el.get("numberFormat")),
                    "list_item_index": item_index,
                })

        # graphic / image / toc / group with no text → skip

    # ── Deduplicate paragraph_objects by (text, type) ─────────────────────
    # Complex pages (cover pages, TOC pages) can produce the same text element
    # multiple times: once from a section-ref list promotion, once from the
    # actual heading element, and once from overlapping PDF visual regions.
    # Deduplication here prevents the same object from reaching two different
    # section chunks and being indexed twice in OpenSearch.
    seen_po: set[tuple] = set()
    deduped_po: list[dict] = []
    for po in paragraph_objects:
        key = (po.get("text", "").strip().lower(), po.get("type", ""))
        if key in seen_po:
            continue
        seen_po.add(key)
        deduped_po.append(po)
    paragraph_objects = deduped_po

    # Every native Apryse source span must carry its owning PDF page. This is
    # essential when a semantic object is later merged across a page boundary:
    # the object may remain one paragraph/sentence candidate, but the UI must
    # receive page-local geometry groups.
    page_number = props.get("pageNumber", 0)
    for _po in paragraph_objects:
        for _sp in (_po.get("spans") or []):
            if isinstance(_sp, dict):
                _sp.setdefault("page", page_number)

    # Build canonical page text directly from Apryse paragraph-object order.
    # Geometry does not depend on page-relative character offsets.
    parts: list[str] = [
        str(layout_obj.get("text", ""))
        for layout_obj in paragraph_objects
        if layout_obj.get("text")
    ]

    return {
        "page_number":  props.get("pageNumber", 0),
        "width":        props.get("width", 612.0),
        "height":       props.get("height", 792.0),
        "headings":     headings,
        "paragraphs":   paragraphs,
        "headers":          headers,
        "footers":          footers,
        "tables":           tables,
        "lists":            lists,
        "paragraph_objects": paragraph_objects,
        "raw_text":         "\n".join(p for p in parts if p),
        "doc_structure":    page,    # raw Apryse page — preserved verbatim
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _union_rects(rects: list) -> list:
    """Return the union bbox of existing Apryse rectangles only."""
    valid = [r for r in rects if isinstance(r, (list, tuple)) and len(r) >= 4]
    if not valid:
        return []
    return [
        min(float(r[0]) for r in valid),
        min(float(r[1]) for r in valid),
        max(float(r[2]) for r in valid),
        max(float(r[3]) for r in valid),
    ]


def _parse_section_items(items: list[str]) -> tuple[list[tuple[str, str]], bool]:
    """
    Analyse extracted list items to determine whether they are section
    references in a regulatory document.

    Two forms are recognised
    ------------------------
    *Pure*   : "5.2.2.8."    — section number only, no title
    *Hybrid* : "5.2.2.8.1. Hematology"  — number followed by title

    Only section numbers with **at least 2 dots** (3-level depth, e.g.
    ``5.2.2``) are treated as references.  Single-level numbers like
    ``1.`` are plain ordered-list items and are left as content.

    Returns
    -------
    (section_refs, all_are_refs)
        section_refs  — list of (number_str, title_str) for each item;
                        title_str is "" for pure items.
        all_are_refs  — True when EVERY item is a reference.
    """
    refs: list[tuple[str, str]] = []
    for item in items:
        stripped = item.strip()
        # Pure: only a section number (e.g. "5.2.2.8" or "5.2.2.8.")
        plain = stripped.rstrip(".")
        if _SECTION_NUM_RE.match(plain) and plain.count(".") >= 2:
            refs.append((plain, ""))
            continue
        # Hybrid: "N.N.N. Title text"
        m = _re.match(r'^(\d+(\.\d+)*)\.\s+(.+)$', stripped)
        if m and m.group(1).count(".") >= 2:
            refs.append((m.group(1), m.group(3).strip()))
            continue
        # Not a reference
        break   # short-circuit: one non-ref means all_are_refs = False
    else:
        # loop completed without break → ALL items are refs
        return refs, bool(refs)
    # Loop broke → at least one item is NOT a ref
    # Still return what we found so callers can inspect, but flag as mixed
    return refs, False


def _all_section_numbers(items: list[str]) -> bool:  # kept for backward compat
    _, all_refs = _parse_section_items(items)
    return all_refs


def _is_bold_heading(el: dict, text: str) -> bool:
    """
    True if this paragraph looks like an unlabelled section heading that
    Apryse classified as ``paragraph``.

    Heuristics (all must pass):
    - Text is short (≤ 100 chars) with at least 2 words
    - Does not end with a sentence-ending period
    - Is not itself a section number (e.g. "5.2.2.8.")
    - All text spans carry bold styling
    """
    stripped = text.strip()
    if len(stripped) > 100 or len(stripped.split()) < 2:
        return False
    if stripped.endswith("."):
        return False
    if _SECTION_NUM_RE.match(stripped.rstrip(".")):
        return False
    spans = [c for c in el.get("contents", []) if c.get("type") == "span"]
    if not spans:
        return False
    return all(
        c.get("style", {}).get("bold", False)
        or c.get("style", {}).get("weight", 400) > 400
        for c in spans
    )


def _all_section_numbers(items: list[str]) -> bool:
    """
    True when every item string looks like a section number
    (e.g. "5.2.2.8" or "5.2.2.8.").
    """
    return bool(items) and all(
        _SECTION_NUM_RE.match(i.rstrip(".").strip()) for i in items
    )


def _spans_text(contents: list) -> str:
    """
    Join text from span elements inside a paragraph or heading.
    Spans are the leaf text nodes in Apryse's model.
    """
    return " ".join(
        c["text"]
        for c in contents
        if c.get("type") == "span" and c.get("text", "").strip()
    )


def _extract_span_geometry(contents: list) -> list[dict]:
    """
    Extract Apryse leaf spans together with their native geometry.

    IMPORTANT:
        This function deliberately does NOT manufacture character offsets.

    The previous implementation assigned synthetic ``start``/``end`` offsets
    by assuming that ``_spans_text()`` was always exactly:

        span_1 + " " + span_2 + " " + ...

    That assumption is fragile because Apryse text nodes can contain their own
    whitespace, line breaks, hidden/empty nodes, or formatting boundaries.
    Those synthetic offsets were later used to decide which rectangles belonged
    to a sentence.

    Geometry is now represented by the source text + its native rect only.
    Downstream sentence construction matches the sentence text against this
    ordered source-span stream and selects the corresponding rects. No
    persisted character offsets are required for geometry.

    Returns:
        [
            {"text": str, "rect": list[float], "span_index": int},
            ...
        ]
    """
    spans_with_geo: list[dict] = []

    for span_index, c in enumerate(contents):
        if c.get("type") != "span":
            continue

        text = c.get("text", "")
        if not text or not text.strip():
            continue

        rect = c.get("rect", [])

        spans_with_geo.append({
            "text": text,
            "rect": rect,
            "span_index": span_index,
        })

    return spans_with_geo


def _extract_table_cell_content(
    contents: list[dict],
    *,
    table_id: str,
    row_index: int,
    row_start: int,
    col_start: int,
    cell_id: str,
    parent_cell_rect: list | tuple,
) -> dict:
    """Preserve semantic content nested inside a table cell.

    The table_cell keeps its aggregate text/geometry, while native listItems
    inside the cell are additionally emitted as independent searchable objects.
    Direct paragraphs/headings remain part of the cell aggregate (they are not
    duplicated as separate table children). A listItem whose body is a heading
    is promoted to ``heading`` using the same rule as top-level lists.
    """
    text_parts: list[str] = []
    spans: list[dict] = []
    objects: list[dict] = []

    def process_list(lst: dict, inherited_meta: dict | None = None) -> None:
        meta = {
            "list_id": lst.get("listId", (inherited_meta or {}).get("list_id")),
            "list_level": lst.get("level", (inherited_meta or {}).get("list_level")),
            "list_number_format": lst.get(
                "numberFormat", (inherited_meta or {}).get("list_number_format")
            ),
        }
        for item in lst.get("items", []):
            if item.get("type") == "list":
                process_list(item, meta)
                continue
            if item.get("type") != "listItem":
                continue

            label = item.get("label", {}).get("text", "").strip()
            item_parts: list[str] = []
            item_spans: list[dict] = []
            item_types: list[str] = []

            body_contents = item.get("body", {}).get("contents", [])
            if not body_contents:
                body_contents = item.get("contents", []) or []

            for child in body_contents:
                child_kind = child.get("type", "")
                if child_kind in ("paragraph", "heading"):
                    child_text = _spans_text(child.get("contents", []))
                    if child_text.strip():
                        item_parts.append(child_text.strip())
                        item_types.append(child_kind)
                    item_spans.extend(_extract_span_geometry(child.get("contents", [])))
                elif child_kind == "list":
                    process_list(child, meta)
                else:
                    child_text = _recursive_text(child)
                    if child_text.strip():
                        item_parts.append(child_text.strip())
                        item_types.append(child_kind or "other")

            content = " ".join(item_parts).strip()
            full_text = (
                f"{label} {content}".strip()
                if label and len(label) > 2
                else (content or label)
            )
            if not full_text:
                continue

            item_rect = item.get("rect") or item.get("body", {}).get("rect") or parent_cell_rect or []
            content_type = (
                "heading"
                if item_types and all(t == "heading" for t in item_types)
                else "list_item"
            )
            objects.append({
                "text": full_text,
                "rect": list(item_rect) if isinstance(item_rect, (list, tuple)) else [],
                "type": content_type,
                "spans": item_spans,
                "table_id": table_id,
                "cell_id": cell_id,
                "row_index": row_index,
                "row_start": row_start,
                "col_start": col_start,
                "table_role": "body",
                "list_id": meta.get("list_id"),
                "list_level": meta.get("list_level"),
                "list_label": label or None,
                "list_number_format": meta.get("list_number_format"),
            })
            text_parts.append(full_text)
            spans.extend(item_spans)

    for element in contents or []:
        kind = element.get("type", "")
        if kind in ("paragraph", "heading"):
            text = _spans_text(element.get("contents", []))
            if text.strip():
                text_parts.append(text.strip())
            spans.extend(_extract_span_geometry(element.get("contents", [])))
        elif kind == "list":
            process_list(element)
        else:
            text = _recursive_text(element)
            if text.strip():
                text_parts.append(text.strip())

    return {
        "text": " ".join(p for p in text_parts if p).strip(),
        "spans": spans,
        "objects": objects,
    }

def _extract_table_rows_with_spans(table: dict) -> list[dict]:
    """
    Extract table rows with cell-level span geometry.

    Returns list of {
        "text": str,       # row text joined with " | "
        "spans": list[dict] # [{"text", "rect", "start", "end"}, ...]
    }

    Character offsets account for " | " separator between cells.
    """
    # Get the basic grid first (as per _extract_table_rows)
    max_row = 0
    max_col = 0
    for tr in table.get("trs", []):
        for td in tr.get("tds", []):
            r = td.get("rowStart", 0) + td.get("rowSpan", 1)
            c = td.get("colStart", 0) + td.get("colSpan", 1)
            max_row = max(max_row, r)
            max_col = max(max_col, c)

    if max_row == 0:
        return []

    # For each row, collect text AND spans
    rows_with_spans = []
    
    for row_idx in range(max_row):
        row_cells = [None] * max_col  # Will be {text, spans} dicts
        
        for tr in table.get("trs", []):
            for td in tr.get("tds", []):
                row_start = td.get("rowStart", 0)
                col_start = td.get("colStart", 0)
                
                if row_start == row_idx and col_start < max_col:
                    # Extract text and spans from this cell
                    cell_spans = []
                    cell_parts = []
                    
                    for c in td.get("contents", []):
                        if c.get("type") in ("paragraph", "heading"):
                            # Extract both text and spans from this paragraph
                            para_text = _spans_text(c.get("contents", []))
                            if para_text:
                                cell_parts.append(para_text)
                                
                            # Extract span geometry, adjusted for current cell position
                            para_spans = _extract_span_geometry(c.get("contents", []))
                            cell_spans.extend(para_spans)
                    
                    cell_text = " ".join(p for p in cell_parts if p).strip()
                    row_cells[col_start] = {
                        "text": cell_text,
                        "spans": cell_spans if cell_spans else []
                    }
        
        # Join cells with " | ". Geometry is kept as native source-span
        # geometry; no synthetic row-level character offsets are created.
        all_row_spans = []

        for cell_dict in row_cells:
            if cell_dict is None:
                cell_dict = {"text": "", "spans": []}

            all_row_spans.extend(
                cell_dict.get("spans", [])
            )

        row_text = " | ".join(c["text"] if c else "" for c in row_cells)
        if row_text.strip():
            rows_with_spans.append({
                "text": row_text,
                "spans": all_row_spans,
            })
    
    return rows_with_spans


def _extract_list_items_with_spans(lst: dict, _parent_meta: dict | None = None) -> list[dict]:
    """
    Extract list items with body-level span geometry.

    Returns list of {
        "text": str,       # item text (may include label)
        "spans": list[dict] # [{"text", "rect", "span_index"}, ...]
    }

    Geometry is preserved directly from the Apryse source spans. No synthetic
    character offsets are generated here.
    """
    items_with_spans = []
    parent_meta = dict(_parent_meta or {})
    current_meta = {
        "list_id": lst.get("listId", parent_meta.get("list_id")),
        "list_level": lst.get("level", parent_meta.get("list_level")),
        "list_number_format": lst.get("numberFormat", parent_meta.get("list_number_format")),
    }

    for item in lst.get("items", []):
        item_type = item.get("type", "")

        # Nested sub-list — recurse
        if item_type == "list":
            items_with_spans.extend(_extract_list_items_with_spans(item, current_meta))
            continue

        # Standard listItem
        label = item.get("label", {}).get("text", "").strip()
        
        # Extract text, native semantic type, and spans from body.
        content_parts = []
        content_spans = []
        content_types = []

        for c in item.get("body", {}).get("contents", []):
            c_type = c.get("type", "")
            if c_type in ("paragraph", "heading"):
                t = _spans_text(c.get("contents", []))
                if t.strip():
                    content_parts.append(t.strip())
                    content_types.append(c_type)
                content_spans.extend(_extract_span_geometry(c.get("contents", [])))
            else:
                t = _recursive_text(c)
                if t.strip():
                    content_parts.append(t.strip())
                    content_types.append(c_type or "other")

        # Fallback to item.contents if body is empty.
        if not content_parts:
            for c in item.get("contents", []):
                c_type = c.get("type", "")
                if c_type in ("paragraph", "heading"):
                    t = _spans_text(c.get("contents", []))
                    if t.strip():
                        content_parts.append(t.strip())
                        content_types.append(c_type)
                    content_spans.extend(_extract_span_geometry(c.get("contents", [])))
                else:
                    t = _recursive_text(c)
                    if t.strip():
                        content_parts.append(t.strip())
                        content_types.append(c_type or "other")

        content = " ".join(content_parts).lstrip("\t").strip()
        
        # Build final text with label if meaningful
        meaningful_label = label and len(label) > 2
        if meaningful_label:
            full_text = f"{label} {content}".strip()
            # The label normally has no independent Apryse text span. Keep
            # the body geometry untouched; sentence/display matching can fall
            # back to the parent bbox when the label prevents an exact source
            # text match.
            adjusted_spans = list(content_spans)
        else:
            full_text = content if content else label
            adjusted_spans = content_spans
        
        if full_text and len(full_text.strip()) > 1:
            items_with_spans.append({
                "text": full_text,
                "spans": adjusted_spans if adjusted_spans else [],
                "rect": item.get("rect") or item.get("body", {}).get("rect") or lst.get("rect", []),
                "label": label,
                "list_id": current_meta.get("list_id"),
                "list_level": current_meta.get("list_level"),
                "list_label": label or None,
                "list_number_format": current_meta.get("list_number_format"),
                # Preserve the native semantic type carried by the list item's
                # body. A listItem containing only a heading is a heading
                # semantically; ordinary list content remains list_item.
                "content_type": (
                    "heading"
                    if content_types and all(t == "heading" for t in content_types)
                    else "list_item"
                ),
            })
    
    return items_with_spans


def _recursive_text(el: dict) -> str:
    """
    Recursively extract all text from any element by walking its contents.
    Used for header / footer / group / textbox which can nest arbitrarily.
    """
    el_type = el.get("type", "")

    if el_type == "span":
        return el.get("text", "")

    if el_type in ("paragraph", "heading"):
        return _spans_text(el.get("contents", []))

    if el_type in ("header", "footer", "group", "textbox"):
        texts = [_recursive_text(c) for c in el.get("contents", [])]
        return " ".join(t for t in texts if t.strip())

    if el_type == "table":
        rows = _extract_table_rows(el)
        return _table_to_text(rows)

    if el_type == "list":
        return "\n".join(_extract_list_items(el))

    return ""


def _extract_table_rows(table: dict) -> list[list[str]]:
    """
    Walk trs → tds → contents (paragraphs) → spans and return a 2-D list of
    cell text strings.  Preserves the grid position via rowStart / colStart so
    spanned cells are placed correctly.
    """
    # Figure out grid dimensions first
    max_row = 0
    max_col = 0
    for tr in table.get("trs", []):
        for td in tr.get("tds", []):
            r = td.get("rowStart", 0) + td.get("rowSpan", 1)
            c = td.get("colStart", 0) + td.get("colSpan", 1)
            max_row = max(max_row, r)
            max_col = max(max_col, c)

    if max_row == 0:
        return []

    grid: list[list[str]] = [[""] * max_col for _ in range(max_row)]

    for tr in table.get("trs", []):
        for td in tr.get("tds", []):
            row_start = td.get("rowStart", 0)
            col_start = td.get("colStart", 0)
            cell_parts: list[str] = []
            for c in td.get("contents", []):
                if c.get("type") in ("paragraph", "heading"):
                    cell_parts.append(_spans_text(c.get("contents", [])))
            cell_text = " ".join(p for p in cell_parts if p).strip()
            if row_start < max_row and col_start < max_col:
                grid[row_start][col_start] = cell_text

    return grid


def _extract_list_items(lst: dict) -> list[str]:
    """
    Extract text from each listItem.

    Apryse stores bullet text in ``item["body"]["contents"]`` (NOT in
    ``item["contents"]``, which is always empty in the observed output).
    Nested sub-lists within ``items`` are handled recursively.
    Leading tab stops (inserted by Apryse after the bullet glyph) are
    stripped from the assembled text.
    Labels longer than 2 characters (e.g. "1.1", "a)") are prepended;
    single-character bullet glyphs (•, –, tab) are discarded.
    """
    texts: list[str] = []
    for item in lst.get("items", []):
        item_type = item.get("type", "")

        # Nested sub-list — recurse and collect its items
        if item_type == "list":
            texts.extend(_extract_list_items(item))
            continue

        # Standard listItem
        label = item.get("label", {}).get("text", "").strip()

        content_parts: list[str] = []
        # Primary path: item["body"]["contents"]
        for c in item.get("body", {}).get("contents", []):
            t = _recursive_text(c)
            if t.strip():
                content_parts.append(t.strip())
        # Fallback: item["contents"] (alternate Apryse versions / PDF structures)
        if not content_parts:
            for c in item.get("contents", []):
                t = _recursive_text(c)
                if t.strip():
                    content_parts.append(t.strip())

        # Strip the leading \t that Apryse inserts after bullet glyphs
        content = " ".join(content_parts).lstrip("\t").strip()

        # Only prepend the label when it is a meaningful marker (>2 chars)
        meaningful_label = label and len(label) > 2
        full_text = (f"{label} {content}" if meaningful_label else content).strip()
        if not full_text:
            full_text = content if content else label
        if full_text and len(full_text.strip()) > 1:
            texts.append(full_text)
    return texts


def _table_to_text(rows: list[list[str]]) -> str:
    return "\n".join(" | ".join(cell for cell in row) for row in rows)
