"""
Apryse DocStructure Parser
===========================
Pure-Python helpers that turn an Apryse ``e_DocStructure`` JSON result into
the structured page dicts used by the rest of the pipeline.

This module has zero external dependencies — it only uses the stdlib.
It is used by:
  - lambdas/document/extraction/lambda_function.py  (production)
  - tests/local_pipeline_test.py                     (mock extraction)

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
                    })
                else:
                    paragraphs.append(text.strip())
                    paragraph_objects.append({
                        "text": text.strip(),
                        "rect": el.get("rect", []),
                        "type": "paragraph",
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
            rows = _extract_table_rows(el)
            tables.append({"rect": el.get("rect", []), "rows": rows})
            # Treat each non-empty row as a layout object for sentence extraction
            for row in rows:
                row_text = " | ".join(c for c in row if c).strip()
                if row_text:
                    paragraph_objects.append({
                        "text": row_text,
                        "rect": el.get("rect", []),
                        "type": "table_row",
                    })

        elif el_type == "list":
            items = _extract_list_items(el)
            section_refs, all_are_refs = _parse_section_items(items)
            if all_are_refs:
                # Every item is a section reference.
                # De-duplicate: only one section_marker per Y-position (several
                # nested list elements share the same rect for the same group).
                y_pos = el.get("rect", [0, 0])[1]
                if not any(
                    o.get("type") == "section_marker"
                    and abs(o.get("rect", [0, 0])[1] - y_pos) < 2
                    for o in paragraph_objects
                ):
                    best = max(section_refs, key=lambda r: r[0].count("."))
                    paragraph_objects.append({
                        "text":            best[0],
                        "rect":            el.get("rect", []),
                        "type":            "section_marker",
                        "section_numbers": [r[0] for r in section_refs],
                        "section_title":   best[1] or None,
                    })
                    # Promote each hybrid ref (number + title) to a proper heading
                    # instead of a list object, eliminating the duplicate pair.
                    for num, title in section_refs:
                        if title:
                            depth = num.count(".") + 1
                            headings.append({"level": depth, "text": title,
                                             "rect": el.get("rect", [])})
                            paragraph_objects.append({
                                "text":  title,
                                "rect":  el.get("rect", []),
                                "type":  "heading",
                                "level": depth,
                            })
            elif items:
                lists.append({"items": items})
                # Emit ONE object for the whole list; individual items become
                # display_spans in the extraction lambda so the whole list is
                # embedded as a single semantic unit instead of N fragments.
                combined = "\n".join(item for item in items if item.strip())
                if combined:
                    paragraph_objects.append({
                        "text": combined,
                        "rect": el.get("rect", []),
                        "type": "list",
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

    # Build raw_text in reading order: headings → paragraphs → tables → lists
    parts: list[str] = []
    for h in headings:
        parts.append(h["text"])
    parts.extend(paragraphs)
    for t in tables:
        parts.append(_table_to_text(t["rows"]))
    for lst in lists:
        parts.extend(lst["items"])

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
        "paragraph_objects": paragraph_objects,   # NEW: layout-aware sentence units
        "raw_text":         "\n".join(p for p in parts if p),
        "doc_structure":    page,    # raw Apryse page — preserved verbatim
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

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
