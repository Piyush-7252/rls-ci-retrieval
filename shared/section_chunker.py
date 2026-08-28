"""
section_chunker.py
──────────────────
Hierarchical, section-aware chunker for clinical protocol documents.

Instead of splitting by page count, this module walks the paragraph_objects
stream produced by the Apryse parser in document order, detects heading
boundaries from Apryse's own heading-level tags (not regexp page scanning),
and emits one SectionChunk per logical subsection.

Key features
────────────
• Relies on Apryse heading levels (1/2/3…) — no hard-coded page patterns.
• Canonical section category assigned from heading text rules.
• Smart merging: tiny chunks (< min_words) folded into the preceding chunk.
• Smart splitting: oversized chunks (> max_words) split at paragraph boundaries.
• Every chunk carries a full heading breadcrumb for reranker context enrichment.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, replace as _dc_replace
from typing import Callable, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Heading filters
# ─────────────────────────────────────────────────────────────────────────────

# Navigation / structural headings that appear in front-matter (TOC, list of
# tables, list of figures).  These reset the heading stack completely so they
# never become ancestor headings for clinical sections that follow.
_STRUCTURAL_HEADING_RE = re.compile(
    r"^("                                    # --- TOC / navigation headings ---
    r"LIST\s+OF\s+(IN.TEXT\s+)?(TABLES?|FIGURES?|ABBREVIATIONS?|ACRONYMS?)"
    r"|TABLE\s+OF\s+CONTENTS?"
    r"|LIST\s+OF\s+TABLES"
    r"|LIST\s+OF\s+FIGURES"
    r"|INDEX"
    r")",
    re.I,
)

# Caption-style headings: "Figure 1:", "Fig. 2.", "FIGURE 4 " etc.
# Apryse sometimes tags figure captions as heading elements because they're bold.
# Treat them as regular paragraphs — they should NOT become section ancestors.
# NOTE: "Table N:" is intentionally excluded — table headings like
#       "Table 2: Schedule of Activities" are real semantic headings.
_CAPTION_HEADING_RE = re.compile(
    r"^(Figure|Fig\.|FIGURE)\s+\d+[:.\.\s]",
    re.I,
)


def _is_structural_heading(text: str) -> bool:
    """True for TOC / navigation headings that must NOT become section ancestors."""
    return bool(_STRUCTURAL_HEADING_RE.match(text.strip()))


def _is_caption_heading(text: str) -> bool:
    """True for figure/table caption lines Apryse mislabels as headings."""
    return bool(_CAPTION_HEADING_RE.match(text.strip()))


# ─────────────────────────────────────────────────────────────────────────────
# Canonical section categories
# ─────────────────────────────────────────────────────────────────────────────
# Ordered list of (category, compiled_pattern).  First match wins.
# Checked from the most-specific heading in the breadcrumb down to the root.

_CATEGORY_RULES: list[tuple[str, re.Pattern]] = [
    # Administrative front-matter: cover page, sponsor, IND, compliance
    ("ADMINISTRATIVE",  re.compile(
        r"^(confidential(ity)?|sponsor|ind|status|approval|signature"
        r"|principal\s+invest|coordinating\s+invest"
        r"|statement\s+of\s+compliance|protocol\s+status|amendment\s+history"
        r"|clinical\s+protocol|cover\s+page)$",
        re.I,
    )),
    # Very specific first
    ("SYNOPSIS",        re.compile(r"synops|executive\s+summ|study\s+overview|study\s+abstract", re.I)),
    ("OBJECTIVES",      re.compile(r"\bobjective", re.I)),
    ("ENDPOINTS",       re.compile(r"\bendpoint", re.I)),
    ("ELIGIBILITY",     re.compile(r"eligib|inclusion\s+crit|exclusion\s+crit|enroll|criteria\s+for\s+(incl|excl)", re.I)),
    ("DESIGN",          re.compile(r"study\s+design|overall\s+design|trial\s+design|study\s+schema|study\s+overview|schema", re.I)),
    ("BACKGROUND",      re.compile(r"\bbackground\b|rationale|introduction|disease\s+overview|unmet\s+need", re.I)),
    ("SAFETY",          re.compile(r"\bsafety\b|adverse\s+event|tolerab|toxicity|stopping\s+rule|dose\s+limit", re.I)),
    ("PK",              re.compile(r"pharmacokinetic|pharmacodynamic|\bPK\b|\bPD\b|pk\/pd", re.I)),
    ("STATISTICS",      re.compile(r"statistic|sample\s+size|power\s+calc|analysis\s+plan|estimand|hypothesis", re.I)),
    ("TREATMENT",       re.compile(r"treatment\s+plan|treatment\s+arm|dosing\s+regimen|dose\s+modif|dose\s+esc|administration|study\s+treatment|description\s+of\s+study", re.I)),
    ("PROCEDURES",      re.compile(r"procedure|assessment|visit\s+schedule|schedule\s+of\s+activit|study\s+evaluati|pro\s+eval|patient.reported", re.I)),
    ("POPULATION",      re.compile(r"study\s+population|patient\s+population|subject\s+population|demograph|number\s+of\s+participant|number\s+of\s+subject", re.I)),
    ("BIOMARKER",       re.compile(r"biomarker|translational|genomic|genetic|correlative|mrd", re.I)),
    ("EFFICACY",        re.compile(r"\befficacy\b|response\s+criteria|outcome\s+measure|efficacy\s+eval", re.I)),
    ("BENEFIT_RISK",    re.compile(r"benefit.risk|risk.benefit", re.I)),
    ("APPENDIX",        re.compile(r"appendix|abbreviation|glossary|\breference\b|amendment\s+summ|protocol\s+amendment", re.I)),
]


def _obj_group(kind: str, obj: dict | None = None) -> str:
    """Map an object to a broad content group for section boundary detection.

    A list_item nested inside a table cell remains table content for chunking;
    it must travel with its parent table rather than create a paragraph/list
    boundary of its own.
    """
    if kind in ("table", "table_header", "table_row", "table_cell"):
        return "table"
    if kind == "list_item" and isinstance(obj, dict) and obj.get("table_id"):
        return "table"
    if kind == "list":
        return "list"
    return "paragraph"


# Regex that identifies column-header rows inside amendment tables (not topic rows).
_TABLE_HEADER_ROW_RE = re.compile(
    r"description\s+of\s+change|section\s+(number|name)|brief\s+rational"
    r"|document\s+history|date\s+of\s+amendment|original\s+protocol",
    re.I,
)


def _table_row_topic(text: str) -> Optional[str]:
    """
    Return the canonical category for the *section* referenced in a table row,
    or None if the row is a column header, too short, or unrecognisable.

    Amendment summary tables use a pipe-delimited format:
        ``Section N.N  SectionName | Description of Change | Rationale``
    The first column (before the first ``|``) carries the section reference.
    Category rules are run against that first column only.
    """
    if not text or "|" not in text:
        return None
    first_col = text.split("|")[0].strip()
    if not first_col or len(first_col) < 4:
        return None
    if _TABLE_HEADER_ROW_RE.search(first_col):
        return None
    for cat, pat in _CATEGORY_RULES:
        if cat == "ADMINISTRATIVE":
            continue
        if pat.search(first_col):
            return cat
    return None


def _categorize_with_conf(heading_path: list[str]) -> tuple[str, float]:
    """
    Like ``categorize_heading`` but also returns a confidence score (0.0–1.0).

    Confidence reflects both how specific the matching heading is and how deep
    in the breadcrumb the match was found:
      - Match on the immediate heading  → base 0.90
      - Match on parent heading          → base 0.75
      - Match on grandparent+            → base 0.60
      - No match (OTHER)                 → 0.30
    """
    for depth, text in enumerate(reversed(heading_path)):
        for cat, pat in _CATEGORY_RULES:
            if pat.search(text):
                conf = round(max(0.50, 0.90 - depth * 0.15), 2)
                return cat, conf
    return "OTHER", 0.30


def categorize_heading(heading_path: list[str]) -> str:
    """
    Map a heading breadcrumb path to a canonical section category.

    Checks from the most-specific (last) heading toward the root.
    Returns the first matching category, or "OTHER" if none match.
    """
    for text in reversed(heading_path):
        for cat, pat in _CATEGORY_RULES:
            if pat.search(text):
                return cat
    return "OTHER"


# ─────────────────────────────────────────────────────────────────────────────
# SectionChunk dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SectionChunk:
    """
    One semantic unit — a heading and its body content.

    ``objects`` is a list of ``(page_number, paragraph_object_dict)`` in
    document order.  Heading objects from the section title are NOT included
    in ``objects`` — the heading context is carried via ``heading_path``.
    """
    heading_path:      list[str]   # ["2. Study Objectives", "2.1 Primary Objective"]
    heading_text:      str         # last element of heading_path (immediate heading)
    heading_level:     int         # Apryse level (1 = top, 2 = sub, …)
    section:           str         # one level above (parent context)
    subsection:        str         # immediate heading text (same as heading_text)
    section_category:  str         # canonical category string
    page_start:        int
    page_end:          int
    objects:            list        # [(pg_num, obj_dict), …]
    word_count:         int            = 0
    section_confidence: float          = 0.80  # how strongly the category matched (0–1)
    document_position:  float          = 0.0   # page_start / total_pages; 0 if unknown
    chunk_idx:          int            = -1    # ordinal position in final chunk list
    parent_chunk_idx:   Optional[int]  = None  # nearest ancestor by heading_path prefix
    prev_chunk_idx:     Optional[int]  = None  # previous chunk in document order
    next_chunk_idx:     Optional[int]  = None  # next chunk in document order

    @property
    def text(self) -> str:
        """
        Plain-text for this chunk: heading breadcrumb followed by body.
        This is stored as raw_text and used for embedding.
        """
        crumb = " > ".join(self.heading_path)
        body  = "\n\n".join(
            o["text"] for _, o in self.objects
            if o.get("type") not in ("section_marker",)
            and (o.get("text") or "").strip()
        )
        if crumb and body:
            return f"{crumb}\n\n{body}"
        return crumb or body

    @property
    def virtual_pages(self) -> list[dict]:
        """
        Group objects by page number into virtual page dicts that are
        compatible with sentence_builder._build_objects().
        """
        page_objs: dict[int, list] = defaultdict(list)
        for pg_num, obj in self.objects:
            page_objs[pg_num].append(obj)
        return [
            {"page_number": pg, "height": 792.0, "paragraph_objects": objs}
            for pg, objs in sorted(page_objs.items())
        ]

    @property
    def semantic_path(self) -> list[str]:
        """
        Inferred semantic navigation path, e.g.
        ['SAFETY', 'Daratumumab SC Risks'] or ['TREATMENT', 'Dose Modifications'].

        Walks the heading_path noting category transitions, ensures
        section_category is always represented (handles table-topic chunks
        where the category comes from row content not heading structure),
        and appends the cleaned immediate heading text.
        """
        parts: list[str] = []
        seen:  set[str]  = set()
        for i in range(len(self.heading_path)):
            cat = categorize_heading(self.heading_path[:i + 1])
            if cat not in seen and cat not in ("OTHER", "ADMINISTRATIVE"):
                parts.append(cat)
                seen.add(cat)
        # Ensure the chunk's actual category is present (may differ from
        # the heading path when derived from a table row topic).
        if self.section_category not in seen and self.section_category not in ("OTHER", "ADMINISTRATIVE"):
            parts.insert(0, self.section_category)
            seen.add(self.section_category)
        # Append the immediate heading, stripping leading section numbers.
        clean = re.sub(r"^\d+(\.(\d+))*\s*[.\-\s]+\s*", "", self.heading_text or "").strip()
        label = clean or self.heading_text or ""
        if label and label not in parts:
            parts.append(label)
        return parts

    @property
    def heading_embedding_text(self) -> str:
        """Short heading-only text for a focused second embedding."""
        sem = " > ".join(self.semantic_path)
        return sem if sem else (" > ".join(self.heading_path) if self.heading_path else "")


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def build_section_chunks(
    pages:                list[dict],
    min_words:            int = 40,
    max_words:            int = 400,
    similarity_fn:        Optional[Callable[[str, str], float]] = None,
    similarity_threshold: float = 0.60,
    total_pages:          int   = 0,
) -> list[SectionChunk]:
    """
    Walk parsed pages (output of ``apryse_parser.parse_pages()``) and return
    a list of SectionChunk objects, one per logical subsection.

    Algorithm
    ─────────
    Phase 1 — stream walk
      For every paragraph_object across all pages (in document order):
      • heading     → flush + update heading stack.
      • obj-type Δ  → flush when object group changes (paragraph↔table↔list).
      • content     → append; force-flush when word count exceeds max_words.
    After the last page, flush any remaining objects.

    Phase 1b — similarity split (optional)
      If similarity_fn is provided, paragraph chunks with ≥ 3 paragraphs are
      split at their lowest-similarity adjacent boundary when that score falls
      below similarity_threshold.

    Phase 2 — merge tiny chunks
      Consecutive chunks with the same section_category whose combined word
      count is still below min_words are merged into the preceding chunk.

    Parameters
    ──────────
    pages                : output from apryse_parser.parse_pages()
    min_words            : chunks smaller than this are merged into preceding
    max_words            : chunks larger than this are force-split
    similarity_fn        : optional fn(text_a, text_b) → float 0-1; enables
                           similarity-based paragraph boundary detection
    similarity_threshold : split when similarity drops below this value
    total_pages          : total pages in source document; used to compute
                           document_position (0.0 = start, 1.0 = end).
                           Pass 0 to leave document_position as 0.0.
    """
    if not pages:
        return []

    heading_stack: list[tuple[int, str]] = []   # [(level, text), …]
    cur_objects:   list                  = []   # [(pg_num, obj_dict), …]
    cur_pg_start:  int                   = pages[0].get("page_number", 1)
    cur_pg_end:    int                   = cur_pg_start
    cur_group:     str                   = "paragraph"  # current object-type group
    raw_chunks:    list[SectionChunk]    = []

    def _flush(next_pg: Optional[int] = None) -> None:
        nonlocal cur_objects, cur_pg_start, cur_pg_end
        if not cur_objects:
            return
        path      = [t for _, t in heading_stack]
        top       = heading_stack[-1]  if heading_stack else (0, "")
        sec       = heading_stack[-2][1] if len(heading_stack) >= 2 else top[1]
        sub       = top[1]
        cat, conf = _categorize_with_conf(path)
        wc        = sum(len(o["text"].split()) for _, o in cur_objects if o.get("text"))
        pos       = round(cur_pg_start / total_pages, 3) if total_pages > 0 else 0.0
        raw_chunks.append(SectionChunk(
            heading_path       = path[:],
            heading_text       = top[1],
            heading_level      = top[0],
            section            = sec,
            subsection         = sub,
            section_category   = cat,
            page_start         = cur_pg_start,
            page_end           = cur_pg_end,
            objects            = cur_objects[:],
            word_count         = wc,
            section_confidence = conf,
            document_position  = pos,
        ))
        cur_objects = []
        if next_pg is not None:
            cur_pg_start = next_pg
            cur_pg_end   = next_pg

    for page in pages:
        pg_num = page.get("page_number", 0)
        for obj in page.get("paragraph_objects", []):
            kind  = obj.get("type", "paragraph")
            level = obj.get("level")
            text  = (obj.get("text") or "").strip()
            if not text:
                continue

            if kind == "heading" and level is not None:
                # Caption headings ("Figure 1:", "Table 3.") are Apryse artefacts
                # — treat them as paragraphs so they don't pollute the hierarchy.
                if _is_caption_heading(text):
                    cur_objects.append((pg_num, obj))
                    cur_pg_end = pg_num
                    continue

                # Structural / navigation headings (TOC, list of tables) reset
                # the stack entirely but never become section ancestors.
                if _is_structural_heading(text):
                    _flush(next_pg=pg_num)
                    heading_stack.clear()
                    cur_pg_start = pg_num
                    cur_pg_end   = pg_num
                    continue

                # Flush accumulated content, then update heading stack.
                # IMPORTANT: keep the native heading object in the new section.
                # The heading is both section metadata AND a searchable semantic
                # object with its own canonical geometry.
                _flush(next_pg=pg_num)
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, text))
                cur_pg_start = pg_num
                cur_pg_end   = pg_num
                cur_group = "paragraph"
                cur_objects.append((pg_num, obj))

            elif kind != "section_marker":
                # Split when object type group changes (paragraph ↔ table ↔ list)
                new_group = _obj_group(kind, obj)
                # A section-start heading belongs to the section it starts.
                # Do not strand it in a heading-only chunk when the first body
                # object is a table/list; keep it with that section content.
                heading_only = (
                    cur_objects
                    and all(o.get("type") == "heading" for _, o in cur_objects)
                )
                if cur_objects and new_group != cur_group and not heading_only:
                    _flush(next_pg=pg_num)
                cur_group = new_group

                cur_objects.append((pg_num, obj))
                cur_pg_end = pg_num

                # Force-split when chunk is growing too large
                if sum(len(o["text"].split()) for _, o in cur_objects if o.get("text")) > max_words:
                    _flush(next_pg=pg_num)

    _flush()  # flush anything remaining after the last page

    if not raw_chunks:
        return []

    # ── Phase 1c: table topic splitting ─────────────────────────────────────────
    # Amendment summary tables interleave rows from multiple protocol sections
    # (Safety, Objectives, Eligibility …).  Split at rows whose leading text
    # announces a new section topic so each topic becomes its own retrieval unit.
    topic_out: list[SectionChunk] = []
    for chunk in raw_chunks:
        # Only pure table-group chunks with enough rows benefit
        non_empty = [(pg, o) for pg, o in chunk.objects if (o.get("text") or "").strip()]
        is_table  = all(_obj_group(o.get("type", ""), o) == "table" for _, o in non_empty)
        if not is_table or len(non_empty) < 3:
            topic_out.append(chunk)
            continue

        # Walk rows and record where the section topic changes
        split_before: list[int] = []
        prev_topic: Optional[str] = None
        for i, (_, o) in enumerate(chunk.objects):
            text  = (o.get("text") or "").strip()
            topic = _table_row_topic(text)
            if topic is not None and topic != prev_topic:
                if prev_topic is not None:
                    split_before.append(i)
                prev_topic = topic

        if not split_before:
            topic_out.append(chunk)
            continue

        # Build sub-chunks at each topic boundary
        boundaries = [0] + split_before + [len(chunk.objects)]
        for j in range(len(boundaries) - 1):
            objs_sub = chunk.objects[boundaries[j]: boundaries[j + 1]]
            if not objs_sub:
                continue
            # Assign a better category from the first topic-label row
            sub_cat, sub_conf = chunk.section_category, chunk.section_confidence
            for _, o in objs_sub[:2]:
                rc = _table_row_topic((o.get("text") or "").strip())
                if rc:
                    sub_cat  = rc
                    sub_conf = 0.72
                    break
            wc = sum(len(o["text"].split()) for _, o in objs_sub if o.get("text"))
            topic_out.append(_dc_replace(
                chunk,
                objects            = objs_sub,
                page_start         = objs_sub[0][0],
                page_end           = objs_sub[-1][0],
                section_category   = sub_cat,
                section_confidence = sub_conf,
                word_count         = wc,
            ))
    raw_chunks = topic_out

    # ── Phase 1b: similarity-based paragraph splitting ────────────────────────
    # Only applied when caller provides a similarity_fn; skips table/list chunks
    # and chunks too small to benefit from splitting.
    if similarity_fn is not None:
        sim_out: list[SectionChunk] = []
        for chunk in raw_chunks:
            # Only paragraph-group chunks with enough content
            para_idxs = [
                i for i, (_, o) in enumerate(chunk.objects)
                if _obj_group(o.get("type", "paragraph"), o) == "paragraph"
                and (o.get("text") or "").strip()
            ]
            if len(para_idxs) < 3 or chunk.word_count < max_words // 2:
                sim_out.append(chunk)
                continue
            texts  = [(chunk.objects[i][1].get("text") or "") for i in para_idxs]
            scores = [similarity_fn(texts[j], texts[j + 1]) for j in range(len(texts) - 1)]
            weak   = min(range(len(scores)), key=lambda j: scores[j])
            if scores[weak] < similarity_threshold:
                split_at = para_idxs[weak + 1]   # object index to split before
                objs_a, objs_b = chunk.objects[:split_at], chunk.objects[split_at:]
                if objs_a and objs_b:
                    wc_a = sum(len(o["text"].split()) for _, o in objs_a if o.get("text"))
                    wc_b = sum(len(o["text"].split()) for _, o in objs_b if o.get("text"))
                    sim_out.append(_dc_replace(
                        chunk, objects=objs_a, page_end=objs_a[-1][0], word_count=wc_a,
                    ))
                    sim_out.append(_dc_replace(
                        chunk, objects=objs_b, page_start=objs_b[0][0], word_count=wc_b,
                    ))
                    continue
            sim_out.append(chunk)
        raw_chunks = sim_out

    # ── Phase 2: merge tiny consecutive chunks ────────────────────────────────
    # A tiny chunk (< min_words) gets absorbed into the preceding chunk when
    # they share the same section category.  This avoids single-sentence stubs.
    merged: list[SectionChunk] = [raw_chunks[0]]
    for chunk in raw_chunks[1:]:
        prev = merged[-1]
        if chunk.word_count < min_words and prev.section_category == chunk.section_category:
            merged[-1] = _dc_replace(
                prev,
                page_end   = chunk.page_end,
                objects    = prev.objects + chunk.objects,
                word_count = prev.word_count + chunk.word_count,
            )
        else:
            merged.append(chunk)

    # ── Phase 3: assign chunk indices and adjacency / parent–child links ────────
    n = len(merged)
    for i in range(n):
        merged[i] = _dc_replace(
            merged[i],
            chunk_idx      = i,
            prev_chunk_idx = i - 1 if i > 0     else None,
            next_chunk_idx = i + 1 if i < n - 1 else None,
        )
    # Assign parent: most recent earlier chunk whose heading_path is a strict
    # prefix of the current chunk’s heading_path.
    for i, chunk in enumerate(merged):
        if not chunk.heading_path:
            continue
        for j in range(i - 1, -1, -1):
            c = merged[j]
            plen = len(c.heading_path)
            if (c.heading_path
                    and plen < len(chunk.heading_path)
                    and chunk.heading_path[:plen] == c.heading_path):
                merged[i] = _dc_replace(merged[i], parent_chunk_idx=j)
                break

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-printer (used by --show-sections CLI mode)
# ─────────────────────────────────────────────────────────────────────────────

def print_section_tree(chunks: list[SectionChunk], verbose: bool = False) -> None:
    """
    Print a human-readable section tree for cross-validation.

    Shows: index, page range, category, heading breadcrumb, word count.
    In verbose mode also prints the first 120 chars of the chunk text.
    """
    cat_counts: dict[str, int] = {}
    print(f"\n{'─'*90}")
    print(f"  {'#':>4}  {'PAGES':>10}  {'WORDS':>6}  {'CATEGORY':<15}  HEADING PATH")
    print(f"{'─'*90}")

    for i, c in enumerate(chunks):
        cat_counts[c.section_category] = cat_counts.get(c.section_category, 0) + 1
        indent = "  " * max(0, c.heading_level - 1)
        path_str = " > ".join(c.heading_path) if c.heading_path else "(pre-heading)"
        print(f"  {i:>4}  p{c.page_start:>3}–p{c.page_end:<3}  "
              f"{c.word_count:>5}w  {c.section_category:<15}  {indent}{path_str[:65]}")
        if verbose and c.text:
            preview = c.text.replace("\n", " ")[:120]
            print(f"        preview: {preview}")

    print(f"{'─'*90}")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Category breakdown:")
    for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat:<20} {n:>3} chunk(s)")
    print(f"{'─'*90}\n")
