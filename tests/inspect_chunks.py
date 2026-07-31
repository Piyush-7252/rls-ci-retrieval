"""
inspect_chunks.py
─────────────────
Inspect section-aware chunks produced by the new hierarchical chunker.

Runs locally — no AWS, no OpenSearch.
Reads the PDF from localfiles/pdfs/, runs Apryse extraction, then feeds
the output through shared/section_chunker.py.

Usage
─────
    python tests/inspect_chunks.py [--pages 1-10] [--verbose] [--out localfiles/inspection]

Examples
────────
    # Default: 10993 document, pages 1-10
    python tests/inspect_chunks.py

    # Full document
    python tests/inspect_chunks.py --pages 1-101

    # Show first 400 chars of each chunk body
    python tests/inspect_chunks.py --pages 1-10 --verbose

    # Different document
    python tests/inspect_chunks.py --doc 10987 --pages 1-20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Apryse config (mirrored from s3_pipeline_test.py) ─────────────────────────
APRYSE_KEY           = "Real Life Sciences  LLC:OEM:RL Protect Docs::L+:AMS(20280530):1CA5D5FD04E73F3F77A8B83043360FF960617FEB89B17A92FB446D82ED34C3AE42B231F5C7"
APRYSE_RESOURCE_PATH = os.path.expanduser("~/.apryse_modules")

PDF_DIR = ROOT / "localfiles" / "pdfs"
OUT_DIR = ROOT / "localfiles" / "inspection"

# Known document → filename mapping
_DOC_FILES = {
    "10993": "10993_REDACTED_Protocol-Amendment-1-FD-64407564MMY3002-218114_1245209.pdf",
    "10987": "10987_REDACTED_Protocol-FD-64407564-MonumenTAL-1_Phase12.pdf",
}

W = 80


# ─────────────────────────────────────────────────────────────────────────────
# Apryse extraction (local)
# ─────────────────────────────────────────────────────────────────────────────

def _run_apryse(pdf_path: Path) -> dict:
    import json as _json
    from apryse_sdk import PDFNet, DataExtractionModule, DataExtractionOptions

    PDFNet.Initialize(APRYSE_KEY)
    PDFNet.AddResourceSearchPath(APRYSE_RESOURCE_PATH)

    print(f"  Running Apryse e_DocStructure on {pdf_path.name} …", flush=True)
    options  = DataExtractionOptions()
    raw_json = DataExtractionModule.ExtractData(
        str(pdf_path),
        DataExtractionModule.e_DocStructure,
        options,
    )
    doc = _json.loads(raw_json)
    print(f"  Extraction done — {len(doc.get('pages', []))} pages total", flush=True)
    return doc


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

_CATEGORY_ICONS = {
    "OBJECTIVES":    "🎯",
    "ENDPOINTS":     "📊",
    "ELIGIBILITY":   "✅",
    "DESIGN":        "🔬",
    "BACKGROUND":    "📖",
    "SAFETY":        "⚠️ ",
    "SYNOPSIS":      "📋",
    "TREATMENT":     "💊",
    "PROCEDURES":    "📅",
    "STATISTICS":    "📈",
    "PK":            "🧬",
    "POPULATION":    "👥",
    "EFFICACY":      "⭐",
    "BIOMARKER":     "🔭",
    "BENEFIT_RISK":  "⚖️ ",
    "APPENDIX":      "📎",
    "ADMINISTRATIVE":"🗂️",
    "OTHER":         "  ",
}

_OBJ_COLORS = {
    "heading":   "H",
    "paragraph": "P",
    "table_row": "T",
    "list":      "L",
    "metadata":  "M",
}


def _wrap(text: str, width: int = 74, indent: str = "    ") -> list[str]:
    text = text.replace("\n", " ").strip()
    out  = []
    while len(text) > width:
        cut = text.rfind(" ", 0, width)
        if cut <= 0:
            cut = width
        out.append(f"{indent}{text[:cut]}")
        text = text[cut:].lstrip()
    if text:
        out.append(f"{indent}{text}")
    return out or [f"{indent}(empty)"]


def fmt_chunk(idx: int, chunk, verbose: bool = False) -> str:
    """Console summary — truncated for readability."""
    lines: list[str] = []

    icon     = _CATEGORY_ICONS.get(chunk.section_category, "  ")
    path_str = " > ".join(chunk.heading_path) if chunk.heading_path else "(pre-heading content)"
    pg_range = f"p{chunk.page_start}" if chunk.page_start == chunk.page_end \
               else f"p{chunk.page_start}–p{chunk.page_end}"
    lvl_str  = f"L{chunk.heading_level}" if chunk.heading_level else "L?"

    lines.append(f"\n┌{'─'*(W-1)}")
    lines.append(f"│  [{idx:>3}]  {icon} {chunk.section_category:<14}  "
                 f"{pg_range:<12}  {lvl_str}  {chunk.word_count:>4} words  "
                 f"conf={chunk.section_confidence:.2f}  pos={chunk.document_position:.3f}")
    lines.append(f"│  PATH: {path_str[:W-10]}")
    lines.append(f"│")

    # Object breakdown
    type_counts: dict[str, int] = {}
    for _, obj in chunk.objects:
        t = obj.get("type", "paragraph")
        type_counts[t] = type_counts.get(t, 0) + 1

    obj_summary = "  ".join(f"{_OBJ_COLORS.get(t,t)}×{n}" for t, n in type_counts.items())
    lines.append(f"│  Objects ({len(chunk.objects)}): {obj_summary}")

    if verbose:
        lines.append(f"│")
        lines.append(f"│  CONTENT PREVIEW:")
        shown = 0
        for _, obj in chunk.objects:
            t    = obj.get("type", "paragraph")
            text = (obj.get("text") or "").strip()
            if not text:
                continue
            tag = _OBJ_COLORS.get(t, t)
            if shown < 4:
                lines.append(f"│  [{tag}] {text[:120]}")
                if len(text) > 120:
                    lines.append(f"│      … ({len(text)} chars total)")
            else:
                preview = text[:80].replace("\n", " ")
                lines.append(f"│  [{tag}] {preview}…" if len(text) > 80 else f"│  [{tag}] {text}")
            shown += 1
            if shown >= 10:
                remaining = len(chunk.objects) - shown
                if remaining > 0:
                    lines.append(f"│  … {remaining} more object(s) — see saved files for full content")
                break

    lines.append(f"└{'─'*(W-1)}")
    return "\n".join(lines)


def fmt_chunk_full(idx: int, chunk) -> str:
    """Full untruncated chunk — every object, written only to the saved text file."""
    lines: list[str] = []
    path_str = " > ".join(chunk.heading_path) if chunk.heading_path else "(pre-heading content)"
    pg_range = f"p{chunk.page_start}" if chunk.page_start == chunk.page_end \
               else f"p{chunk.page_start}–{chunk.page_end}"

    sep = "=" * W
    lines.append(f"\n{sep}")
    lines.append(f"CHUNK {idx:03d}  |  {chunk.section_category}  |  "
                 f"{pg_range}  |  L{chunk.heading_level}  |  {chunk.word_count} words")
    lines.append(f"PATH : {path_str}")
    lines.append(sep)
    lines.append("")

    for _, obj in chunk.objects:
        kind = obj.get("type", "paragraph")
        text = (obj.get("text") or "").strip()
        if not text:
            continue
        tag = _OBJ_COLORS.get(kind, kind)
        lines.append(f"[{tag}] {text}")
        lines.append("")

    lines.append(f"{'─'*W}")
    lines.append("RAW CHUNK TEXT (as stored in index):")
    lines.append(f"{'─'*W}")
    lines.append(chunk.text)
    return "\n".join(lines)


def fmt_summary(chunks, doc_name: str, page_start: int, page_end: int) -> str:
    lines: list[str] = []
    lines.append(f"\n{'═'*W}")
    lines.append(f"  SECTION CHUNKER INSPECTION")
    lines.append(f"  Document : {doc_name}")
    lines.append(f"  Pages    : {page_start}–{page_end}")
    lines.append(f"  Chunks   : {len(chunks)}")
    lines.append(f"{'═'*W}")

    # Category breakdown
    cat_counts: dict[str, int] = {}
    cat_words:  dict[str, int] = {}
    for c in chunks:
        cat_counts[c.section_category] = cat_counts.get(c.section_category, 0) + 1
        cat_words[c.section_category]  = cat_words.get(c.section_category, 0) + c.word_count

    lines.append(f"\n  CATEGORY BREAKDOWN  (chunks / total words):")
    for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
        icon = _CATEGORY_ICONS.get(cat, "  ")
        w    = cat_words[cat]
        bar  = "█" * min(n * 3, 30)
        lines.append(f"    {icon} {cat:<15}  {n:>3} chunks  {w:>5} words  {bar}")

    # Size distribution
    words = [c.word_count for c in chunks]
    lines.append(f"\n  CHUNK SIZE  (words):")
    lines.append(f"    min={min(words)}  median={sorted(words)[len(words)//2]}  "
                 f"max={max(words)}  total={sum(words)}")
    tiny  = sum(1 for w in words if w < 40)
    large = sum(1 for w in words if w > 500)
    if tiny:
        lines.append(f"    ⚠  {tiny} tiny chunk(s) < 40 words")
    if large:
        lines.append(f"    ⚠  {large} large chunk(s) > 500 words")

    lines.append(f"\n{'═'*W}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect section-aware chunks — local, no AWS"
    )
    parser.add_argument(
        "--pages", default="1-10",
        help="Page range, e.g. '1-10' or '1-101'  (default: 1-10)",
    )
    parser.add_argument(
        "--doc", default="10993",
        choices=list(_DOC_FILES.keys()),
        help="Document key (default: 10993)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show chunk content preview on console (files always have full content)",
    )
    parser.add_argument(
        "--out", default=str(OUT_DIR),
        help="Output directory for saved reports (default: localfiles/inspection)",
    )
    args = parser.parse_args()

    # Resolve PDF path
    pdf_name = _DOC_FILES.get(args.doc)
    if not pdf_name:
        print(f"ERROR: unknown doc key '{args.doc}'. Known: {list(_DOC_FILES)}")
        sys.exit(1)
    pdf_path = PDF_DIR / pdf_name
    if not pdf_path.exists():
        # try partial match
        matches = list(PDF_DIR.glob(f"{args.doc}*.pdf"))
        if matches:
            pdf_path = matches[0]
        else:
            print(f"ERROR: PDF not found at {pdf_path}")
            sys.exit(1)

    # Parse page range
    parts      = args.pages.split("-")
    page_start = int(parts[0])
    page_end   = int(parts[1]) if len(parts) > 1 else page_start

    # Run Apryse
    doc_structure = _run_apryse(pdf_path)
    total_pages   = len(doc_structure.get("pages", []))
    page_end      = min(page_end, total_pages)

    print(f"  Parsing pages {page_start}–{page_end} …", flush=True)
    from shared.apryse_parser   import parse_pages
    from shared.section_chunker import build_section_chunks

    all_pages = parse_pages(doc_structure, page_start, page_end)
    print(f"  Parsed {len(all_pages)} page(s)")

    print(f"  Running section chunker …", flush=True)
    chunks = build_section_chunks(all_pages, total_pages=total_pages)
    print(f"  Produced {len(chunks)} chunk(s)\n")

    # ── Print console report ──────────────────────────────────────────────────
    summary = fmt_summary(chunks, pdf_path.name, page_start, page_end)
    print(summary)

    print(f"\n  CHUNK SUMMARY  (verbose={args.verbose}  |  full content in saved files):")
    for i, chunk in enumerate(chunks):
        print(fmt_chunk(i, chunk, verbose=args.verbose))

    # ── Always save output files ──────────────────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{ts}_chunks_{args.doc}_p{page_start}-{page_end}"

    # 1. Full text report — every object, no truncation
    full_parts: list[str] = [
        f"SECTION CHUNKER FULL REPORT",
        f"Document : {pdf_path.name}",
        f"Pages    : {page_start}–{page_end}",
        f"Chunks   : {len(chunks)}",
        f"Generated: {ts}",
        "=" * W,
        summary,
    ]
    for i, chunk in enumerate(chunks):
        full_parts.append(fmt_chunk_full(i, chunk))
    txt_file = out_dir / f"{stem}.txt"
    txt_file.write_text("\n".join(full_parts), encoding="utf-8")

    # 2. Structured JSON — complete metadata + full text per chunk
    json_out = [
        {
            "index":              i,
            "chunk_idx":          c.chunk_idx,
            "parent_chunk_idx":   c.parent_chunk_idx,
            "prev_chunk_idx":     c.prev_chunk_idx,
            "next_chunk_idx":     c.next_chunk_idx,
            "heading_path":       c.heading_path,
            "heading_text":       c.heading_text,
            "heading_level":      c.heading_level,
            "semantic_path":      c.semantic_path,
            "heading_embedding_text": c.heading_embedding_text,
            "section":            c.section,
            "subsection":         c.subsection,
            "section_category":   c.section_category,
            "section_confidence": round(c.section_confidence, 3),
            "document_position":  round(c.document_position, 4),
            "page_start":         c.page_start,
            "page_end":           c.page_end,
            "word_count":         c.word_count,
            "n_objects":        len(c.objects),
            "object_types":     {
                t: sum(1 for _, o in c.objects if o.get("type") == t)
                for t in sorted({o.get("type") for _, o in c.objects})
            },
            "full_text": c.text,
            "objects": [
                {
                    "page":  pg,
                    "type":  o.get("type"),
                    "level": o.get("level"),
                    "text":  o.get("text", ""),
                }
                for pg, o in c.objects
            ],
        }
        for i, c in enumerate(chunks)
    ]
    json_file = out_dir / f"{stem}.json"
    json_file.write_text(json.dumps(json_out, indent=2, ensure_ascii=False), encoding="utf-8")

    # 3. Raw Apryse doc_structure JSON — only the pages in the requested range
    apryse_pages = [
        p for p in doc_structure.get("pages", [])
        if page_start <= p.get("properties", {}).get("pageNumber", 0) <= page_end
    ]
    apryse_out = {
        "properties": doc_structure.get("properties", {}),
        "pages":      apryse_pages,
    }
    apryse_file = out_dir / f"{stem}_apryse.json"
    apryse_file.write_text(json.dumps(apryse_out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  Saved to {out_dir}/")
    print(f"    {txt_file.name}  (full text, no truncation)")
    print(f"    {json_file.name}  (structured chunks JSON)")
    print(f"    {apryse_file.name}  (raw Apryse doc_structure)")


if __name__ == "__main__":
    main()
