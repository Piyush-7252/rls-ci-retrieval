"""
Dry-run extractor: download PDF from S3, run Apryse, parse and build
semantic objects for a single page — WITHOUT touching OpenSearch.

Shows two views side-by-side and saves both to localfiles/dry_run/:

  1. RAW APRYSE   — verbatim Apryse e_DocStructure page JSON
  2. PARSED PAGE  — output of parse_pages() (headings, paragraphs, tables, lists …)
  3. OBJECTS      — final semantic objects from _build_objects() (what gets indexed)

Usage:
    python tests/dry_run_page.py --page 12
    python tests/dry_run_page.py --page 12 --save-only   # skip console print
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.apryse_parser import parse_pages  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("dry_run")

# ── Constants ─────────────────────────────────────────────────────────────────
S3_BUCKET   = os.environ.get("S3_BUCKET",   "rls-file-bucket-eu")
S3_KEY      = os.environ.get("S3_KEY",
              "RLS CIM/29/documents/"
              "20260507091101630_00y2ql0_10987_REDACTED_CO-FD-JNJ-64407564-AAA-498425_1245108.pdf")
DOCUMENT_ID = "10987-co-jnj-64407564"
AWS_REGION  = os.environ.get("AWS_REGION", "eu-west-1")
APRYSE_KEY  = os.environ.get("APRYSE_LICENSE_KEY", "")
OUT_DIR     = ROOT / "localfiles" / "dry_run"

W = 76  # print width


# ── Load extraction lambda helper ─────────────────────────────────────────────

def _load_extraction_module():
    lf_path = ROOT / "lambdas" / "document" / "extraction" / "lambda_function.py"
    sys.path.insert(0, str(lf_path.parent))
    spec = importlib.util.spec_from_file_location("doc_extraction", lf_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── S3 download ───────────────────────────────────────────────────────────────

def download_pdf() -> bytes:
    import boto3
    logger.info("Downloading s3://%s/%s", S3_BUCKET, S3_KEY)
    resp = boto3.client("s3", region_name=AWS_REGION).get_object(
        Bucket=S3_BUCKET, Key=S3_KEY
    )
    data = resp["Body"].read()
    logger.info("Downloaded %d bytes", len(data))
    return data


# ── Apryse extraction ─────────────────────────────────────────────────────────

def run_apryse(pdf_bytes: bytes) -> dict:
    from apryse_sdk import PDFNet, DataExtractionModule, DataExtractionOptions
    PDFNet.Initialize(APRYSE_KEY)
    PDFNet.AddResourceSearchPath(os.environ.get("APRYSE_RESOURCE_PATH", "/opt/apryse"))

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
        fh.write(pdf_bytes)
        tmp_path = fh.name

    try:
        options  = DataExtractionOptions()
        raw_json = DataExtractionModule.ExtractData(
            tmp_path, DataExtractionModule.e_DocStructure, options
        )
    finally:
        os.unlink(tmp_path)

    doc_structure = json.loads(raw_json)
    total_pages   = len(doc_structure.get("pages", []))
    logger.info("Apryse done — %d pages in document", total_pages)
    return doc_structure


# ── Pretty printer ────────────────────────────────────────────────────────────

def _hr(label: str = "", char: str = "─") -> str:
    if label:
        pad = (W - len(label) - 2) // 2
        return char * pad + f" {label} " + char * (W - pad - len(label) - 2)
    return char * W


def print_raw_apryse(page_struct: dict, page: int) -> str:
    lines = [
        "",
        "=" * W,
        f"  VIEW 1 — RAW APRYSE e_DocStructure  (page {page})",
        "=" * W,
    ]
    elements = page_struct.get("elements", [])
    lines.append(f"  {len(elements)} top-level element(s)\n")
    for i, el in enumerate(elements):
        etype = el.get("type", "?")
        rect  = [round(x, 1) for x in el.get("rect", [])]
        lines.append(f"  [{i:>3}]  type={etype:<14}  rect={rect}")
        # Show text content preview
        text = _element_text(el)
        if text:
            lines.append(f"         text: {text[:100].replace(chr(10),' ')!r}")
    return "\n".join(lines)


def _element_text(el: dict) -> str:
    etype = el.get("type", "")
    if etype in ("paragraph", "heading"):
        return "".join(
            sp.get("text", "") for sp in el.get("contents", [])
            if sp.get("type") == "span"
        )
    if etype == "list":
        items = []
        for item in el.get("items", []):
            for para in item.get("contents", []):
                items.append("".join(
                    sp.get("text", "") for sp in para.get("contents", [])
                    if sp.get("type") == "span"
                ))
        return " | ".join(items[:3]) + (" ..." if len(items) > 3 else "")
    if etype == "table":
        rows = el.get("trs", [])
        if rows:
            first_row = rows[0]
            cells = []
            for td in first_row.get("tds", []):
                for para in td.get("contents", []):
                    cells.append("".join(
                        sp.get("text", "") for sp in para.get("contents", [])
                        if sp.get("type") == "span"
                    ))
            return " | ".join(cells[:4])
    return ""


def print_parsed_page(pages: list[dict], page: int) -> str:
    pg = next((p for p in pages if p.get("page_number") == page), None)
    if not pg:
        return f"  Page {page} not found in parse_pages() output"

    lines = [
        "",
        "=" * W,
        f"  VIEW 2 — PARSED PAGE  (parse_pages output, page {page})",
        "=" * W,
        f"  Size        : {pg.get('width', '?')} × {pg.get('height', '?')}",
        f"  Headings    : {len(pg.get('headings', []))}",
        f"  Paragraphs  : {len(pg.get('paragraphs', []))}",
        f"  Tables      : {len(pg.get('tables', []))}",
        f"  Lists       : {len(pg.get('lists', []))}",
        f"  Headers     : {len(pg.get('headers', []))}",
        f"  Footers     : {len(pg.get('footers', []))}",
        f"  Para objects: {len(pg.get('paragraph_objects', []))}",
        "",
    ]

    for h in pg.get("headings", []):
        lines.append(f"  [HEADING L{h.get('level',1)}]  {h.get('text','')[:100]}")

    for i, obj in enumerate(pg.get("paragraph_objects", [])):
        kind = obj.get("type", "paragraph")
        text = obj.get("text", "")
        rect = [round(x, 1) for x in obj.get("rect", [])]
        searchable = obj.get("searchable", True)
        flag = "" if searchable else "  ⚠ non-searchable"
        lines.append(f"\n  [{i:>3}]  type={kind:<12}  rect={rect}{flag}")
        lines.append(f"         {text[:110].replace(chr(10),' ')!r}")

    return "\n".join(lines)


def print_objects(objects: list[dict], page: int) -> str:
    page_objs = [o for o in objects if o.get("page") == page]
    lines = [
        "",
        "=" * W,
        f"  VIEW 3 — SEMANTIC OBJECTS  (_build_objects output, page {page})",
        f"  {len(page_objs)} object(s)  |  global_position range: "
        f"{page_objs[0]['global_position'] if page_objs else '—'} – "
        f"{page_objs[-1]['global_position'] if page_objs else '—'}",
        "=" * W,
    ]

    for obj in page_objs:
        gpos  = obj["global_position"]
        kind  = obj["type"]
        text  = obj["text"]
        bbox  = [round(x, 1) for x in obj.get("bbox", [])]
        spans = obj.get("display_spans", [])
        section = obj.get("section", "")
        norm  = obj.get("normalized_text", "")[:80]
        searchable = obj.get("searchable", True)

        lines.append(f"\n┌─ [{gpos:>4}]  type={kind:<12}  page={obj['page']}  "
                     f"searchable={searchable}")
        lines.append(f"│  section : {section!r}")
        lines.append(f"│  bbox    : {bbox}")
        lines.append(f"│")
        lines.append(f"│  TEXT ({len(text)} chars):")
        for chunk in _wrap(text, 68):
            lines.append(f"│    {chunk}")
        lines.append(f"│  NORM   : {norm!r}")
        lines.append(f"│")
        lines.append(f"│  DISPLAY SPANS ({len(spans)}):")
        for j, sp in enumerate(spans, 1):
            sp_bbox = [round(x, 1) for x in sp.get("bbox", [])]
            lines.append(f"│    [{j}]  type={sp.get('type','?'):<10}  "
                         f"[{sp.get('start',0)}:{sp.get('end',0)}]  "
                         f"bbox={sp_bbox}")
            sp_text = sp.get("text", "")
            lines.append(f"│         {sp_text[:90].replace(chr(10),' ')!r}")
        lines.append("└" + "─" * (W - 1))

    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    text  = text.replace("\n", " ").strip()
    lines = []
    while len(text) > width:
        cut = text.rfind(" ", 0, width) or width
        lines.append(text[:cut])
        text = text[cut:].lstrip()
    if text:
        lines.append(text)
    return lines


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run page extractor")
    parser.add_argument("--page",      type=int, default=12,
                        help="Page to inspect (default: 12)")
    parser.add_argument("--save-only", action="store_true",
                        help="Write files but skip console output")
    args = parser.parse_args()
    page = args.page

    # ── Step 1: Download PDF ──────────────────────────────────────────────────
    pdf_bytes = download_pdf()

    # ── Step 2: Run Apryse extraction ─────────────────────────────────────────
    logger.info("Running Apryse e_DocStructure extraction …")
    doc_structure = run_apryse(pdf_bytes)

    # Extract raw page struct for the requested page
    all_pages_raw = doc_structure.get("pages", [])
    raw_page = next(
        (p for p in all_pages_raw
         if p.get("properties", {}).get("pageNumber") == page),
        None,
    )
    if not raw_page:
        logger.error("Page %d not found in document (has %d pages)", page, len(all_pages_raw))
        sys.exit(1)

    # ── Step 3: parse_pages ───────────────────────────────────────────────────
    logger.info("Running parse_pages for page %d …", page)
    parsed = parse_pages(doc_structure, page, page)

    # ── Step 4: _build_objects ────────────────────────────────────────────────
    logger.info("Building semantic objects …")
    extraction_mod = _load_extraction_module()
    chunk_id       = f"{DOCUMENT_ID}_chunk_dryrun"
    objects        = extraction_mod._build_objects(chunk_id, parsed, global_offset=0)

    # ── Step 5: Build views ───────────────────────────────────────────────────
    view1 = print_raw_apryse(raw_page, page)
    view2 = print_parsed_page(parsed, page)
    view3 = print_objects(objects, page)
    full_report = "\n".join([view1, view2, view3])

    if not args.save_only:
        print(full_report)

    # ── Step 6: Save ──────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Human-readable report
    txt_path = OUT_DIR / f"{ts}_page{page}_dry_run.txt"
    txt_path.write_text(full_report, encoding="utf-8")

    # Raw Apryse JSON (verbatim)
    raw_path = OUT_DIR / f"{ts}_page{page}_apryse_raw.json"
    raw_path.write_text(json.dumps(raw_page, indent=2, ensure_ascii=False), encoding="utf-8")

    # Parsed page JSON
    parsed_path = OUT_DIR / f"{ts}_page{page}_parsed.json"
    parsed_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")

    # Final semantic objects JSON
    objects_path = OUT_DIR / f"{ts}_page{page}_objects.json"
    objects_path.write_text(json.dumps(objects, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved to {OUT_DIR}/")
    print(f"  {txt_path.name}      — full report")
    print(f"  {raw_path.name}    — raw Apryse JSON")
    print(f"  {parsed_path.name}      — parse_pages() output")
    print(f"  {objects_path.name}     — _build_objects() output (pre-index)")


if __name__ == "__main__":
    main()
