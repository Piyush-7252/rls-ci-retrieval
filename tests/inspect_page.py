"""
Inspect semantic objects for a specific page.

Usage:
    python tests/inspect_page.py [--page 1] [--document 10987-co-jnj-64407564] [--out localfiles/inspection]

Connects to OpenSearch, fetches all semantic objects for the given page,
and prints + saves a structured view of:
  - global_position
  - type  (paragraph | heading | table_row | list_item)
  - text  (the embedding unit)
  - display_spans  (sentences / rows / fields — each with bbox)
  - entities
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Config ────────────────────────────────────────────────────────────────────
OPENSEARCH_ENDPOINT = os.environ.get(
    "OPENSEARCH_ENDPOINT",
    "search-rls-dev-rhitzxwnctmuyq2l4kny5kwelu.eu-west-1.es.amazonaws.com",
)
SEMANTIC_OBJECTS_INDEX = os.environ.get("SEMANTIC_OBJECTS_INDEX", "semantic-objects")
AWS_REGION             = os.environ.get("AWS_REGION", "eu-west-1")
DEFAULT_DOCUMENT_ID    = "Combined_REDACTED_CSR-Full-co-jnj-64407564"
DEFAULT_OUT_DIR        = str(ROOT / "localfiles" / "inspection")


# ── OpenSearch client ─────────────────────────────────────────────────────────

def _get_os():
    import boto3
    from opensearchpy import OpenSearch, RequestsHttpConnection
    from requests_aws4auth import AWS4Auth

    frozen  = boto3.Session().get_credentials().get_frozen_credentials()
    awsauth = AWS4Auth(
        frozen.access_key, frozen.secret_key, AWS_REGION, "es",
        session_token=frozen.token,
    )
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_ENDPOINT, "port": 443}],
        http_auth=awsauth, use_ssl=True, verify_certs=True,
        connection_class=RequestsHttpConnection,
    )


# ── Query ─────────────────────────────────────────────────────────────────────

def fetch_objects(document_id: str, page: int) -> list[dict]:
    """
    Retrieve ALL semantic objects for *document_id* / *page* from OpenSearch.

    Uses search_after pagination so no result is dropped regardless of how
    many objects are indexed for the page.  All _source fields are returned
    so no indexed data is hidden.
    """
    os_client = _get_os()
    results: list[dict] = []
    search_after = None
    PAGE_SIZE = 500

    while True:
        body: dict = {
            "size": PAGE_SIZE,
            "query": {"bool": {"filter": [
                {"term": {"document_id": document_id}},
                {"term": {"page": page}},
            ]}},
            # Sort by global_position then _id for a stable, consistent cursor
            "sort": [
                {"global_position": "asc"},
                {"_id": "asc"},
            ],
            "_source": {"excludes": ["*embedding*", "*vector*"]},  # skip dense vectors
            "track_total_hits": True,
        }
        if search_after is not None:
            body["search_after"] = search_after

        resp = os_client.search(index=SEMANTIC_OBJECTS_INDEX, body=body)
        hits = resp["hits"]["hits"]
        if not hits:
            break

        results.extend(h["_source"] for h in hits)

        # Stop when we got fewer results than requested (last page)
        if len(hits) < PAGE_SIZE:
            break

        search_after = hits[-1]["sort"]

    return results


# ── Formatting ────────────────────────────────────────────────────────────────

W = 76

def fmt_objects(objects: list[dict], document_id: str, page: int) -> str:
    lines: list[str] = []
    lines.append("=" * W)
    lines.append(f"  Document : {document_id}")
    lines.append(f"  Page     : {page}")
    lines.append(f"  Objects  : {len(objects)}")
    lines.append("=" * W)

    # Fields rendered explicitly below — everything else shown as "extra"
    KNOWN_FIELDS = {
        "object_id", "global_position", "position", "type", "page",
        "bbox", "text", "display_spans", "entities", "parent_chunk_id",
    }

    for obj in objects:
        gpos  = obj.get("global_position", "?")
        lpos  = obj.get("position", "?")
        kind  = obj.get("type", "?")
        chunk = obj.get("parent_chunk_id", "?")
        bbox  = [round(x, 1) for x in obj.get("bbox", [])]
        text  = obj.get("text", "")
        spans = obj.get("display_spans", [])
        ents  = obj.get("entities", [])

        lines.append(f"\n\u250c\u2500 global={gpos}  local={lpos}  type={kind}  chunk={chunk}")
        lines.append(f"\u2502  bbox : {bbox}")
        lines.append(f"\u2502")

        # Full object text (embedding unit)
        lines.append(f"\u2502  OBJECT TEXT ({len(text)} chars):")
        for chunk_line in _wrap(text, 70):
            lines.append(f"\u2502    {chunk_line}")

        # Display spans
        lines.append(f"\u2502")
        lines.append(f"\u2502  DISPLAY SPANS ({len(spans)}):")
        for i, sp in enumerate(spans, 1):
            sp_bbox  = [round(x, 1) for x in sp.get("bbox", [])]
            sp_type  = sp.get("type", "?")
            sp_start = sp.get("start", 0)
            sp_end   = sp.get("end", 0)
            sp_text  = sp.get("text", "")
            lines.append(f"\u2502    [{i:>2}]  type={sp_type:<12}  "
                         f"[{sp_start}:{sp_end}]  bbox={sp_bbox}")
            for span_line in _wrap(sp_text, 66, prefix="\u2502         "):
                lines.append(span_line)

        # Entities
        if ents:
            lines.append(f"\u2502")
            lines.append(f"\u2502  ENTITIES ({len(ents)}):")
            for e in ents[:10]:
                lines.append(f"\u2502    {e.get('type','?'):<20}  \"{e.get('text','')}\"  "
                              f"score={e.get('score', 0):.2f}")

        # Any extra fields returned by OpenSearch that are not in the known set
        extra = {k: v for k, v in obj.items() if k not in KNOWN_FIELDS}
        if extra:
            lines.append(f"\u2502")
            lines.append(f"\u2502  EXTRA FIELDS ({len(extra)}):")
            for k, v in sorted(extra.items()):
                v_str = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
                for el in _wrap(f"{k}: {v_str}", 70):
                    lines.append(f"\u2502    {el}")

        lines.append("\u2514" + "\u2500" * (W - 1))

    return "\n".join(lines)


def _wrap(text: str, width: int, prefix: str = "│    ") -> list[str]:
    text = text.replace("\n", " ").strip()
    lines = []
    while len(text) > width:
        cut = text.rfind(" ", 0, width)
        if cut == -1:
            cut = width
        lines.append(f"{prefix}{text[:cut]}")
        text = text[cut:].lstrip()
    if text:
        lines.append(f"{prefix}{text}")
    return lines


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect semantic objects for a page")
    parser.add_argument("--page",     type=int, default=1,
                        help="Page number to inspect (default: 1)")
    parser.add_argument("--document", default=DEFAULT_DOCUMENT_ID,
                        help="Document ID")
    parser.add_argument("--out",      default=DEFAULT_OUT_DIR,
                        help="Output directory for saved inspection files")
    args = parser.parse_args()

    print(f"Fetching ALL objects for page {args.page} of '{args.document}' …")
    objects = fetch_objects(args.document, args.page)
    print(f"Retrieved {len(objects)} objects.")

    if not objects:
        print(f"No objects found for page {args.page} — is the index populated?")
        sys.exit(1)

    report = fmt_objects(objects, args.document, args.page)
    print(report)

    # Save to file
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"{ts}_page{args.page}_{args.document}.txt"
    out_file.write_text(report, encoding="utf-8")

    # Also save raw JSON
    json_file = out_dir / f"{ts}_page{args.page}_{args.document}.json"
    json_file.write_text(json.dumps(objects, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved:\n  {out_file}\n  {json_file}")


if __name__ == "__main__":
    main()
