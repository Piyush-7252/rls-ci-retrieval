"""
S3 Pipeline Test — Real AWS, No Mocks
=======================================
Downloads the pre-computed Apryse DocStructure (full_tables.json) from S3,
then passes every chunk through the full document enrichment pipeline using
real AWS services.  No local Apryse SDK call needed.

Use ``tests/index_cis.py`` for CI indexing.

Services used (all real, no mocks)
------------------------------------
  S3                  rls-file-bucket-eu  (eu-west-1)
  Normalization       pure Python  (no AWS)
  NER                 GLiNER / Comprehend Medical  (eu-west-1)
  Ontology            pure Python  (no AWS)
  Bedrock Titan       dense + sparse embeddings  (eu-west-1)
  OpenSearch          rls-dev cluster  (eu-west-1)

Pre-requisites
--------------
  AWS credentials must be exported in the calling shell:
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    export AWS_SESSION_TOKEN=...

Usage
-----
    python tests/s3_pipeline_test.py
    python tests/s3_pipeline_test.py --pages 1-10 --chunk-size 5
    python tests/s3_pipeline_test.py --skip-index      # dry-run, no OpenSearch write
    python tests/s3_pipeline_test.py --show-sections   # print section tree and exit
    python tests/s3_pipeline_test.py --verbose
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import tempfile
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from shared.geometry_trace import trace, trace_raw_apryse

# ─── project root on sys.path ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — all real values, no mocks
# ─────────────────────────────────────────────────────────────────────────────

W = "─" * 70   # banner separator

S3_BUCKET  = "rls-file-bucket-eu"
S3_KEY     = (
    "Patterns Check Run/18/documents/"
    "20260726062234599_4xs0l7p_10993_REDACTED_Protocol-Amendment-1-FD-64407564MMY3002-218114_1245209"
)
DOCUMENT_ID = "20260726062234599_4xs0l7p_10993_REDACTED_Protocol-Amendment-1-FD-64407564MMY3002-218114_1245209"

# Pre-computed Apryse DocStructure (all 176 pages, no demo-mode cap)
FULL_TABLES_S3_KEY = (
    "extractions/Patterns Check Run/18/"
    "20260726062234599_4xs0l7p_10993_REDACTED_Protocol-Amendment-1-FD-64407564MMY3002-218114_1245209/"
    "full_tables.json"
)

# Allow batch_run.py to override per-document via env vars
# PIPELINE_DOCUMENT_ID  -> overrides DOCUMENT_ID
# PIPELINE_S3_FOLDER    -> derives FULL_TABLES_S3_KEY and S3_KEY metadata
if os.environ.get("PIPELINE_DOCUMENT_ID"):
    DOCUMENT_ID = os.environ["PIPELINE_DOCUMENT_ID"]
if os.environ.get("PIPELINE_S3_FOLDER"):
    _s3f = os.environ["PIPELINE_S3_FOLDER"]
    FULL_TABLES_S3_KEY = f"extractions/Patterns Check Run/18/{_s3f}/full_tables.json"
    S3_KEY = f"Patterns Check Run/18/extraction/{_s3f}"  # metadata only

OPENSEARCH_ENDPOINT = (
    "search-rls-dev-rhitzxwnctmuyq2l4kny5kwelu.eu-west-1.es.amazonaws.com"
)
AWS_REGION      = "eu-west-1"
APRYSE_KEY      = "demo:1744408704147:6136e23d030000000053544a6ccf2b965334c4d0169bc2693bc540885b"
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"

# Path to the Apryse StructuredOutput native module (extracted from StructuredOutputMac.zip)
APRYSE_RESOURCE_PATH = os.path.expanduser("~/.apryse_modules")

# ─── env vars (must be set before any Lambda module is imported) ──────────────
os.environ.update(
    {
        "AWS_DEFAULT_REGION":     AWS_REGION,
        "AWS_REGION":             AWS_REGION,
        "OPENSEARCH_ENDPOINT":    OPENSEARCH_ENDPOINT,
        "OPENSEARCH_INDEX":       "document-chunks",
        "OPENSEARCH_CI_INDEX":    "ci-objects",
        "SEMANTIC_OBJECTS_INDEX": "semantic-objects",
        "NER_MODEL":              "gliner",
        "EMBEDDING_MODEL":        EMBEDDING_MODEL,
        "APRYSE_LICENSE_KEY":     APRYSE_KEY,
        # Fan-out ARNs unused in sequential mode — set empty to satisfy env reads
        "EXTRACTION_LAMBDA_ARN":  "",
        "NORMALIZE_LAMBDA_ARN":   "",
        "NER_LAMBDA_ARN":         "",
        "ONTOLOGY_LAMBDA_ARN":    "",
        "EMBEDDING_LAMBDA_ARN":   "",
        "INDEX_LAMBDA_ARN":       "",
        "CHUNK_SIZE":             "20",
    }
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Lambda module loader
#   Same pattern as local_pipeline_test.py: loads each lambda_function.py
#   directly so we can call _process() without triggering fan-out.
# ─────────────────────────────────────────────────────────────────────────────

_loaded: dict[str, types.ModuleType] = {}


def _load(rel_path: str, alias: str) -> types.ModuleType:
    if alias in _loaded:
        return _loaded[alias]
    lf_path = ROOT / "lambdas" / rel_path / "lambda_function.py"
    spec    = importlib.util.spec_from_file_location(alias, lf_path)
    mod     = importlib.util.module_from_spec(spec)
    lf_dir  = str(lf_path.parent)
    if lf_dir not in sys.path:
        sys.path.insert(0, lf_dir)
    spec.loader.exec_module(mod)
    _loaded[alias] = mod
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# Real OpenSearch client
#   Built once with frozen STS credentials so the token is resolved correctly
#   before being passed to AWS4Auth.
# ─────────────────────────────────────────────────────────────────────────────

_os_client = None
_os_client_built_at: float = 0.0
_OS_CLIENT_TTL = 3000  # 50 min — safely within 1-hour STS token lifetime


def _build_os_client():
    global _os_client, _os_client_built_at
    now = time.monotonic()
    if _os_client is not None and (now - _os_client_built_at) < _OS_CLIENT_TTL:
        return _os_client

    from opensearchpy import OpenSearch, RequestsHttpConnection

    if os.environ.get("USE_LOCAL_OPENSEARCH"):
        # Local Docker OpenSearch — no auth, plain HTTP
        _os_client = OpenSearch(
            hosts        = [{"host": "localhost", "port": 9200}],
            use_ssl      = False,
            verify_certs = False,
        )
    else:
        import boto3
        from requests_aws4auth import AWS4Auth

        # Re-resolve credentials every TTL seconds so STS tokens never go stale
        frozen  = boto3.Session().get_credentials().get_frozen_credentials()
        awsauth = AWS4Auth(
            frozen.access_key,
            frozen.secret_key,
            AWS_REGION,
            "es",
            session_token=frozen.token,
        )
        _os_client = OpenSearch(
            hosts            = [{"host": OPENSEARCH_ENDPOINT, "port": 443}],
            http_auth        = awsauth,
            use_ssl          = True,
            verify_certs     = True,
            connection_class = RequestsHttpConnection,
        )
    _os_client_built_at = now
    return _os_client


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Download PDF from S3
# ─────────────────────────────────────────────────────────────────────────────

def download_pdf() -> bytes:
    import boto3
    logger.info("Downloading s3://%s/%s", S3_BUCKET, S3_KEY)
    s3       = boto3.client("s3", region_name=AWS_REGION)
    response = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
    data     = response["Body"].read()
    logger.info("Downloaded %d bytes", len(data))
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Download pre-computed DocStructure from S3 (with local cache)
# ─────────────────────────────────────────────────────────────────────────────

def _doc_cache_path() -> Path:
    """
    Local cache path for the current FULL_TABLES_S3_KEY.
    Stored under localfiles/doc_cache/<folder_name>/full_tables.json
    so each unique S3 extraction folder gets its own slot.
    """
    folder_name = Path(FULL_TABLES_S3_KEY).parent.name or "default"
    return ROOT / ".cache" / folder_name / "full_tables.json"


def download_doc_structure() -> dict:
    """
    Load the pre-computed Apryse DocStructure, using a local cache when
    available to avoid re-downloading the (potentially 1.4 GB) file from S3.

    Cache location: localfiles/doc_cache/<extraction_folder>/full_tables.json
    To force a fresh download, delete that file.

    Multiple chunks (if present) are merged in chunk-index order so the
    returned dict is always ``{"properties": {...}, "pages": [p1, p2, ...]}``.
    """
    cache_path = _doc_cache_path()

    if cache_path.exists():
        logger.info("Loading docStructure from local cache: %s", cache_path)
        raw = json.loads(cache_path.read_bytes())
    else:
        import boto3
        logger.info("Downloading pre-computed docStructure from s3://%s/%s",
                    S3_BUCKET, FULL_TABLES_S3_KEY)
        s3_client = boto3.client("s3", region_name=AWS_REGION)
        resp  = s3_client.get_object(Bucket=S3_BUCKET, Key=FULL_TABLES_S3_KEY)
        data  = resp["Body"].read()
        # Persist to local cache before parsing so future runs skip S3
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
        logger.info("Cached docStructure to: %s  (%d MB)",
                    cache_path, len(data) // 1_000_000)
        raw = json.loads(data)

    trace_raw_apryse("INDEX_DOCUMENT:after_docstructure_load", raw)

    # Merge pages from all chunks (sorted by chunkIndex) into one flat structure
    all_pages: list[dict] = []
    merged_props: dict = {}
    for chunk in sorted(raw.get("chunks", []), key=lambda c: c.get("chunkIndex", 0)):
        ds = chunk.get("docStructure", {})
        if not merged_props:
            merged_props = ds.get("properties", {})
        all_pages.extend(ds.get("pages", []))

    total = len(all_pages)
    logger.info("Pre-computed docStructure loaded — %d pages total", total)
    merged = {"properties": merged_props, "pages": all_pages}
    trace("INDEX_DOCUMENT:after_docstructure_merge", merged)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Build section-aware chunks from extracted pages
# ─────────────────────────────────────────────────────────────────────────────

def build_chunks(doc_structure: dict, page_start: int, page_end: int,
                 chunk_size: int = 20) -> list[dict]:
    """
    Build hierarchical section-aware chunks from an Apryse doc_structure.

    Each chunk represents one logical subsection of the protocol (heading +
    its body paragraphs).  Section boundaries are detected from Apryse heading
    levels, not from page counts or regexp patterns.

    ``chunk_size`` is accepted for API compatibility but no longer controls
    page-count splitting — section boundaries take precedence.
    """
    from shared.apryse_parser  import parse_pages
    from shared.section_chunker import build_section_chunks

    # Parse all pages in one pass — heading continuity requires the full range
    all_pages = parse_pages(doc_structure, page_start, page_end)
    geom_state = trace("INDEX_DOCUMENT:after_parse_pages", {"pages": all_pages})
    if not all_pages:
        logger.warning("No pages extracted for range %d–%d", page_start, page_end)
        return []

    sections = build_section_chunks(all_pages, total_pages=len(doc_structure.get("pages", [])))
    section_state = trace("INDEX_DOCUMENT:after_section_chunker", {
        "objects": [obj for sec in sections for _, obj in sec.objects]
    }, geom_state)
    if not sections:
        logger.warning("Section chunker produced no chunks for pages %d–%d",
                       page_start, page_end)
        return []

    # Log section tree at DEBUG (can be very large for long documents)
    logger.info("Section tree built: %d sections for pages %d–%d",
                len(sections), page_start, page_end)
    if logger.isEnabledFor(logging.DEBUG):
        for i, sec in enumerate(sections):
            indent = "  " * max(0, sec.heading_level - 1)
            logger.debug("  [%05d] p%d–p%d  %-14s %s%s  (%d w)",
                         i, sec.page_start, sec.page_end,
                         sec.section_category, indent,
                         sec.heading_text[:55] or "(pre-heading)",
                         sec.word_count)

    extraction = _load("extraction", "extraction")
    chunks: list[dict] = []
    global_obj_counter = 0

    for sec in sections:
        chunk_id = f"{DOCUMENT_ID}_chunk_{len(chunks):04d}"

        objects = extraction._build_objects(
            chunk_id, sec.virtual_pages, global_offset=global_obj_counter
        )
        trace(
            f"INDEX_DOCUMENT:after_extraction_build_objects:{chunk_id}",
            {"objects": objects},
            section_state,
        )
        # Enrich each object with the chunk's canonical section metadata
        for obj in objects:
            obj["section_category"]   = sec.section_category
            obj["heading_path"]       = " > ".join(sec.heading_path)
            obj["semantic_path"]      = " > ".join(sec.semantic_path)
            obj["section_confidence"] = sec.section_confidence
            obj["document_position"]  = sec.document_position
            obj["chunk_idx"]          = sec.chunk_idx
            obj["parent_chunk_idx"]   = sec.parent_chunk_idx
            obj["prev_chunk_idx"]     = sec.prev_chunk_idx
            obj["next_chunk_idx"]     = sec.next_chunk_idx

        global_obj_counter += len(objects)

        chunks.append({
            "document_id":      DOCUMENT_ID,
            "chunk_id":         chunk_id,
            "s3_bucket":        S3_BUCKET,
            "s3_key":           S3_KEY,
            "page_start":       sec.page_start,
            "page_end":         sec.page_end,
            # Section metadata stored alongside extraction
            "section":            sec.section,
            "subsection":         sec.subsection,
            "section_category":   sec.section_category,
            "heading_path":       " > ".join(sec.heading_path),
            "heading_level":      sec.heading_level,
            "word_count":         sec.word_count,
            "chunk_idx":          sec.chunk_idx,
            "parent_chunk_idx":   sec.parent_chunk_idx,
            "prev_chunk_idx":     sec.prev_chunk_idx,
            "next_chunk_idx":     sec.next_chunk_idx,
            "section_confidence": sec.section_confidence,
            "document_position":  sec.document_position,
            "semantic_path":          " > ".join(sec.semantic_path),
            "heading_embedding_text":  sec.heading_embedding_text,
            "extraction": {
                "raw_text": sec.text,
                "pages":    sec.virtual_pages,
                "objects":  objects,
            },
        })

    logger.info("Built %d section-chunk(s) for pages %d–%d  (%d total objects)",
                len(chunks), page_start, page_end, global_obj_counter)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Document pipeline  (Stages 3 – 7, sequential)
# ─────────────────────────────────────────────────────────────────────────────

def enrich_chunk(chunk: dict, skip_index: bool,
                 _print_lock: threading.Lock | None = None) -> dict:
    """
    Enrich one chunk through Normalize → NER → Embedding (parallel) → Ontology → Index.

    Inner parallelism: NER and Embedding are dispatched concurrently after
    Normalize because they read only normalisation output and are independent.
    """
    normalize = _load("normalize", "normalize")
    ner       = _load("ner",       "ner")
    ontology  = _load("ontology",  "ontology")
    embedding = _load("embedding", "embedding")
    idx       = _load("index",     "idx")
    idx._os_client = _build_os_client()

    chunk_id = chunk["chunk_id"]
    lines: list[str] = []
    _pt: dict[str, float] = {}   # per-stage pipeline timings
    _t_chunk = time.perf_counter()

    def _log(msg: str) -> None:
        lines.append(msg)

    # ─ Geometry baseline immediately before enrichment ──────────────────────
    geom_state = trace(f"INDEX_DOCUMENT:before_normalize:{chunk_id}", chunk)

    # ─ Normalize (pure Python) ────────────────────────────────────────────────
    _log(f"  [{chunk_id}] Normalize …")
    _t0 = time.perf_counter(); chunk = normalize._process_document(chunk)
    _pt["normalize"] = round(time.perf_counter() - _t0, 3)
    geom_state = trace(f"INDEX_DOCUMENT:after_normalize:{chunk_id}", chunk, geom_state)

    # ─ NER (entities + per-object enrichment) ────────────────────────────────
    _log(f"  [{chunk_id}] NER …")
    _t0 = time.perf_counter(); chunk = ner._process_document(chunk)
    _pt["ner"] = round(time.perf_counter() - _t0, 3)
    geom_state = trace(f"INDEX_DOCUMENT:after_ner:{chunk_id}", chunk, geom_state)

    # ─ Ontology (reads ner.entities — must follow NER) ─────────────────────
    _log(f"  [{chunk_id}] Ontology …")
    _t0 = time.perf_counter(); chunk = ontology._process_document(chunk)
    _pt["ontology"] = round(time.perf_counter() - _t0, 3)
    geom_state = trace(f"INDEX_DOCUMENT:after_ontology:{chunk_id}", chunk, geom_state)

    # ─ Embedding (reads extraction.objects — must follow NER enrichment) ──────
    _log(f"  [{chunk_id}] Embedding …")
    _t0 = time.perf_counter(); chunk = embedding._process_document(chunk)
    _pt["embedding"] = round(time.perf_counter() - _t0, 3)
    geom_state = trace(f"INDEX_DOCUMENT:after_embedding:{chunk_id}", chunk, geom_state)

    # ─ Index → OpenSearch ────────────────────────────────────────────────────
    _t0 = time.perf_counter()
    if not skip_index:
        _log(f"  [{chunk_id}] Index → OpenSearch …")
        geom_state = trace(f"INDEX_DOCUMENT:before_index_lambda:{chunk_id}", chunk, geom_state)
        idx._process_document(chunk)
        _pt["index"] = round(time.perf_counter() - _t0, 3)
        _log(f"  [{chunk_id}] → indexed ✓")
    else:
        _pt["index"] = 0.0
        _log(f"  [{chunk_id}] Index SKIPPED (dry-run)")

    _pt["total"] = round(time.perf_counter() - _t_chunk, 3)
    chunk["_pipeline_timings"] = _pt

    # Flush buffered lines atomically
    lock = _print_lock or threading.Lock()
    with lock:
        for line in lines:
            print(line)

    return chunk


# ─────────────────────────────────────────────────────────────────────────────
# Print helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sep(char: str = "─", width: int = 72) -> str:
    return char * width


def print_chunk(chunk: dict) -> None:
    print(f"\n{_sep()}")
    print(f"  CHUNK  {chunk['chunk_id']}  (pages {chunk['page_start']}–{chunk['page_end']})")
    print(_sep())

    for pg in chunk["extraction"]["pages"]:
        dims = f"  ({int(pg['width'])}×{int(pg['height'])})" if "width" in pg and "height" in pg else ""
        print(f"  Page {pg['page_number']}{dims}")
        if pg.get("headings"):
            print(f"    Headings   : {[h['text'][:60] for h in pg['headings']]}")
        if pg.get("paragraphs"):
            print(f"    Paragraphs : {len(pg['paragraphs'])} block(s)")
        if pg.get("tables"):
            print(f"    Tables     : {len(pg['tables'])} table(s)  "
                  f"rows={[len(t['rows']) for t in pg['tables']]}")
        if pg.get("lists"):
            print(f"    Lists      : {len(pg['lists'])} list(s)  "
                  f"items={[len(l['items']) for l in pg['lists']]}")
        if pg.get("headers"):
            print(f"    Header     : {pg['headers'][0][:70]}")
        if pg.get("footers"):
            print(f"    Footer     : {pg['footers'][0][:70]}")

    n = chunk["normalization"]
    print(f"  Tokens         : {len(n['tokens'])}")
    abbrs = list(n["abbreviations_found"].keys())
    if abbrs:
        print(f"  Abbreviations  : {abbrs}")

    entities = chunk.get("ner", {}).get("entities", [])
    if entities:
        print(f"  NER entities   : {[e['text'][:45] for e in entities[:8]]}"
              + ("  ..." if len(entities) > 8 else ""))

    expansions = chunk.get("ontology", {}).get("expansions", [])
    if expansions:
        print(f"  Ontology       : {[e['original'] for e in expansions]}")

    emb = chunk.get("embedding", {})
    print(f"  Embedding dims : {emb.get('dimensions', '—')}")
    print(f"  Sparse terms   : {len(emb.get('sparse_vector', {}))}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="S3 pipeline test — document enrichment with real AWS services"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=3,
        help="Pages per chunk  (default: 3)",
    )
    parser.add_argument(
        "--skip-index", action="store_true",
        help="Skip writing to OpenSearch  (dry-run)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Delete existing document-chunks and semantic-objects for this document "
             "before re-indexing (full clean re-index)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel chunk workers  (default: 4)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG logging",
    )
    parser.add_argument(
        "--show-sections", action="store_true",
        help="Dry-run: extract document, print the section tree, and exit",
    )
    parser.add_argument(
        "--document-id", default=None,
        help="Override DOCUMENT_ID constant (e.g. 10990-co-jnj-64407564)",
    )
    parser.add_argument(
        "--s3-folder", default=None,
        help="Extraction s3_folder key; derives FULL_TABLES_S3_KEY automatically",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Use local Docker OpenSearch at localhost:9200 instead of AWS (no SigV4 auth)",
    )
    args = parser.parse_args()

    # Apply per-document overrides from CLI (take precedence over env vars)
    global DOCUMENT_ID, FULL_TABLES_S3_KEY, S3_KEY
    if args.document_id:
        DOCUMENT_ID = args.document_id
    if args.local:
        os.environ["USE_LOCAL_OPENSEARCH"] = "1"
        os.environ["OPENSEARCH_ENDPOINT"]  = "localhost"
    if args.s3_folder:
        FULL_TABLES_S3_KEY = (
            f"Patterns Check Run/18/extraction/{args.s3_folder}/full_tables.json"
        )
        S3_KEY = f"Patterns Check Run/18/extraction/{args.s3_folder}"  # metadata only

    logging.basicConfig(
        level  = logging.DEBUG if args.verbose else logging.INFO,
        format = "%(asctime)s %(levelname)-8s %(message)s",
        datefmt= "%H:%M:%S",
    )

    # Validate AWS credentials are present (supports env vars AND ~/.aws/credentials)
    try:
        import boto3
        creds = boto3.Session().get_credentials()
        if creds is None:
            raise RuntimeError("no credentials")
        creds.get_frozen_credentials()  # force resolution
    except Exception as _cred_err:
        print(f"\nERROR: AWS credentials not found: {_cred_err}")
        print("  Configure via: aws configure  OR  export AWS_ACCESS_KEY_ID/SECRET_ACCESS_KEY")
        sys.exit(1)

    # ── --show-sections: local dry-run, no AWS needed ─────────────────────────
    if args.show_sections:
        print(f"\nSection tree for: {DOCUMENT_ID}")
        _doc_structure = download_doc_structure()
        _total = len(_doc_structure.get("pages", []))
        print(f"Pages 1\u2013{_total}\n")
        from shared.apryse_parser    import parse_pages
        from shared.section_chunker  import build_section_chunks, print_section_tree
        _all_pages = parse_pages(_doc_structure, 1, _total)
        _sections  = build_section_chunks(_all_pages)
        print_section_tree(_sections, verbose=args.verbose)
        return

    print(f"\n{W}")
    print("  CLINICAL CONFIDENTIAL INFORMATION RETRIEVAL ENGINE")
    print("  S3 Pipeline Test  ·  Real AWS  ·  Sequential")
    print(f"  Document   : {S3_KEY.split('/')[-1]}")
    print(f"  Pages      : all  (chunk_size={args.chunk_size})")
    os_status = "SKIP (dry-run)" if args.skip_index else OPENSEARCH_ENDPOINT
    print(f"  OpenSearch : {os_status}")
    print(W)
    # ── --force: delete existing index data for this document ─────────────────────
    if args.force and not args.skip_index:
        print(f"\n  [force] Deleting existing data for document_id={DOCUMENT_ID} …")
        _os = _build_os_client()
        _dq = {"query": {"term": {"document_id": DOCUMENT_ID}}}
        for _idx in ("document-chunks", "semantic-objects"):
            try:
                _r = _os.delete_by_query(index=_idx, body=_dq, params={"refresh": "true"})
                print(f"  [force]   {_idx}: deleted={_r.get('deleted', 0)}  "
                      f"failures={len(_r.get('failures', []))}")
            except Exception as _exc:
                print(f"  [force]   {_idx}: WARNING — {_exc}")
        print()
    doc_results: list[dict] = []


    # ── Document pipeline ─────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  DOCUMENT PIPELINE")
    print(f"{'─'*72}")

    print(f"\n  Downloading pre-computed DocStructure from S3 ...")
    doc_structure = download_doc_structure()
    total_pages   = len(doc_structure.get("pages", []))
    page_start    = 1
    page_end      = total_pages
    print(f"  Document has {total_pages} pages total")

    chunks = build_chunks(doc_structure, page_start, page_end, args.chunk_size)

    n_workers = min(args.workers, len(chunks))
    print(f"  Processing {len(chunks)} chunk(s) in parallel "
          f"({total_pages} pages, section-aware chunking, "
          f"{n_workers} worker(s))\n")

    # Pre-load all lambda modules on the main thread to avoid importlib races
    for rel, alias in [
        ("normalize", "normalize"),
        ("ner",       "ner"),
        ("ontology",  "ontology"),
        ("embedding", "embedding"),
        ("index",     "idx"),
    ]:
        _load(rel, alias)
    _build_os_client()   # resolve STS credentials once

    doc_results = [None] * len(chunks)
    _print_lock  = threading.Lock()
    _t_wall_start = time.perf_counter()

    def _process_chunk(idx_chunk: tuple[int, dict]) -> tuple[int, dict]:
        idx, ch = idx_chunk
        return idx, enrich_chunk(ch, skip_index=args.skip_index,
                                 _print_lock=_print_lock)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_process_chunk, (i, ch)): i
            for i, ch in enumerate(chunks)
        }
        for future in as_completed(futures):
            orig_idx, result = future.result()
            doc_results[orig_idx] = result
            with _print_lock:
                print_chunk(result)

    doc_results = [r for r in doc_results if r is not None]

    # ── Timing summary ────────────────────────────────────────────────────────
    _wall_time = time.perf_counter() - _t_wall_start
    all_pt     = [r.get("_pipeline_timings", {}) for r in doc_results]
    n_chunks   = len(doc_results) or 1
    stages     = ["normalize", "ner", "ontology", "embedding", "index", "total"]

    def _tot(k: str) -> float:
        return sum(p.get(k, 0.0) for p in all_pt)

    print(f"\n{'═' * 62}")
    print(f"  TIMING BREAKDOWN  ({n_chunks} chunk{'s' if n_chunks != 1 else ''},  "
          f"wall-clock {_wall_time:.1f}s)")
    print(f"{'═' * 62}")
    print(f"  {'Stage':<16}  {'Total':>9}  {'Avg/chunk':>10}  {'Share':>7}")
    print(f"  {'─' * 16}  {'─' * 9}  {'─' * 10}  {'─' * 7}")
    total_pipeline = _tot("total")
    denom = total_pipeline if total_pipeline > 0 else 1.0
    for stage in stages[:-1]:
        tot = _tot(stage)
        print(f"  {stage:<16}  {tot:>8.2f}s  {tot/n_chunks:>9.2f}s  "
              f"{100*tot/denom:>6.1f}%")
    print(f"  {'─' * 16}  {'─' * 9}  {'─' * 10}  {'─' * 7}")
    print(f"  {'TOTAL':<16}  {total_pipeline:>8.2f}s  "
          f"{total_pipeline/n_chunks:>9.2f}s  100.0%")
    print(f"{'═' * 62}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{W}")
    print("  COMPLETE")
    print(f"  Chunks enriched  : {len(doc_results)}")
    print(f"  OpenSearch write : {'SKIPPED (dry-run)' if args.skip_index else 'YES'}")
    print(W + "\n")


if __name__ == "__main__":
    main()
