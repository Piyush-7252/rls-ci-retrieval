"""
Compute per-retriever score caps from the live OpenSearch index.

Run this after a major re-index (new BM25 analyser, new embedding model,
significantly larger corpus) to update the RETRIEVER_CAP_* environment
variables used by the aggregator.

Usage:
    python tests/compute_retriever_caps.py [--percentile 95] [--sample 500]

Output:
    Prints the recommended env var exports so you can copy-paste or source them.

How it works:
    1. Samples up to --sample documents from the semantic-objects index.
    2. Runs a set of generic queries (common clinical terms) against each
       retriever and collects the raw score distributions.
    3. Computes the requested percentile per retriever and rounds up to the
       nearest 5 (so small corpus fluctuations don't flip the cap).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── Sample queries — representative of the clinical domain ───────────────────
_SAMPLE_QUERIES = [
    "overall response rate ORR",
    "progression-free survival PFS",
    "dose escalation cohort RP2D",
    "adverse events safety tolerability",
    "primary endpoint efficacy",
    "treatment-naive relapsed refractory",
    "complete response stringent CR",
    "pharmacokinetics AUC Cmax",
    "inclusion exclusion criteria",
    "minimal residual disease MRD",
]

# ── BM25 and literal search index names ──────────────────────────────────────
SEMANTIC_OBJECTS_INDEX = os.environ.get("SEMANTIC_OBJECTS_INDEX", "semantic-objects")
DOCUMENT_CHUNKS_INDEX  = os.environ.get("OPENSEARCH_INDEX",       "document-chunks")
AWS_REGION             = os.environ.get("AWS_REGION",             "eu-west-1")
OPENSEARCH_ENDPOINT    = os.environ.get(
    "OPENSEARCH_ENDPOINT",
    "search-rls-dev-rhitzxwnctmuyq2l4kny5kwelu.eu-west-1.es.amazonaws.com",
)


def _get_os():
    import boto3
    from opensearchpy import OpenSearch, RequestsHttpConnection
    from requests_aws4auth import AWS4Auth
    frozen  = boto3.Session().get_credentials().get_frozen_credentials()
    auth    = AWS4Auth(frozen.access_key, frozen.secret_key, AWS_REGION, "es",
                       session_token=frozen.token)
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_ENDPOINT, "port": 443}],
        http_auth=auth, use_ssl=True, verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=30,
    )


def _collect_bm25_scores(osc, query_text: str, top_k: int = 20) -> list[float]:
    """BM25 search against semantic-objects; return raw _score list."""
    try:
        resp = osc.search(index=SEMANTIC_OBJECTS_INDEX, body={
            "size": top_k,
            "query": {"multi_match": {
                "query":  query_text,
                "fields": ["text^2"],
                "type":   "best_fields",
            }},
            "_source": ["object_id"],
        }, params={"request_timeout": 15})
        return [h["_score"] for h in resp["hits"]["hits"]]
    except Exception as exc:
        print(f"  [warn] bm25 query failed: {exc}", file=sys.stderr)
        return []


def _collect_literal_scores(osc, query_text: str, top_k: int = 20) -> list[float]:
    """Phrase search against document-chunks; return raw _score list."""
    try:
        resp = osc.search(index=DOCUMENT_CHUNKS_INDEX, body={
            "size": top_k,
            "query": {"match_phrase": {
                "raw_text": {"query": query_text, "slop": 2},
            }},
            "_source": ["chunk_id"],
        }, params={"request_timeout": 15})
        return [h["_score"] for h in resp["hits"]["hits"]]
    except Exception as exc:
        print(f"  [warn] literal query failed: {exc}", file=sys.stderr)
        return []


def _collect_fact_scores(osc, query_text: str, top_k: int = 20) -> list[float]:
    """Facts-field search against semantic-objects; return raw _score list."""
    try:
        resp = osc.search(index=SEMANTIC_OBJECTS_INDEX, body={
            "size": top_k,
            "query": {"multi_match": {
                "query":  query_text,
                "fields": ["facts.drug^3", "facts.endpoint^2", "facts.adverse_event"],
                "type":   "cross_fields",
            }},
            "_source": ["object_id"],
        }, params={"request_timeout": 15})
        return [h["_score"] for h in resp["hits"]["hits"]]
    except Exception as exc:
        print(f"  [warn] fact query failed: {exc}", file=sys.stderr)
        return []


def _collect_ontology_scores(osc, query_text: str, top_k: int = 20) -> list[float]:
    """Fuzzy/synonym search against semantic-objects; return raw _score list."""
    try:
        resp = osc.search(index=SEMANTIC_OBJECTS_INDEX, body={
            "size": top_k,
            "query": {"multi_match": {
                "query":    query_text,
                "fields":   ["text"],
                "fuzziness": "AUTO",
                "type":     "best_fields",
            }},
            "_source": ["object_id"],
        }, params={"request_timeout": 15})
        return [h["_score"] for h in resp["hits"]["hits"]]
    except Exception as exc:
        print(f"  [warn] ontology query failed: {exc}", file=sys.stderr)
        return []


def _percentile(data: list[float], p: float) -> float:
    """p-th percentile (0–100) of data."""
    if not data:
        return 0.0
    data_sorted = sorted(data)
    idx = (p / 100) * (len(data_sorted) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(data_sorted) - 1)
    frac = idx - lo
    return data_sorted[lo] * (1 - frac) + data_sorted[hi] * frac


def _round_up_to_5(value: float) -> float:
    """Round up to the nearest multiple of 5 so minor corpus changes don't flip the cap."""
    return math.ceil(value / 5) * 5.0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--percentile", type=float, default=95.0,
                        help="Score percentile to use as the cap (default: 95)")
    args = parser.parse_args()

    print(f"Connecting to OpenSearch at {OPENSEARCH_ENDPOINT} ...")
    osc = _get_os()

    all_scores: dict[str, list[float]] = {
        "bm25": [], "literal": [], "fact": [], "ontology": [],
    }

    for i, q in enumerate(_SAMPLE_QUERIES, 1):
        print(f"  [{i}/{len(_SAMPLE_QUERIES)}] querying: {q!r}")
        all_scores["bm25"]    += _collect_bm25_scores(osc, q)
        all_scores["literal"] += _collect_literal_scores(osc, q)
        all_scores["fact"]    += _collect_fact_scores(osc, q)
        all_scores["ontology"]+= _collect_ontology_scores(osc, q)

    print()
    p = args.percentile
    caps: dict[str, float] = {}
    for retriever, scores in all_scores.items():
        if not scores:
            print(f"  {retriever:10s}: no data — keeping default")
            continue
        raw_cap = _percentile(scores, p)
        cap     = _round_up_to_5(raw_cap)
        caps[retriever] = cap
        print(
            f"  {retriever:10s}: n={len(scores):4d}  "
            f"min={min(scores):.1f}  mean={statistics.mean(scores):.1f}  "
            f"p{int(p)}={raw_cap:.1f}  → cap={cap:.0f}"
        )

    print()
    print("# ── Recommended environment variables ──────────────────────────────")
    env_map = {
        "bm25":    "RETRIEVER_CAP_BM25",
        "literal": "RETRIEVER_CAP_LITERAL",
        "fact":    "RETRIEVER_CAP_FACT",
        "ontology":"RETRIEVER_CAP_ONTOLOGY",
    }
    for retriever, env_var in env_map.items():
        cap = caps.get(retriever)
        if cap:
            print(f'export {env_var}="{cap:.0f}"')
    print()
    print("# Vector, NER, and regex scores are already 0–1; caps stay at 1.0.")


if __name__ == "__main__":
    main()
