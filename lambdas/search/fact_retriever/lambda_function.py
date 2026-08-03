"""
Search Pipeline — Fact Retriever
==================================
Queries the semantic-objects index on pre-computed `facts.*` fields rather than
raw text, giving high-precision retrieval before semantic search even begins.

How it works
------------
At indexing time, every semantic object is enriched by `enrich_object` which
extracts structured clinical facts: drug names, endpoints, study arms, etc.
CIs are enriched the same way at CI-indexing time.

At search time this retriever reads the CI's `facts` dict and builds an
OpenSearch query that matches objects sharing those clinical entities:

    facts.drug:     "teclistamab"
    facts.endpoint: "ORR"

High-specificity slots (drug, study_id, study_arm) must match at least one
term — this stops the retriever from returning everything when only weak
facts are present.  All matched slots boost the score proportionally.

Input:  classified search request (ci.facts must be populated)
Output: { "retriever": "fact", "hits": list[Hit] }
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OPENSEARCH_ENDPOINT    = os.environ.get("OPENSEARCH_ENDPOINT", "localhost")
SEMANTIC_OBJECTS_INDEX = os.environ.get("SEMANTIC_OBJECTS_INDEX", "semantic-objects")
AWS_REGION             = os.environ.get("AWS_REGION", "us-east-1")
TOP_K                  = int(os.environ.get("RETRIEVER_TOP_K", "10"))

_os_client = None

def _get_os():
    global _os_client
    if _os_client is None:
        import boto3
        from opensearchpy import OpenSearch, RequestsHttpConnection
        from requests_aws4auth import AWS4Auth
        frozen  = boto3.Session().get_credentials().get_frozen_credentials()
        awsauth = AWS4Auth(frozen.access_key, frozen.secret_key, AWS_REGION, "es",
                          session_token=frozen.token)
        _os_client = OpenSearch(
            hosts=[{"host": OPENSEARCH_ENDPOINT, "port": 443}],
            http_auth=awsauth, use_ssl=True, verify_certs=True,
            connection_class=RequestsHttpConnection,
        )
    return _os_client


# ─── Slot configuration ───────────────────────────────────────────────────────

# Slots that indicate a specific clinical entity — at least one must match
# (used as a hard filter to prevent retrieving everything when only weak facts exist)
_HIGH_SPECIFICITY_SLOTS: frozenset[str] = frozenset({"drug", "study_id", "study_arm"})

# Score boosts per slot — mirrors how clinically specific each entity type is
_SLOT_BOOST: dict[str, float] = {
    "drug":               4.0,
    "study_id":           4.0,
    "study_arm":          3.0,
    "endpoint":           2.5,
    "adverse_event":      2.0,
    "biomarker":          2.0,
    "study_population":   1.5,
    "dose":               1.5,
    "phase":              1.2,
    "response_criterion": 1.0,
    "statistical_method": 1.0,
    # "action" intentionally excluded — too generic
}

# Entity slots that can appear as keys in a clinical_relation dict
_RELATION_ENTITY_SLOTS: frozenset[str] = frozenset(_SLOT_BOOST)

# Relation search field boosts
_REL_DRUG_BOOST = float(os.environ.get("FACT_REL_DRUG_BOOST", "3.0"))
_REL_SLOT_BOOST = float(os.environ.get("FACT_REL_SLOT_BOOST", "2.0"))
_REL_TYPE_BOOST = float(os.environ.get("FACT_REL_TYPE_BOOST", "1.5"))

# Merge: combined = max_score + MERGE_ALPHA * second_best_score
MERGE_ALPHA = float(os.environ.get("FACT_MERGE_ALPHA", "0.30"))

# Whether to use Painless script confidence boosting in the relation query
USE_CONFIDENCE_BOOST = os.environ.get("FACT_USE_CONFIDENCE_BOOST", "true").lower() == "true"

# Anchor priority for relation search: highest-priority entity present becomes the must clause
_ANCHOR_PRIORITY: list[str] = [
    "study_id",          # most specific — uniquely identifies one trial
    "drug",              # eliminates cross-drug contamination
    "study_arm",         # discriminates between arms of the same trial
    "endpoint",          # good fallback when no drug/arm present
    "adverse_event",
    "biomarker",
    "study_population",
    "dose",
    "phase",
    "response_criterion",
    "statistical_method",
]

# _source fields requested from every OpenSearch search
# Semantic layer contract: effective_facts + canonical identity only.
# Raw entities, neighbor text, and provenance fields are excluded here;
# the reranker and verifier read those from the full document if needed.
_SOURCE_FIELDS: list[str] = [
    "object_id", "parent_chunk_id", "document_id",
    "position", "global_position", "document_position",
    "type", "text", "page", "bbox",
    "display_spans",
    "section_category", "heading_path", "semantic_path", "section_confidence",
    "prev_sentence_text", "next_sentence_text", "paragraph_text",
    # NER
    "entities",
    # raw + propagated fact slots
    "facts", "own_facts", "effective_facts", "inherited_slots", "slot_provenance",
    # classification
    "study_context", "statement_type", "object_subtype", "modality",
    # relations
    "clinical_relations",
    # identity layer
    "clinical_identity", "treatment_identity", "endpoint_identity",
    "population_identity", "temporal_context",
    # structural / provenance
    "study_hierarchy", "negated_slots", "clinical_signature",
    # numeric / statistical
    "statistical_identity",
]


# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    search_id = event.get("search_id", "unknown")
    logger.info("[Fact Retriever] start search_id=%s", search_id)
    try:
        result = _process(event)
    except Exception as exc:
        logger.error("[Fact Retriever] failed search_id=%s error=%s", search_id, exc)
        raise
    logger.info("[Fact Retriever] done search_id=%s hits=%d", search_id, len(result["hits"]))
    return result


def _process(req: dict) -> dict:
    ci               = req.get("ci", {})
    # effective_facts includes inherited heading/paragraph context — more complete
    # than own facts alone; fall back to facts if not yet populated.
    ci_facts         = ci.get("effective_facts") or ci.get("facts", {})
    ci_relations     = ci.get("clinical_relations", [])
    ci_stmt_type     = ci.get("statement_type", "")
    ci_study_context = ci.get("study_context", "GENERAL")
    document_id      = req.get("document_id")

    # Stage 1: flat-fact + statement_type + study_context search
    fact_hits = _fact_search(ci_facts, ci_stmt_type, ci_study_context, document_id)

    # Stage 2: clinical-relation search (skip if no relations extracted)
    rel_hits  = _relation_search(ci_relations, document_id) if ci_relations else []

    return {
        "retriever": "fact",
        "hits":      _merge_hits(fact_hits, rel_hits)[:TOP_K],
    }


# ─── Stage 1: flat-fact search ───────────────────────────────────────────────

def _fact_search(
    ci_facts:         dict,
    ci_stmt_type:     str,
    ci_study_context: str,
    document_id:      str | None,
) -> list[dict]:
    body = _build_fact_query(ci_facts, ci_stmt_type, ci_study_context, document_id)
    if body is None:
        logger.debug("[Fact Retriever] no usable facts — skipping stage 1")
        return []
    try:
        resp = _get_os().search(index=SEMANTIC_OBJECTS_INDEX, body=body)
    except Exception as exc:
        logger.warning("[Fact Retriever] fact search failed: %s", exc)
        return []
    return _parse_hits(resp)


def _build_fact_query(
    ci_facts:         dict,
    ci_stmt_type:     str,
    ci_study_context: str,
    document_id:      str | None,
) -> dict | None:
    """
    Stage 1 query: one `match` clause per entity value (not joined into a
    single string), plus statement_type and study_context boosts.
    """
    usable = {slot: vals for slot, vals in ci_facts.items() if slot in _SLOT_BOOST and vals}
    if not usable:
        return None

    filter_clause = [{"term": {"document_id.keyword": document_id}}] if document_id else []
    should_clauses: list[dict] = []

    # One clause per value, per slot — not joined
    for slot, values in usable.items():
        boost = _SLOT_BOOST[slot]
        for v in values:
            should_clauses.append({"match": {f"effective_facts.{slot}": {"query": v, "boost": boost}}})

    # statement_type alignment
    if ci_stmt_type:
        should_clauses.append(
            {"match": {"statement_type": {"query": ci_stmt_type, "boost": 2.0}}}
        )

    # study_context preference — soft boost for CURRENT, not a hard filter
    if ci_study_context in ("CURRENT", "GENERAL"):
        should_clauses.append(
            {"term": {"study_context": {"value": "CURRENT", "boost": 1.5}}}
        )

    # Hard anchor: at least one high-specificity slot must match
    high_spec = [s for s in usable if s in _HIGH_SPECIFICITY_SLOTS]
    must_clauses: list[dict] = []
    if high_spec:
        must_clauses.append({
            "bool": {
                "should": [
                    {"match": {f"effective_facts.{s}": {"query": v}}}
                    for s in high_spec for v in usable[s]
                ],
                "minimum_should_match": 1,
            }
        })

    bool_query: dict = {"filter": filter_clause, "should": should_clauses}
    if must_clauses:
        bool_query["must"] = must_clauses
    else:
        bool_query["minimum_should_match"] = 1

    return {"size": TOP_K, "query": {"bool": bool_query}, "_source": _SOURCE_FIELDS}


# ─── Stage 2: clinical-relation search ─────────────────────────────────────────

def _relation_search(ci_relations: list[dict], document_id: str | None) -> list[dict]:
    body = _build_relation_query(ci_relations, document_id)
    if body is None:
        return []
    try:
        resp = _get_os().search(index=SEMANTIC_OBJECTS_INDEX, body=body)
    except Exception as exc:
        logger.warning("[Fact Retriever] relation search failed: %s", exc)
        return []
    return _parse_hits(resp)


def _build_relation_query(ci_relations: list[dict], document_id: str | None) -> dict | None:
    """
    Stage 2 query: one `bool` clause per CI relation, each anchored on the drug
    and boosted by the other entity slot and relation type.

    All per-relation clauses are wrapped in a top-level `should` so a document
    matching ANY CI relation is retrieved.

    Confidence boosting
    -------------------
    When USE_CONFIDENCE_BOOST is true, wraps the query in a `function_score`
    that uses a Painless script to find the max confidence across all
    clinical_relations in the document and adds it to the relevance score.
    This means a document where the relation was extracted with confidence=0.97
    ranks above one where it was 0.52, all else being equal.
    Gracefully disabled via FACT_USE_CONFIDENCE_BOOST=false if the field is
    not indexed as a numeric type in the cluster.
    """
    if not ci_relations:
        return None

    filter_clause = [{"term": {"document_id.keyword": document_id}}] if document_id else []
    per_relation_clauses: list[dict] = []

    for rel in ci_relations:
        anchor = _pick_relation_anchor(rel)
        if anchor is None:
            continue  # no usable entity in this relation — skip
        anchor_slot, anchor_value = anchor

        rel_type    = rel.get("relation", "")
        other_slots = [
            (k, v) for k, v in rel.items()
            if k in _RELATION_ENTITY_SLOTS and k != anchor_slot
            and isinstance(v, str) and v
        ]

        should_sub: list[dict] = []
        if rel_type:
            should_sub.append({
                "match": {"clinical_relations.relation": {"query": rel_type, "boost": _REL_TYPE_BOOST}}
            })
        for slot, value in other_slots:
            should_sub.append({
                "match": {f"clinical_relations.{slot}": {"query": value, "boost": _REL_SLOT_BOOST}}
            })

        clause: dict = {
            "bool": {
                "must": [
                    {"match": {f"clinical_relations.{anchor_slot}": {"query": anchor_value, "boost": _REL_DRUG_BOOST}}}
                ],
            }
        }
        if should_sub:
            clause["bool"]["should"] = should_sub
        per_relation_clauses.append(clause)

    if not per_relation_clauses:
        return None

    bool_body: dict = {
        "filter":               filter_clause,
        "should":               per_relation_clauses,
        "minimum_should_match": 1,
    }

    if USE_CONFIDENCE_BOOST:
        # Add max(clinical_relations.confidence) to the relevance score.
        # The script returns 0.5 (neutral) when the field is absent or empty.
        query_body: dict = {
            "function_score": {
                "query": {"bool": bool_body},
                "functions": [{
                    "script_score": {
                        "script": {
                            "lang": "painless",
                            "source": (
                                "double best = 0.5;"
                                "if (doc.containsKey('clinical_relations.confidence')"
                                "    && doc['clinical_relations.confidence'].size() > 0) {"
                                "  for (double c : doc['clinical_relations.confidence'])"
                                "    { if (c > best) best = c; }"
                                "}"
                                "return best;"
                            ),
                        }
                    },
                    "weight": 1.0,
                }],
                "boost_mode": "sum",
                "score_mode": "sum",
            }
        }
    else:
        query_body = {"bool": bool_body}

    return {"size": TOP_K, "query": query_body, "_source": _SOURCE_FIELDS}


# ─── Merge + parse ────────────────────────────────────────────────────────────

def _merge_hits(fact_hits: list[dict], rel_hits: list[dict]) -> list[dict]:
    """
    Merge stage 1 and stage 2 hits with a combined score formula:

        combined = max_score + MERGE_ALPHA * second_best_score

    When a chunk appears in both stages the second signal adds partial
    evidence rather than being discarded.  MERGE_ALPHA=0.30 means the
    relation score contributes 30% of its value on top of the fact score
    (or vice versa).  Set FACT_MERGE_ALPHA=0 to restore pure-max behaviour.

    The matched_object from the higher-scoring stage is kept.
    """
    # Collect all scores per chunk, retaining the best matched_object
    by_chunk: dict[str, dict] = {}
    for h in fact_hits + rel_hits:
        cid = h.get("chunk_id", "")
        if cid not in by_chunk:
            by_chunk[cid] = {"best_hit": h, "scores": [h["score"]]}
        else:
            by_chunk[cid]["scores"].append(h["score"])
            if h["score"] > by_chunk[cid]["best_hit"]["score"]:
                by_chunk[cid]["best_hit"] = h

    merged: list[dict] = []
    for data in by_chunk.values():
        scores  = sorted(data["scores"], reverse=True)
        combined = scores[0] + MERGE_ALPHA * scores[1] if len(scores) > 1 else scores[0]
        merged.append({**data["best_hit"], "score": round(combined, 4)})

    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged


def _pick_relation_anchor(rel: dict) -> tuple[str, str] | None:
    """
    Return (slot, value) for the highest-priority entity present in this relation.

    Using study_id or drug as the anchor when available gives the tightest
    filter.  Falling through to endpoint or AE handles CI types where the
    relation is not drug-centric (biomarker CIs, endpoint-only queries, etc.).
    Returns None if no recognised entity is found.
    """
    for slot in _ANCHOR_PRIORITY:
        v = rel.get(slot, "")
        if isinstance(v, str) and v:
            return slot, v
    return None


def _parse_hits(resp: dict) -> list[dict]:
    hits = []
    for h in resp.get("hits", {}).get("hits", []):
        src = h.get("_source", {})
        hits.append({
            "chunk_id":   src.get("parent_chunk_id", ""),
            "score":      round(h.get("_score", 0.0), 4),
            "page_start": src.get("page", 0),
            "page_end":   src.get("page", 0),
            "snippet":    src.get("text", "")[:200],
            "matched_object": {
                "object_id":           src.get("object_id"),
                "parent_chunk_id":     src.get("parent_chunk_id"),
                "document_id":         src.get("document_id"),
                "position":            src.get("position"),
                "global_position":     src.get("global_position"),
                "document_position":   src.get("document_position"),
                "type":                src.get("type"),
                "text":                src.get("text"),
                "page":                src.get("page"),
                "bbox":                src.get("bbox", []),
                "display_spans":       src.get("display_spans", []),
                "section_category":    src.get("section_category"),
                "heading_path":        src.get("heading_path"),
                "semantic_path":       src.get("semantic_path"),
                "section_confidence":  src.get("section_confidence"),
                "prev_sentence_text":  src.get("prev_sentence_text"),
                "next_sentence_text":  src.get("next_sentence_text"),
                "paragraph_text":      src.get("paragraph_text"),
                # NER + raw facts
                "entities":            src.get("entities", []),
                "facts":               src.get("facts", {}),
                "own_facts":           src.get("own_facts", {}),
                # Semantic layer — full ENRICHMENT_DEFAULTS parity with bm25_retriever
                "effective_facts":     src.get("effective_facts", {}),
                "inherited_slots":     src.get("inherited_slots", []),
                "slot_provenance":     src.get("slot_provenance", {}),
                "clinical_identity":   src.get("clinical_identity", {}),
                "treatment_identity":  src.get("treatment_identity", {}),
                "endpoint_identity":   src.get("endpoint_identity", {}),
                "population_identity": src.get("population_identity", {}),
                "temporal_context":    src.get("temporal_context", {}),
                "modality":            src.get("modality", "GENERAL"),
                "object_subtype":      src.get("object_subtype", "GENERAL"),
                "clinical_relations":  src.get("clinical_relations", []),
                "statement_type":      src.get("statement_type"),
                "study_context":       src.get("study_context", "GENERAL"),
                "study_hierarchy":     src.get("study_hierarchy", {}),
                "negated_slots":       src.get("negated_slots", []),
                "clinical_signature":  src.get("clinical_signature", {}),
                "statistical_identity": src.get("statistical_identity", {}),
            },
        })
    return hits
