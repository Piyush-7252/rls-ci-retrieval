"""
export_results_csv.py
---------------------
Convert a search-result JSON (produced by search_test.py) into a flat CSV.

Row expansion
-------------
  rows = sum(CIs) x sum(verified candidates per CI)

  Every verified candidate gets its own row - YES/MAYBE -> FINAL,
  NO -> REJECTED, SKIP -> SKIPPED.  CIs with zero candidates get a single
  NO_HIT placeholder row.

  Legacy JSON (no candidates[] array) falls back to
  final_hits + rejected_hits + skipped_hits.

Column groups (~200+ columns)
------------------------------
  ci_*              CI identity, NER, facts, ontology, clinical relations
  hit_*             Verdict, page, match span, evidence classification
  cmp_{dim}_*       Comparator per-dimension: outcome, score, severity,
                    ci_values, candidate_values, ci_group, candidate_group,
                    slot, conflicts, relation  (11 dims x 12 fields)
  sb_*              score_breakdown scalar fields (all flattened individually)
  sb_cr_*           clinical_reasoning (matched, conflicts, decision)
  sb_struct_*       struct_detail flattened
  sb_contra_detail  contra_detail as compact JSON
  agg_*             agg_score_breakdown (all fields individually)
  eff_ci_*          effective CI facts from score_breakdown.ci_facts
  eff_cand_*        effective candidate facts from score_breakdown.cand_facts
  enr_ci_*          enrichment_status per CI field (T/F per enrichment slot)
  enr_cand_*        enrichment_status per candidate field
  enrd_ci_*         enrichment_diagnostics.ci (stage, status, reason, missing)
  enrd_cand_*       enrichment_diagnostics.candidate
  ctx_*             Full text: prev_sentence, object, next_sentence,
                    paragraph, chunk, heading_path, section, semantic_path
  cand_*            Candidate NER, facts, clinical relations

Usage
-----
  python tools/export_results_csv.py localfiles/search_results/v45.json
  python tools/export_results_csv.py v45.json --out v45.csv
  python tools/export_results_csv.py v45.json --out -   # stdout
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _join(vals: Any, sep: str = " | ") -> str:
    if vals is None:
        return ""
    if isinstance(vals, (list, tuple)):
        return sep.join(str(v) for v in vals if v is not None)
    if isinstance(vals, set):
        return sep.join(sorted(str(v) for v in vals))
    return str(vals)


def _facts_str(facts: dict | None, slot: str) -> str:
    if not facts:
        return ""
    return _join(facts.get(slot, []))


def _ner_entities(entities: list[dict]) -> str:
    return " | ".join(
        f"{e.get('text', '')}:{e.get('label', e.get('type', '?'))}"
        for e in (entities or [])
    )


def _ner_labels(entities: list[dict]) -> str:
    labels = sorted({e.get("label", e.get("type", "")) for e in (entities or []) if e})
    return " | ".join(labels)


def _relations_str(rels: list[dict]) -> str:
    parts = []
    for r in (rels or []):
        rel   = r.get("relation", "?")
        slots = {k: v for k, v in r.items() if k not in {"relation", "confidence", "verb"}}
        vals  = list(slots.values())
        left  = vals[0] if vals else ""
        right = vals[1] if len(vals) > 1 else ""
        parts.append(f"{left}->{rel}->{right}")
    return " | ".join(parts)


def _ontology_terms(expansions: Any) -> str:
    if not expansions:
        return ""
    if isinstance(expansions, list):
        terms: list[str] = []
        for item in expansions:
            if isinstance(item, dict):
                terms.append(item.get("term") or item.get("text") or str(item))
            else:
                terms.append(str(item))
        return " | ".join(terms)
    if isinstance(expansions, dict):
        return " | ".join(str(v) for v in expansions.values())
    return str(expansions)


def _f(val: Any) -> str:
    if val is None:
        return ""
    return str(round(val, 6)) if isinstance(val, float) else str(val)


def _b(val: Any) -> str:
    if val is None:
        return ""
    return str(val)


def _compact(val: Any) -> str:
    if val is None or val == {} or val == []:
        return ""
    try:
        return json.dumps(val, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(val)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FACT_SLOTS = [
    "drug", "endpoint", "dose", "population", "biomarker",
    "adverse_event", "response_criterion", "phase", "study_arm",
    "treatment_regimen", "statistical_method",
    # additional slots found in data
    "sample_size", "action", "study_id", "assessment", "disease",
]

_CMP_DIMS = [
    "drug", "endpoint", "population", "phase", "biomarker",
    "study_arm", "modality", "regimen", "temporal", "negation", "statistical",
]

_SB_SCALAR_KEYS = [
    "ce", "retrieval", "drug_identity", "intent_align", "entity_family",
    "fact_slot", "study_id", "source", "entity_term", "asset", "pos_bonus",
    "struct_penalty", "contra_penalty", "interaction", "gran_factor",
    "section_mult", "profile", "composite", "pre_veto_composite",
    "validation_severity",
]

_AGG_SCALAR_KEYS = [
    "vector", "bm25", "literal", "ontology", "fact_ret", "numeric_ret",
    "ret_component", "entity_olap", "fact_olap", "relation", "gran",
    "contradiction", "identity_overlap", "identity_bonus",
    "zero_id_pen", "zero_enrich_pen", "sect_drift_pen",
    "raw", "sect_mult", "ctx_mult",
]

_ENR_STATUS_FIELDS = [
    "entities", "facts", "effective_facts",
    "clinical_identity", "treatment_identity", "endpoint_identity",
    "population_identity", "clinical_relations",
    "statement_type", "modality",
]


# ---------------------------------------------------------------------------
# CI block
# ---------------------------------------------------------------------------

def _ci_block(ci: dict) -> dict:
    facts     = ci.get("facts") or {}
    own_facts = ci.get("own_facts") or {}
    entities  = ci.get("entities", [])
    rels      = ci.get("clinical_relations", [])
    ont_exp   = ci.get("ontology_expansions") or ci.get("ontology", {})
    if isinstance(ont_exp, dict):
        ont_exp = ont_exp.get("expansions") or ont_exp.get("synonyms") or []

    assets = _join([a.get("code", a.get("name", "")) for a in ci.get("assets", []) if a])
    ns = ci.get("negated_slots") or []
    neg = _join(list(ns.keys()) if isinstance(ns, dict) else ns)

    cat = ci.get("category")
    cat_str = cat.get("name", "") if isinstance(cat, dict) else str(cat or "")

    row: dict[str, str] = {
        "ci_id":                 str(ci.get("id", "")),
        "ci_text":               (ci.get("text") or "").replace("\n", " ").strip(),
        "ci_type":               ci.get("ci_type", ""),
        "ci_category":           cat_str,
        "ci_assets":             assets,
        "ci_strategies":         _join(ci.get("strategies", [])),
        "ci_modality":           ci.get("modality", ""),
        "ci_statement_type":     ci.get("statement_type", ""),
        "ci_negated_slots":      neg,
        "ci_ner_model":          ci.get("ner_model", ""),
        "ci_ner_labels":         _ner_labels(entities),
        "ci_ner_entities":       _ner_entities(entities),
        "ci_ontology_terms":     _ontology_terms(ont_exp),
        "ci_clinical_relations": _relations_str(rels),
        "ci_justification":      (ci.get("justification_text") or "").replace("\n", " ")[:400],
    }
    for slot in _FACT_SLOTS:
        row[f"ci_facts_{slot}"]     = _facts_str(facts, slot)
        row[f"ci_own_facts_{slot}"] = _facts_str(own_facts, slot)
    return row


# ---------------------------------------------------------------------------
# Candidate block - everything flattened
# ---------------------------------------------------------------------------

def _candidate_block(cand: dict, hit_type: str, rank: int) -> dict:
    sb      = cand.get("score_breakdown") or {}
    asb     = cand.get("agg_score_breakdown") or {}
    ct      = sb.get("comparator_trace") or {}
    io      = cand.get("indexed_object") or {}
    io_f    = io.get("facts") or {}
    io_e    = io.get("entities") or []
    io_r    = io.get("clinical_relations") or []
    eff_ci  = sb.get("ci_facts") or {}
    eff_cnd = sb.get("cand_facts") or {}
    cr      = sb.get("clinical_reasoning") or {}
    enr_raw = sb.get("enrichment_status") or {}
    enrd    = sb.get("enrichment_diagnostics") or {}

    row: dict[str, str] = {}

    # Hit identity
    row["hit_type"]    = hit_type
    row["hit_rank"]    = str(rank)
    row["chunk_id"]    = cand.get("chunk_id", "")
    row["page_start"]  = str(cand.get("page_start", ""))
    row["page_end"]    = str(cand.get("page_end", ""))
    row["match_page"]  = str(cand.get("match_page", ""))
    row["sources"]     = _join(cand.get("sources", []))
    row["retriever"]   = cand.get("retriever", "")
    row["verdict"]         = cand.get("verdict", "")
    row["confidence"]      = _f(cand.get("confidence"))
    row["verifier_reason"] = (cand.get("verifier_reason") or cand.get("reason", ""))[:400]
    row["evidence_type"]       = cand.get("evidence_type", "")
    row["evidence_confidence"] = _f(cand.get("evidence_confidence"))
    row["evidence_reason"]     = (cand.get("evidence_reason") or "")[:300]
    row["cross_encoder_score"] = _f(cand.get("cross_encoder_score"))
    row["agg_score"]           = _f(cand.get("agg_score"))
    row["match_span"]       = (cand.get("match_span") or "").strip()
    row["context_sentence"] = (cand.get("context_sentence") or "").replace("\n", " ")[:500]
    row["highlight_score"]  = _f(cand.get("highlight_score"))
    row["match_method"]     = cand.get("match_method", "")
    row["retrieval_object_type"]  = cand.get("retrieval_object_type", "")
    row["retrieval_object_id"]    = cand.get("retrieval_object_id", "")
    row["retrieval_section"]      = cand.get("retrieval_section", "")
    row["retrieval_heading_path"] = _join(cand.get("retrieval_heading_path") or [])
    row["rejection_reason"] = (cand.get("reason") or "")[:300]

    # Comparator trace - every dimension fully flattened
    for dim in _CMP_DIMS:
        entry = ct.get(dim) or {}
        ev    = entry.get("evidence") or {}
        row[f"cmp_{dim}_outcome"]          = entry.get("outcome", "")
        row[f"cmp_{dim}_score"]            = _f(entry.get("score"))
        row[f"cmp_{dim}_severity"]         = entry.get("severity", "")
        row[f"cmp_{dim}_ci_values"]        = _join(ev.get("ci") or ev.get("ci_values") or [])
        row[f"cmp_{dim}_candidate_values"] = _join(ev.get("candidate") or ev.get("candidate_values") or [])
        row[f"cmp_{dim}_ci_group"]         = ev.get("ci_group", "")
        row[f"cmp_{dim}_candidate_group"]  = ev.get("candidate_group", "")
        row[f"cmp_{dim}_ci_modality"]      = ev.get("ci_modality", "")
        row[f"cmp_{dim}_cand_modality"]    = ev.get("candidate_modality", "")
        row[f"cmp_{dim}_slot"]             = ev.get("slot", "")
        row[f"cmp_{dim}_conflicts"]        = _join(ev.get("conflicts") or [])
        row[f"cmp_{dim}_relation"]         = ev.get("relation", "")

    # score_breakdown scalars
    for key in _SB_SCALAR_KEYS:
        row[f"sb_{key}"] = _f(sb.get(key))

    # clinical_reasoning
    if isinstance(cr, dict):
        row["sb_cr_matched"]   = _join(cr.get("matched") or [])
        row["sb_cr_conflicts"] = _join(cr.get("conflicts") or [])
        row["sb_cr_decision"]  = (cr.get("decision") or "")[:400]
    else:
        row["sb_cr_matched"]   = ""
        row["sb_cr_conflicts"] = ""
        row["sb_cr_decision"]  = str(cr)[:400]

    # struct_detail
    sd = sb.get("struct_detail") or {}
    mr = sd.get("missing_relation") or {}
    ms = sd.get("missing_slots") or {}
    row["sb_struct_missing_relation_weight"]    = _f(mr.get("weight"))
    row["sb_struct_missing_relation_severity"]  = mr.get("severity", "")
    row["sb_struct_missing_relation_ci_rels"]   = _f((mr.get("evidence") or {}).get("ci_relations"))
    row["sb_struct_missing_relation_cand_rels"] = _f((mr.get("evidence") or {}).get("candidate_relations"))
    row["sb_struct_missing_slots_weight"]       = _f(ms.get("weight"))
    row["sb_struct_missing_slots_severity"]     = ms.get("severity", "")
    row["sb_struct_missing_slots_list"]         = _join((ms.get("evidence") or {}).get("missing_slots") or [])
    row["sb_struct_detail_json"]                = _compact(sd)
    row["sb_contra_detail_json"]                = _compact(sb.get("contra_detail"))
    row["sb_enrichment_status_json"]            = _compact(enr_raw)

    # agg_score_breakdown scalars
    for key in _AGG_SCALAR_KEYS:
        row[f"agg_{key}"] = _f(asb.get(key))
    row["agg_contra_detail_json"] = _compact(asb.get("contra_detail"))

    # Effective facts from comparator
    for slot in _FACT_SLOTS:
        row[f"eff_ci_{slot}"]   = _facts_str(eff_ci, slot)
        row[f"eff_cand_{slot}"] = _facts_str(eff_cnd, slot)

    # Enrichment status - per field, per side
    enr_ci   = (enr_raw.get("ci")        or {}) if isinstance(enr_raw, dict) else {}
    enr_cand = (enr_raw.get("candidate") or {}) if isinstance(enr_raw, dict) else {}
    for field in _ENR_STATUS_FIELDS:
        row[f"enr_ci_{field}"]   = _b(enr_ci.get(field))
        row[f"enr_cand_{field}"] = _b(enr_cand.get(field))

    # Enrichment diagnostics
    enrd_ci   = (enrd.get("ci")        or {}) if isinstance(enrd, dict) else {}
    enrd_cand = (enrd.get("candidate") or {}) if isinstance(enrd, dict) else {}
    row["enrd_ci_stage"]           = enrd_ci.get("stage", "")
    row["enrd_ci_status"]          = enrd_ci.get("status", "")
    row["enrd_ci_reason"]          = (enrd_ci.get("reason") or "")[:300]
    row["enrd_ci_missing_slots"]   = _join(enrd_ci.get("missing_slots") or [])
    row["enrd_cand_stage"]         = enrd_cand.get("stage", "")
    row["enrd_cand_status"]        = enrd_cand.get("status", "")
    row["enrd_cand_reason"]        = (enrd_cand.get("reason") or "")[:300]
    row["enrd_cand_missing_slots"] = _join(enrd_cand.get("missing_slots") or [])

    # Full text context
    row["ctx_prev_sentence"]   = (io.get("prev_sentence_text") or "").replace("\n", " ")
    row["ctx_object_text"]     = (io.get("text") or "").replace("\n", " ")
    row["ctx_next_sentence"]   = (io.get("next_sentence_text") or "").replace("\n", " ")
    row["ctx_paragraph_text"]  = (io.get("paragraph_text") or "").replace("\n", " ")
    row["ctx_chunk_text"]      = (io.get("context_chunk_text") or "").replace("\n", " ")
    row["ctx_heading_path"]    = _join(io.get("heading_path") or [])
    row["ctx_section"]         = io.get("section_category", "")
    row["ctx_semantic_path"]   = _join(io.get("semantic_path") or [])
    row["ctx_object_type"]     = io.get("type", "")
    row["ctx_page"]            = str(io.get("page", ""))
    row["ctx_statement_type"]  = io.get("statement_type", "")
    row["ctx_study_context"]   = io.get("study_context", "")
    row["ctx_bbox"]            = _compact(io.get("bbox"))
    row["ctx_position"]        = str(io.get("position", ""))
    row["ctx_global_position"] = str(io.get("global_position", ""))

    # Candidate NER, facts, clinical relations
    row["cand_ner_labels"]         = _ner_labels(io_e)
    row["cand_ner_entities"]       = _ner_entities(io_e)
    row["cand_clinical_relations"] = _relations_str(io_r)
    for slot in _FACT_SLOTS:
        row[f"cand_facts_{slot}"] = _facts_str(io_f, slot)

    eff_facts = io.get("effective_facts") or {}
    for slot in _FACT_SLOTS:
        row[f"cand_eff_facts_{slot}"] = _facts_str(eff_facts, slot)

    return row


# ---------------------------------------------------------------------------
# Expansion: result list -> flat row list
# ---------------------------------------------------------------------------

def _verdict_to_hit_type(verdict: str | None) -> str:
    if verdict in ("YES", "MAYBE"):
        return "FINAL"
    if verdict == "NO":
        return "REJECTED"
    if verdict == "SKIP":
        return "SKIPPED"
    return "UNKNOWN"


def expand_results(data: dict) -> list[dict]:
    rows: list[dict] = []

    for result in data.get("results", []):
        ci = result.get("ci") or {}
        cb = _ci_block(ci)

        candidates = result.get("candidates", [])

        if candidates:
            counters: dict[str, int] = {}
            for cand in candidates:
                verdict  = cand.get("verdict") or ""
                hit_type = _verdict_to_hit_type(verdict)

                if hit_type == "FINAL":
                    chunk_id = cand.get("chunk_id", "")
                    for fh in result.get("final_hits", []):
                        if chunk_id in (fh.get("chunk_ids") or []) or chunk_id == fh.get("chunk_id", ""):
                            cand = {
                                **cand,
                                "evidence_type":       fh.get("evidence_type", ""),
                                "evidence_confidence": fh.get("evidence_confidence"),
                                "evidence_reason":     fh.get("evidence_reason", ""),
                                "match_span":          cand.get("match_span") or fh.get("match_span", ""),
                                "context_sentence":    cand.get("context_sentence") or fh.get("context_sentence", ""),
                                "highlight_score":     cand.get("highlight_score") or fh.get("highlight_score"),
                                "match_method":        cand.get("match_method") or fh.get("match_method", ""),
                                "match_page":          cand.get("match_page") or fh.get("match_page"),
                            }
                            break

                counters[hit_type] = counters.get(hit_type, 0) + 1
                rows.append({**cb, **_candidate_block(cand, hit_type, counters[hit_type])})

        else:
            final_hits    = result.get("final_hits", []) or []
            rejected_hits = result.get("rejected_hits", []) or []
            skipped_hits  = result.get("skipped_hits", []) or []

            total = len(final_hits) + len(rejected_hits) + len(skipped_hits)
            if total == 0:
                rows.append({**cb, **_candidate_block({}, "NO_HIT", 0)})
                continue

            for rank, hit in enumerate(final_hits, 1):
                adapted = {**hit, "verifier_reason": hit.get("evidence_reason", ""), "verdict": "YES"}
                rows.append({**cb, **_candidate_block(adapted, "FINAL", rank)})
            for rank, hit in enumerate(rejected_hits, 1):
                rows.append({**cb, **_candidate_block(hit, "REJECTED", rank)})
            for rank, hit in enumerate(skipped_hits, 1):
                rows.append({**cb, **_candidate_block(hit, "SKIPPED", rank)})

        if not rows or rows[-1].get("ci_id") != str(ci.get("id", "")):
            rows.append({**cb, **_candidate_block({}, "NO_HIT", 0)})

    return rows


# ---------------------------------------------------------------------------
# Canonical column order
# ---------------------------------------------------------------------------

def _column_order() -> list[str]:
    cols: list[str] = []

    cols += ["ci_id", "ci_text", "ci_type", "ci_category", "ci_assets",
             "ci_strategies", "ci_modality", "ci_statement_type",
             "ci_negated_slots", "ci_justification",
             "ci_ner_model", "ci_ner_labels", "ci_ner_entities",
             "ci_ontology_terms", "ci_clinical_relations"]
    for slot in _FACT_SLOTS:
        cols.append(f"ci_facts_{slot}")
    for slot in _FACT_SLOTS:
        cols.append(f"ci_own_facts_{slot}")

    cols += ["hit_type", "hit_rank", "chunk_id",
             "page_start", "page_end", "match_page", "sources", "retriever",
             "verdict", "confidence", "verifier_reason",
             "evidence_type", "evidence_confidence", "evidence_reason",
             "agg_score", "cross_encoder_score",
             "match_span", "context_sentence", "highlight_score", "match_method",
             "retrieval_object_type", "retrieval_object_id",
             "retrieval_section", "retrieval_heading_path",
             "rejection_reason"]

    for dim in _CMP_DIMS:
        cols += [
            f"cmp_{dim}_outcome",    f"cmp_{dim}_score",
            f"cmp_{dim}_severity",   f"cmp_{dim}_ci_values",
            f"cmp_{dim}_candidate_values", f"cmp_{dim}_ci_group",
            f"cmp_{dim}_candidate_group",  f"cmp_{dim}_ci_modality",
            f"cmp_{dim}_cand_modality",    f"cmp_{dim}_slot",
            f"cmp_{dim}_conflicts",        f"cmp_{dim}_relation",
        ]

    for key in _SB_SCALAR_KEYS:
        cols.append(f"sb_{key}")

    cols += ["sb_cr_matched", "sb_cr_conflicts", "sb_cr_decision",
             "sb_struct_missing_relation_weight",
             "sb_struct_missing_relation_severity",
             "sb_struct_missing_relation_ci_rels",
             "sb_struct_missing_relation_cand_rels",
             "sb_struct_missing_slots_weight",
             "sb_struct_missing_slots_severity",
             "sb_struct_missing_slots_list",
             "sb_struct_detail_json",
             "sb_contra_detail_json",
             "sb_enrichment_status_json"]

    for key in _AGG_SCALAR_KEYS:
        cols.append(f"agg_{key}")
    cols.append("agg_contra_detail_json")

    for slot in _FACT_SLOTS:
        cols.append(f"eff_ci_{slot}")
    for slot in _FACT_SLOTS:
        cols.append(f"eff_cand_{slot}")

    for field in _ENR_STATUS_FIELDS:
        cols.append(f"enr_ci_{field}")
    for field in _ENR_STATUS_FIELDS:
        cols.append(f"enr_cand_{field}")

    cols += ["enrd_ci_stage", "enrd_ci_status", "enrd_ci_reason", "enrd_ci_missing_slots",
             "enrd_cand_stage", "enrd_cand_status", "enrd_cand_reason", "enrd_cand_missing_slots"]

    cols += ["ctx_prev_sentence", "ctx_object_text", "ctx_next_sentence",
             "ctx_paragraph_text", "ctx_chunk_text",
             "ctx_heading_path", "ctx_section", "ctx_semantic_path",
             "ctx_object_type", "ctx_page", "ctx_statement_type", "ctx_study_context",
             "ctx_bbox", "ctx_position", "ctx_global_position"]

    cols += ["cand_ner_labels", "cand_ner_entities", "cand_clinical_relations"]
    for slot in _FACT_SLOTS:
        cols.append(f"cand_facts_{slot}")
    for slot in _FACT_SLOTS:
        cols.append(f"cand_eff_facts_{slot}")

    return cols


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a search-result JSON file to a flat CSV."
    )
    parser.add_argument("json_file", help="Path to result JSON")
    parser.add_argument("--out", "-o", default=None,
                        help="Output path (default: <json>.csv). Use '-' for stdout.")
    args = parser.parse_args()

    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"ERROR: file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    with json_path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    rows    = expand_results(data)
    columns = _column_order()

    all_keys: list[str] = list(columns)
    seen: set[str]      = set(all_keys)
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    if args.out == "-":
        out_fh    = sys.stdout
        close_out = False
    else:
        out_path  = Path(args.out) if args.out else json_path.with_suffix(".csv")
        out_fh    = out_path.open("w", newline="", encoding="utf-8")
        close_out = True

    try:
        writer = csv.DictWriter(out_fh, fieldnames=all_keys,
                                extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if close_out:
            out_fh.close()

    if args.out != "-":
        n_cis = len({r["ci_id"] for r in rows})
        by_type: dict[str, int] = {}
        for r in rows:
            t = r.get("hit_type", "?")
            by_type[t] = by_type.get(t, 0) + 1
        breakdown = "  ".join(f"{t}={n}" for t, n in sorted(by_type.items()))
        print(f"Wrote {len(rows)} rows ({n_cis} CIs)  {len(all_keys)} columns  [{breakdown}]  -> {out_path}")


if __name__ == "__main__":
    main()
