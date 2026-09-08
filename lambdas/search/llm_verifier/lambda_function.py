"""
Search Pipeline — Stage 6: LLM Verifier
=========================================
Asks Bedrock Claude for a YES/NO/MAYBE verdict on each top-ranked candidate.

Prompt returns structured JSON with verdict, confidence, identity, and explicit
constraint statuses for temporal/numeric requirements.

Input:  re-ranked search request  (must have "ranked_candidates")
Appends: "verified_candidates": list[VerifiedCandidate]

VerifiedCandidate = RankedCandidate + { "verdict": str, "reason": str,
                                         "confidence": float }
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BEDROCK_REGION       = os.environ.get("BEDROCK_REGION", os.environ.get("AWS_REGION", "us-east-1"))
BEDROCK_MODEL        = os.environ.get("VERIFIER_MODEL", "eu.anthropic.claude-haiku-4-5-20251001-v1:0")
MIN_RERANK_SCORE     = float(os.environ.get("MIN_RERANK_SCORE", "0.0"))

_aws: dict = {}

def _get(service: str, region: str | None = None):
    key = f"{service}:{region or ''}"
    if key not in _aws:
        import boto3
        _aws[key] = boto3.client(service, region_name=region) if region else boto3.client(service)
    return _aws[key]


# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    search_id = event.get("search_id", "unknown")
    logger.info("[LLM Verifier] start search_id=%s", search_id)
    try:
        result = _process(event)
    except Exception as exc:
        logger.error("[LLM Verifier] failed search_id=%s error=%s", search_id, exc)
        raise
    verified_count = sum(1 for c in result["verified_candidates"]
                         if c.get("verdict") == "YES")
    logger.info("[LLM Verifier] done search_id=%s verified=%d", search_id, verified_count)

    return result


def _process(req: dict) -> dict:
    ci_text    = req["ci"].get("knownCI", "")
    ci_assets  = req["ci"].get("assets", [])
    doc_ctx    = req.get("document_context", {})
    ranked     = req.get("ranked_candidates", [])

    # Verify all candidates that cleared the reranker score threshold (no position cap)
    to_verify  = [c for c in ranked]
    skip       = []

    verified = _verify_batch(ci_text, to_verify, doc_ctx, ci_assets)

    # Candidates below threshold are marked SKIP without an LLM call
    for cand in skip:
        verified.append({**cand, "verdict": "SKIP", "reason": "below reranker threshold",
                         "confidence": 0.0})

    return {
        **req,
        "verified_candidates": verified,
    }


_MAX_VERIFY_BATCH = 40  # max candidates per Bedrock call (8000 token output cap)


_TEMPORAL_CONSTRAINT_PATTERNS = [
    r'\b(?:within|over|during|after|before|prior to|following|for|at)\s+'
    r'(?:approximately\s+)?\d+(?:\.\d+)?\s*'
    r'(?:days?|weeks?|months?|years?|hours?|minutes?)\b',
    # Also recognize a bare duration such as "26 weeks".
    r'\b(?:approximately\s+)?\d+(?:\.\d+)?\s*'
    r'(?:days?|weeks?|months?|years?|hours?|minutes?)\b',
    r'\b(?:cycle\s*\d+\s*day\s*\d+|c\d+\s*d\d+)\b',
    r'\b(?:week|day|month|year)\s*\d+\b',
    r'\b(?:baseline|screening|randomization|first\s+dose|last\s+dose|'
    r'end\s+of\s+(?:treatment|study)|follow[\s-]?up)\b',
]

_NUMERIC_CONSTRAINT_PATTERNS = [
    r'\b(?:n|N)\s*=\s*\(?\s*\d+(?:\.\d+)?\s*\)?',
    r'\b\d+(?:\.\d+)?\s*(?:patients?|subjects?|participants?)\b',
    r'\bp(?:\s*[-_ ]?\s*value)?\s*[<>=≤≥]\s*\d+(?:\.\d+)?',
    r'\bhazard\s+ratio\s*[<>=≤≥]?\s*\d+(?:\.\d+)?',
    r'\bHR\s*[<>=≤≥]?\s*\d+(?:\.\d+)?',
    r'\bodds\s+ratio\s*[<>=≤≥]?\s*\d+(?:\.\d+)?',
    r'\bOR\s*[<>=≤≥]?\s*\d+(?:\.\d+)?',
    r'\b\d+(?:\.\d+)?\s*%\b',
    r'\b(?:score|range|value)\s+(?:of\s+)?\d+(?:\.\d+)?\s*(?:to|-|–|—)\s*\d+(?:\.\d+)?\b',
]

def _dedupe_constraints(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        normalized = re.sub(r'\s+', ' ', value or '').strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def _extract_explicit_constraints(ci_text: str) -> dict:
    """Extract explicit CI constraints for LLM + deterministic guarding.

    Explicit numeric/temporal constraints must be supported by the authoritative
    candidate text, never by surrounding context.
    """
    text = ci_text or ""
    temporal, numeric = [], []

    for pattern in _TEMPORAL_CONSTRAINT_PATTERNS:
        temporal.extend(m.group(0) for m in re.finditer(pattern, text, flags=re.IGNORECASE))
    for pattern in _NUMERIC_CONSTRAINT_PATTERNS:
        numeric.extend(m.group(0) for m in re.finditer(pattern, text, flags=re.IGNORECASE))

    return {
        "temporal": _dedupe_constraints(temporal)[:16],
        "numeric": _dedupe_constraints(numeric)[:16],
    }


def _strict_constraint_instructions(ci_text: str) -> str:
    constraints = _extract_explicit_constraints(ci_text)
    if not constraints["temporal"] and not constraints["numeric"]:
        return ""

    return (
        "<strict_constraints>\n"
        "These constraints are MANDATORY, not ranking hints.\n"
        f"Temporal constraints: {json.dumps(constraints['temporal'], ensure_ascii=False)}\n"
        f"Numeric/statistical constraints: {json.dumps(constraints['numeric'], ensure_ascii=False)}\n\n"
        "SOURCE AUTHORITY RULE: CANDIDATE TEXT is the only authoritative evidence "
        "for satisfying a constraint. SUPPORTING CONTEXT is context only. "
        "A value or fact appearing only in SUPPORTING CONTEXT does NOT satisfy the CI.\n"
        "For EVERY candidate, evaluate EVERY explicit constraint independently.\n"
        "1. CONTRADICTED constraint => verdict MUST be NO.\n"
        "2. Missing/unsupported required constraint in CANDIDATE TEXT => verdict MUST NOT be YES.\n"
        "3. Use MAYBE only when CANDIDATE TEXT is genuinely insufficient to determine whether "
        "the constraint is satisfied.\n"
        "4. Semantic similarity NEVER substitutes for an explicit numeric or temporal constraint.\n"
        "5. Preserve operators and ranges exactly: =, <, <=, >, >=, ≤, ≥.\n"
        "6. Preserve temporal meaning and units. '26 weeks', 'within 26 weeks', "
        "'after 26 weeks', and 'Week 26' are different constraints.\n"
        "7. Never invent a value from SUPPORTING CONTEXT or document profile metadata.\n"
        "8. A candidate that matches the entity but omits a required constraint is incomplete.\n\n"
        "<examples>\n"
        "<example>CI: 26 weeks; CANDIDATE TEXT: Participants were followed for 16 weeks; "
        "SUPPORTING CONTEXT: another section says 26 weeks. Result: NO.</example>\n"
        "<example>CI: 26 weeks; CANDIDATE TEXT: The assessment occurred at 26 weeks; "
        "Result: YES if identity is also supported.</example>\n"
        "<example>CI: RPLS occurring within 26 weeks; CANDIDATE TEXT: RPLS. "
        "SUPPORTING CONTEXT: RPLS occurred within 26 weeks. Result: NO.</example>\n"
        "<example>CI: RPLS occurring within 26 weeks; "
        "CANDIDATE TEXT: RPLS occurring within 26 weeks. Result: YES.</example>\n"
        "<example>CI: N=8 patients; CANDIDATE TEXT: 12 patients. Result: NO — contradicted.</example>\n"
        "<example>CI: pValue >= 0.05; CANDIDATE TEXT: pValue = 0.06. Result: YES.</example>\n"
        "<example>CI: pValue >= 0.05; CANDIDATE TEXT: pValue = 0.03. Result: NO.</example>\n"
        "</examples>\n"
        "</strict_constraints>\n\n"
    )


def _normalize_verdict_item(item: dict) -> dict:
    verdict = str(item.get("verdict", "MAYBE")).upper()
    if verdict not in {"YES", "NO", "MAYBE"}:
        verdict = "MAYBE"
    try:
        confidence = float(item.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    identity = item.get("identity", {})
    if not isinstance(identity, dict):
        identity = {}
    constraints = item.get("constraints", {})
    if not isinstance(constraints, dict):
        constraints = {}

    return {
        "verdict": verdict,
        "reason": str(item.get("reason", "")),
        "confidence": confidence,
        "identity": identity,
        "constraints": constraints,
    }


def _constraint_status_summary(item: dict) -> tuple[bool, bool, str]:
    constraints = item.get("constraints", {})
    if not isinstance(constraints, dict):
        return False, False, ""

    statuses = []
    for key in ("temporal", "numeric"):
        values = constraints.get(key, [])
        if isinstance(values, list):
            for entry in values:
                if isinstance(entry, dict):
                    status = str(entry.get("status", "")).upper()
                    required = entry.get("required") or entry.get("constraint") or key
                    if status:
                        statuses.append((status, str(required)))

    failed = [v for s, v in statuses if s in {"CONTRADICTS", "FAIL", "FAILED"}]
    unknown = [v for s, v in statuses if s in {
        "DOES_NOT_MENTION", "UNKNOWN", "UNSUPPORTED", "MISSING"
    }]

    if failed:
        return True, False, f"Required constraint contradicted: {failed[0]}"
    if unknown:
        return False, True, f"Required constraint not supported by candidate: {unknown[0]}"
    return False, False, ""


def _candidate_authoritative_text(candidate: dict) -> str:
    """Return only text that is allowed to satisfy a CI constraint."""
    ctx = candidate.get("context", {}) or {}
    if isinstance(ctx, dict) and ctx.get("current_text"):
        return str(ctx["current_text"]).strip()

    text = candidate.get("text")
    return str(text).strip() if text else ""


def _supporting_context_text(candidate: dict) -> str:
    """Return context for interpretation only; never use this for constraint checks."""
    ctx = candidate.get("context", {}) or {}
    if not isinstance(ctx, dict):
        ctx = {}

    current = str(ctx.get("current_text", "") or "").strip()
    parts = [
        ctx.get("prev_text", ""),
        ctx.get("next_text", ""),
        candidate.get("context_text", ""),
    ]
    out = []
    for part in parts:
        part = str(part or "").strip()
        if part and part != current:
            out.append(part)
    return "\n".join(out)[:3000]


def _constraint_presence_guard(
    ci_text: str, candidate_text: str
) -> tuple[str | None, str | None]:
    """Ensure explicit numeric/temporal tokens exist in authoritative candidate text.

    This closes the context-leakage path even if the LLM mistakenly marks a
    context-only constraint as SATISFIES.
    """
    constraints = _extract_explicit_constraints(ci_text)
    if not constraints["temporal"] and not constraints["numeric"]:
        return None, None

    candidate = re.sub(r"\s+", " ", candidate_text or "").strip().casefold()

    for required in constraints["temporal"]:
        r = required.casefold().replace("–", "-").replace("—", "-")
        m = re.search(
            r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>days?|weeks?|months?|years?|hours?|minutes?)",
            r,
            flags=re.IGNORECASE,
        )
        if m:
            if not re.search(
                rf"\b{re.escape(m.group('num'))}\s*{re.escape(m.group('unit'))}\b",
                candidate,
                flags=re.IGNORECASE,
            ):
                return "NO", f"Required temporal constraint not present in candidate text: {required}"
        elif r and r not in candidate:
            return "NO", f"Required temporal constraint not present in candidate text: {required}"

    for required in constraints["numeric"]:
        r = required.casefold().replace("–", "-").replace("—", "-")
        if r in candidate:
            continue

        nums = re.findall(r"\d+(?:\.\d+)?", r)
        if nums and not any(
            re.search(rf"(?<!\d){re.escape(n)}(?!\d)", candidate)
            for n in nums
        ):
            return "NO", f"Required numeric constraint not present in candidate text: {required}"

    return None, None


def _apply_strict_constraint_guard(
    ci_text: str, item: dict, candidate_text: str = ""
) -> dict:
    """Apply LLM-reported and deterministic constraint guards.

    Supporting context is intentionally excluded from candidate_text.
    """
    normalized = _normalize_verdict_item(item)

    failed, unknown, reason = _constraint_status_summary(normalized)
    if failed:
        normalized["verdict"] = "NO"
        normalized["confidence"] = min(normalized["confidence"], 0.99)
        normalized["reason"] = reason
        return normalized

    forced_verdict, forced_reason = _constraint_presence_guard(ci_text, candidate_text)
    if forced_verdict == "NO":
        normalized["verdict"] = "NO"
        normalized["confidence"] = min(normalized["confidence"], 0.99)
        normalized["reason"] = forced_reason or "Required constraint is absent from candidate text."
        return normalized

    if unknown and normalized["verdict"] == "YES":
        normalized["verdict"] = "NO"
        normalized["confidence"] = min(normalized["confidence"], 0.95)
        normalized["reason"] = reason

    return normalized


def _verify_batch(
    ci_text: str,
    candidates: list[dict],
    doc_ctx: dict | None = None,
    ci_assets: list | None = None,
) -> list[dict]:
    """Verify all candidates in a single Bedrock call; falls back to sequential on error."""
    if not candidates:
        return []
    if len(candidates) == 1:
        return [_verify(ci_text, candidates[0], doc_ctx, ci_assets)]
    # Split oversized batches to avoid hitting the 8000-token output cap
    if len(candidates) > _MAX_VERIFY_BATCH:
        results = []
        for i in range(0, len(candidates), _MAX_VERIFY_BATCH):
            results.extend(_verify_batch(ci_text, candidates[i:i+_MAX_VERIFY_BATCH], doc_ctx, ci_assets))
        return results

    import re as _re

    # Build shared header (doc profile + drug note)
    doc_profile = ""
    if doc_ctx:
        drugs   = ", ".join(doc_ctx.get("primary_drugs", [])[:2])
        studies = ", ".join(doc_ctx.get("study_ids", [])[:1])
        disease = ", ".join(doc_ctx.get("disease", [])[:1])
        phase   = ", ".join(doc_ctx.get("phase", []))
        doc_profile = (f"DOCUMENT PROFILE:\n  Drug(s): {drugs}\n  Study:   {studies}\n"
                       f"  Disease: {disease}\n  Phase:   {phase}\n\n")

    ci_drug_note = ""
    asset_ctx    = ""
    if ci_assets:
        ci_drug_names = [a.get("name") or a.get("genericName") or a.get("code", "")
                         for a in ci_assets if a]
        ci_drug_names = [n for n in ci_drug_names if n]
        desc = next((a.get("description", "") for a in ci_assets if a and a.get("description")), "")
        desc_lower = _re.sub(r"<[^>]+>", " ", desc).lower()
        if ci_drug_names:
            doc_drugs_lower = {d.lower() for d in doc_ctx.get("primary_drugs", [])} if doc_ctx else set()
            ci_drugs_lower  = {n.lower() for n in ci_drug_names}
            name_overlap = bool(doc_drugs_lower & ci_drugs_lower)
            desc_overlap = any(drug in desc_lower for drug in doc_drugs_lower)
            if doc_drugs_lower and not name_overlap and not desc_overlap:
                doc_drug_str = ", ".join(doc_ctx.get("primary_drugs", [])[:2])
                ci_drug_str  = ", ".join(ci_drug_names)
                ci_drug_note = (f"Context: CI linked to [{ci_drug_str}], document covers "
                                f"[{doc_drug_str}]. Score on content.\n\n")
            else:
                ci_drug_note = f"CI Drug: {', '.join(ci_drug_names)}\n\n"
        if desc:
            asset_ctx = f"Drug/Regimen Context: {_re.sub(chr(60)+'[^>]+>','',desc).strip()[:500]}\n\n"

    # Keep authoritative evidence physically separate from supporting context.
    blocks = []
    for i, c in enumerate(candidates, 1):
        candidate_text = _candidate_authoritative_text(c)
        support_text = _supporting_context_text(c)
        blocks.append(
            f"--- CANDIDATE {i} (p{c.get('page_start')}–{c.get('page_end')}) ---\n"
            f"CANDIDATE TEXT (AUTHORITATIVE EVIDENCE):\n"
            f"{candidate_text[:2500] or '[empty]'}\n\n"
            f"SUPPORTING CONTEXT (NOT EVIDENCE):\n"
            f"{support_text[:3000] or '[none]'}"
        )

    prompt = (
        f"You are a clinical document reviewer.\n\n"
        f"{doc_profile}{ci_drug_note}{asset_ctx}"
        f'Confidential Information (CI): "{ci_text}"\n\n'
        f"{_strict_constraint_instructions(ci_text)}"
        f"For each candidate below, decide if the excerpt contains or directly identifies the CI.\n\n"
        f"CANDIDATE TEXT is the authoritative evidence. SUPPORTING CONTEXT may only help interpret it.\n"
        f"Never satisfy a CI using information that appears only in SUPPORTING CONTEXT.\n\n"
        f"For each identity dimension answer true or false based primarily on CANDIDATE TEXT:\n"
        f"  same_drug       — candidate discusses the same drug/regimen as the CI\n"
        f"  same_study      — candidate is from the same trial/study as the CI\n"
        f"  same_objective  — candidate shares the same primary/secondary objective\n"
        f"  same_endpoint   — candidate uses the same primary endpoint (PFS, ORR, etc.)\n"
        f"  same_comparator — candidate uses the same comparator arms/regimens\n\n"
        f"identity_score: fraction of dimensions that are true (0.0–1.0)\n"
        f"semantic_score: semantic similarity of CANDIDATE TEXT to the CI (0.0–1.0)\n\n"
        f"Reply ONLY with a JSON ARRAY of {len(candidates)} objects in the same order:\n"
        f'[{{"verdict":"YES"|"NO"|"MAYBE","reason":"<one sentence>",'
        f'"confidence":<0.0-1.0>,"identity":{{"same_drug":<bool>,"same_study":<bool>,'
        f'"same_objective":<bool>,"same_endpoint":<bool>,"same_comparator":<bool>,'
        f'"identity_score":<0.0-1.0>,"semantic_score":<0.0-1.0}},'
        f'"constraints":{{"temporal":[{{"required":"<constraint>",'
        f'"status":"SATISFIES"|"CONTRADICTS"|"DOES_NOT_MENTION"|"UNKNOWN"}}],'
        f'"numeric":[{{"required":"<constraint>",'
        f'"status":"SATISFIES"|"CONTRADICTS"|"DOES_NOT_MENTION"|"UNKNOWN"}}]}}}}, ...]\n\n'
        + "\n\n".join(blocks)
    )

    try:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": min(200 * len(candidates), 8000),  # ~85 actual/candidate; cap at API limit
            "messages": [{"role": "user", "content": prompt}],
        }
        resp      = _get("bedrock-runtime", BEDROCK_REGION).invoke_model(
            modelId=BEDROCK_MODEL, contentType="application/json",
            accept="application/json", body=json.dumps(body).encode(),
        )
        resp_body = json.loads(resp["body"].read())
        raw       = resp_body["content"][0]["text"].strip()
        # Strip code fences then find the array — NOT _strip_code_fence which seeks {
        import re as _re2
        raw = _re2.sub(r"^```(?:json)?\s*", "", raw)
        raw = _re2.sub(r"\s*```$", "", raw.strip())
        bracket = raw.find("[")
        if bracket > 0:
            raw = raw[bracket:]
        text      = raw
        usage     = resp_body.get("usage", {})
        in_tok    = usage.get("input_tokens", 0)
        out_tok   = usage.get("output_tokens", 0)
        parsed    = json.loads(text)
        if not isinstance(parsed, list) or len(parsed) == 0:
            raise ValueError(f"Expected list of {len(candidates)}, got {len(parsed) if isinstance(parsed,list) else type(parsed)}")
        # Pad if Claude returned fewer items than expected rather than doing full sequential fallback
        while len(parsed) < len(candidates):
            parsed.append({"verdict": "MAYBE", "confidence": 0.5, "reason": "batch_missing"})
        parsed = parsed[:len(candidates)]  # truncate any extra items Claude occasionally adds
        results = []
        per_tok = max(1, in_tok // len(candidates)), max(1, out_tok // len(candidates))
        for cand, item in zip(candidates, parsed):
            guarded = _apply_strict_constraint_guard(ci_text, item, _candidate_authoritative_text(c))
            results.append({
                **cand,
                "verdict":    guarded["verdict"],
                "reason":     guarded["reason"],
                "confidence": guarded["confidence"],
                "identity":   guarded["identity"],
                "constraints": guarded["constraints"],
                "_tokens":    {"input": per_tok[0], "output": per_tok[1]},
            })
        constraint_count = _extract_explicit_constraints(ci_text)
        logger.info(
            "[LLM Verifier] batch n=%d in_tok=%d out_tok=%d temporal_constraints=%d numeric_constraints=%d",
            len(candidates), in_tok, out_tok,
            len(constraint_count["temporal"]), len(constraint_count["numeric"]),
        )
        return results
    except Exception as exc:
        logger.warning("[LLM Verifier] batch failed (%s) — falling back to sequential", exc)
        return [_verify(ci_text, c, doc_ctx, ci_assets) for c in candidates]


def _strip_code_fence(text: str) -> str:
    """Remove markdown code fences that Claude sometimes wraps JSON in."""
    import re
    # Strip ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    # If there's still preamble before the first {, trim it
    brace = text.find("{")
    if brace > 0:
        text = text[brace:]
    return text.strip()


def _verify(ci_text: str, candidate: dict, doc_ctx: dict | None = None,
            ci_assets: list | None = None) -> dict:
    candidate_text = _candidate_authoritative_text(candidate)
    supporting_context = _supporting_context_text(candidate)

    # Document profile header
    doc_profile = ""
    if doc_ctx:
        drugs   = ", ".join(doc_ctx.get("primary_drugs", [])[:2])
        studies = ", ".join(doc_ctx.get("study_ids", [])[:1])
        disease = ", ".join(doc_ctx.get("disease", [])[:1])
        phase   = ", ".join(doc_ctx.get("phase", []))
        doc_profile = (
            f"DOCUMENT PROFILE:\n"
            f"  Drug(s): {drugs}\n"
            f"  Study:   {studies}\n"
            f"  Disease: {disease}\n"
            f"  Phase:   {phase}\n\n"
        )

    # CI drug note — contextual, never a hard gate
    # Drug identity is a ranking signal, not a filter: combination regimens, comparator arms,
    # and mechanism discussions legitimately cross drug boundaries.
    ci_drug_note = ""
    asset_ctx    = ""
    import re as _re
    if ci_assets:
        ci_drug_names = [
            a.get("name") or a.get("genericName") or a.get("code", "")
            for a in ci_assets if a
        ]
        ci_drug_names = [n for n in ci_drug_names if n]

        # Build a single description blob for overlap checking
        desc = next((a.get("description", "") for a in ci_assets if a and a.get("description")), "")
        desc_lower = _re.sub(r"<[^>]+>", " ", desc).lower()

        if ci_drug_names:
            doc_drugs_lower = {d.lower() for d in doc_ctx.get("primary_drugs", [])} if doc_ctx else set()
            ci_drugs_lower  = {n.lower() for n in ci_drug_names}

            # Overlap via name/code OR via description text
            # (e.g. "Tec-Tal" description mentions "talquetamab" → counts as overlap)
            name_overlap = bool(doc_drugs_lower & ci_drugs_lower)
            desc_overlap = any(drug in desc_lower for drug in doc_drugs_lower)

            if doc_drugs_lower and not name_overlap and not desc_overlap:
                # Genuinely different drug families — note it softly, do NOT hard-gate
                doc_drug_str = ", ".join(doc_ctx.get("primary_drugs", [])[:2])
                ci_drug_str  = ", ".join(ci_drug_names)
                ci_drug_note = (
                    f"Context: This CI is linked to [{ci_drug_str}] and the document "
                    f"primarily covers [{doc_drug_str}]. Score based on content — evidence "
                    f"from comparator arms, related mechanisms, or cross-study references "
                    f"is valid supporting evidence.\n\n"
                )
            else:
                ci_drug_note = f"CI Drug: {', '.join(ci_drug_names)}\n\n"

        # Asset description as drug/regimen context
        if desc:
            desc_clean = _re.sub(r"<[^>]+>", " ", desc).strip()[:500]
            asset_ctx = f"Drug/Regimen Context: {desc_clean}\n\n"

    prompt = (
        f"You are a clinical document reviewer.\n\n"
        f"{doc_profile}"
        f"{ci_drug_note}"
        f"{asset_ctx}"
        f"Confidential Information (CI): \"{ci_text}\"\n\n"
        f"{_strict_constraint_instructions(ci_text)}"
        f"CANDIDATE TEXT (AUTHORITATIVE EVIDENCE; pages {candidate.get('page_start')}–"
        f"{candidate.get('page_end')}):\n{candidate_text[:3000] or '[empty]'}\n\n"
        f"SUPPORTING CONTEXT (NOT EVIDENCE):\n{supporting_context[:3000] or '[none]'}\n\n"
        f"Does CANDIDATE TEXT contain or directly identify the CI?\n\n"
        f"CANDIDATE TEXT is the authoritative evidence. SUPPORTING CONTEXT may only help interpret it.\n"
        f"Never satisfy a CI using information that appears only in SUPPORTING CONTEXT.\n\n"
        f"For each identity dimension answer true or false based primarily on CANDIDATE TEXT:\n"
        f"  same_drug       — candidate discusses the same drug/regimen as the CI\n"
        f"  same_study      — candidate is from the same trial/study as the CI\n"
        f"  same_objective  — candidate shares the same primary/secondary objective\n"
        f"  same_endpoint   — candidate uses the same primary endpoint (PFS, ORR, etc.)\n"
        f"  same_comparator — candidate uses the same comparator arms/regimens\n\n"
        f"identity_score: fraction of dimensions that are true (0.0–1.0)\n"
        f"semantic_score: semantic similarity of CANDIDATE TEXT to the CI (0.0–1.0)\n\n"
        f"Reply ONLY with valid JSON:\n"
        f'{{"verdict": "YES"|"NO"|"MAYBE", "reason": "<one sentence>", '
        f'"confidence": <0.0-1.0>, '
        f'"identity": {{"same_drug": <bool>, "same_study": <bool>, '
        f'"same_objective": <bool>, "same_endpoint": <bool>, '
        f'"same_comparator": <bool>, '
        f'"identity_score": <0.0-1.0>, "semantic_score": <0.0-1.0>}}, '
        f'"constraints": {{"temporal": [{{"required": "<constraint>", '
        f'"status": "SATISFIES"|"CONTRADICTS"|"DOES_NOT_MENTION"|"UNKNOWN"}}], '
        f'"numeric": [{{"required": "<constraint>", '
        f'"status": "SATISFIES"|"CONTRADICTS"|"DOES_NOT_MENTION"|"UNKNOWN"}}]}}}}' 
    )

    try:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens":        500,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp     = _get("bedrock-runtime", BEDROCK_REGION).invoke_model(
            modelId     = BEDROCK_MODEL,
            contentType = "application/json",
            accept      = "application/json",
            body        = json.dumps(body).encode(),
        )
        resp_body = json.loads(resp["body"].read())
        text      = resp_body["content"][0]["text"].strip()
        usage     = resp_body.get("usage", {})
        input_tokens  = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        text   = _strip_code_fence(text)
        parsed = json.loads(text)
        guarded = _apply_strict_constraint_guard(ci_text, parsed, candidate_text)
        verdict  = guarded["verdict"]
        reason   = guarded["reason"]
        conf     = guarded["confidence"]
        identity = guarded["identity"]
        constraints = guarded["constraints"]
    except json.JSONDecodeError as exc:
        logger.warning("[LLM Verifier] JSON parse failed chunk=%s: %s | raw=%r",
                       candidate.get("chunk_id"), exc, text[:200] if "text" in dir() else "")
        verdict, reason, conf, identity, constraints = "MAYBE", "LLM response was not valid JSON", 0.3, {}, {}
        input_tokens, output_tokens = 0, 0
    except Exception as exc:
        logger.warning("[LLM Verifier] call failed chunk=%s error=%s",
                       candidate.get("chunk_id"), exc)
        verdict, reason, conf, identity, constraints = "MAYBE", str(exc), 0.0, {}, {}
        input_tokens, output_tokens = 0, 0

    return {
        **candidate,
        "verdict":    verdict,
        "reason":     reason,
        "confidence": conf,
        "identity":   identity,
        "constraints": constraints,
        "_tokens":    {"input": input_tokens, "output": output_tokens},
    }
