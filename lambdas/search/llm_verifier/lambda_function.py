"""
Search Pipeline — Stage 6: LLM Verifier
=========================================
Asks Bedrock Claude for a YES/NO/MAYBE verdict on each top-ranked candidate.
Only processes the top N candidates (controlled by LLM_VERIFY_TOP_N env var).

Prompt returns structured JSON: { "verdict": "YES"|"NO"|"MAYBE", "reason": str,
                                   "confidence": float 0-1 }

Input:  re-ranked search request  (must have "ranked_candidates")
Appends: "verified_candidates": list[VerifiedCandidate]

VerifiedCandidate = RankedCandidate + { "verdict": str, "reason": str,
                                         "confidence": float }
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BEDROCK_REGION       = os.environ.get("BEDROCK_REGION", os.environ.get("AWS_REGION", "us-east-1"))
BEDROCK_MODEL        = os.environ.get("VERIFIER_MODEL", "eu.anthropic.claude-haiku-4-5-20251001-v1:0")
MERGER_LAMBDA_ARN    = os.environ.get("MERGER_LAMBDA_ARN", "")
LLM_VERIFY_TOP_N     = int(os.environ.get("LLM_VERIFY_TOP_N", "20"))
MIN_RERANK_SCORE     = float(os.environ.get("MIN_RERANK_SCORE", "3.0"))

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
    if MERGER_LAMBDA_ARN:
        _get("lambda").invoke(
            FunctionName   = MERGER_LAMBDA_ARN,
            InvocationType = "Event",
            Payload        = json.dumps(result).encode(),
        )
    return result


def _process(req: dict) -> dict:
    ci_text    = req["ci"].get("knownCI", "")
    ci_assets  = req["ci"].get("assets", [])
    doc_ctx    = req.get("document_context", {})
    ranked     = req.get("ranked_candidates", [])

    # Verify all candidates that cleared the reranker score threshold (no position cap)
    to_verify  = [c for c in ranked
                  if c.get("cross_encoder_score", 0.0) >= MIN_RERANK_SCORE]
    skip       = [c for c in ranked
                  if c.get("cross_encoder_score", 0.0) < MIN_RERANK_SCORE]

    verified = [_verify(ci_text, cand, doc_ctx, ci_assets) for cand in to_verify]

    # Candidates below threshold are marked SKIP without an LLM call
    for cand in skip:
        verified.append({**cand, "verdict": "SKIP", "reason": "below reranker threshold",
                         "confidence": 0.0})

    return {
        **req,
        "verified_candidates": verified,
    }


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
    ctx      = candidate.get("context", {})
    combined = "\n".join(filter(None, [
        ctx.get("prev_text", ""),
        ctx.get("current_text", ""),
        ctx.get("next_text", ""),
    ]))[:3000]

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
        f"Document excerpt (pages {candidate.get('page_start')}–"
        f"{candidate.get('page_end')}):\n{combined}\n\n"
        f"Does this excerpt contain or directly identify the CI?\n\n"
        f"For each identity dimension answer true or false:\n"
        f"  same_drug       — excerpt discusses the same drug/regimen as the CI\n"
        f"  same_study      — excerpt is from the same trial/study as the CI\n"
        f"  same_objective  — excerpt shares the same primary/secondary objective\n"
        f"  same_endpoint   — excerpt uses the same primary endpoint (PFS, ORR, etc.)\n"
        f"  same_comparator — excerpt uses the same comparator arms/regimens\n\n"
        f"identity_score: fraction of dimensions that are true (0.0–1.0)\n"
        f"semantic_score: how semantically similar excerpt is to the CI (0.0–1.0)\n\n"
        f"Reply ONLY with valid JSON:\n"
        f"{{\"verdict\": \"YES\"|\"NO\"|\"MAYBE\", \"reason\": \"<one sentence>\", "
        f"\"confidence\": <0.0-1.0>, "
        f"\"identity\": {{\"same_drug\": <bool>, \"same_study\": <bool>, "
        f"\"same_objective\": <bool>, \"same_endpoint\": <bool>, "
        f"\"same_comparator\": <bool>, "
        f"\"identity_score\": <0.0-1.0>, \"semantic_score\": <0.0-1.0>}}}}"
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
        verdict  = parsed.get("verdict", "MAYBE")
        reason   = parsed.get("reason", "")
        conf     = float(parsed.get("confidence", 0.5))
        identity = parsed.get("identity", {})
    except json.JSONDecodeError as exc:
        logger.warning("[LLM Verifier] JSON parse failed chunk=%s: %s | raw=%r",
                       candidate.get("chunk_id"), exc, text[:200] if "text" in dir() else "")
        verdict, reason, conf, identity = "MAYBE", "LLM response was not valid JSON", 0.3, {}
        input_tokens, output_tokens = 0, 0
    except Exception as exc:
        logger.warning("[LLM Verifier] call failed chunk=%s error=%s",
                       candidate.get("chunk_id"), exc)
        verdict, reason, conf, identity = "MAYBE", str(exc), 0.0, {}
        input_tokens, output_tokens = 0, 0

    return {
        **candidate,
        "verdict":    verdict,
        "reason":     reason,
        "confidence": conf,
        "identity":   identity,
        "_tokens":    {"input": input_tokens, "output": output_tokens},
    }
