"""
NER Lambda  (Unified — CI and Document)
========================================
Routes on ``event["source_type"]``:
  "ci"       → CI NER path       (Stage 2 in the CI pipeline)
  "document" → Document NER path (Stage 4 in the document pipeline)

Both paths share the same GLiNER model, the same Comprehend Medical backend,
the same label set, and the same post-processing rules.  All enrichment is
performed through ``shared.clinical_enrichment_pipeline`` so both CIs and
document objects receive an identical ClinicalObject schema.

Fan-out targets
---------------
  CI path       : ONTOLOGY_LAMBDA_ARN  (same unified lambda as document)
  Document path : ONTOLOGY_LAMBDA_ARN

Adding a new NER label
-----------------------
1. Add the label string to ``_GLINER_LABELS``.
2. Add the mapping to ``_GLINER_LABEL_MAP``.
3. Add a post-processing rule to ``_postprocess_entities()`` if needed.
That is all — both CI and document objects pick it up automatically.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ── ensure shared/ is importable when running as a loaded module ──────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from shared.clinical_dict import match_entities as _dict_match, _LABEL_FAMILY as _DICT_LABEL_FAMILY
except ImportError:
    logger.warning("shared.clinical_dict not found — dictionary NER disabled")
    def _dict_match(text: str) -> list[dict]:
        return []
    _DICT_LABEL_FAMILY: dict = {}

# ── env config ────────────────────────────────────────────────────────────────

ONTOLOGY_LAMBDA_ARN    = os.environ.get("ONTOLOGY_LAMBDA_ARN", "")
NER_MODEL              = os.environ.get("NER_MODEL", "gliner")
GLINER_MODEL_NAME      = os.environ.get("GLINER_MODEL", "urchade/gliner_mediumv2.1")

# ─── lazy AWS clients ─────────────────────────────────────────────────────────
_aws: dict = {}

def _get(service: str):
    if service not in _aws:
        import boto3
        _aws[service] = boto3.client(service)
    return _aws[service]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def handler(event: dict, context: Any) -> dict:
    source_type = event.get("source_type", "document")

    if source_type == "ci":
        ci_id = event.get("id", "unknown")
        logger.info("[NER] start source=ci ci_id=%s model=%s", ci_id, NER_MODEL)
        try:
            result = _process_ci(event)
        except Exception as exc:
            logger.error("[NER] failed source=ci ci_id=%s error=%s", ci_id, exc)
            raise
        logger.info("[NER] done source=ci ci_id=%s entities=%d",
                    ci_id, len(result["ner"]["entities"]))
        if ONTOLOGY_LAMBDA_ARN:
            _get("lambda").invoke(
                FunctionName   = ONTOLOGY_LAMBDA_ARN,
                InvocationType = "Event",
                Payload        = json.dumps(result).encode(),
            )
        return result

    else:
        chunk_id = event.get("chunk_id", "unknown")
        logger.info("[NER] start source=document chunk_id=%s model=%s", chunk_id, NER_MODEL)
        try:
            result = _process_document(event)
        except Exception as exc:
            logger.error("[NER] failed source=document chunk_id=%s error=%s", chunk_id, exc)
            raise
        logger.info("[NER] done source=document chunk_id=%s entities=%d",
                    chunk_id, len(result["ner"]["entities"]))
        if ONTOLOGY_LAMBDA_ARN:
            _get("lambda").invoke(
                FunctionName   = ONTOLOGY_LAMBDA_ARN,
                InvocationType = "Event",
                Payload        = json.dumps(result).encode(),
            )
        return result


# ─────────────────────────────────────────────────────────────────────────────
# CI path
# ─────────────────────────────────────────────────────────────────────────────

def _process_ci(ci: dict) -> dict:
    """
    NER + enrichment for a single CI text.

    Uses the same multi-layer NER pipeline as documents (dict + GLiNER +
    post-processing) so CI entities have the same quality as document entities.
    Enrichment is routed through enrich_ci() so the CI gets the full
    ClinicalObject schema: facts, own_facts, effective_facts, study_hierarchy,
    clinical_identity, endpoint_identity, population_identity, etc.
    """
    text     = ci["normalization"]["normalized_text"]
    entities = _run_ner(text)
    entities = _postprocess_entities(entities)
    for e in entities:
        if "family" not in e:
            e["family"] = _DICT_LABEL_FAMILY.get(e.get("label", ""), "OTHER")

    enrichment: dict = {}
    try:
        from shared.clinical_enrichment_pipeline import enrich_ci
        # enrich_ci expects object_start/object_end; _run_ner uses start/end
        adapted = [
            {**e, "object_start": e.get("start", 0), "object_end": e.get("end", 0)}
            for e in entities
        ]
        # Pass the CI's category name (e.g. "Sample Size") as section_category
        # so enrich_object can use it to fix statistical-notation misclassification.
        _ci_cat       = ci.get("category") or {}
        _ci_cat_name  = _ci_cat.get("name") or ""
        enrichment = enrich_ci(text, adapted, section_category=_ci_cat_name)
    except Exception as exc:
        logger.warning("[NER ci] enrich_ci failed: %s", exc)

    return {
        **ci,
        **enrichment,
        "ner": {
            "entities": entities,
            "model":    NER_MODEL,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Document path
# ─────────────────────────────────────────────────────────────────────────────

def _process_document(chunk: dict) -> dict:
    """
    Multi-layer NER + per-object enrichment for a document chunk.

    Layer 1: Clinical dictionary (high-precision, deterministic, score=1.0)
    Layer 2: GLiNER or Comprehend Medical (fills gaps not covered by dict)
    Layer 3: Post-processing (domain-aware reclassification)
    """
    text     = chunk["normalization"]["normalized_text"]
    entities = _run_ner(text)
    entities = _postprocess_entities(entities)
    for e in entities:
        if "family" not in e:
            e["family"] = _DICT_LABEL_FAMILY.get(e.get("label", ""), "OTHER")

    objects = chunk.get("extraction", {}).get("objects", [])
    if objects:
        _assign_entities_to_objects(objects, entities)
        _enrich_objects_with_facts(objects)
        chunk = {**chunk, "extraction": {**chunk["extraction"], "objects": objects}}

    return {
        **chunk,
        "ner": {
            "entities": entities,
            "model":    NER_MODEL,
        },
    }


def _enrich_objects_with_facts(objects: list[dict]) -> None:
    """
    Add study_context, statement_type, facts, and all Knowledge Layer fields
    to each semantic object using spaCy dep parse + GLiNER entities.

    Enrichment is routed through the centralized pipeline:
      - enrich_object()            → per-object ClinicalObject fields
      - enrich_document_objects()  → context propagation (heading → paragraph → sentence)

    Mutates objects in-place.
    """
    try:
        from shared.clinical_fact_extractor import enrich_object
    except ImportError:
        logger.warning("[NER document] clinical_fact_extractor not importable — skipping enrichment")
        return

    for obj in objects:
        if not obj.get("searchable", True):
            continue
        obj_text = obj.get("text", "")
        if not obj_text:
            continue
        try:
            enrichment = enrich_object(
                text             = obj_text,
                entities         = obj.get("entities", []),
                section_category = obj.get("section_category", ""),
                heading_path     = obj.get("heading_path") or [],
            )
            obj.update(enrichment)
        except Exception as exc:
            logger.debug("[NER document] enrich_object failed for %s: %s",
                         obj.get("object_id", "?"), exc)

    try:
        from shared.clinical_enrichment_pipeline import enrich_document_objects
        enrich_document_objects(objects)
    except Exception as exc:
        logger.warning(
            "[NER document] effective_facts propagation FAILED: %s — "
            "promoting own_facts → effective_facts for missed objects",
            exc, exc_info=True,
        )
        # Fallback: any object that didn't get effective_facts set during propagation
        # gets its own_facts promoted so comparators have data rather than UNKNOWN.
        for _obj in objects:
            if not _obj.get("effective_facts"):
                _own = {k: list(v) for k, v in (_obj.get("facts") or {}).items() if v}
                _obj["effective_facts"] = _own
                _obj["own_facts"]       = _own
                _obj["inherited_slots"] = []
                _obj["slot_provenance"] = {k: "explicit" for k in _own}


# ─────────────────────────────────────────────────────────────────────────────
# NER backends (shared by CI and document paths)
# ─────────────────────────────────────────────────────────────────────────────

def _run_ner(text: str) -> list[dict]:
    """
    Multi-layer NER: clinical dictionary takes priority; model fills the gaps.

    Layer 1 (always): clinical dictionary — high-precision, score=1.0
    Layer 2 (model):  GLiNER or Comprehend Medical fills unclaimed positions
    """
    dict_entities = _dict_match(text)
    dict_spans    = {(e["start"], e["end"]) for e in dict_entities}

    if NER_MODEL == "gliner":
        model_entities = _run_gliner(text)
    elif NER_MODEL == "dictionary":
        model_entities = []
    else:
        model_entities = _run_comprehend(text)

    covered = bytearray(len(text))
    for e in dict_entities:
        for i in range(e["start"], e["end"]):
            covered[i] = 1

    merged = list(dict_entities)
    for e in model_entities:
        s, en = e["start"], e["end"]
        if any(covered[s:en]):
            continue
        for i in range(s, en):
            covered[i] = 1
        if "canonical" not in e:
            e["canonical"] = ""
        merged.append(e)

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Comprehend Medical backend
# ─────────────────────────────────────────────────────────────────────────────

def _run_comprehend(text: str) -> list[dict]:
    """AWS Comprehend Medical — 20 000-character window with offset correction."""
    MAX_CHARS     = 20_000
    all_entities: list[dict] = []

    for offset in range(0, len(text), MAX_CHARS):
        window   = text[offset : offset + MAX_CHARS]
        response = _get("comprehendmedical").detect_entities_v2(Text=window)
        for ent in response.get("Entities", []):
            all_entities.append({
                "text":      ent["Text"],
                "label":     ent["Category"],
                "sub_type":  ent.get("Type", ""),
                "canonical": "",
                "start":     ent["BeginOffset"] + offset,
                "end":       ent["EndOffset"]   + offset,
                "score":     round(ent["Score"], 4),
            })

    return all_entities


# ─────────────────────────────────────────────────────────────────────────────
# GLiNER backend
# ─────────────────────────────────────────────────────────────────────────────

# 17-label clinical trial schema (used for both CI and document objects)
_GLINER_LABELS: list[str] = [
    "Drug",
    "Clinical Endpoint",
    "Clinical Response Status",
    "Biomarker",
    "Disease",
    "Study Population",
    "Questionnaire",
    "Statistical Method",
    "Organization",
    "Protocol Identifier",
    "Regulatory Identifier",
    "Study Arm",
    "Gene / Protein",
    "Adverse Event",
    "Study Design",
    "Study Visit",
    "Treatment Phase",
]

_GLINER_LABEL_MAP: dict[str, tuple[str, str]] = {
    "Drug":                    ("MEDICATION",          "DRUG"),
    "Clinical Endpoint":       ("CLINICAL_ENDPOINT",   ""),
    "Clinical Response Status":("CLINICAL_RESPONSE",   "CRITERIA"),
    "Biomarker":               ("BIOMARKER",           ""),
    "Disease":                 ("MEDICAL_CONDITION",   ""),
    "Study Population":        ("STUDY_POPULATION",    ""),
    "Questionnaire":           ("QUESTIONNAIRE",       "PRO"),
    "Statistical Method":      ("STATISTICAL_METHOD",  ""),
    "Organization":            ("ORGANIZATION",        ""),
    "Protocol Identifier":     ("PROTOCOL_ID",         "STUDY_ID"),
    "Regulatory Identifier":   ("REGULATORY_ID",       "REGULATORY_IDENTIFIER"),
    "Study Arm":               ("STUDY_ARM",            "STUDY_ARM"),
    "Gene / Protein":          ("GENE_PROTEIN",         ""),
    "Adverse Event":           ("ADVERSE_EVENT",        "AE"),
    "Study Design":            ("STUDY_DESIGN",         ""),
    "Study Visit":             ("STUDY_VISIT",          ""),
    "Treatment Phase":         ("TREATMENT_PHASE",      "PHASE"),
}

_GLINER_THRESHOLD = 0.30
_gliner_model     = None


def _get_gliner():
    global _gliner_model
    if _gliner_model is None:
        try:
            import threading, warnings, shutil, os
            # tqdm >= 4.66 removed _lock as a class-level attribute;
            # huggingface_hub accesses it during model download.
            import tqdm as _tqdm_mod
            if not hasattr(_tqdm_mod.tqdm, "_lock"):
                _tqdm_mod.tqdm._lock = threading.RLock()
            from gliner import GLiNER

            # If the model was baked into the image at /var/task/models (read-only
            # at Lambda runtime), copy it to /tmp so HF hub can write lock files.
            baked = Path("/var/task/models")
            tmp_cache = Path("/tmp/hf_cache")
            if baked.exists() and not tmp_cache.exists():
                logger.info("[NER] copying baked model cache to /tmp")
                shutil.copytree(str(baked), str(tmp_cache), symlinks=True)
            if tmp_cache.exists():
                os.environ["HF_HOME"] = str(tmp_cache)

            logger.info("[NER] loading GLiNER: %s", GLINER_MODEL_NAME)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Asking to truncate")
                _gliner_model = GLiNER.from_pretrained(GLINER_MODEL_NAME)
            logger.info("[NER] GLiNER model loaded")
        except Exception as exc:
            logger.warning("[NER] GLiNER load failed (%s) — falling back to Comprehend", exc)
            _gliner_model = None
    return _gliner_model


def _run_gliner(text: str) -> list[dict]:
    """
    GLiNER inference with a 400-word sliding window (50-word overlap).
    Handles both short CI texts (<= 400 words) and long document paragraphs.
    """
    model = _get_gliner()
    if model is None:
        logger.warning("[NER] GLiNER unavailable — using Comprehend Medical")
        return _run_comprehend(text)

    WINDOW_WORDS = 400
    OVERLAP      = 50
    words = text.split()

    if len(words) <= WINDOW_WORDS:
        windows = [(text, 0)]
    else:
        word_starts: list[int] = []
        pos = 0
        for w in words:
            p = text.index(w, pos)
            word_starts.append(p)
            pos = p + len(w)

        step    = WINDOW_WORDS - OVERLAP
        windows = []
        for i in range(0, len(words), step):
            end_word   = min(i + WINDOW_WORDS, len(words))
            char_start = word_starts[i]
            char_end   = word_starts[end_word - 1] + len(words[end_word - 1])
            windows.append((text[char_start:char_end], char_start))
            if end_word >= len(words):
                break

    seen: dict[tuple[int, int], dict] = {}

    for chunk_text, char_offset in windows:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Asking to truncate")
                predictions = model.predict_entities(
                    chunk_text, _GLINER_LABELS, threshold=_GLINER_THRESHOLD
                )
        except Exception as exc:
            logger.warning("[NER] GLiNER prediction failed: %s", exc)
            continue

        for pred in predictions:
            gliner_label        = pred.get("label", "")
            label, sub_type     = _GLINER_LABEL_MAP.get(gliner_label, (gliner_label, ""))
            start = pred["start"] + char_offset
            end   = pred["end"]   + char_offset
            score = round(pred.get("score", 0.0), 4)
            key   = (start, end)
            if key not in seen or score > seen[key]["score"]:
                seen[key] = {
                    "text":      text[start:end],
                    "label":     label,
                    "sub_type":  sub_type,
                    "canonical": "",
                    "start":     start,
                    "end":       end,
                    "score":     score,
                }

    return list(seen.values())


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing  (shared by CI and document paths)
# ─────────────────────────────────────────────────────────────────────────────

_ORG_SUFFIXES_RE = re.compile(
    r"\b(LLC|Inc\.?|Corp\.?|Ltd\.?|GmbH|BV|NV|AG|SE|PLC|R&D|"
    r"Research|Development|Pharmaceuticals?|Biotech|Sciences?|Medical)\b",
    re.IGNORECASE,
)
_PROTOCOL_ID_RE = re.compile(
    r"\b([A-Z]{2,6}-\d{6,}|[A-Z]{1,5}[0-9]{4,}|NCT\d{6,}|"
    r"Protocol[-\s](?:No\.?\s*)?[A-Z0-9-]+)\b",
    re.IGNORECASE,
)
_STUDY_ARM_RE = re.compile(
    r"\b(Arm\s+[A-Z0-9]+|Treatment\s+Arm|Study\s+Arm|Control\s+Arm|"
    r"Arm\s*[1-9]|[A-Z]+\s+Arm)\b",
    re.IGNORECASE,
)
_REGULATORY_ID_RE = re.compile(
    r"\b(IND[\s:#]+\d+|EudraCT[\s:#]*[\d-]+|CTA[\s-]\d+|JNDA[\s-]\d+|"
    r"CTRI[\s/]\d+)\b",
    re.IGNORECASE,
)
_BARE_ARM_RE      = re.compile(r"\barm\b", re.IGNORECASE)
_STAT_EPONYMS_RE  = re.compile(
    r"\b(Kaplan|Meier|Mantel|Haenszel|Cochran|Cox|Bonferroni|Dunnett|Hochberg|"
    r"Holm|Simes|Breslow|Greenwood|Aalen|Nelson)\b",
    re.IGNORECASE,
)
_CITATION_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

_ENDPOINT_ABBREVS: frozenset = frozenset({
    "PFS", "OS", "ORR", "DOR", "DoR", "PFS2", "EFS", "DFS", "RFS", "TTP",
    "TTR", "TTNT", "MFS", "TFS", "DCR", "CBR", "BICR", "IRC",
    "AUC", "Cmax", "Tmax", "t½", "CL", "MRD",
})
# Case-normalised variant used for case-insensitive membership tests
# (GLiNER may emit lowercase abbreviations such as "orr", "pfs").
_ENDPOINT_ABBREVS_UPPER: frozenset = frozenset(a.upper() for a in _ENDPOINT_ABBREVS)
# Dose-finding terms that NER (especially GLiNER) sometimes misclassifies as BIOMARKER.
# Reclassify to DOSAGE so they route to facts["dose"] rather than facts["biomarker"].
# Stored all-uppercase so the text.upper() membership test is consistent.
_DOSE_FINDING_TERMS: frozenset = frozenset({"RP2D", "RP2DS", "MTD", "MAD", "PAD"})
_PRO_ABBREVS: frozenset = frozenset({
    "HRQoL", "QoL", "PRO", "MRU", "HRU",
    "EQ-5D", "EQ5D", "BPI", "SF-36", "PGIC", "VAS", "NRS",
    "FACT-MM", "FACT", "EORTC", "EORTC-QLQ",
})
_REGIMEN_ABBREV_RE = re.compile(
    r"^[A-Z]{2,5}[a-z]$"
    r"|^[A-Za-z]{2,8}-[A-Z]{1,5}[a-z]?$"
    r"|^[A-Z]-[A-Z]{2,5}[a-z]?$",
    re.ASCII,
)


def _postprocess_entities(entities: list[dict]) -> list[dict]:
    """
    Domain-aware reclassification applied after every NER backend.

    Applied to BOTH CI and document entities so the same correction rules
    are guaranteed to run in both pipelines.
    """
    out: list[dict] = []
    for ent in entities:
        label    = ent["label"]
        sub_type = ent.get("sub_type", "")
        text     = ent["text"]

        # Suppress bare "n" / "N" (with optional trailing period) labelled
        # STUDY_POPULATION — it is statistical sample-size notation, not a
        # clinical population descriptor.  Genuine population terms ("patients",
        # "subjects", "participants") are multi-character and are never affected.
        if label == "STUDY_POPULATION" and text.strip().lower().rstrip(".") in {"n"}:
            continue

        if label == "ANATOMY" and (_STUDY_ARM_RE.search(text) or _BARE_ARM_RE.fullmatch(text.strip())):
            ent = {**ent, "label": "STUDY_ARM", "sub_type": "STUDY_ARM"}

        elif label == "PROTECTED_HEALTH_INFORMATION":
            if _STAT_EPONYMS_RE.fullmatch(text.strip()):
                ent = {**ent, "label": "STATISTICAL_METHOD", "sub_type": "EPONYM"}
            elif _CITATION_YEAR_RE.fullmatch(text.strip()) or sub_type in ("DATE", "AGE"):
                ent = {**ent, "label": "CITATION", "sub_type": "YEAR"}
            elif _ORG_SUFFIXES_RE.search(text):
                ent = {**ent, "label": "ORGANIZATION", "sub_type": "SPONSOR"}
            elif _REGULATORY_ID_RE.search(text):
                ent = {**ent, "label": "REGULATORY_ID", "sub_type": "REGULATORY_IDENTIFIER"}
            elif _PROTOCOL_ID_RE.search(text) or sub_type in ("ID", "MEDICAL_RECORD_NUMBER"):
                ent = {**ent, "label": "PROTOCOL_ID", "sub_type": "STUDY_ID"}

        elif label == "PHONE_OR_FAX" and _REGULATORY_ID_RE.search(text):
            ent = {**ent, "label": "REGULATORY_ID", "sub_type": "REGULATORY_IDENTIFIER"}

        elif label == "CLINICAL_RESPONSE" and text.upper() in _ENDPOINT_ABBREVS_UPPER:
            ent = {**ent, "label": "CLINICAL_ENDPOINT", "sub_type": "ENDPOINT"}

        elif label == "CLINICAL_RESPONSE" and text in _PRO_ABBREVS:
            ent = {**ent, "label": "QUESTIONNAIRE", "sub_type": "PRO"}

        elif label == "BIOMARKER" and text.upper() in _ENDPOINT_ABBREVS_UPPER:
            # ORR, PFS, OS, DOR etc. are clinical endpoints, not biomarkers.
            # Reclassify so they flow to facts["endpoint"] and the endpoint
            # comparator handles them rather than the biomarker comparator.
            ent = {**ent, "label": "CLINICAL_ENDPOINT", "sub_type": "ENDPOINT"}

        elif label == "BIOMARKER" and text.upper() in _DOSE_FINDING_TERMS:
            # RP2D / MTD are dose-finding targets, not biomarkers.
            # Reclassify so they flow to facts["dose"] and treatment_identity
            # rather than polluting facts["biomarker"].
            ent = {**ent, "label": "DOSAGE", "sub_type": "DOSE_FINDING"}

        elif label == "MEDICATION" and _REGIMEN_ABBREV_RE.match(text):
            ent = {**ent, "label": "TREATMENT_NAME", "sub_type": "REGIMEN"}

        out.append(ent)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Entity-to-object assignment  (document path only)
# ─────────────────────────────────────────────────────────────────────────────

def _assign_entities_to_objects(objects: list[dict], entities: list[dict]) -> None:
    """
    Map chunk-level NER entities onto individual semantic objects.

    Computes object-relative offsets (``object_start`` / ``object_end``) so
    downstream components (highlight extractor, scorers) can use
    ``obj["text"][s:e]`` directly.  Original chunk-level offsets are preserved
    as ``document_start`` / ``document_end``.

    Mutates objects in-place.
    """
    for obj in objects:
        obj_text  = obj["text"]
        obj_lower = obj_text.lower()
        assigned: list[dict] = []

        for e in entities:
            ent_lower = e["text"].lower()
            if ent_lower not in obj_lower:
                continue

            doc_start = e["start"]
            best_pos  = obj_lower.find(ent_lower)
            cursor    = 0
            while True:
                pos = obj_lower.find(ent_lower, cursor)
                if pos == -1:
                    break
                if abs(pos - doc_start) < abs(best_pos - doc_start):
                    best_pos = pos
                cursor = pos + 1

            assigned.append({
                "text":           e["text"],
                "label":          e["label"],
                "sub_type":       e.get("sub_type", ""),
                "score":          e.get("score", 0.0),
                "family":         e.get("family", ""),
                "canonical":      e.get("canonical", ""),
                "normalized":     e.get("normalized", ""),
                "abbreviation":   e.get("abbreviation", ""),
                "object_start":   best_pos,
                "object_end":     best_pos + len(e["text"]),
                "document_start": e.get("start", 0),
                "document_end":   e.get("end", 0),
            })

        obj["entities"] = _dedup_entities(assigned)


def _dedup_entities(entities: list[dict]) -> list[dict]:
    """
    Deduplicate NER entities within one object.

    Key: (text.lower(), offset_bucket=object_start//5).
    Pass 1 keeps highest-confidence per span.
    Pass 2 removes spans fully contained within a longer accepted entity.
    """
    best: dict[tuple, dict] = {}
    for ent in entities:
        bucket = ent.get("object_start", ent.get("document_start", 0)) // 5
        key    = (ent["text"].lower(), bucket)
        if key not in best or ent["score"] > best[key]["score"]:
            best[key] = ent

    candidates = sorted(best.values(), key=lambda e: e["score"], reverse=True)
    accepted: list[dict] = []
    for cand in candidates:
        cs = cand.get("object_start", 0)
        ce = cand.get("object_end",   cs + len(cand["text"]))
        dominated = any(
            a.get("object_start", 0) <= cs
            and a.get("object_end",   0) >= ce
            and a["text"].lower() != cand["text"].lower()
            for a in accepted
        )
        if not dominated:
            accepted.append(cand)
    return accepted
