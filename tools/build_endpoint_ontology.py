#!/usr/bin/env python3
"""
Build tool: derive the full clinical vocabulary from NCI Thesaurus concept IDs.

Covers every domain in clinical_ontology.py:
  • Clinical endpoints  (ORR, PFS, OS, DOR, TTF, MRD, safety, PK, dose-finding, biomarkers)
  • Response criteria   (CR, sCR, VGPR, PR, SD, PD)
  • Diseases            (MM and subtypes)
  • Drugs               (daratumumab, talquetamab, teclistamab, lenalidomide, …)
  • Adverse events      (CRS, ICANS, IRR, SAE, TEAE)
  • Lab tests           (ANC, HGB, PLT, LDH, SFLC, …)
  • Statistical methods (Kaplan-Meier, RECIST, ECOG)

Architecture
------------
  Build time  : pronto walks NCIt concept subtrees → synonym expansion →
                shared/data/clinical_concepts.json  (primary)
                lambdas/.../reranker/data/clinical_concepts.json  (mirror)

  Runtime     : Lambda cold start loads the JSON (~20 KB) in < 5 ms.
                clinical_ontology.py, reranker, aggregator, query_normalizer
                all read from the same file — single source of truth.

  The Lambda contains ZERO hardcoded synonym strings.  All clinical knowledge
  lives either in NCIt or in the additional_terms lists below (project-specific
  items that NCIt does not model: JNJ codes, combination regimen names, dosing
  schedule abbreviations, sponsor-specific identifiers).

Usage
-----
    pip install pronto requests

    python tools/build_endpoint_ontology.py --download        # download NCIt OBO
    python tools/build_endpoint_ontology.py --owl /tmp/ncit_owl/ncit.obo
    python tools/build_endpoint_ontology.py --no-ontology     # additional_terms only
    python tools/build_endpoint_ontology.py --owl ... --dry-run

NCIt concept IDs verified against NCIt r2026-03-19.
"""
from __future__ import annotations

import argparse, json, logging, re, sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# Full vocabulary category definitions
# ─────────────────────────────────────────────────────────────────────────────
# ncit_anchors  : stable NCIt concept IDs — the graph address, not synonyms
# subtree_depth : 0=anchor only, 1=+direct children, 2=+grandchildren
# additional_terms: items not modelled in NCIt (project-specific codes, etc.)
#                   These live HERE in the build tool — never in the Lambda.
# group         : logical grouping for clinical_ontology.py to reconstruct
#                 ABBREVIATIONS / DRUG_SYNONYMS / DISEASE_SYNONYMS etc.

_CATEGORIES: dict[str, dict] = {

    # ── Clinical endpoints ────────────────────────────────────────────────────

    "orr": {
        "ncit_anchors": ["C94502"],   # Response Rate (closest parent)
        "subtree_depth": 1,
        "group": "endpoint",
        "additional_terms": [
            "orr", "overall response rate", "objective response rate",
            "overall response", "response rate", "tumour response rate",
            "tumor response rate", "best overall response",
        ],
    },
    "pfs": {
        "ncit_anchors": ["C28234"],   # Progression-free Survival
        "subtree_depth": 1,
        "group": "endpoint",
        "additional_terms": [
            "pfs", "progression-free survival", "progression free survival",
            "event-free survival", "relapse-free survival",
        ],
    },
    "os": {
        # Space-padded " os " guard applied at load time (prevents matching
        # "those", "dose", "close").
        "ncit_anchors": ["C125201"],  # Overall Survival
        "subtree_depth": 1,
        "group": "endpoint",
        "additional_terms": ["overall survival", "os"],
    },
    "dor": {
        "ncit_anchors": [],           # No generic DOR concept in NCIt
        "subtree_depth": 0,
        "group": "endpoint",
        "additional_terms": [
            "dor", "duration of response", "response duration",
            "duration of remission", "time in response",
        ],
    },
    "ttf": {
        "ncit_anchors": [],           # No generic TTF concept in NCIt
        "subtree_depth": 0,
        "group": "endpoint",
        "additional_terms": [
            "ttf", "time to first response", "time to response",
            "time to best response", "time to first confirmed response",
        ],
    },
    "safety": {
        "ncit_anchors": ["C41331", "C41332"],  # Adverse Event + Adverse Reaction
        "subtree_depth": 1,
        "group": "endpoint",
        "additional_terms": [
            "ae", "adverse event", "adverse reaction", "tolerability",
            "toxicity", "safety profile", "side effect",
        ],
    },
    "pk": {
        "ncit_anchors": ["C15299"],   # Pharmacokinetics
        "subtree_depth": 1,
        "group": "endpoint",
        "additional_terms": [
            "pk", "pk/pd", "pharmacokinetic", "pharmacokinetics",
            "auc", "cmax", "half-life", "clearance", "bioavailability", "exposure",
        ],
    },
    "dose_finding": {
        "ncit_anchors": ["C94489"],   # Maximum Tolerated Dose
        "subtree_depth": 1,
        "group": "endpoint",
        "additional_terms": [
            "rp2d", "recommended phase 2 dose", "recommended phase ii dose",
            "mtd", "maximum tolerated dose",
            "dose limiting toxicity", "dose-limiting toxicity", "dlt",
            "dose escalation", "dose expansion", "starting dose",
        ],
    },
    "mrd": {
        "ncit_anchors": ["C3896"],    # Measurable Residual Disease
        "subtree_depth": 1,
        "group": "endpoint",
        "additional_terms": [
            "mrd", "minimal residual disease", "measurable residual disease",
            "mrd negativity", "mrd-negative", "mrd negative",
        ],
    },
    "biomarker": {
        "ncit_anchors": ["C16342"],   # Biomarker (depth=0 — subtree is too broad)
        "subtree_depth": 0,
        "group": "endpoint",
        "additional_terms": [
            "biomarker", "ctdna", "circulating tumor dna", "circulating tumour dna",
            "bcma expression", "cd38 expression", "genomic marker",
            "predictive biomarker", "pharmacodynamic marker",
        ],
    },

    # ── Response criteria ─────────────────────────────────────────────────────

    "response_cr": {
        "ncit_anchors": ["C4870"],    # Complete Remission
        "subtree_depth": 1,
        "group": "response_criteria",
        "additional_terms": ["cr", "complete response", "complete remission", "complete remission/response"],
    },
    "response_scr": {
        "ncit_anchors": [],
        "subtree_depth": 0,
        "group": "response_criteria",
        "additional_terms": ["scr", "stringent complete response", "stringent complete remission"],
    },
    "response_vgpr": {
        "ncit_anchors": ["C123618"],  # Very Good Partial Response
        "subtree_depth": 0,
        "group": "response_criteria",
        "additional_terms": ["vgpr", "very good partial response", "very good partial remission"],
    },
    "response_pr": {
        "ncit_anchors": ["C18212"],   # Partial Response
        "subtree_depth": 0,
        "group": "response_criteria",
        "additional_terms": ["pr", "partial response", "partial remission"],
    },
    "response_mr": {
        "ncit_anchors": [],
        "subtree_depth": 0,
        "group": "response_criteria",
        "additional_terms": ["mr", "minimal response", "minor response"],
    },
    "response_sd": {
        "ncit_anchors": [],
        "subtree_depth": 0,
        "group": "response_criteria",
        "additional_terms": ["sd", "stable disease"],
    },
    "response_pd": {
        "ncit_anchors": [],
        "subtree_depth": 0,
        "group": "response_criteria",
        "additional_terms": ["pd", "progressive disease", "disease progression"],
    },

    # ── Diseases ──────────────────────────────────────────────────────────────

    "disease_mm": {
        "ncit_anchors": ["C3242"],    # Multiple Myeloma
        "subtree_depth": 0,
        "group": "disease",
        "canonical_name": "multiple myeloma",
        "additional_terms": [
            "mm", "multiple myeloma", "plasma cell myeloma",
            "kahler disease", "myelomatosis",
        ],
    },
    "disease_rrmm": {
        "ncit_anchors": [],           # NCIt has no RRMM-specific concept
        "subtree_depth": 0,
        "group": "disease",
        "canonical_name": "relapsed/refractory multiple myeloma",
        "additional_terms": [
            "rrmm", "r/r mm", "relapsed/refractory multiple myeloma",
            "relapsed refractory multiple myeloma",
            "relapsed or refractory multiple myeloma",
        ],
    },
    "disease_ndmm": {
        "ncit_anchors": [],
        "subtree_depth": 0,
        "group": "disease",
        "canonical_name": "newly diagnosed multiple myeloma",
        "additional_terms": [
            "ndmm", "newly diagnosed multiple myeloma", "newly diagnosed mm",
            "treatment-naive multiple myeloma",
        ],
    },
    "disease_smm": {
        "ncit_anchors": ["C7149"],    # Smoldering Multiple Myeloma
        "subtree_depth": 0,
        "group": "disease",
        "canonical_name": "smoldering multiple myeloma",
        "additional_terms": ["smm", "smoldering multiple myeloma", "smoldering myeloma"],
    },

    # ── Drug targets / biomarkers ─────────────────────────────────────────────
    # NCIt codes for targets are gene/protein entries; use additional_terms
    # for the clinical abbreviations used in protocols.

    "target_bcma": {
        "ncit_anchors": [],
        "subtree_depth": 0,
        "group": "target",
        "additional_terms": [
            "bcma", "b-cell maturation antigen", "tnfrsf17",
            "anti-bcma", "bcma-directed",
        ],
    },
    "target_cd38": {
        "ncit_anchors": [],
        "subtree_depth": 0,
        "group": "target",
        "additional_terms": [
            "cd38", "cluster of differentiation 38",
            "anti-cd38", "cd38-directed",
        ],
    },
    "target_gprc5d": {
        "ncit_anchors": [],
        "subtree_depth": 0,
        "group": "target",
        "additional_terms": [
            "gprc5d", "g protein-coupled receptor class c group 5 member d",
            "anti-gprc5d", "gprc5d-directed",
        ],
    },
    "target_cd19": {
        "ncit_anchors": [],
        "subtree_depth": 0,
        "group": "target",
        "additional_terms": ["cd19", "cluster of differentiation 19", "anti-cd19"],
    },
    "target_cd20": {
        "ncit_anchors": [],
        "subtree_depth": 0,
        "group": "target",
        "additional_terms": ["cd20", "cluster of differentiation 20", "anti-cd20"],
    },

    # ── Drugs (depth=0 to avoid noisy chemical-name children) ─────────────────

    "drug_daratumumab": {
        "ncit_anchors": ["C74007"],   # Daratumumab
        "subtree_depth": 0,
        "group": "drug",
        "canonical_name": "daratumumab",
        "additional_terms": [
            "daratumumab", "darzalex", "dara", "anti-cd38 antibody",
            "cd38 monoclonal antibody",
        ],
    },
    "drug_talquetamab": {
        "ncit_anchors": ["C171840"],  # Talquetamab
        "subtree_depth": 0,
        "group": "drug",
        "canonical_name": "talquetamab",
        "additional_terms": [
            "talquetamab", "talvey", "jnj-64407564", "jnj64407564",
            "anti-gprc5d", "gprc5d bispecific", "gprc5d\u00d7cd3",
        ],
    },
    "drug_teclistamab": {
        "ncit_anchors": ["C136823"],  # Teclistamab
        "subtree_depth": 0,
        "group": "drug",
        "canonical_name": "teclistamab",
        "additional_terms": [
            "teclistamab", "tecvayli", "jnj-64007957", "jnj64007957",
            "anti-bcma", "bcma bispecific", "bcma\u00d7cd3",
        ],
    },
    "drug_lenalidomide": {
        "ncit_anchors": ["C2668"],    # Lenalidomide
        "subtree_depth": 0,
        "group": "drug",
        "canonical_name": "lenalidomide",
        "additional_terms": ["lenalidomide", "revlimid", "len", "cc-5013"],
    },
    "drug_bortezomib": {
        "ncit_anchors": ["C1851"],    # Bortezomib
        "subtree_depth": 0,
        "group": "drug",
        "canonical_name": "bortezomib",
        "additional_terms": ["bortezomib", "velcade", "bort", "ps-341"],
    },
    "drug_carfilzomib": {
        "ncit_anchors": ["C52196"],   # Carfilzomib
        "subtree_depth": 0,
        "group": "drug",
        "canonical_name": "carfilzomib",
        "additional_terms": ["carfilzomib", "kyprolis", "cfz", "pr-171"],
    },
    "drug_pomalidomide": {
        "ncit_anchors": ["C72560"],   # Pomalidomide
        "subtree_depth": 0,
        "group": "drug",
        "canonical_name": "pomalidomide",
        "additional_terms": ["pomalidomide", "pomalyst", "poma", "cc-4047"],
    },
    "drug_dexamethasone": {
        "ncit_anchors": ["C422"],     # Dexamethasone
        "subtree_depth": 0,
        "group": "drug",
        "canonical_name": "dexamethasone",
        "additional_terms": ["dexamethasone", "dex", "decadron"],
    },
    "drug_isatuximab": {
        "ncit_anchors": ["C90578"],   # Isatuximab
        "subtree_depth": 0,
        "group": "drug",
        "canonical_name": "isatuximab",
        "additional_terms": ["isatuximab", "sarclisa", "isa", "sar650984"],
    },

    # ── Adverse event subtypes ────────────────────────────────────────────────

    "ae_crs": {
        "ncit_anchors": ["C78251"],   # Cytokine Release Syndrome
        "subtree_depth": 0,
        "group": "ae_type",
        "additional_terms": ["crs", "cytokine release syndrome", "cytokine storm"],
    },
    "ae_icans": {
        "ncit_anchors": ["C162909"],  # ICANS
        "subtree_depth": 0,
        "group": "ae_type",
        "additional_terms": [
            "icans", "immune effector cell associated neurotoxicity syndrome",
            "immune-effector cell-associated neurotoxicity syndrome",
            "neurotoxicity",
        ],
    },
    "ae_irr": {
        "ncit_anchors": ["C78361"],   # Infusion-Related Reaction
        "subtree_depth": 0,
        "group": "ae_type",
        "additional_terms": ["irr", "infusion-related reaction", "infusion related reaction"],
    },
    "ae_sae": {
        "ncit_anchors": ["C41335"],   # Serious Adverse Event
        "subtree_depth": 0,
        "group": "ae_type",
        "additional_terms": ["sae", "serious adverse event"],
    },
    "ae_teae": {
        "ncit_anchors": [],
        "subtree_depth": 0,
        "group": "ae_type",
        "additional_terms": [
            "teae", "treatment-emergent adverse event",
            "treatment emergent adverse event",
        ],
    },

    # ── Lab tests ─────────────────────────────────────────────────────────────

    "lab_anc": {
        "ncit_anchors": ["C63321"],   # Absolute Neutrophil Count
        "subtree_depth": 0,
        "group": "lab",
        "additional_terms": ["anc", "absolute neutrophil count"],
    },
    "lab_hgb": {
        "ncit_anchors": ["C16676"],   # Hemoglobin
        "subtree_depth": 0,
        "group": "lab",
        "additional_terms": ["hgb", "hb", "hemoglobin", "haemoglobin"],
    },
    "lab_plt": {
        "ncit_anchors": ["C51951"],   # Platelet Count
        "subtree_depth": 0,
        "group": "lab",
        "additional_terms": ["plt", "platelet count", "thrombocyte count"],
    },
    "lab_ldh": {
        "ncit_anchors": [],
        "subtree_depth": 0,
        "group": "lab",
        "additional_terms": ["ldh", "lactate dehydrogenase", "lactic dehydrogenase"],
    },
    "lab_sflc": {
        "ncit_anchors": [],
        "subtree_depth": 0,
        "group": "lab",
        "additional_terms": [
            "sflc", "serum free light chain", "flc", "free light chain",
            "kappa free light chain", "lambda free light chain",
        ],
    },

    # ── Statistical methods / frameworks ─────────────────────────────────────

    "stats_km": {
        "ncit_anchors": ["C85436"],   # Kaplan-Meier Survival Plot
        "subtree_depth": 0,
        "group": "stats",
        "additional_terms": ["kaplan-meier", "kaplan meier", "km curve", "km estimate"],
    },
    "stats_recist": {
        "ncit_anchors": ["C124414", "C124415"],  # RECIST 1.0 + 1.1
        "subtree_depth": 0,
        "group": "stats",
        "additional_terms": [
            "recist", "recist 1.1", "recist 1.0",
            "response evaluation criteria in solid tumors",
        ],
    },
    "stats_ecog": {
        "ncit_anchors": ["C102116"],  # ECOG Performance Status
        "subtree_depth": 0,
        "group": "stats",
        "additional_terms": ["ecog", "ecog ps", "ecog performance status",
                              "eastern cooperative oncology group"],
    },
    "stats_imwg": {
        "ncit_anchors": [],
        "subtree_depth": 0,
        "group": "stats",
        "additional_terms": [
            "imwg", "international myeloma working group",
            "imwg criteria", "imwg response criteria",
        ],
    },

    # ── Clinical trial roles ──────────────────────────────────────────────────
    # NCIt IDs for trial personnel roles are sparse; additional_terms cover the
    # standard role names used in protocols and regulatory documents.
    # Add NCIt anchors here when verified (e.g. NCIt "Principal Investigator").

    "role_pi": {
        "ncit_anchors": [],
        "subtree_depth": 0,
        "group": "role",
        "canonical_name": "principal investigator",
        "additional_terms": [
            "principal investigator", "pi",
            "lead investigator", "site principal investigator",
            "qualified investigator", "study investigator",
        ],
    },
    "role_ci": {
        "ncit_anchors": [],
        "subtree_depth": 0,
        "group": "role",
        "canonical_name": "chief investigator",
        "additional_terms": [
            "chief investigator", "ci",
            "coordinating investigator", "lead investigator",
        ],
    },
    "role_si": {
        "ncit_anchors": [],
        "subtree_depth": 0,
        "group": "role",
        "canonical_name": "sub-investigator",
        "additional_terms": [
            "sub-investigator", "sub-i",
            "co-investigator", "coinvestigator",
        ],
    },
    "role_coordinator": {
        "ncit_anchors": [],
        "subtree_depth": 0,
        "group": "role",
        "canonical_name": "study coordinator",
        "additional_terms": [
            "study coordinator", "clinical research coordinator", "crc",
            "site coordinator", "research coordinator",
        ],
    },
    "role_monitor": {
        "ncit_anchors": [],
        "subtree_depth": 0,
        "group": "role",
        "canonical_name": "monitor",
        "additional_terms": [
            "monitor", "clinical research associate", "cra",
            "site monitor", "clinical monitor",
        ],
    },
    "role_sponsor": {
        "ncit_anchors": [],
        "subtree_depth": 0,
        "group": "role",
        "canonical_name": "sponsor",
        "additional_terms": [
            "sponsor", "study sponsor", "sponsor-investigator",
        ],
    },
}

_NCIT_OBO_URL = "https://purl.obolibrary.org/obo/ncit.obo"
_REPO_ROOT    = Path(__file__).resolve().parent.parent

# Canonical: shared/data/ — accessible from every Lambda and every shared module.
# Mirror:    reranker/data/ — fallback when shared/ is not in the Lambda package.
_PRIMARY_PATH  = _REPO_ROOT / "shared" / "data" / "clinical_concepts.json"
_RERANKER_PATH = _REPO_ROOT / "lambdas" / "search" / "reranker" / "data" / "clinical_concepts.json"


# ── Download ──────────────────────────────────────────────────────────────────

def download_ncit_obo(dest: Path) -> Path:
    try:
        import requests
    except ImportError:
        logger.error("pip install requests")
        sys.exit(1)
    dest.mkdir(parents=True, exist_ok=True)
    obo_path = dest / "ncit.obo"
    if obo_path.exists():
        logger.info("Cached OBO at %s — skipping download.", obo_path)
        return obo_path
    logger.info("Downloading NCIt OBO (~237 MB) …")
    with requests.get(_NCIT_OBO_URL, stream=True, timeout=600, allow_redirects=True) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        done  = 0
        with obo_path.open("wb") as f:
            for chunk in resp.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done/total*100:5.1f}%  ({done>>20}/{total>>20} MB)",
                          end="", flush=True)
    print()
    logger.info("Saved %.1f MB → %s", obo_path.stat().st_size / (1<<20), obo_path)
    return obo_path


# ── Synonym expansion ─────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\u2013\u2014\u2012]", "-", s)
    return re.sub(r"\s+", " ", s)

def _is_noise(s: str) -> bool:
    """
    Return True only for strings that carry no lexical information at all:
    blank, single-character, or purely numeric/whitespace.

    Chemical/IUPAC names, long brand names, enzyme codes, CAS numbers, etc.
    are all kept — callers may need them for full-text matching in CIs.
    """
    return (
        len(s) < 2
        or bool(re.fullmatch(r'[\d\s\-\.]+', s))
        or s.startswith('[')
    )

def expand_with_ncit(obo_path: Path) -> dict[str, list[str]]:
    """Walk NCIt subtrees for each anchor and merge with additional_terms."""
    try:
        import pronto
    except ImportError:
        logger.error("pip install pronto")
        sys.exit(1)

    logger.info("Parsing %s (~60 s) …", obo_path.name)
    onto = pronto.Ontology(str(obo_path))
    logger.info("Loaded %d terms.", len(onto.terms()))

    child_index: dict[str, list[str]] = {}
    for term in onto.terms():
        for parent in term.superclasses(distance=1, with_self=False):
            child_index.setdefault(parent.id, []).append(term.id)

    def _subtree(anchor_id: str, depth: int) -> list[str]:
        visited: set[str] = set()
        queue = [(anchor_id, 0)]
        while queue:
            cid, d = queue.pop(0)
            if cid in visited:
                continue
            visited.add(cid)
            if d < depth:
                for child in child_index.get(cid, []):
                    queue.append((child, d + 1))
        return list(visited)

    def _all_labels(term) -> list[str]:
        labels = [term.name] if term.name else []
        for syn in term.synonyms:
            txt = getattr(syn, "description", None) or (syn if isinstance(syn, str) else "")
            if txt:
                labels.append(txt)
        return labels

    result:    dict[str, set[str]] = {cat: set() for cat in _CATEGORIES}
    ncit_hits: dict[str, int]      = {cat: 0      for cat in _CATEGORIES}

    for cat, cfg in _CATEGORIES.items():
        for short_id in cfg["ncit_anchors"]:
            full_id = f"NCIT:{short_id}" if ":" not in short_id else short_id
            for cid in _subtree(full_id, cfg["subtree_depth"]):
                term = onto.get_term(cid)
                if term is None:
                    continue
                for lbl in _all_labels(term):
                    n = _norm(lbl)
                    if not _is_noise(n):
                        result[cat].add(n)
                ncit_hits[cat] += 1
        for t in cfg["additional_terms"]:
            result[cat].add(_norm(t))

    for cat in _CATEGORIES:
        logger.info("  %-22s  %3d NCIt  →  %3d synonyms",
                    cat, ncit_hits[cat], len(result[cat]))

    return {cat: sorted(syns) for cat, syns in result.items()}


def no_ontology_output() -> dict[str, list[str]]:
    return {cat: sorted(set(_norm(t) for t in cfg["additional_terms"]))
            for cat, cfg in _CATEGORIES.items()}


# ── Output ────────────────────────────────────────────────────────────────────

def write_output(categories: dict[str, list[str]], source: str,
                 dry_run: bool, ncit_ver: str = "") -> None:
    total = sum(len(v) for v in categories.values())

    # Include group metadata so loaders can reconstruct structured dicts.
    groups: dict[str, list[str]] = {}
    for cat, cfg in _CATEGORIES.items():
        g = cfg.get("group", "other")
        groups.setdefault(g, []).append(cat)

    # Per-category metadata — lets clinical_ontology.py auto-populate
    # DRUG_SYNONYMS, DISEASE_SYNONYMS, ROLE_SYNONYMS without any hardcoding.
    metadata = {
        cat: {k: cfg[k] for k in ("group", "canonical_name") if k in cfg}
        for cat, cfg in _CATEGORIES.items()
    }

    payload: dict = {
        "built_at":   datetime.now(timezone.utc).isoformat(),
        "source":     source,
        "groups":     groups,
        "metadata":   metadata,
        "categories": categories,
    }
    if ncit_ver:
        payload["ncit_version"] = ncit_ver

    text = json.dumps(payload, indent=2, ensure_ascii=False)

    if dry_run:
        logger.info("Dry-run — not writing.  %d categories, %d total synonyms.", len(categories), total)
        for cat, syns in categories.items():
            logger.info("  %-22s  %d", cat, len(syns))
        return

    for path, label in [(_PRIMARY_PATH, "primary"), (_RERANKER_PATH, "reranker mirror")]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        logger.info("Written (%s) → %s  (%.1f KB)", label, path, path.stat().st_size / 1024)

    logger.info("Total: %d categories, %d synonyms.", len(categories), total)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--download",     action="store_true", help="Download NCIt OBO automatically.")
    src.add_argument("--owl",          metavar="PATH",      help="Path to a local ncit.obo file.")
    src.add_argument("--no-ontology",  action="store_true", help="additional_terms only.")
    p.add_argument("--owl-dir",   default="/tmp/ncit_owl",  help="Cache dir (default: /tmp/ncit_owl).")
    p.add_argument("--dry-run",   action="store_true",      help="Show counts without writing.")
    args = p.parse_args()

    if args.no_ontology or (not args.download and not args.owl):
        cats, src_desc, ver = no_ontology_output(), "additional_terms only", ""
    elif args.download:
        obo = download_ncit_obo(Path(args.owl_dir))
        cats, src_desc, ver = expand_with_ncit(obo), "NCIt OBO (OBO Foundry) + additional_terms", ""
    else:
        obo = Path(args.owl)
        if not obo.exists():
            logger.error("File not found: %s", obo)
            sys.exit(1)
        m   = re.search(r'\d{4}-\d{2}-\d{2}', obo.name)
        cats, src_desc, ver = (expand_with_ncit(obo),
                               f"NCIt OBO ({obo.name}) + additional_terms",
                               m.group(0) if m else "")

    write_output(cats, src_desc, args.dry_run, ver)


if __name__ == "__main__":
    main()
