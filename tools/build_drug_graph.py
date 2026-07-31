#!/usr/bin/env python3
"""
Build tool: construct drug_graph.json from NCIt OBO + RxNorm + custom aliases.

Architecture
────────────
  NCIt OBO        ──┐
  RxNorm API      ──┼──► DrugGraphBuilder ──► localfiles/drug_graph.json
  drug_aliases.json ──┘

  The resulting graph is loaded at Lambda cold-start by shared/drug_identity.py.
  No clinical knowledge is hardcoded in the runtime — it all lives in the graph.

Usage
─────
  pip install pronto requests

  # Full build (NCIt synonyms + RxNorm brand names + custom aliases):
  python tools/build_drug_graph.py --ncit /tmp/ncit_owl/ncit.obo --rxnorm

  # NCIt + custom aliases only (no RxNorm API calls):
  python tools/build_drug_graph.py --ncit /tmp/ncit_owl/ncit.obo

  # Custom aliases only (fastest — no external dependencies):
  python tools/build_drug_graph.py --no-ontology

  # Dry-run (print summary, do not write):
  python tools/build_drug_graph.py --ncit /tmp/ncit_owl/ncit.obo --dry-run

Output schema (localfiles/drug_graph.json)
──────────────────────────────────────────
  {
    "_meta": { "built": "...", "sources": [...] },
    "nodes": {
      "<canonical_id>": {
        "canonical_id": str,
        "display":      str,           # proper-case display name
        "aliases":      [str, ...],    # lower-cased; includes NCIt synonyms + custom
        "ncit_code":    str | null,    # e.g. "C136823"
        "rxcui":        str | null,    # RxNorm concept unique identifier
        "family":       str | null,    # mechanism class key
        "target":       str | null     # molecular target
      }
    },
    "families": {
      "<family_key>": { "related": [<family_key>, ...] }
    },
    "combos": [
      { "id": str, "pattern": str (regex), "components": [canonical_id, ...] }
    ]
  }
"""
from __future__ import annotations

import argparse, json, logging, re, sys, time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

_REPO_ROOT    = Path(__file__).resolve().parent.parent
_ALIASES_PATH = _REPO_ROOT / "localfiles" / "drug_aliases.json"
_OUT_PATH     = _REPO_ROOT / "localfiles" / "drug_graph.json"

# ─────────────────────────────────────────────────────────────────────────────
# Seed configuration
# ─────────────────────────────────────────────────────────────────────────────
# This is the ONLY place family/target metadata is declared.
# NCIt supplies synonyms; RxNorm supplies brand names; custom aliases supply
# trial codes. The seed just tells us WHICH NCIt concept to look up and WHAT
# clinical family it belongs to.
#
# NCIt codes verified against NCIt r2026-03-19.

_DRUG_SEEDS: dict[str, dict] = {

    # ── BCMA bispecific antibodies ────────────────────────────────────────────
    "teclistamab": {
        "ncit_code": "C136823", "display": "Teclistamab",
        "family": "bcma_bispecific", "target": "BCMA",
    },
    "elranatamab": {
        "ncit_code": "C146860", "display": "Elranatamab",
        "family": "bcma_bispecific", "target": "BCMA",
    },

    # ── GPRC5D bispecific antibodies ──────────────────────────────────────────
    "talquetamab": {
        "ncit_code": "C171840", "display": "Talquetamab",
        "family": "gprc5d_bispecific", "target": "GPRC5D",
    },

    # ── FcRH5 bispecific antibodies ───────────────────────────────────────────
    "cevostamab": {
        "ncit_code": "C139549", "display": "Cevostamab",
        "family": "fcrh5_bispecific", "target": "FcRH5",
    },

    # ── Anti-CD38 monoclonal antibodies ───────────────────────────────────────
    "daratumumab": {
        "ncit_code": "C74007",  "display": "Daratumumab",
        "family": "anti_cd38", "target": "CD38",
    },
    "isatuximab": {
        "ncit_code": "C90578",  "display": "Isatuximab",
        "family": "anti_cd38", "target": "CD38",
    },

    # ── Proteasome inhibitors ─────────────────────────────────────────────────
    "bortezomib": {
        "ncit_code": "C1851",   "display": "Bortezomib",
        "family": "proteasome_inhibitor", "target": "Proteasome",
    },
    "carfilzomib": {
        "ncit_code": "C52196",  "display": "Carfilzomib",
        "family": "proteasome_inhibitor", "target": "Proteasome",
    },
    "ixazomib": {
        "ncit_code": "C97940",  "display": "Ixazomib",
        "family": "proteasome_inhibitor", "target": "Proteasome",
    },

    # ── IMiDs ─────────────────────────────────────────────────────────────────
    "thalidomide": {
        "ncit_code": "C853",    "display": "Thalidomide",
        "family": "imid", "target": "CRBN",
    },
    "lenalidomide": {
        "ncit_code": "C2668",   "display": "Lenalidomide",
        "family": "imid", "target": "CRBN",
    },
    "pomalidomide": {
        "ncit_code": "C72560",  "display": "Pomalidomide",
        "family": "imid", "target": "CRBN",
    },

    # ── CELMoDs (next-gen CRBN modulators) ───────────────────────────────────
    "iberdomide": {
        "ncit_code": "C129048", "display": "Iberdomide",
        "family": "celmod", "target": "CRBN",
    },
    "mezigdomide": {
        "ncit_code": "C146660", "display": "Mezigdomide",
        "family": "celmod", "target": "CRBN",
    },

    # ── Corticosteroids ───────────────────────────────────────────────────────
    "dexamethasone": {
        "ncit_code": "C422",    "display": "Dexamethasone",
        "family": "steroid", "target": None,
    },
    "prednisone": {
        "ncit_code": "C770",    "display": "Prednisone",
        "family": "steroid", "target": None,
    },

    # ── Alkylating agents ─────────────────────────────────────────────────────
    "cyclophosphamide": {
        "ncit_code": "C405",    "display": "Cyclophosphamide",
        "family": "alkylating_agent", "target": None,
    },
    "melphalan": {
        "ncit_code": "C642",    "display": "Melphalan",
        "family": "alkylating_agent", "target": None,
    },
    "bendamustine": {
        "ncit_code": "C73261",  "display": "Bendamustine",
        "family": "alkylating_agent", "target": None,
    },

    # ── Anti-SLAMF7 ───────────────────────────────────────────────────────────
    "elotuzumab": {
        "ncit_code": "C66982",  "display": "Elotuzumab",
        "family": "anti_slamf7", "target": "SLAMF7",
    },

    # ── HDAC inhibitors ───────────────────────────────────────────────────────
    "panobinostat": {
        "ncit_code": "C66948",  "display": "Panobinostat",
        "family": "hdac_inhibitor", "target": "HDAC",
    },

    # ── BCL-2 inhibitors ──────────────────────────────────────────────────────
    "venetoclax": {
        "ncit_code": "C103147", "display": "Venetoclax",
        "family": "bcl2_inhibitor", "target": "BCL-2",
    },

    # ── BCMA CAR-T therapies ──────────────────────────────────────────────────
    "ciltacabtagene_autoleucel": {
        "ncit_code": "C148498", "display": "Ciltacabtagene Autoleucel",
        "family": "bcma_cart", "target": "BCMA",
    },
    "idecabtagene_vicleucel": {
        "ncit_code": "C117729", "display": "Idecabtagene Vicleucel",
        "family": "bcma_cart", "target": "BCMA",
    },
    "orvacabtagene_autoleucel": {
        "ncit_code": "C147523", "display": "Orvacabtagene Autoleucel",
        "family": "bcma_cart", "target": "BCMA",
    },

    # ── Anthracyclines ────────────────────────────────────────────────────────
    "doxorubicin": {
        "ncit_code": "C456",    "display": "Doxorubicin",
        "family": "anthracycline", "target": None,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\u2013\u2014\u2012]", "-", s)
    return re.sub(r"\s+", " ", s)


def _is_noise(s: str) -> bool:
    """Reject chemical IUPAC names, CAS numbers, Greek-letter strings, etc."""
    if len(s) < 2:
        return True
    if re.fullmatch(r"[\d\s\-\.]+", s):      # purely numeric / CAS
        return True
    if s.startswith("[") or s.startswith("("):
        return True
    # Very long strings are usually IUPAC or SMILES-like
    if len(s) > 120:
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# NCIt synonym extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_ncit_synonyms(obo_path: Path) -> dict[str, list[str]]:
    """
    Return {canonical_id: [synonym, ...]} using NCIt OBO parsed by pronto.
    Only processes concept IDs listed in _DRUG_SEEDS.
    """
    try:
        import pronto  # type: ignore
    except ImportError:
        logger.error("pip install pronto  (required for --ncit mode)")
        sys.exit(1)

    logger.info("Parsing %s …", obo_path.name)
    onto = pronto.Ontology(str(obo_path))
    logger.info("Loaded %d terms.", len(onto.terms()))

    code_to_canonical = {f"NCIT:{cfg['ncit_code']}": cid
                         for cid, cfg in _DRUG_SEEDS.items()
                         if cfg.get("ncit_code")}

    result: dict[str, list[str]] = {}
    for full_id, canonical_id in code_to_canonical.items():
        term = onto.get_term(full_id)
        if term is None:
            logger.warning("NCIt term not found: %s (%s)", full_id, canonical_id)
            continue
        syns: set[str] = set()
        if term.name:
            syns.add(_norm(term.name))
        for syn in term.synonyms:
            raw = getattr(syn, "description", None) or (syn if isinstance(syn, str) else "")
            if raw:
                n = _norm(raw)
                if not _is_noise(n):
                    syns.add(n)
        result[canonical_id] = sorted(syns)
        logger.info("  %-38s  %d NCIt synonyms", canonical_id, len(syns))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# RxNorm synonym enrichment
# ─────────────────────────────────────────────────────────────────────────────

_RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"
_RXNORM_TTY  = "IN+BN+BPCK"  # Ingredient, Brand Name, Branded Pack


def _rxnorm_get(path: str) -> dict:
    try:
        import requests  # type: ignore
    except ImportError:
        logger.error("pip install requests  (required for --rxnorm mode)")
        sys.exit(1)
    url = f"{_RXNORM_BASE}{path}"
    resp = requests.get(url, timeout=15, headers={"Accept": "application/json"})
    if resp.status_code == 200:
        return resp.json()
    return {}


def _fetch_rxcui(name: str) -> str | None:
    data = _rxnorm_get(f"/rxcui.json?name={name}&search=2")
    ids = (data.get("idGroup") or {}).get("rxnormId") or []
    return ids[0] if ids else None


def _fetch_rxnorm_synonyms(rxcui: str) -> list[str]:
    data = _rxnorm_get(f"/rxcui/{rxcui}/related.json?tty={_RXNORM_TTY}")
    concept_groups = (data.get("relatedGroup") or {}).get("conceptGroup") or []
    syns: set[str] = set()
    for group in concept_groups:
        for prop in (group.get("conceptProperties") or []):
            name = prop.get("name") or ""
            if name:
                n = _norm(name)
                if not _is_noise(n):
                    syns.add(n)
    return sorted(syns)


def enrich_with_rxnorm(nodes: dict) -> dict[str, str | None]:
    """
    Lookup RxCUI for each canonical drug and fetch brand name synonyms.
    Returns {canonical_id: rxcui | None}.
    Adds brand names directly into nodes[canonical_id]["aliases"].
    """
    rxcui_map: dict[str, str | None] = {}
    for canonical_id, node in nodes.items():
        display = _DRUG_SEEDS[canonical_id]["display"]
        rxcui = _fetch_rxcui(display)
        rxcui_map[canonical_id] = rxcui
        if rxcui:
            brand_syns = _fetch_rxnorm_synonyms(rxcui)
            existing   = set(node["aliases"])
            new_syns   = [s for s in brand_syns if s not in existing and not _is_noise(s)]
            node["aliases"] = sorted(set(node["aliases"]) | set(new_syns))
            logger.info("  %-38s  rxcui=%-10s  +%d RxNorm synonyms",
                        canonical_id, rxcui, len(new_syns))
        else:
            logger.warning("  %-38s  no RxCUI found", canonical_id)
        time.sleep(0.12)   # RxNorm rate-limit: ~10 req/s
    return rxcui_map


# ─────────────────────────────────────────────────────────────────────────────
# Custom alias layer
# ─────────────────────────────────────────────────────────────────────────────

def load_custom_aliases(path: Path) -> dict:
    """Load drug_aliases.json; return the full dict (aliases, combos, families)."""
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        logger.warning("drug_aliases.json not found at %s — skipping custom layer.", path)
        return {"aliases": {}, "combos": [], "families": {}}
    # Strip comment keys (keys starting with _)
    aliases = {k: v for k, v in raw.get("aliases", {}).items()
               if not k.startswith("_")}
    return {
        "aliases":  aliases,
        "combos":   raw.get("combos", []),
        "families": {k: v for k, v in raw.get("families", {}).items()
                     if not k.startswith("_")},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(
    ncit_synonyms: dict[str, list[str]],
    rxcui_map:     dict[str, str | None],
    custom:        dict,
) -> dict:
    """
    Assemble the final graph from all sources.
    Priority: NCIt synonyms < custom aliases (custom aliases win on conflicts).
    """
    nodes: dict[str, dict] = {}

    for canonical_id, seed in _DRUG_SEEDS.items():
        aliases: set[str] = set()

        # Seed canonical_id itself always resolves to itself
        aliases.add(canonical_id.replace("_", " "))  # e.g. "ciltacabtagene autoleucel"
        aliases.add(canonical_id)

        # NCIt-derived synonyms
        for syn in ncit_synonyms.get(canonical_id, []):
            if not _is_noise(syn):
                aliases.add(syn)

        nodes[canonical_id] = {
            "canonical_id": canonical_id,
            "display":      seed["display"],
            "aliases":      sorted(aliases),
            "ncit_code":    seed.get("ncit_code"),
            "rxcui":        rxcui_map.get(canonical_id),
            "family":       seed.get("family"),
            "target":       seed.get("target"),
        }

    # Custom aliases — add to the target node (or log if unknown)
    for alias, target_id in custom["aliases"].items():
        a = _norm(alias)
        if target_id in nodes:
            existing = nodes[target_id]["aliases"]
            if a not in existing:
                nodes[target_id]["aliases"] = sorted(set(existing) | {a})
        else:
            logger.warning("Custom alias '%s' → '%s': target not in seeds — ignored.", alias, target_id)

    return {
        "_meta": {
            "built":   datetime.now(timezone.utc).isoformat(),
            "sources": _active_sources,
        },
        "nodes":    nodes,
        "families": custom.get("families", {}),
        "combos":   custom.get("combos", []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

_active_sources: list[str] = []


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--ncit",         metavar="PATH",  help="Path to ncit.obo file.")
    src.add_argument("--no-ontology",  action="store_true",
                     help="Skip NCIt; use custom aliases only.")
    p.add_argument("--rxnorm",   action="store_true",
                   help="Enrich with RxNorm brand names (requires internet).")
    p.add_argument("--aliases",  default=str(_ALIASES_PATH),
                   metavar="PATH", help="Path to drug_aliases.json.")
    p.add_argument("--out",      default=str(_OUT_PATH),
                   metavar="PATH", help="Output path for drug_graph.json.")
    p.add_argument("--dry-run",  action="store_true",
                   help="Print summary without writing output.")
    args = p.parse_args()

    # 1. NCIt synonyms
    ncit_synonyms: dict[str, list[str]] = {}
    if not args.no_ontology and args.ncit:
        obo_path = Path(args.ncit)
        if not obo_path.exists():
            logger.error("NCIt OBO not found: %s", obo_path)
            sys.exit(1)
        ncit_synonyms = _extract_ncit_synonyms(obo_path)
        _active_sources.append(f"NCIt OBO ({obo_path.name})")
    else:
        logger.info("Skipping NCIt OBO — aliases only.")
    _active_sources.append("custom aliases")

    # 2. Load custom aliases (always)
    custom = load_custom_aliases(Path(args.aliases))

    # 3. Build initial graph (without RxNorm yet)
    rxcui_map: dict[str, str | None] = {cid: None for cid in _DRUG_SEEDS}
    graph = build_graph(ncit_synonyms, rxcui_map, custom)

    # 4. Optional RxNorm enrichment
    if args.rxnorm:
        logger.info("Fetching RxNorm synonyms …")
        rxcui_map = enrich_with_rxnorm(graph["nodes"])
        # Update rxcui fields in graph
        for cid, rxcui in rxcui_map.items():
            if cid in graph["nodes"]:
                graph["nodes"][cid]["rxcui"] = rxcui
        _active_sources.append("RxNorm API")

    # 5. Summary
    total_aliases = sum(len(n["aliases"]) for n in graph["nodes"].values())
    logger.info("Graph: %d drugs, %d total aliases, %d combo patterns",
                len(graph["nodes"]), total_aliases, len(graph["combos"]))
    for cid, node in sorted(graph["nodes"].items()):
        logger.info("  %-38s  family=%-22s  aliases=%d",
                    cid, node["family"] or "—", len(node["aliases"]))

    if args.dry_run:
        logger.info("Dry-run — not writing.")
        return

    # 6. Write
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(graph, indent=2, ensure_ascii=False))
    logger.info("Written → %s  (%.1f KB)", out_path, out_path.stat().st_size / 1024)


if __name__ == "__main__":
    main()
