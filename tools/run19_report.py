"""
run19_report.py
---------------
Generate an HTML report for all run19 search result JSONs.
Covers: per-document timing, cost, hit quality, and object type stats.

Usage:
  python3.12 tools/run19_report.py
  python3.12 tools/run19_report.py --run-dir localfiles/search_results/run19 --out localfiles/run19_report.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

ROOT    = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "localfiles" / "search_results" / "run19"
OUT     = ROOT / "localfiles" / "run19_report.html"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _short_doc(document_id: str) -> str:
    """Return a readable short name from a document_id."""
    parts = document_id.split("_")
    # Find the first all-digit chunk that looks like a doc number (5+ digits)
    for p in parts:
        if p.isdigit() and len(p) >= 4:
            # grab everything from here onward (up to REDACTED or end)
            idx = document_id.index(p)
            tail = document_id[idx:]
            # strip trailing long hash-like prefix tokens
            tail = tail.replace("_REDACTED_", " · ").replace("_Redacted_", " · ")
            return tail[:80]
    return document_id[-60:]


def _ci_set(filename: str) -> str:
    for tag in ("ahmedCis", "christineCIs", "random", "ahmedFalseNumaricCis"):
        if tag in filename:
            return tag
    return "?"


def load_all(run_dir: Path) -> list[dict]:
    records = []
    for f in sorted(run_dir.glob("*.json")):
        if "fixed" in f.name:
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  SKIP {f.name}: {e}")
            continue

        run     = data.get("run", {})
        summary = data.get("summary", {})
        timing  = data.get("timing_summary", {})
        cost    = data.get("cost_estimate", {})

        doc_id   = run.get("document_id", f.stem)
        ci_file  = Path(run.get("ci_file", "")).stem

        records.append({
            "file":          f.name,
            "doc_id":        doc_id,
            "doc_short":     _short_doc(doc_id),
            "ci_set":        _ci_set(f.name),
            "timestamp":     run.get("timestamp", ""),
            "model":         cost.get("model", ""),
            "parallelism":   run.get("parallelism", ""),
            # Summary
            "cis_searched":      summary.get("cis_searched", 0),
            "total_final_hits":  summary.get("total_final_hits", 0),
            "direct_hits":       summary.get("direct_hits", 0),
            "related_hits":      summary.get("related_hits", 0),
            "same_study_hits":   summary.get("same_study_hits", 0),
            "total_rejected":    summary.get("total_rejected", 0),
            "total_skipped":     summary.get("total_skipped", 0),
            "obj_stats":         summary.get("object_type_stats", {}),
            "related_breakdown": summary.get("related_breakdown", {}),
            # Timing (seconds)
            "t_total":       timing.get("total", {}).get("wall_clock_s", 0),
            "t_retrievers":  timing.get("retrievers_total", {}).get("wall_clock_s", 0),
            "t_reranker":    timing.get("reranker", {}).get("wall_clock_s", 0),
            "t_context":     timing.get("context_expander", {}).get("wall_clock_s", 0),
            "t_verifier":    timing.get("llm_verifier", {}).get("wall_clock_s", 0),
            "t_evidence":    timing.get("evidence_classification", {}).get("wall_clock_s", 0),
            "t_aggregator":  timing.get("aggregator", {}).get("wall_clock_s", 0),
            "t_highlight":   timing.get("highlight_extractor", {}).get("wall_clock_s", 0),
            "t_merger":      timing.get("merger", {}).get("wall_clock_s", 0),
            "t_classifier":  timing.get("classifier", {}).get("wall_clock_s", 0),
            # Cost
            "cost_total":    cost.get("combined_est_cost_usd", 0),
            "cost_verifier": cost.get("llm_verifier", {}).get("est_cost_usd", 0),
            "cost_evidence": cost.get("evidence_classification", {}).get("est_cost_usd", 0),
            "verifier_candidates": cost.get("llm_verifier", {}).get("candidates_passed_to_verifier", 0),
            "verifier_calls":      cost.get("llm_verifier", {}).get("actual_bedrock_calls", 0),
            "verifier_skipped":    cost.get("llm_verifier", {}).get("skipped_below_threshold", 0),
            "verifier_tokens":     cost.get("llm_verifier", {}).get("total_tokens", 0),
            "evidence_calls":      cost.get("evidence_classification", {}).get("bedrock_calls", 0),
            "evidence_tokens":     cost.get("evidence_classification", {}).get("total_tokens", 0),
        })
    return records


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _bar(value: float, total: float, color: str = "#4a90d9") -> str:
    pct = min(100, round(value / total * 100)) if total else 0
    return (
        f'<div style="background:#eee;border-radius:3px;height:12px;width:120px;display:inline-block;vertical-align:middle">'
        f'<div style="background:{color};width:{pct}%;height:100%;border-radius:3px"></div></div>'
        f' <small>{value:.1f}s</small>'
    )


def _pct(num, denom):
    if not denom:
        return "–"
    return f"{num/denom*100:.0f}%"


STAGE_COLORS = {
    "Retrievers":  "#4a90d9",
    "Reranker":    "#7b68ee",
    "Context Exp": "#e07b39",
    "LLM Verify":  "#d94a4a",
    "Evidence":    "#4ab87e",
    "Aggregator":  "#aaa",
    "Highlight":   "#ccc",
    "Merger":      "#ddd",
}

OBJ_COLORS = {
    "sentence":  "#4a90d9",
    "paragraph": "#7b68ee",
    "list":      "#e07b39",
    "table_row": "#4ab87e",
    "heading":   "#aaa",
}

CI_SET_COLORS = {
    "ahmedCis":              "#1a6fb0",
    "christineCIs":          "#7b35a8",
    "random":                "#2e8b57",
    "ahmedFalseNumaricCis":  "#c0392b",
}


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_html(records: list[dict]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    total_cost = sum(r["cost_total"] for r in records)
    total_time = sum(r["t_total"] for r in records)
    total_tp   = sum(r["total_final_hits"] for r in records)
    total_cis  = sum(r["cis_searched"] for r in records)
    n_docs     = len(records)

    # ---- grouped by ci_set for summary cards ----
    by_set: dict[str, list] = {}
    for r in records:
        by_set.setdefault(r["ci_set"], []).append(r)

    cards_html = ""
    for ci_set, recs in sorted(by_set.items()):
        color = CI_SET_COLORS.get(ci_set, "#555")
        cards_html += f"""
        <div class="card" style="border-top:4px solid {color}">
          <div class="card-title" style="color:{color}">{ci_set}</div>
          <div class="card-row"><span>Documents</span><strong>{len(recs)}</strong></div>
          <div class="card-row"><span>CIs Searched</span><strong>{sum(r['cis_searched'] for r in recs)}</strong></div>
          <div class="card-row"><span>Final Hits (TP)</span><strong>{sum(r['total_final_hits'] for r in recs)}</strong></div>
          <div class="card-row"><span>Total Time</span><strong>{sum(r['t_total'] for r in recs):.1f}s</strong></div>
          <div class="card-row"><span>Est. Cost</span><strong>${sum(r['cost_total'] for r in recs):.4f}</strong></div>
        </div>"""

    # ---- overview table ----
    overview_rows = ""
    for r in records:
        color = CI_SET_COLORS.get(r["ci_set"], "#555")
        hit_rate = _pct(r["total_final_hits"], r["cis_searched"])
        precision = _pct(r["total_final_hits"], r["total_final_hits"] + r["total_rejected"])
        doc_name = r["doc_short"]
        ts = r["timestamp"][:16].replace("T", " ") if r["timestamp"] else ""
        overview_rows += f"""
        <tr>
          <td><span class="badge" style="background:{color}">{r['ci_set']}</span></td>
          <td class="doc-name" title="{r['doc_id']}">{doc_name}</td>
          <td>{ts}</td>
          <td class="num">{r['cis_searched']}</td>
          <td class="num green"><strong>{r['total_final_hits']}</strong></td>
          <td class="num">{r['direct_hits']}</td>
          <td class="num">{r['related_hits']}</td>
          <td class="num red">{r['total_rejected']}</td>
          <td class="num gray">{r['total_skipped']}</td>
          <td class="num">{hit_rate}</td>
          <td class="num">{precision}</td>
          <td class="num"><strong>{r['t_total']:.1f}s</strong></td>
          <td class="num green"><strong>${r['cost_total']:.4f}</strong></td>
        </tr>"""

    # ---- timing table ----
    timing_rows = ""
    max_total = max((r["t_total"] for r in records), default=1)
    for r in records:
        color = CI_SET_COLORS.get(r["ci_set"], "#555")
        timing_rows += f"""
        <tr>
          <td><span class="badge" style="background:{color}">{r['ci_set']}</span></td>
          <td class="doc-name" title="{r['doc_id']}">{r['doc_short']}</td>
          <td class="num">{_bar(r['t_retrievers'],  max_total, '#4a90d9')}</td>
          <td class="num">{_bar(r['t_reranker'],    max_total, '#7b68ee')}</td>
          <td class="num">{_bar(r['t_context'],     max_total, '#e07b39')}</td>
          <td class="num">{_bar(r['t_verifier'],    max_total, '#d94a4a')}</td>
          <td class="num">{_bar(r['t_evidence'],    max_total, '#4ab87e')}</td>
          <td class="num total-col">{r['t_total']:.1f}s</td>
        </tr>"""

    # ---- cost table ----
    cost_rows = ""
    max_cost = max((r["cost_total"] for r in records), default=1)
    for r in records:
        color = CI_SET_COLORS.get(r["ci_set"], "#555")
        tok_per_ci = f"{r['verifier_tokens']//r['cis_searched']:,}" if r['cis_searched'] else "–"
        cost_rows += f"""
        <tr>
          <td><span class="badge" style="background:{color}">{r['ci_set']}</span></td>
          <td class="doc-name" title="{r['doc_id']}">{r['doc_short']}</td>
          <td class="num">{r['verifier_candidates']:,}</td>
          <td class="num">{r['verifier_calls']:,}</td>
          <td class="num">{r['verifier_skipped']:,}</td>
          <td class="num">{r['verifier_tokens']:,}</td>
          <td class="num">${r['cost_verifier']:.4f}</td>
          <td class="num">{r['evidence_calls']:,}</td>
          <td class="num">{r['evidence_tokens']:,}</td>
          <td class="num">${r['cost_evidence']:.4f}</td>
          <td class="num green"><strong>${r['cost_total']:.4f}</strong></td>
          <td class="num">{tok_per_ci}</td>
        </tr>"""

    # ---- object type stats table ----
    obj_rows = ""
    obj_types = ["sentence", "paragraph", "list", "table_row", "heading"]
    for r in records:
        color = CI_SET_COLORS.get(r["ci_set"], "#555")
        stats = r["obj_stats"]
        cells = ""
        for ot in obj_types:
            s = stats.get(ot, {})
            retr = s.get("retrieved", 0)
            yes  = s.get("final_yes", 0)
            rej  = s.get("rejected", 0)
            oc   = OBJ_COLORS.get(ot, "#aaa")
            if retr:
                cells += f'<td class="num"><span style="color:{oc};font-weight:600">{yes}</span>/<small>{retr}</small></td>'
            else:
                cells += '<td class="num gray">–</td>'
        obj_rows += f"""
        <tr>
          <td><span class="badge" style="background:{color}">{r['ci_set']}</span></td>
          <td class="doc-name" title="{r['doc_id']}">{r['doc_short']}</td>
          {cells}
          <td class="num total-col">{r['total_final_hits']}</td>
        </tr>"""

    obj_headers = "".join(f'<th>{ot}</th>' for ot in obj_types)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Run 19 — Search Pipeline Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         font-size: 13px; background: #f5f7fa; color: #2d3748; }}
  header {{ background: #1a202c; color: #fff; padding: 24px 32px; }}
  header h1 {{ font-size: 22px; font-weight: 700; }}
  header .sub {{ color: #a0aec0; font-size: 13px; margin-top: 4px; }}
  .kpi-bar {{ display: flex; gap: 16px; padding: 20px 32px; flex-wrap: wrap; }}
  .kpi {{ background: #fff; border-radius: 8px; padding: 14px 20px;
           min-width: 140px; flex: 1; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  .kpi-val {{ font-size: 26px; font-weight: 700; color: #2b6cb0; }}
  .kpi-label {{ font-size: 11px; color: #718096; margin-top: 2px; text-transform: uppercase; letter-spacing: .5px; }}
  .cards {{ display: flex; gap: 14px; padding: 0 32px 20px; flex-wrap: wrap; }}
  .card {{ background: #fff; border-radius: 8px; padding: 16px 20px; min-width: 190px;
           flex: 1; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  .card-title {{ font-weight: 700; font-size: 14px; margin-bottom: 10px; }}
  .card-row {{ display: flex; justify-content: space-between; padding: 3px 0;
               border-bottom: 1px solid #f0f0f0; font-size: 12px; }}
  section {{ padding: 0 32px 28px; }}
  h2 {{ font-size: 16px; font-weight: 700; margin-bottom: 12px; color: #2d3748;
         border-left: 4px solid #4a90d9; padding-left: 10px; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff;
           border-radius: 8px; overflow: hidden;
           box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  th {{ background: #2d3748; color: #e2e8f0; text-align: left;
        padding: 9px 10px; font-size: 11px; text-transform: uppercase; letter-spacing: .4px; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #edf2f7; vertical-align: middle; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f7faff; }}
  .num {{ text-align: right; white-space: nowrap; }}
  .doc-name {{ max-width: 280px; overflow: hidden; text-overflow: ellipsis;
               white-space: nowrap; font-size: 12px; color: #4a5568; }}
  .badge {{ color: #fff; font-size: 10px; font-weight: 700; padding: 2px 6px;
            border-radius: 10px; white-space: nowrap; }}
  .green {{ color: #276749; }}
  .red {{ color: #c53030; }}
  .gray {{ color: #a0aec0; }}
  .total-col {{ font-weight: 700; }}
  .note {{ font-size: 11px; color: #718096; margin-bottom: 10px; }}
  .legend {{ display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 10px; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 11px; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}
</style>
</head>
<body>

<header>
  <h1>Run 19 — Clinical CI Search Pipeline Report</h1>
  <div class="sub">Generated {now} &nbsp;·&nbsp; {n_docs} documents &nbsp;·&nbsp; Model: claude-haiku-4-5</div>
</header>

<!-- KPI bar -->
<div class="kpi-bar">
  <div class="kpi"><div class="kpi-val">{n_docs}</div><div class="kpi-label">Documents</div></div>
  <div class="kpi"><div class="kpi-val">{total_cis:,}</div><div class="kpi-label">CIs Searched</div></div>
  <div class="kpi"><div class="kpi-val">{total_tp:,}</div><div class="kpi-label">Final Hits (TP)</div></div>
  <div class="kpi"><div class="kpi-val">{total_time:.0f}s</div><div class="kpi-label">Total Wall Time</div></div>
  <div class="kpi"><div class="kpi-val" style="color:#276749">${total_cost:.4f}</div><div class="kpi-label">Est. Total Cost (USD)</div></div>
  <div class="kpi"><div class="kpi-val">${total_cost/n_docs:.4f}</div><div class="kpi-label">Avg Cost / Doc</div></div>
  <div class="kpi"><div class="kpi-val">{total_time/n_docs:.0f}s</div><div class="kpi-label">Avg Time / Doc</div></div>
</div>

<!-- CI set summary cards -->
<div class="cards">{cards_html}</div>

<!-- 1. Overview -->
<section>
  <h2>1 · Overview — Hits &amp; Quality</h2>
  <p class="note">
    Hit Rate = Final Hits ÷ CIs Searched &nbsp;|&nbsp;
    Precision = Final Hits ÷ (Final Hits + Rejected) &nbsp;|&nbsp;
    Direct = DIRECT evidence &nbsp;|&nbsp; Related = SUPPORTING/RELATED evidence
  </p>
  <table>
    <thead><tr>
      <th>CI Set</th><th>Document</th><th>Timestamp</th>
      <th>CIs</th><th>Final TP</th><th>Direct</th><th>Related</th>
      <th>Rejected</th><th>Skipped</th><th>Hit Rate</th><th>Precision</th>
      <th>Total Time</th><th>Cost USD</th>
    </tr></thead>
    <tbody>{overview_rows}</tbody>
  </table>
</section>

<!-- 2. Timing -->
<section>
  <h2>2 · Pipeline Timing Breakdown</h2>
  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#4a90d9"></div>Retrievers</div>
    <div class="legend-item"><div class="legend-dot" style="background:#7b68ee"></div>Reranker</div>
    <div class="legend-item"><div class="legend-dot" style="background:#e07b39"></div>Context Expander</div>
    <div class="legend-item"><div class="legend-dot" style="background:#d94a4a"></div>LLM Verifier</div>
    <div class="legend-item"><div class="legend-dot" style="background:#4ab87e"></div>Evidence Classification</div>
  </div>
  <p class="note">Bar scale = proportion of {max_total:.0f}s (slowest document)</p>
  <table>
    <thead><tr>
      <th>CI Set</th><th>Document</th>
      <th>Retrievers</th><th>Reranker</th><th>Context Exp</th>
      <th>LLM Verify</th><th>Evidence</th><th>Total</th>
    </tr></thead>
    <tbody>{timing_rows}</tbody>
  </table>
</section>

<!-- 3. Cost -->
<section>
  <h2>3 · Cost &amp; Token Analysis</h2>
  <p class="note">
    Candidates = chunks passed into verifier pipeline &nbsp;|&nbsp;
    Calls = actual Bedrock API calls (after skip-threshold filtering) &nbsp;|&nbsp;
    Tokens/CI = verifier tokens ÷ CIs searched
  </p>
  <table>
    <thead><tr>
      <th>CI Set</th><th>Document</th>
      <th>V Candidates</th><th>V Calls</th><th>V Skipped</th><th>V Tokens</th><th>V Cost</th>
      <th>Ev Calls</th><th>Ev Tokens</th><th>Ev Cost</th>
      <th>Total Cost</th><th>Tokens/CI</th>
    </tr></thead>
    <tbody>{cost_rows}</tbody>
  </table>
</section>

<!-- 4. Object type stats -->
<section>
  <h2>4 · Retrieved Object Types  (Final ÷ Retrieved)</h2>
  <p class="note">
    Format: <strong style="color:#4a90d9">Final Yes</strong> / <small>Total Retrieved</small>
  </p>
  <table>
    <thead><tr>
      <th>CI Set</th><th>Document</th>
      {obj_headers}
      <th>Total Final</th>
    </tr></thead>
    <tbody>{obj_rows}</tbody>
  </table>
</section>

</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Generate HTML report for run19 results.")
    p.add_argument("--run-dir", default=str(RUN_DIR))
    p.add_argument("--out",     default=str(OUT))
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    out     = Path(args.out)

    print(f"Loading JSONs from {run_dir} ...")
    records = load_all(run_dir)
    print(f"  {len(records)} documents loaded")

    html = generate_html(records)
    out.write_text(html, encoding="utf-8")
    print(f"\nReport written → {out}")
    print(f"Open in browser: open '{out}'")


if __name__ == "__main__":
    main()
