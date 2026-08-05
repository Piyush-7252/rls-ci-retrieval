"""
batch_search_run19.py — Run search for all run19 doc/CI combinations sequentially.

Groups:
  Group A: 12 docs  × ahmedCis.json          (34 CIs)
  Group B:  5 docs  × christineCIs.json       (61 CIs)
  Group C:  4 docs  × random.json             (11 CIs)
  Group D:  1 doc   × ahmedFalseNumaricCis.json (13 CIs)

Output: localfiles/search_results/run19/{timestamp}_{ci_stem}_{doc_id}.json
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = ROOT / "localfiles" / "search_results" / "run19"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WORKERS = 10

RUNS: list[tuple[str, str, int]] = [
    # ── Group A: ahmedCis × 12 docs ─────────────────────────────────────────
    ("localfiles/ci/ahmedCis.json", "20260726062234599_4xs0l7p_10993_REDACTED_Protocol-Amendment-1-FD-64407564MMY3002-218114_1245209", 34),
    # ("localfiles/ci/ahmedCis.json", "20260727131413555_kma65kc_10991_REDACTED_SCS-FD-JNJ-64407564-AAA-498431_1245203", 34),
    # ("localfiles/ci/ahmedCis.json", "20260727131420119_56fiqkl_10992_REDACTED_PopPK-Full-64407564MMY1001-867391_1245204", 34),
    # ("localfiles/ci/ahmedCis.json", "20260727131528867_exspu63_10999_REDACTED_SAP-FD-64407564MMY1001-745917_1245323", 34),
    # ("localfiles/ci/ahmedCis.json", "20260727131537112_e8ijab8_11008_REDACTED_ISS-Pub-JNJ-64407564-AAA-884948_1245347", 34),
    # ("localfiles/ci/ahmedCis.json", "20260727131631457_4pz079j_15973_REDACTED_Protocol-Amend_5-64407564MMY3009-694326_1720663", 34),
    # ("localfiles/ci/ahmedCis.json", "20260727131636662_61zgqnl_10990_REDACTED_SCE-FD-JNJ-64407564-AAA-827851_1245113", 34),
    ("localfiles/ci/ahmedCis.json", "20260727131644799_jjryatd_10996_REDACTED_CSR_Protocol_and_Amendments-FD-64407564MMY1001-869239_1245312", 34),
    # ("localfiles/ci/ahmedCis.json", "20260727131823070_1glpw5m_12981_REDACTED_Protocol-Amend-2-64007957MMY1008-1124499_1430988", 34),
    # ("localfiles/ci/ahmedCis.json", "20260727131828604_zg1px2h_14473_REDACTED_SAP-Amend_2-64007957MMY1001-192385_1099540", 34),
    ("localfiles/ci/ahmedCis.json", "20260727131832855_1bchble_14475_REDACTED_CSR_Protocol_Appdx-FD-64007957MMY1001-1242306_1565113", 34),
    # ("localfiles/ci/ahmedCis.json", "20260727131845808_cy9r8kq_14706_Redacted-Master_ICF_Parts_1-2-64407564MMY1001-1359988", 34),
    # ── Group B: christineCIs × 5 docs ──────────────────────────────────────
    # ("localfiles/ci/christineCIs.json", "20260727133608401_s2bgig2_011_vbp15-006-study-report_Marking_and_Anon", 61),
    # ("localfiles/ci/christineCIs.json", "20260727133613754_trhhhzi_018_vbp15-006-16-1-01-prot-amendments_Marking_and_Anon", 61),
    # ("localfiles/ci/christineCIs.json", "20260727133623334_8s9ljsh_022_extrapolation-report-15-august-2025-0037_Marking_and_Anon", 61),
    # ("localfiles/ci/christineCIs.json", "20260727133628608_8v1o5jc_023_intiquan-2024-model-rep-pooled-poppk-paed-adult-dmd_Marking_and_Anon", 61),
    # ("localfiles/ci/christineCIs.json", "20260727133731703_h2242ab_035_mRNA-1083-P301-CSR-body-Final_Version_2", 61),
    # ── Group C: random × 4 docs ─────────────────────────────────────────────
    # ("localfiles/ci/random.json", "20260727133508767_7twc5sw_Anonymize_fixture", 11),
    # ("localfiles/ci/random.json", "20260727133514575_vk1juqo_CDISC", 11),
    # ("localfiles/ci/random.json", "20260727133523404_tg2hsvm_ProtocoI_301", 11),
    # ("localfiles/ci/random.json", "20260727133527469_x5ucao8_RXP_0521_CSR-original", 11),
    # ── Group D: ahmedFalseNumaricCis × 1 doc ───────────────────────────────
    # ("localfiles/ci/ahmedFalseNumaricCis.json", "20260727133531679_e78ivkk_RxPharmaProtocolv1", 13),
]


def main() -> None:
    total = len(RUNS)
    passed = 0
    failed: list[str] = []

    print(f"[{datetime.now():%H:%M:%S}] Starting run19 — {total} combinations")
    print(f"  Output dir : {OUTPUT_DIR}")
    print(f"  Workers    : {WORKERS}")
    print()

    for idx, (ci_file, doc_id, max_cis) in enumerate(RUNS, 1):
        ci_stem = Path(ci_file).stem
        out_path = OUTPUT_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{ci_stem}_{doc_id}.json"
        cmd = [
            sys.executable, str(ROOT / "tests" / "search_new.py"),
            "--ci-file",     ci_file,
            "--document-id", doc_id,
            "--max-cis",     str(max_cis),
            "--workers",     str(WORKERS),
            "--output",      str(out_path),
        ]
        print(f"[{datetime.now():%H:%M:%S}] [{idx:>2}/{total}] {ci_stem}  ×  {doc_id[:50]}...")
        t0 = time.time()
        result = subprocess.run(cmd, cwd=ROOT)
        elapsed = time.time() - t0
        if result.returncode == 0:
            passed += 1
            print(f"  ✓  done in {elapsed:.0f}s  →  {out_path.name}")
        else:
            failed.append(f"{ci_stem} × {doc_id}")
            print(f"  ✗  FAILED (rc={result.returncode}) after {elapsed:.0f}s")
        print()

    print(f"[{datetime.now():%H:%M:%S}] Finished — {passed}/{total} passed", end="")
    if failed:
        print(f"  |  {len(failed)} failed:")
        for f in failed:
            print(f"    - {f}")
    else:
        print(" — all clean")


if __name__ == "__main__":
    main()
