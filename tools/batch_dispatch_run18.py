"""
Batch dispatch for all documents under s3://rls-file-bucket-eu/Patterns Check Run/18/

Runs dispatch_chunks_to_sqs.py sequentially for each document.
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

S3_BUCKET    = "rls-file-bucket-eu"
S3_PREFIX    = "Patterns Check Run/18"
QUEUE_URL    = "https://sqs.eu-west-1.amazonaws.com/064051750322/rls-ci-chunk-queue"
REGION       = "eu-west-1"
CACHE_BASE   = Path(".cache/run18")

DOCUMENTS = [
    # "20260726062234599_4xs0l7p_10993_REDACTED_Protocol-Amendment-1-FD-64407564MMY3002-218114_1245209",
    # "20260727131413555_kma65kc_10991_REDACTED_SCS-FD-JNJ-64407564-AAA-498431_1245203",
    # "20260727131420119_56fiqkl_10992_REDACTED_PopPK-Full-64407564MMY1001-867391_1245204",
    # "20260727131528867_exspu63_10999_REDACTED_SAP-FD-64407564MMY1001-745917_1245323",
    # "20260727131537112_e8ijab8_11008_REDACTED_ISS-Pub-JNJ-64407564-AAA-884948_1245347",
    # "20260727131631457_4pz079j_15973_REDACTED_Protocol-Amend_5-64407564MMY3009-694326_1720663",
    # "20260727131636662_61zgqnl_10990_REDACTED_SCE-FD-JNJ-64407564-AAA-827851_1245113",
    "20260727131644799_jjryatd_10996_REDACTED_CSR_Protocol_and_Amendments-FD-64407564MMY1001-869239_1245312",
    # "20260727131823070_1glpw5m_12981_REDACTED_Protocol-Amend-2-64007957MMY1008-1124499_1430988",
    # "20260727131828604_zg1px2h_14473_REDACTED_SAP-Amend_2-64007957MMY1001-192385_1099540",
    # "20260727131832855_1bchble_14475_REDACTED_CSR_Protocol_Appdx-FD-64007957MMY1001-1242306_1565113",
    # "20260727131845808_cy9r8kq_14706_Redacted-Master_ICF_Parts_1-2-64407564MMY1001-1359988",
#    -----------Large with 14000 pages -------------------
    # "Combined_REDACTED_CSR-Full-co-jnj-64407564",
#    -----------Large end  -------------------
    # "20260727133508767_7twc5sw_Anonymize_fixture",
    # "20260727133514575_vk1juqo_CDISC",
    # "20260727133523404_tg2hsvm_ProtocoI_301",
    # "20260727133527469_x5ucao8_RXP_0521_CSR-original",
    # "20260727133531679_e78ivkk_RxPharmaProtocolv1",
    # "20260727133608401_s2bgig2_011_vbp15-006-study-report_Marking_and_Anon",
    # "20260727133613754_trhhhzi_018_vbp15-006-16-1-01-prot-amendments_Marking_and_Anon",
    # "20260727133623334_8s9ljsh_022_extrapolation-report-15-august-2025-0037_Marking_and_Anon",
    # "20260727133628608_8v1o5jc_023_intiquan-2024-model-rep-pooled-poppk-paed-adult-dmd_Marking_and_Anon",
    # "20260727133731703_h2242ab_035_mRNA-1083-P301-CSR-body-Final_Version_2",
]

def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def dispatch(doc_id: str, index: int, total: int, suffix: str = "") -> bool:
    s3_key          = f"{S3_PREFIX}/documents/{doc_id}.pdf"
    full_tables_key = f"{S3_PREFIX}/extraction/{doc_id}/full_tables.json"
    cache_path      = CACHE_BASE / doc_id / "full_tables.json"

    print(f"\n{'='*70}")
    print(f"[{index}/{total}] {now_utc()}")
    print(f"  doc_id : {doc_id}{f'  suffix={suffix!r}' if suffix else ''}")
    print(f"{'='*70}", flush=True)

    cmd = [
        sys.executable, "tools/dispatch_chunks_to_sqs.py",
        "--document-id",    doc_id,
        "--s3-bucket",      S3_BUCKET,
        "--s3-key",         s3_key,
        "--full-tables-key", full_tables_key,
        "--queue-url",      QUEUE_URL,
        "--cache-path",     str(cache_path),
        "--region",         REGION,
    ]
    if suffix:
        cmd += ["--suffix", suffix]

    t0 = time.time()
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"  ✓ done in {elapsed:.0f}s", flush=True)
        return True
    else:
        print(f"  ✗ FAILED (exit {result.returncode}) after {elapsed:.0f}s", flush=True)
        return False


def main():
    import argparse as _ap
    ap = _ap.ArgumentParser()
    ap.add_argument("--suffix", default="",
                    help="Suffix appended to document_id for all dispatched chunks (e.g. 'run19+50+fix-latest')")
    args = ap.parse_args()

    total    = len(DOCUMENTS)
    failed   = []

    print(f"=== BATCH DISPATCH RUN18 — {now_utc()} ===")
    print(f"  {total} documents  |  queue: {QUEUE_URL}{f'  suffix={args.suffix!r}' if args.suffix else ''}")

    for i, doc_id in enumerate(DOCUMENTS, start=1):
        ok = dispatch(doc_id, i, total, suffix=args.suffix)
        if not ok:
            failed.append(doc_id)

    print(f"\n{'='*70}")
    print(f"=== BATCH DONE — {now_utc()} ===")
    print(f"  {total - len(failed)}/{total} succeeded")
    if failed:
        print(f"  FAILED ({len(failed)}):")
        for d in failed:
            print(f"    {d}")


if __name__ == "__main__":
    main()
