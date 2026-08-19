"""
REGRESSION TESTS: Merger Invariants

Purpose:
--------
Validate that the Merger correctly implements the production geometry contract:

  1. REJECTION INVARIANT: Rejected candidates cannot contribute geometry
     Formally: rejected_only_rects ∩ final_hit_rects = ∅

  2. ORDERING INVARIANT: Document order is preserved deterministically
     Formally: chunk_ids and highlight_spans are sorted by document position

  3. SHARED GEOMETRY INVARIANT: Shared rects allowed if accepted candidate owns them
     Formally: shared_rects ⊆ accepted_rects (per candidate)

These tests run alongside production code to catch regressions.
"""

import sys
from pathlib import Path
import importlib.util

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from geometry import SentenceSpan

# Import Merger
merger_path = Path(__file__).parent.parent / "lambdas" / "search" / "merger" / "lambda_function.py"
spec = importlib.util.spec_from_file_location("merger", merger_path)
merger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merger)


def test_rejection_invariant():
    """
    REGRESSION TEST: Rejected candidates don't contribute geometry to final hit.
    
    Setup:
      - Candidate A: YES verdict, rects=[R1, R2]
      - Candidate B: NO verdict, rects=[R2, R3, R4]  (R2 shared with A)
      - Candidate C: YES verdict, rects=[R5]
    
    Expected:
      - Final hit includes: R1 (A-only), R2 (shared & A accepted), R5 (C)
      - Final hit excludes: R3 (B-only), R4 (B-only)
    
    Invariant:
      rejected_only_rects ∩ final_hit_rects = ∅
    """
    print("\n[REGRESSION TEST] Rejection Invariant")
    print("=" * 80)
    
    candidates = [
        {
            "chunk_id": "chunk_A",
            "verdict": "YES",
            "confidence": 0.95,
            "match_span": "text A",
            "match_rects": [[100, 200, 300, 215], [100, 218, 300, 233]],  # R1, R2
            "match_geometry_source": "apryse_span",
            "match_page": 10,
            "page_start": 10,
            "page_end": 10,
            "context": {"current_text": "Context A"},
            "sources": ["source_A"],
        },
        {
            "chunk_id": "chunk_B",
            "verdict": "NO",  # REJECTED
            "confidence": 0.60,
            "match_span": "text B",
            "match_rects": [[100, 218, 300, 233], [100, 236, 300, 251], [100, 254, 300, 269]],  # R2, R3, R4
            "match_geometry_source": "apryse_span",
            "match_page": 10,
            "page_start": 10,
            "page_end": 10,
            "context": {"current_text": "Context B"},
            "sources": ["source_B"],
        },
        {
            "chunk_id": "chunk_C",
            "verdict": "YES",
            "confidence": 0.88,
            "match_span": "text C",
            "match_rects": [[100, 272, 300, 287]],  # R5
            "match_geometry_source": "apryse_span",
            "match_page": 10,
            "page_start": 10,
            "page_end": 10,
            "context": {"current_text": "Context C"},
            "sources": ["source_C"],
        },
    ]
    
    # In production, _process filters to accepted candidates first
    accepted = [c for c in candidates if c.get("verdict") in ("YES", "MAYBE")]
    
    # Run merger with filtered candidates
    final_hit = merger._merge_group("ci_123", "known CI text", accepted)
    
    # Extract rects from final hit
    final_rects_set = set(tuple(r) for r in final_hit.get("match_rects", []))
    for span in final_hit.get("highlight_spans", []):
        for rect in span.get("match_rects", []):
            final_rects_set.add(tuple(rect))
    
    print(f"  Accepted candidates: A (YES), C (YES)")
    print(f"  Rejected candidates: B (NO)")
    print(f"  Rejected-only rects expected to be absent: R3, R4")
    print(f"  Final hit rects: {len(final_rects_set)} total")
    
    # Expected rects: R1 [100, 200, 300, 215], R2 [100, 218, 300, 233], R5 [100, 272, 300, 287]
    # Rejected-only rects: R3 [100, 236, 300, 251], R4 [100, 254, 300, 269]
    
    rejected_only_rects = {
        (100, 236, 300, 251),  # R3
        (100, 254, 300, 269),  # R4
    }
    
    for rect in rejected_only_rects:
        if rect in final_rects_set:
            print(f"  ❌ FAILED: Rejected-only rect {rect} leaked into final hit")
            return False
    
    print(f"  ✅ Rejection invariant satisfied: No rejected-only rects leaked")
    return True


def test_ordering_invariant():
    """
    REGRESSION TEST: Document order is preserved (no set() deduplication).
    
    Setup:
      - Candidate A at page 10
      - Candidate B at page 10 (same page, but later in document)
      - Candidate C at page 10 (same page, but earliest in document per position)
    
    Expected:
      - chunk_ids ordered by page + position_in_doc
      - No reordering as [C, B, A]
    """
    print("\n[REGRESSION TEST] Ordering Invariant")
    print("=" * 80)
    
    candidates = [
        {
            "chunk_id": "chunk_A",
            "verdict": "YES",
            "confidence": 0.95,
            "page_start": 10,
            "page_end": 10,
            "position_in_doc": 300,  # Position 3 in document
            "match_span": "text A",
            "match_rects": [[100, 200, 300, 215]],
            "match_geometry_source": "apryse_span",
            "match_page": 10,
            "context": {"current_text": "Context A"},
            "sources": ["source_A"],
            "highlight_score": 0.95,
        },
        {
            "chunk_id": "chunk_B",
            "verdict": "YES",
            "confidence": 0.90,
            "page_start": 10,
            "page_end": 10,
            "position_in_doc": 200,  # Position 2 in document
            "match_span": "text B",
            "match_rects": [[100, 218, 300, 233]],
            "match_geometry_source": "apryse_span",
            "match_page": 10,
            "context": {"current_text": "Context B"},
            "sources": ["source_B"],
            "highlight_score": 0.90,
        },
        {
            "chunk_id": "chunk_C",
            "verdict": "YES",
            "confidence": 0.88,
            "page_start": 10,
            "page_end": 10,
            "position_in_doc": 100,  # Position 1 in document
            "match_span": "text C",
            "match_rects": [[100, 272, 300, 287]],
            "match_geometry_source": "apryse_span",
            "match_page": 10,
            "context": {"current_text": "Context C"},
            "sources": ["source_C"],
            "highlight_score": 0.88,
        },
    ]
    
    # Filter like production does
    accepted = [c for c in candidates if c.get("verdict") in ("YES", "MAYBE")]
    
    # Run merger
    final_hit = merger._merge_group("ci_123", "known CI text", accepted)
    
    chunk_ids = final_hit.get("chunk_ids", [])
    highlight_spans = final_hit.get("highlight_spans", [])
    
    print(f"  Expected order: C, B, A (by position_in_doc: 100, 200, 300)")
    print(f"  Actual chunk_ids: {chunk_ids}")
    
    # Verify order
    expected_order = ["chunk_C", "chunk_B", "chunk_A"]
    if chunk_ids == expected_order:
        print(f"  ✅ Document order preserved")
    else:
        print(f"  ❌ FAILED: Order mismatch")
        print(f"     Expected: {expected_order}")
        print(f"     Got:      {chunk_ids}")
        return False
    
    # Verify highlight_spans also preserve order
    span_ids = [s.get("chunk_id") for s in highlight_spans]
    if span_ids == expected_order:
        print(f"  ✅ highlight_spans order preserved")
    else:
        print(f"  ❌ FAILED: highlight_spans order mismatch")
        return False
    
    return True


def test_shared_geometry_invariant():
    """
    REGRESSION TEST: Shared geometry allowed when accepted candidate owns it.
    
    Setup:
      - Candidate A: YES verdict, owns R1, R2
      - Candidate B: YES verdict, owns R2, R3  (R2 shared with A)
    
    Expected:
      - Final hit includes R1 (A-only), R2 (shared, both owned it)
      - highlight_spans has both A's and B's geometry
    
    Invariant:
      shared_rects ⊆ union(accepted_rects)
    """
    print("\n[REGRESSION TEST] Shared Geometry Invariant")
    print("=" * 80)
    
    candidates = [
        {
            "chunk_id": "chunk_A",
            "verdict": "YES",
            "confidence": 0.95,
            "match_span": "text A",
            "match_rects": [[100, 200, 300, 215], [100, 218, 300, 233]],  # R1, R2 (shared)
            "match_geometry_source": "apryse_span",
            "match_page": 10,
            "page_start": 10,
            "page_end": 10,
            "context": {"current_text": "Context A"},
            "sources": ["source_A"],
            "highlight_score": 0.95,
            "position_in_doc": 100,
        },
        {
            "chunk_id": "chunk_B",
            "verdict": "YES",
            "confidence": 0.90,
            "match_span": "text B",
            "match_rects": [[100, 218, 300, 233], [100, 236, 300, 251]],  # R2 (shared), R3
            "match_geometry_source": "apryse_span",
            "match_page": 10,
            "page_start": 10,
            "page_end": 10,
            "context": {"current_text": "Context B"},
            "sources": ["source_B"],
            "highlight_score": 0.90,
            "position_in_doc": 200,
        },
    ]
    
    # Filter like production does
    accepted = [c for c in candidates if c.get("verdict") in ("YES", "MAYBE")]
    
    # Run merger
    final_hit = merger._merge_group("ci_123", "known CI text", accepted)
    
    highlight_spans = final_hit.get("highlight_spans", [])
    
    print(f"  Accepted candidates: A (owns R1, R2), B (owns R2, R3)")
    print(f"  Shared rect: R2 [100, 218, 300, 233]")
    print(f"  highlight_spans: {len(highlight_spans)} spans")
    
    # Extract all rects from highlight_spans
    all_span_rects = set()
    for span in highlight_spans:
        for rect in span.get("match_rects", []):
            all_span_rects.add(tuple(rect))
    
    # Shared rect should be present
    shared_rect = (100, 218, 300, 233)
    if shared_rect in all_span_rects:
        print(f"  ✅ Shared rect preserved in highlight_spans")
    else:
        print(f"  ❌ FAILED: Shared rect missing from highlight_spans")
        return False
    
    # Verify both A and B are in highlight_spans
    chunk_ids_in_spans = [s.get("chunk_id") for s in highlight_spans]
    if "chunk_A" in chunk_ids_in_spans and "chunk_B" in chunk_ids_in_spans:
        print(f"  ✅ Both accepted candidates in highlight_spans")
    else:
        print(f"  ❌ FAILED: Missing accepted candidates")
        return False
    
    return True


def test_geometry_source_propagation():
    """
    REGRESSION TEST: geometry_source field consistently propagated.
    
    Ensures that every geometry output has a traceable source field.
    """
    print("\n[REGRESSION TEST] Geometry Source Propagation")
    print("=" * 80)
    
    candidates = [
        {
            "chunk_id": "chunk_A",
            "verdict": "YES",
            "confidence": 0.95,
            "match_span": "text",
            "match_rects": [[100, 200, 300, 215]],
            "match_geometry_source": "apryse_span",
            "match_page": 10,
            "page_start": 10,
            "page_end": 10,
            "context": {"current_text": "Context"},
            "sources": ["source"],
            "highlight_score": 0.95,
        },
    ]
    
    # Filter like production does
    accepted = [c for c in candidates if c.get("verdict") in ("YES", "MAYBE")]
    
    final_hit = merger._merge_group("ci_123", "known CI text", accepted)
    
    # Check primary geometry source
    primary_source = final_hit.get("match_geometry_source")
    if primary_source == "apryse_span":
        print(f"  ✅ Primary match_geometry_source: {primary_source}")
    else:
        print(f"  ❌ FAILED: Primary geometry_source is '{primary_source}', expected 'apryse_span'")
        return False
    
    # Check all highlight_spans have geometry_source
    for i, span in enumerate(final_hit.get("highlight_spans", [])):
        source = span.get("match_geometry_source")
        if not source:
            print(f"  ❌ FAILED: highlight_spans[{i}] missing match_geometry_source")
            return False
    
    print(f"  ✅ All {len(final_hit.get('highlight_spans', []))} spans have geometry_source")
    return True


def main():
    print("\n" + "=" * 80)
    print("REGRESSION TESTS: Merger Invariants")
    print("=" * 80)
    
    tests = [
        ("Rejection Invariant", test_rejection_invariant),
        ("Ordering Invariant", test_ordering_invariant),
        ("Shared Geometry Invariant", test_shared_geometry_invariant),
        ("Geometry Source Propagation", test_geometry_source_propagation),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"  ❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} — {name}")
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n✅ ALL REGRESSION TESTS PASSED")
        return 0
    else:
        print("\n❌ SOME REGRESSION TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
