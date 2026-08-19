"""
TEST 11: OpenSearch Serialization — Geometry Preservation Through Indexing

Purpose:
--------
Validate that SentenceSpan geometry survives full round-trip through
OpenSearch serialization/deserialization.

This is CRITICAL for production. We've validated the in-memory pipeline
(Tests 5-10), but geometry can be lost during:
  1. SentenceSpan → JSON (extraction lambda output)
  2. JSON → OpenSearch document
  3. OpenSearch retrieval → JSON
  4. JSON → HighlightExtractor deserialization

Risk: If geometry is truncated or lost during JSON serialization,
the final highlights will be broken.

Test Scenarios:
  - SentenceSpan serialization round-trip
  - Per-line rects preservation (not truncated)
  - Geometry source metadata preserved
  - Multi-line spans work through JSON
  - Empty rects (fallback geometry) handled correctly
"""

import json
import sys
from pathlib import Path

# Add lambdas to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "lambdas" / "extraction"))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from geometry import SentenceSpan


def test_sentence_span_json_roundtrip():
    """Test 11A: SentenceSpan serializes and deserializes through JSON correctly."""
    print("\n[TEST 11A] SentenceSpan JSON Round-Trip")
    print("=" * 80)
    
    # Create a realistic SentenceSpan with per-line geometry
    original = SentenceSpan(
        text="The primary objective was progression-free survival as assessed by independent review committee.",
        page=42,
        char_start=100,
        char_end=200,
        rects=[
            [100.0, 200.0, 500.0, 215.0],  # line 1
            [100.0, 218.0, 510.0, 233.0],  # line 2
            [100.0, 236.0, 380.0, 251.0],  # line 3
        ],
        source_object_id="chunk_001_obj_042",
        source_span_ids=["span_1", "span_2"],
        span_type="sentence",
        geometry_source="apryse_span",
    )
    
    print(f"  Original SentenceSpan:")
    print(f"    text: {original.text[:50]}...")
    print(f"    page: {original.page}")
    print(f"    char_range: [{original.char_start}, {original.char_end})")
    print(f"    rects: {len(original.rects)} lines")
    print(f"    geometry_source: {original.geometry_source}")
    
    # Serialize to dict (as extraction lambda would)
    serialized_dict = original.to_dict()
    print(f"\n  Serialized to dict:")
    print(f"    keys: {sorted(serialized_dict.keys())}")
    
    # Convert to JSON (as OpenSearch would store)
    json_str = json.dumps(serialized_dict)
    print(f"    JSON size: {len(json_str)} bytes")
    
    # Deserialize from JSON (as search lambda would retrieve)
    deserialized_dict = json.loads(json_str)
    restored = SentenceSpan.from_dict(deserialized_dict)
    
    print(f"\n  Restored SentenceSpan:")
    print(f"    text: {restored.text[:50]}...")
    print(f"    page: {restored.page}")
    print(f"    char_range: [{restored.char_start}, {restored.char_end})")
    print(f"    rects: {len(restored.rects)} lines")
    print(f"    geometry_source: {restored.geometry_source}")
    
    # Validate preservation
    assert restored.text == original.text, "Text changed"
    assert restored.page == original.page, "Page changed"
    assert restored.char_start == original.char_start, "char_start changed"
    assert restored.char_end == original.char_end, "char_end changed"
    assert len(restored.rects) == len(original.rects), "Rect count changed"
    assert restored.geometry_source == original.geometry_source, "geometry_source changed"
    
    # Validate precision of rectangles
    for i, (orig_rect, restored_rect) in enumerate(zip(original.rects, restored.rects)):
        assert len(orig_rect) == len(restored_rect), f"Rect {i}: length mismatch"
        for j, (orig_val, restored_val) in enumerate(zip(orig_rect, restored_rect)):
            assert abs(orig_val - restored_val) < 0.001, \
                f"Rect {i}, coord {j}: {orig_val} != {restored_val}"
    
    print(f"\n  ✅ All assertions passed")
    return True


def test_multiline_spans_json_preservation():
    """Test 11B: Multi-line spans with complex geometry survive JSON."""
    print("\n[TEST 11B] Multi-Line Span JSON Preservation")
    print("=" * 80)
    
    # Create a span that spans 5 lines (common in clinical documents)
    span = SentenceSpan(
        text="Study design was randomized, double-blind, placebo-controlled phase 2b trial evaluating safety, tolerability, pharmacokinetics, and preliminary efficacy.",
        page=15,
        char_start=0,
        char_end=160,
        rects=[
            [72.0, 92.75, 540.01, 106.04],
            [72.0, 109.15, 540.01, 122.44],
            [72.0, 125.55, 540.01, 138.84],
            [72.0, 141.95, 540.01, 155.24],
            [72.0, 158.35, 245.67, 171.64],
        ],
        source_object_id="para_123_obj_001",
        span_type="sentence",
        geometry_source="apryse_span",
    )
    
    print(f"  Original span: {len(span.rects)} lines")
    
    # Round-trip through JSON
    json_str = json.dumps(span.to_dict())
    restored = SentenceSpan.from_dict(json.loads(json_str))
    
    print(f"  Restored span: {len(restored.rects)} lines")
    
    # Validate all rects preserved
    assert len(restored.rects) == 5, f"Expected 5 rects, got {len(restored.rects)}"
    
    # Validate specific rect values (check first and last)
    assert restored.rects[0] == span.rects[0], "First rect mismatch"
    assert restored.rects[-1] == span.rects[-1], "Last rect mismatch"
    
    print(f"  ✅ All {len(restored.rects)} rects preserved")
    return True


def test_empty_rects_json_handling():
    """Test 11C: Fallback geometry (empty rects) handled correctly."""
    print("\n[TEST 11C] Empty Rects (Fallback Geometry) JSON Handling")
    print("=" * 80)
    
    # Create span with fallback geometry (empty rects)
    span = SentenceSpan(
        text="This is a sentence without explicit line geometry.",
        page=5,
        char_start=0,
        char_end=50,
        rects=[],  # No geometry available
        source_object_id="merged_para_001",
        span_type="sentence",
        geometry_source="object_bbox",  # Fallback to object-level
    )
    
    print(f"  Original: rects={len(span.rects)}, source={span.geometry_source}")
    
    # Round-trip through JSON
    json_str = json.dumps(span.to_dict())
    restored = SentenceSpan.from_dict(json.loads(json_str))
    
    print(f"  Restored: rects={len(restored.rects)}, source={restored.geometry_source}")
    
    assert len(restored.rects) == 0, "Empty rects should remain empty"
    assert restored.geometry_source == "object_bbox", "geometry_source should be preserved"
    assert restored.source_object_id == "merged_para_001", "source_object_id should be preserved"
    
    print(f"  ✅ Fallback geometry handled correctly")
    return True


def test_geometry_source_field_consistency():
    """Test 11D: geometry_source field consistent across serialization."""
    print("\n[TEST 11D] Geometry Source Field Consistency")
    print("=" * 80)
    
    sources_to_test = [
        ("apryse_span", [[100, 200, 500, 215]]),
        ("object_bbox", [[100, 200, 500, 300]]),
        ("none", []),
    ]
    
    for source, rects in sources_to_test:
        span = SentenceSpan(
            text="Test sentence.",
            page=1,
            char_start=0,
            char_end=14,
            rects=rects,
            geometry_source=source,
        )
        
        # Round-trip
        restored = SentenceSpan.from_dict(json.loads(json.dumps(span.to_dict())))
        
        assert restored.geometry_source == source, \
            f"geometry_source changed: {source} → {restored.geometry_source}"
        print(f"  ✅ {source}: preserved")
    
    print(f"  ✅ All geometry sources consistent")
    return True


def test_opensearch_document_structure():
    """Test 11E: SentenceSpan fits into OpenSearch document structure."""
    print("\n[TEST 11E] OpenSearch Document Structure")
    print("=" * 80)
    
    # Create SentenceSpan as it would appear from extraction
    sentence = SentenceSpan(
        text="Primary endpoint was PFS as assessed by IRC.",
        page=10,
        char_start=50,
        char_end=95,
        rects=[[100, 200, 500, 215], [100, 218, 350, 233]],
        source_object_id="chunk_001_obj_005",
        span_type="sentence",
        geometry_source="apryse_span",
    )
    
    # Create a simulated OpenSearch document (as it would be in extraction lambda)
    opensearch_doc = {
        "ci_id": "CI_12345",
        "page": 10,
        "object_type": "paragraph",
        "text": "Primary endpoint was PFS as assessed by IRC.",
        "display_spans": [
            {
                "type": "sentence",
                "text": sentence.text,
                "start": sentence.char_start,
                "end": sentence.char_end,
                "bbox": [100, 200, 500, 233],
                "_sentence_span": sentence.to_dict(),  # Stored for retrieval
            }
        ],
    }
    
    print(f"  Document structure:")
    print(f"    ci_id: {opensearch_doc['ci_id']}")
    print(f"    page: {opensearch_doc['page']}")
    print(f"    display_spans: {len(opensearch_doc['display_spans'])} items")
    
    # Simulate OpenSearch storage + retrieval (full JSON round-trip)
    json_str = json.dumps(opensearch_doc)
    retrieved_doc = json.loads(json_str)
    
    print(f"    JSON size: {len(json_str)} bytes")
    
    # Simulate retrieval in HighlightExtractor
    display_span = retrieved_doc["display_spans"][0]
    restored_sentence = SentenceSpan.from_dict(display_span["_sentence_span"])
    
    print(f"\n  Retrieved SentenceSpan:")
    print(f"    text: {restored_sentence.text}")
    print(f"    rects: {len(restored_sentence.rects)} lines")
    print(f"    geometry_source: {restored_sentence.geometry_source}")
    
    assert restored_sentence.text == sentence.text, "Text corrupted in OpenSearch"
    assert len(restored_sentence.rects) == 2, "Rects lost in OpenSearch"
    assert restored_sentence.geometry_source == "apryse_span", "geometry_source lost"
    
    print(f"  ✅ OpenSearch document structure validated")
    return True


def main():
    print("\n" + "=" * 80)
    print("TEST 11: OpenSearch Serialization — Geometry Preservation")
    print("=" * 80)
    
    tests = [
        ("11A", test_sentence_span_json_roundtrip),
        ("11B", test_multiline_spans_json_preservation),
        ("11C", test_empty_rects_json_handling),
        ("11D", test_geometry_source_field_consistency),
        ("11E", test_opensearch_document_structure),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, True))
        except Exception as e:
            print(f"\n  ❌ FAILED: {e}")
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
        print("\n✅ TEST 11 PASSED — OpenSearch serialization validated")
        return 0
    else:
        print("\n❌ TEST 11 FAILED — Geometry loss detected in serialization")
        return 1


if __name__ == "__main__":
    exit(main())
