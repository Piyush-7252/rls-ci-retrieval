"""
geometry.py — Geometry-preserving span representation.

Key insight: Don't lose Apryse PDF geometry at sentence-creation time.

Every sentence in a PDF may span multiple lines, each with its own bounding box.
This module ensures that geometry information (page, rects) is preserved from
the original Apryse extraction through sentence splitting, retrieval, LLM
verification, and finally highlight extraction.

The key principle:
  HighlightExtractor should SELECT existing geometry, not CALCULATE it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SentenceSpan:
    """
    A geometry-aware sentence or span within a PDF.

    Carries both text coordinates (char_start, char_end) and document
    coordinates (page, rects) to preserve Apryse line/span-level geometry
    through the retrieval pipeline without reverse-lookup at extraction time.
    
    IMPORTANT: rects may be shared with adjacent sentences when the sentence
    boundary falls within a PDF line. This is expected behavior with line-level
    geometry resolution. The Merger validates that only accepted candidates'
    geometry reaches the final hit.

    Fields
    ------
    text : str
        The actual span text.

    page : int
        Page number (1-based or 0-based depending on context).

    char_start : int
        Character offset within the parent object's text.

    char_end : int
        Character offset within the parent object's text.

    rects : list[list[float]]
        Apryse span/line-level display rectangles for this span.
        For single-line spans: one box [x1, y1, x2, y2]
        For multi-line spans: one box per line
        
        NOTE: Adjacent sentences may share rectangles when the sentence boundary
        falls within a PDF line. This is a direct consequence of line-level
        geometry from Apryse and is expected. Merging handles this correctly:
        shared rects are kept if any owning candidate is accepted.
        
        Empty list means geometry is unknown (fallback to source_object_id).

    source_object_id : Optional[str]
        ID of the retrieval object this span came from (e.g. "chunk_001_obj_042").
        If rects is empty, fallback uses this to locate the source object's bbox.

    source_span_ids : list[str]
        Optional: IDs of intermediate spans this was derived from (for tracing).

    span_type : str
        "sentence", "list_item", "table_row", "heading", etc.

    Example
    -------
    # Multi-line sentence in a PDF
    span = SentenceSpan(
        text="The primary objective was progression-free survival "
             "as assessed by independent review committee "
             "according to RECIST 1.1.",
        page=135,
        char_start=421,
        char_end=487,
        rects=[
            [100.0, 200.0, 500.0, 215.0],  # line 1
            [100.0, 218.0, 510.0, 233.0],  # line 2
            [100.0, 236.0, 380.0, 251.0],  # line 3
        ],
        source_object_id="chunk_001_obj_042",
        span_type="sentence",
    )
    """

    text: str
    page: int
    char_start: int
    char_end: int
    rects: list[list[float]] = field(default_factory=list)
    source_object_id: Optional[str] = None
    source_span_ids: list[str] = field(default_factory=list)
    span_type: str = "sentence"
    geometry_source: str = "none"  # "apryse_span" | "object_bbox" | "none"

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "text": self.text,
            "page": self.page,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "rects": self.rects,
            "source_object_id": self.source_object_id,
            "source_span_ids": self.source_span_ids,
            "span_type": self.span_type,
            "geometry_source": self.geometry_source,
        }

    @staticmethod
    def from_dict(d: dict) -> SentenceSpan:
        """Deserialize from dictionary."""
        return SentenceSpan(
            text=d.get("text", ""),
            page=d.get("page", 0),
            char_start=d.get("char_start", 0),
            char_end=d.get("char_end", 0),
            rects=d.get("rects", []),
            source_object_id=d.get("source_object_id"),
            source_span_ids=d.get("source_span_ids", []),
            span_type=d.get("span_type", "sentence"),
            geometry_source=d.get("geometry_source", "none"),
        )

    def has_geometry(self) -> bool:
        """True if this span has explicit PDF rects (not a fallback)."""
        return self.geometry_source == "apryse_span" and len(self.rects) > 0

    def has_fallback_geometry(self) -> bool:
        """True if this span has parent-object bbox fallback (coarser resolution than line-level)."""
        return self.geometry_source == "object_bbox" and len(self.rects) > 0

    def as_legacy_display_span(self) -> dict:
        """
        Convert to the legacy display_span format for backward compatibility.

        display_span format:
            {"type": str, "text": str, "start": int, "end": int, "bbox": list}

        If we have rects, use the bounding box of all rects combined.
        Otherwise, return the legacy format without bbox info.
        """
        bbox = []
        if self.rects:
            # Compute bounding box of all rects
            x_coords = [r[0] for r in self.rects] + [r[2] for r in self.rects]
            y_coords = [r[1] for r in self.rects] + [r[3] for r in self.rects]
            bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

        return {
            "type": self.span_type,
            "text": self.text,
            "start": self.char_start,
            "end": self.char_end,
            "bbox": bbox,
        }
