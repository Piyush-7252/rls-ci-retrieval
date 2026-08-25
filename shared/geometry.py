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


def _precision_for_source(source: str) -> str:
    """Return the geometry precision implied by a source label."""
    if source == "apryse_span":
        return "exact"
    if source == "apryse_line":
        return "containing"
    if source == "object_bbox":
        return "object"
    return "none"


def _authoritative_for_source(
    source: str,
    precision: str | None = None,
    span_type: str | None = None,
) -> bool:
    """Return whether the geometry is authoritative for the semantic object.

    Native Apryse source-span geometry is authoritative for every span type.
    Object-bbox geometry is also authoritative when the candidate itself is
    the complete semantic object (paragraph, list item, table cell/row, etc.).
    It is deliberately non-authoritative for an NLP sentence because the bbox
    may contain neighbouring text when the sentence does not map exactly to
    Apryse source spans.
    """
    if source == "apryse_span" and (precision in (None, "", "exact")):
        return True
    if source == "object_bbox" and span_type not in ("sentence",):
        return True
    return False


@dataclass
class SentenceSpan:
    """
    A geometry-aware sentence or span within a PDF.

    Carries both text coordinates (char_start, char_end) and document
    coordinates (page, rects) to preserve Apryse line/span-level geometry
    through the retrieval pipeline without reverse-lookup at extraction time.
    
    IMPORTANT: Coordinate system is TWO-TIERED:
      - Object-relative: char_start/char_end (within parent object, e.g. paragraph)
      - Page-relative: page-local character offsets (within page text)
    
    Page-relative coordinates enable precise UI highlighting without reverse lookup:
      Geometry is page-local through rects/page_distribution.

    Fields
    ------
    text : str
        The actual span text.

    page : int
        Page number (1-based).

    char_start : int
        Character offset within the parent object's text (for internal tracing).

    char_end : int
        Character offset within the parent object's text (for internal tracing).
        Character offset within the page's canonical text (FOR UI HIGHLIGHTING).
        Character offset within the page's canonical text (FOR UI HIGHLIGHTING).

    rects : list[list[float]]
        Apryse span/line-level display rectangles for this span.
        For single-line spans: one box [x1, y1, x2, y2]
        For multi-line spans: one box per line
        
        NOTE: Adjacent sentences may share rectangles when the sentence boundary
        falls within a PDF line. This is expected behavior with line-level
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
        char_start=421,              # ← Within paragraph
        char_end=487,                # ← Within paragraph
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
    geometry_source: str = "none"
    # "apryse_span" = exact native source-span ownership
    # "apryse_line" = containing native line/span; NOT authoritative
    # "object_bbox" = geometry of the retrieved semantic object
    # "parent_bbox_text_search" = bbox is only a bounded UI search region
    # "none" = geometry unavailable
    geometry_precision: str = "none"
    # "exact" | "containing" | "object" | "none"
    is_authoritative: bool = False
    geometry_fallback_reason: Optional[str] = None
    parent_bbox: list[list[float]] = field(default_factory=list)
    evidence_level: str = "none"
    # Page-local geometry groups. A single semantic object may cross a PDF page
    # boundary; each entry is independently highlightable by the UI.
    page_distribution: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Keep geometry metadata internally consistent for every producer,
        # including the extraction Lambda's existing _make_display_spans().
        if self.geometry_precision == "none":
            self.geometry_precision = _precision_for_source(self.geometry_source)
        if not self.is_authoritative:
            self.is_authoritative = _authoritative_for_source(
                self.geometry_source,
                self.geometry_precision,
                self.span_type,
            )
        if self.evidence_level == "none":
            self.evidence_level = self.span_type or "none"

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "text": self.text,
            "page": self.page,
            "rects": self.rects,
            "source_object_id": self.source_object_id,
            "source_span_ids": self.source_span_ids,
            "span_type": self.span_type,
            "geometry_source": self.geometry_source,
            "geometry_precision": self.geometry_precision,
            "is_authoritative": self.is_authoritative,
            "geometry_fallback_reason": self.geometry_fallback_reason,
            "parent_bbox": self.parent_bbox,
            "evidence_level": self.evidence_level,
            "page_distribution": self.page_distribution,
        }

    @staticmethod
    def from_dict(d: dict) -> SentenceSpan:
        """Deserialize from dictionary."""
        return SentenceSpan(
            text=d.get("text", ""),
            page=d.get("page", 0),
            char_start=0,
            char_end=0,
            rects=d.get("rects", []),
            source_object_id=d.get("source_object_id"),
            source_span_ids=d.get("source_span_ids", []),
            span_type=d.get("span_type", "sentence"),
            geometry_source=d.get("geometry_source", "none"),
            geometry_precision=d.get("geometry_precision", _precision_for_source(d.get("geometry_source", "none"))),
            is_authoritative=bool(d.get("is_authoritative", _authoritative_for_source(d.get("geometry_source", "none"), d.get("geometry_precision")))),
            geometry_fallback_reason=d.get("geometry_fallback_reason"),
            parent_bbox=d.get("parent_bbox", []),
            evidence_level=d.get("evidence_level", d.get("span_type", "none")),
            page_distribution=d.get("page_distribution", []),
        )

    def has_any_geometry(self) -> bool:
        """True when this span carries any usable rectangles, including object bbox."""
        return bool(self.rects)

    def is_exact_geometry(self) -> bool:
        """True only for exact native Apryse source-span geometry."""
        return self.geometry_source == "apryse_span" and self.geometry_precision == "exact" and bool(self.rects)

    def has_geometry(self) -> bool:
        """True if this span owns complete Apryse source spans exactly."""
        return self.geometry_source == "apryse_span" and len(self.rects) > 0

    def has_fallback_geometry(self) -> bool:
        """True if this span has parent-object bbox fallback (coarser resolution than line-level)."""
        return self.geometry_source == "object_bbox" and len(self.rects) > 0

    def needs_bounded_text_search(self) -> bool:
        """True when parent bbox is only a search boundary, not final highlight geometry."""
        return self.geometry_source == "parent_bbox_text_search" and bool(self.parent_bbox) and bool(self.text)

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
            "page_distribution": self.page_distribution,
        }
