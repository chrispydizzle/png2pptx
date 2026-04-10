"""Data classes used throughout the png2pptx pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WordBox:
    """A single word detected by Tesseract with its bounding box."""

    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    line_num: int = 0
    block_num: int = 0
    par_num: int = 0

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2


@dataclass
class TextBlock:
    """A group of words forming a logical text block (line or paragraph)."""

    words: list[WordBox] = field(default_factory=list)
    color: tuple[int, int, int] = (0, 0, 0)  # RGB

    @property
    def text(self) -> str:
        if not self.words:
            return ""
        # Sort words left-to-right, then join with spaces
        sorted_words = sorted(self.words, key=lambda w: w.x)
        return " ".join(w.text for w in sorted_words)

    @property
    def x(self) -> int:
        return min(w.x for w in self.words)

    @property
    def y(self) -> int:
        return min(w.y for w in self.words)

    @property
    def right(self) -> int:
        return max(w.right for w in self.words)

    @property
    def bottom(self) -> int:
        return max(w.bottom for w in self.words)

    @property
    def width(self) -> int:
        return self.right - self.x

    @property
    def height(self) -> int:
        return self.bottom - self.y

    @property
    def estimated_font_size_px(self) -> float:
        """Estimate font size from median word height in pixels."""
        if not self.words:
            return 12.0
        heights = sorted(w.height for w in self.words)
        mid = len(heights) // 2
        return float(heights[mid])


@dataclass
class SlideData:
    """All data needed to build one PPTX slide."""

    image_path: Path
    image_width: int
    image_height: int
    text_blocks: list[TextBlock] = field(default_factory=list)
