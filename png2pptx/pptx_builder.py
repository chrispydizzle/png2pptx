"""Build a PPTX file with image backgrounds and editable text overlays."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Pt
from PIL import ImageFont

import re

from .models import SlideData, TextBlock

# Pattern to clean OCR artifacts from text edges
_EDGE_JUNK_RE = re.compile(r'^[_\-—–\s]+|[_\-—–\s]+$')
_FONT_FAMILY = "Calibri"
_FONT_WIDTH_PADDING = 1.05
_FONT_HEIGHT_PADDING = 1.02

_FONT_PATH_CANDIDATES = (
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/Library/Fonts/Arial.ttf",
)


EMU_PER_INCH = 914400
EMU_PER_PT = 12700

# Standard 16:9 slide long edge in EMU (≈ 13.33 inches).
# We use this for the LONG edge of the slide so that portrait images
# get a generously-sized slide instead of being squashed.
SLIDE_LONG_EDGE_EMU = 12192000  # 13.33 in


def _compute_slide_dimensions(
    img_width: int, img_height: int
) -> tuple[int, int]:
    """Return (slide_width, slide_height) in EMU matching the image aspect ratio.

    The longer image dimension maps to SLIDE_LONG_EDGE_EMU; the shorter
    dimension is computed from the aspect ratio.
    """
    if img_width >= img_height:
        # Landscape or square
        slide_w = SLIDE_LONG_EDGE_EMU
        slide_h = int(SLIDE_LONG_EDGE_EMU * img_height / img_width)
    else:
        # Portrait
        slide_h = SLIDE_LONG_EDGE_EMU
        slide_w = int(SLIDE_LONG_EDGE_EMU * img_width / img_height)

    return slide_w, slide_h


def build_pptx(slides: list[SlideData], output_path: str | Path) -> Path:
    """Build a PPTX file from slide data.

    Args:
        slides: List of SlideData, each becomes one slide.
        output_path: Where to save the .pptx file.

    Returns:
        Path to the created file.
    """
    output_path = Path(output_path)
    prs = Presentation()

    if slides:
        first = slides[0]
        slide_w, slide_h = _compute_slide_dimensions(
            first.image_width, first.image_height
        )
        prs.slide_width = slide_w
        prs.slide_height = slide_h

    blank_layout = prs.slide_layouts[6]  # Blank layout

    for slide_data in slides:
        slide = prs.slides.add_slide(blank_layout)
        slide_w = prs.slide_width
        slide_h = prs.slide_height

        # Add the PNG as a full-bleed background image
        slide.shapes.add_picture(
            str(slide_data.image_path),
            left=0,
            top=0,
            width=slide_w,
            height=slide_h,
        )

        # Scale factors: image pixels → slide EMUs
        scale_x = slide_w / slide_data.image_width
        scale_y = slide_h / slide_data.image_height

        # Add text boxes for each block
        for block in slide_data.text_blocks:
            _add_text_block(slide, block, scale_x, scale_y)

    prs.save(str(output_path))
    return output_path


def _add_text_block(
    slide,
    block: TextBlock,
    scale_x: float,
    scale_y: float,
) -> None:
    """Add a transparent, single-line text box to the slide for one TextBlock."""
    clean_text = _EDGE_JUNK_RE.sub('', block.text)
    rendered_text = clean_text if clean_text else block.text

    # Convert block pixel coords → slide EMU coords
    left_emu = int(block.x * scale_x)
    top_emu = int(block.y * scale_y)
    width_emu = int(block.width * scale_x)
    height_emu = int(block.height * scale_y)

    # Minimal padding — just enough to avoid clipping the last character
    h_pad = int(width_emu * 0.05)
    width_emu += h_pad

    txbox = slide.shapes.add_textbox(
        Emu(left_emu), Emu(top_emu), Emu(width_emu), Emu(height_emu)
    )

    # Transparent background
    txbox.fill.background()

    tf = txbox.text_frame
    tf.word_wrap = False  # Single-line — never reflow
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Emu(0)
    tf.margin_right = Emu(0)
    tf.margin_top = Emu(0)
    tf.margin_bottom = Emu(0)

    # Anchor text to the vertical middle of the box
    tf.auto_size = None
    txbox.text_frame.paragraphs[0].space_before = Pt(0)
    txbox.text_frame.paragraphs[0].space_after = Pt(0)

    para = tf.paragraphs[0]
    para.alignment = PP_ALIGN.LEFT

    run = para.add_run()
    run.text = rendered_text

    # Fit the font using both width and height so long labels shrink enough
    # while short headings can remain visually large.
    fitted_font_px = _fit_font_size_px(rendered_text, block)
    font_size_pt = (fitted_font_px * scale_y) / EMU_PER_PT
    font_size_pt = max(6.0, min(font_size_pt, 96.0))
    run.font.size = Pt(font_size_pt)
    run.font.name = _FONT_FAMILY

    # Font color
    r, g, b = block.color
    run.font.color.rgb = RGBColor(r, g, b)


def _fit_font_size_px(text: str, block: TextBlock) -> float:
    """Find the largest rendered font size that fits inside the OCR box."""
    if not text.strip():
        return block.estimated_font_size_px

    target_width = max(1, int(block.width * _FONT_WIDTH_PADDING))
    target_height = max(1, int(block.height * _FONT_HEIGHT_PADDING))

    # Use the current geometry-based estimate as a seed, but search a wider range.
    seed = max(4, int(round(block.estimated_font_size_px)))
    low = 4
    high = max(seed * 2, int(block.height * 2.5), 12)
    best = low

    while low <= high:
        mid = (low + high) // 2
        measured_width, measured_height = _measure_text(text, mid)
        if measured_width <= target_width and measured_height <= target_height:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    return float(best)


@lru_cache(maxsize=1)
def _measurement_font_path() -> str | None:
    for candidate in _FONT_PATH_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


@lru_cache(maxsize=512)
def _measure_text(text: str, font_px: int) -> tuple[int, int]:
    """Measure single-line text in source-image pixels."""
    font_path = _measurement_font_path()
    if font_path is None:
        width = max(1, int(len(text) * font_px * 0.6))
        height = max(1, int(font_px))
        return width, height

    font = ImageFont.truetype(font_path, max(1, font_px))
    left, top, right, bottom = font.getbbox(text or "Ag")
    width = max(1, right - left)
    height = max(1, bottom - top)
    return width, height
