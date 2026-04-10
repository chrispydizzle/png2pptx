"""Tests for the styles (color extraction) module."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from png2pptx.models import TextBlock, WordBox
from png2pptx.styles import extract_text_colors


def _make_color_test_image(tmp_path: Path) -> Path:
    """Create an image with known colors: black text on white background."""
    img = Image.new("RGB", (200, 50), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Draw a black rectangle to simulate text area
    draw.rectangle([10, 10, 100, 40], fill=(0, 0, 0))
    path = tmp_path / "color_test.png"
    img.save(path)
    return path


def test_extract_dark_on_light(tmp_path):
    """Should detect dark text color on a light background."""
    img_path = _make_color_test_image(tmp_path)
    block = TextBlock(
        words=[
            WordBox(text="test", x=10, y=10, width=90, height=30,
                    confidence=95.0)
        ]
    )
    result = extract_text_colors(img_path, [block])
    r, g, b = result[0].color
    # The region is all black, so text color should be very dark
    assert r < 100 and g < 100 and b < 100


def test_extract_colored_text(tmp_path):
    """Should detect a specific text color."""
    img = Image.new("RGB", (200, 50), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Red rectangle simulating red text
    draw.rectangle([10, 10, 100, 40], fill=(220, 30, 30))
    path = tmp_path / "red_text.png"
    img.save(path)

    block = TextBlock(
        words=[
            WordBox(text="red", x=10, y=10, width=90, height=30,
                    confidence=95.0)
        ]
    )
    result = extract_text_colors(path, [block])
    r, g, b = result[0].color
    # Should be predominantly red
    assert r > 150
    assert g < 100
    assert b < 100


def test_empty_blocks(tmp_path):
    """Should handle empty blocks list."""
    img_path = _make_color_test_image(tmp_path)
    result = extract_text_colors(img_path, [])
    assert result == []
