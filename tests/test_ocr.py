"""Tests for the OCR module."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import pytest

# Skip all tests in this module if Tesseract is not installed
pytesseract = pytest.importorskip("pytesseract")
try:
    pytesseract.get_tesseract_version()
    HAS_TESSERACT = True
except Exception:
    HAS_TESSERACT = False

pytestmark = pytest.mark.skipif(
    not HAS_TESSERACT, reason="Tesseract not installed"
)

from png2pptx.ocr import extract_words


def _create_test_image(tmp_path: Path, text: str = "Hello World") -> Path:
    """Create a simple test image with text."""
    img = Image.new("RGB", (400, 100), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except OSError:
        font = ImageFont.load_default()
    draw.text((20, 20), text, fill="black", font=font)
    path = tmp_path / "test.png"
    img.save(path)
    return path


@pytest.fixture
def test_image(tmp_path):
    return _create_test_image(tmp_path)


def test_extract_words_returns_list(test_image):
    """extract_words should return a list of WordBox and image dimensions."""
    words, w, h = extract_words(test_image)
    assert isinstance(words, list)
    assert w == 400
    assert h == 100


def test_extract_words_finds_text(test_image):
    """Should find at least some text in a simple image."""
    words, _, _ = extract_words(test_image, confidence_threshold=0)
    # Tesseract should find at least one word
    assert len(words) > 0
    texts = " ".join(w.text for w in words).lower()
    assert "hello" in texts or "world" in texts


def test_confidence_filter(test_image):
    """Higher confidence threshold should return same or fewer words."""
    words_low, _, _ = extract_words(test_image, confidence_threshold=0)
    words_high, _, _ = extract_words(test_image, confidence_threshold=90)
    assert len(words_high) <= len(words_low)
