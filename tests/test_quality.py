from pathlib import Path

import numpy as np
from PIL import Image

from png2pptx.models import SlideData, TextBlock, WordBox
from png2pptx.pptx_builder import build_pptx
from png2pptx.quality import (
    compute_image_delta_metrics,
    discover_example_images,
    render_pptx_overlay,
)


def test_discover_example_images_skips_generated_pngs(tmp_path: Path):
    for name in (
        "sample_input.png",
        "CommandCenter.png",
        "sample_input_baseline_clean.png",
        "sample_input_current_clean.png",
        "sample_input_compare.png",
        "sample_input_overlay.png",
        "sample_input_review.png",
    ):
        (tmp_path / name).write_bytes(b"")

    discovered = discover_example_images(tmp_path)

    assert [path.name for path in discovered] == ["CommandCenter.png", "sample_input.png"]


def test_compute_image_delta_metrics_reports_expected_values(tmp_path: Path):
    reference = tmp_path / "reference.png"
    candidate = tmp_path / "candidate.png"

    Image.new("RGB", (2, 1), color=(0, 0, 0)).save(reference)
    image = Image.new("RGB", (2, 1), color=(0, 0, 0))
    image.putpixel((1, 0), (10, 20, 30))
    image.save(candidate)

    metrics = compute_image_delta_metrics(reference, candidate)

    assert metrics["changed_ratio"] == 0.5
    assert metrics["mae"] == 10.0
    assert metrics["max_diff"] == 30


def test_render_pptx_overlay_creates_highlighted_debug_image(tmp_path: Path):
    image_path = tmp_path / "slide.png"
    Image.new("RGB", (800, 600), color="white").save(image_path)

    block = TextBlock(
        words=[
            WordBox(
                text="Hello",
                x=100,
                y=80,
                width=180,
                height=40,
                confidence=95.0,
                line_num=1,
                block_num=1,
                par_num=1,
            )
        ],
        color=(0, 0, 0),
    )
    pptx_path = tmp_path / "slide.pptx"
    build_pptx(
        [
            SlideData(
                image_path=image_path,
                image_width=800,
                image_height=600,
                text_blocks=[block],
            )
        ],
        pptx_path,
    )

    overlay_path = tmp_path / "overlay.png"
    render_pptx_overlay(pptx_path, image_path, overlay_path)

    overlay = np.asarray(Image.open(overlay_path).convert("RGB"))
    assert np.any((overlay[:, :, 0] > 200) & (overlay[:, :, 1] < 100) & (overlay[:, :, 2] < 100))
