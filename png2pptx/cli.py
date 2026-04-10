from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import click

from .layout import group_into_blocks
from .models import SlideData
from .ocr import extract_words
from .pptx_builder import build_pptx
from .styles import extract_text_colors


@click.group()
@click.version_option(package_name="png2pptx")
def main():
    """Convert PNG infographics to editable PowerPoint files."""


@main.command()
@click.argument("inputs", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "-o", "--output",
    default="output.pptx",
    type=click.Path(),
    help="Output PPTX file path.",
)
@click.option(
    "--confidence",
    default=40.0,
    type=float,
    help="Minimum OCR confidence threshold (0-100).",
)
@click.option(
    "--lang",
    default="eng",
    help="Tesseract language code (e.g. eng, fra, deu).",
)
@click.option(
    "--ocr-mode",
    type=click.Choice(["fast", "aggressive"], case_sensitive=False),
    default="aggressive",
    show_default=True,
    help="OCR quality mode. 'aggressive' runs slower but can recover more text.",
)
@click.option(
    "--remove-text/--no-remove-text",
    default=True,
    help="Remove detected text from the background image (inpainting).",
)
def convert(
    inputs: tuple[str, ...],
    output: str,
    confidence: float,
    lang: str,
    ocr_mode: str,
    remove_text: bool,
):
    """Convert one or more PNG files to an editable PPTX.

    Each input image becomes one slide with the original image as
    background and OCR'd text as editable overlays.

    Use --remove-text to erase detected text from the background so
    only the editable overlay text is visible (no doubling).

    Examples:

        png2pptx convert infographic.png -o result.pptx

        png2pptx convert infographic.png -o result.pptx --ocr-mode aggressive

        png2pptx convert infographic.png -o result.pptx --remove-text

        png2pptx convert *.png -o deck.pptx
    """
    slides: list[SlideData] = []
    temp_files: list[Path] = []

    for i, input_path in enumerate(inputs, 1):
        path = Path(input_path)
        click.echo(f"[{i}/{len(inputs)}] Processing {path.name}...", err=True)

        # OCR
        words, img_w, img_h = extract_words(
            path,
            confidence_threshold=confidence,
            lang=lang,
            ocr_mode=ocr_mode,
        )
        click.echo(f"  Found {len(words)} words", err=True)

        # Group into blocks
        blocks = group_into_blocks(words)
        click.echo(f"  Grouped into {len(blocks)} text blocks", err=True)

        # Extract colors
        blocks = extract_text_colors(path, blocks)

        # Optionally remove text from background image
        bg_path = path
        if remove_text and blocks:
            from .inpaint import remove_text as do_inpaint

            clean_path = Path(tempfile.mktemp(suffix=".png"))
            temp_files.append(clean_path)
            bg_path = do_inpaint(path, blocks, output_path=clean_path)
            click.echo("  Removed text from background (inpainted)", err=True)

        slides.append(
            SlideData(
                image_path=bg_path,
                image_width=img_w,
                image_height=img_h,
                text_blocks=blocks,
            )
        )

    # Build PPTX
    out_path = build_pptx(slides, output)
    click.echo(f"Saved {len(slides)} slide(s) to {out_path}", err=True)

    # Clean up temp inpainted images
    for tf in temp_files:
        try:
            tf.unlink()
        except OSError:
            pass
