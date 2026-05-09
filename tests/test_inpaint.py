from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from png2pptx.inpaint import _build_mask, remove_text
from png2pptx.models import TextBlock, WordBox


def _make_text_block(text: str, bbox: tuple[int, int, int, int], color=(245, 245, 245)) -> TextBlock:
    x0, y0, x1, y1 = bbox
    return TextBlock(
        words=[
            WordBox(
                text=text,
                x=x0,
                y=y0,
                width=x1 - x0,
                height=y1 - y0,
                confidence=95.0,
                line_num=1,
                block_num=1,
                par_num=1,
            )
        ],
        color=color,
    )


def _make_split_background_image() -> tuple[Image.Image, Image.Image, TextBlock, np.ndarray]:
    width, height = 160, 90
    background = Image.new("RGB", (width, height), (14, 42, 78))
    bg_draw = ImageDraw.Draw(background)
    bg_draw.rectangle([width // 2, 0, width, height], fill=(28, 108, 72))

    rendered = background.copy()
    draw = ImageDraw.Draw(rendered)
    font = ImageFont.load_default()
    text = "AIAI"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    draw.text((x, y), text, fill=(245, 245, 245), font=font)

    glyph_mask = Image.new("L", (width, height), 0)
    glyph_draw = ImageDraw.Draw(glyph_mask)
    glyph_draw.text((x, y), text, fill=255, font=font)

    text_bbox = draw.textbbox((x, y), text, font=font)
    block = _make_text_block(text, text_bbox)

    return background, rendered, block, np.array(glyph_mask) > 0


def test_build_mask_targets_glyphs_instead_of_full_word_box():
    _, rendered, block, glyph_mask = _make_split_background_image()

    mask = _build_mask(
        rendered.width,
        rendered.height,
        [block],
        dilate_px=4,
        img_arr=np.array(rendered),
    )
    mask_arr = np.array(mask) > 0

    word = block.words[0]
    rect_width = min(rendered.width, word.right + 4) - max(0, word.x - 4)
    rect_height = min(rendered.height, word.bottom + 4) - max(0, word.y - 4)
    rect_area = rect_width * rect_height

    glyph_coverage = (mask_arr & glyph_mask).sum() / glyph_mask.sum()

    assert glyph_coverage >= 0.85
    assert mask_arr.sum() < rect_area * 0.6


def test_remove_text_preserves_non_text_pixels_inside_word_box(tmp_path: Path):
    background, rendered, block, glyph_mask = _make_split_background_image()

    image_path = tmp_path / "split.png"
    output_path = tmp_path / "split_clean.png"
    rendered.save(image_path)

    remove_text(image_path, [block], output_path=output_path, dilate_px=4, inpaint_radius=3)

    clean_arr = np.array(Image.open(output_path).convert("RGB"))
    background_arr = np.array(background)

    word = block.words[0]
    bbox_mask = np.zeros(glyph_mask.shape, dtype=bool)
    bbox_mask[word.y:word.bottom, word.x:word.right] = True
    non_text_bbox = bbox_mask & ~glyph_mask

    glyph_error = np.abs(clean_arr[glyph_mask].astype(int) - background_arr[glyph_mask].astype(int)).mean()
    non_text_error = np.abs(clean_arr[non_text_bbox].astype(int) - background_arr[non_text_bbox].astype(int)).mean()

    assert glyph_error < 35
    assert non_text_error < 8
