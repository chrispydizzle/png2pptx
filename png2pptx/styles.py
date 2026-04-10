"""Extract text colors from the source image at detected text positions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .models import TextBlock


def extract_text_colors(
    image_path: str | Path,
    blocks: list[TextBlock],
    margin: int = 2,
) -> list[TextBlock]:
    """Sample pixel colors under each text block and set the block's color.

    Uses the median color of pixels in the text region, which handles
    anti-aliased text edges well.

    Args:
        image_path: Path to the source PNG.
        blocks: TextBlocks with positions already set.
        margin: Pixels to shrink the sampling region inward (avoids background bleed).

    Returns:
        The same blocks list with colors populated.
    """
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    img_h, img_w = img_array.shape[:2]

    for block in blocks:
        # Shrink the region slightly to avoid sampling background pixels
        x1 = max(0, block.x + margin)
        y1 = max(0, block.y + margin)
        x2 = min(img_w, block.right - margin)
        y2 = min(img_h, block.bottom - margin)

        if x2 <= x1 or y2 <= y1:
            # Region too small — keep default black
            continue

        region = img_array[y1:y2, x1:x2]
        pixels = region.reshape(-1, 3)

        if len(pixels) == 0:
            continue

        # The text color is typically the minority color in the region
        # (background is the majority). Find the median (≈ background),
        # then select pixels far from it as text candidates.
        median_color = np.median(pixels, axis=0).astype(float)

        # Calculate distance of each pixel from the median (background)
        distances = np.sqrt(np.sum((pixels.astype(float) - median_color) ** 2, axis=1))

        # Use a high percentile to be selective — only the most distinct
        # pixels are likely actual text (sparse text on large background).
        threshold = np.percentile(distances, 85)
        text_mask = distances >= threshold

        if text_mask.any():
            text_pixels = pixels[text_mask]
            text_color = np.median(text_pixels, axis=0).astype(float)

            # Sanity check: if extracted "text" color is very close to the
            # background, it's likely wrong. Fall back to the single pixel
            # with maximum distance from background.
            dist_from_bg = np.sqrt(np.sum((text_color - median_color) ** 2))
            if dist_from_bg < 30:
                brightest_idx = np.argmax(distances)
                text_color = pixels[brightest_idx].astype(float)

            block.color = (int(text_color[0]), int(text_color[1]), int(text_color[2]))
        else:
            block.color = (0, 0, 0)

    return blocks
