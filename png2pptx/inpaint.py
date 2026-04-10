"""Remove detected text from the background image by inpainting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .models import TextBlock


def _build_mask(
    img_width: int,
    img_height: int,
    blocks: list[TextBlock],
    dilate_px: int = 4,
) -> Image.Image:
    """Create a binary mask covering all text block bounding boxes.

    The mask is white (255) where text was detected and black (0) elsewhere.
    A small dilation is applied to cover anti-aliasing fringe pixels.
    """
    mask = Image.new("L", (img_width, img_height), 0)
    draw = ImageDraw.Draw(mask)

    for block in blocks:
        for word in block.words:
            x0 = max(0, word.x - dilate_px)
            y0 = max(0, word.y - dilate_px)
            x1 = min(img_width, word.right + dilate_px)
            y1 = min(img_height, word.bottom + dilate_px)
            draw.rectangle([x0, y0, x1, y1], fill=255)

    return mask


def _inpaint_opencv(
    img: np.ndarray,
    mask: np.ndarray,
    radius: int = 5,
) -> np.ndarray:
    """Inpaint using OpenCV's Telea algorithm."""
    import cv2

    # cv2.inpaint expects uint8 BGR image and uint8 mask
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    result_bgr = cv2.inpaint(bgr, mask, radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


def _inpaint_pil_fallback(
    img: Image.Image,
    mask: Image.Image,
    sample_border: int = 10,
) -> Image.Image:
    """Simple fallback: fill masked regions with median color of surrounding pixels."""
    result = img.copy()
    img_arr = np.array(img)
    mask_arr = np.array(mask)
    h, w = mask_arr.shape

    # Find connected masked regions via simple bounding-box approach
    # (we already have block bounding boxes, but this works from the mask)
    ys, xs = np.where(mask_arr > 0)
    if len(ys) == 0:
        return result

    draw = ImageDraw.Draw(result)

    # Process each text block region individually
    # Re-derive bounding boxes from mask runs for per-region fill
    from scipy import ndimage  # type: ignore

    try:
        labeled, n_features = ndimage.label(mask_arr)
    except ImportError:
        # No scipy — fill entire masked area with global border median
        border_mask = np.zeros_like(mask_arr, dtype=bool)
        border_mask[:sample_border, :] = True
        border_mask[-sample_border:, :] = True
        border_mask[:, :sample_border] = True
        border_mask[:, -sample_border:] = True
        border_pixels = img_arr[border_mask & (mask_arr == 0)]
        if len(border_pixels) > 0:
            fill = tuple(int(v) for v in np.median(border_pixels, axis=0))
        else:
            fill = (0, 0, 0)
        for y_px in range(h):
            for x_px in range(w):
                if mask_arr[y_px, x_px] > 0:
                    draw.point((x_px, y_px), fill=fill)
        return result

    for region_id in range(1, n_features + 1):
        ry, rx = np.where(labeled == region_id)
        y0, y1 = ry.min(), ry.max()
        x0, x1 = rx.min(), rx.max()

        # Sample border pixels around this region
        sy0 = max(0, y0 - sample_border)
        sy1 = min(h, y1 + sample_border + 1)
        sx0 = max(0, x0 - sample_border)
        sx1 = min(w, x1 + sample_border + 1)

        region_img = img_arr[sy0:sy1, sx0:sx1]
        region_mask = mask_arr[sy0:sy1, sx0:sx1]

        # Pixels in the border zone but NOT masked
        bg_pixels = region_img[region_mask == 0]
        if len(bg_pixels) > 0:
            fill = tuple(int(v) for v in np.median(bg_pixels, axis=0))
        else:
            fill = (0, 0, 0)

        draw.rectangle([x0, y0, x1, y1], fill=fill)

    return result


def remove_text(
    image_path: Path,
    blocks: list[TextBlock],
    output_path: Path | None = None,
    dilate_px: int = 4,
    inpaint_radius: int = 5,
) -> Path:
    """Remove detected text regions from the image and save the result.

    Tries OpenCV inpainting first (best quality). Falls back to a simpler
    median-color fill if OpenCV is not installed.

    Args:
        image_path: Path to the original PNG image.
        blocks: Text blocks whose bounding boxes define what to erase.
        output_path: Where to save the cleaned image. Defaults to
            ``<image_path>_clean.png``.
        dilate_px: Extra pixels to expand each text mask box (covers aliasing).
        inpaint_radius: Pixel radius for OpenCV inpainting (ignored in fallback).

    Returns:
        Path to the cleaned image file.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    mask = _build_mask(w, h, blocks, dilate_px=dilate_px)

    try:
        img_arr = np.array(img)
        mask_arr = np.array(mask)
        result_arr = _inpaint_opencv(img_arr, mask_arr, radius=inpaint_radius)
        result = Image.fromarray(result_arr)
    except ImportError:
        result = _inpaint_pil_fallback(img, mask)

    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_clean.png"

    result.save(str(output_path))
    return output_path
