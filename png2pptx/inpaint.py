"""Remove detected text from the background image by inpainting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .models import TextBlock, WordBox


def _build_rectangular_mask(
    img_width: int,
    img_height: int,
    blocks: list[TextBlock],
    dilate_px: int,
) -> Image.Image:
    """Create the legacy rectangle mask used when no image content is available."""
    mask_arr = np.zeros((img_height, img_width), dtype=np.uint8)

    for block in blocks:
        for word in block.words:
            x0 = max(0, word.x - dilate_px)
            y0 = max(0, word.y - dilate_px)
            x1 = min(img_width, word.right + dilate_px)
            y1 = min(img_height, word.bottom + dilate_px)
            mask_arr[y0:y1, x0:x1] = 255

    return Image.fromarray(mask_arr, mode="L")


def _estimate_text_color(
    pixels: np.ndarray,
    fallback: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate local background/text colors from a word crop."""
    fallback = fallback.astype(np.float32)
    if len(pixels) == 0:
        return fallback, fallback

    bg_color = np.median(pixels, axis=0).astype(np.float32)
    distances = np.linalg.norm(pixels.astype(np.float32) - bg_color, axis=1)

    if float(distances.max()) < 8.0:
        return bg_color, fallback

    threshold = np.percentile(distances, 85)
    text_pixels = pixels[distances >= threshold]
    if len(text_pixels) == 0:
        return bg_color, fallback

    local_text = np.median(text_pixels, axis=0).astype(np.float32)
    if np.linalg.norm(local_text - bg_color) >= np.linalg.norm(fallback - bg_color):
        return bg_color, local_text
    return bg_color, fallback


def _filter_components(
    mask: np.ndarray,
    word_area: int,
) -> np.ndarray:
    """Drop obviously over-large mask blobs that are unlikely to be glyphs."""
    import cv2

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    if labels_count <= 1:
        return mask

    max_area = max(12, int(word_area * 0.65))
    filtered = np.zeros_like(mask, dtype=bool)
    for label in range(1, labels_count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area <= max_area:
            filtered |= labels == label

    return filtered if filtered.any() else mask


def _build_word_mask(
    img_arr: np.ndarray,
    word: WordBox,
    block_color: tuple[int, int, int],
    dilate_px: int,
) -> tuple[slice, slice, np.ndarray]:
    """Build a tight mask for one OCR word using local color/contrast cues."""
    import cv2

    img_height, img_width = img_arr.shape[:2]
    context_px = max(2, dilate_px)
    x0 = max(0, word.x - context_px)
    y0 = max(0, word.y - context_px)
    x1 = min(img_width, word.right + context_px)
    y1 = min(img_height, word.bottom + context_px)

    region = img_arr[y0:y1, x0:x1]
    region_h, region_w = region.shape[:2]

    inner_x0 = word.x - x0
    inner_y0 = word.y - y0
    inner_x1 = inner_x0 + word.width
    inner_y1 = inner_y0 + word.height

    search_mask = np.zeros((region_h, region_w), dtype=np.uint8)
    inner_area = np.zeros((region_h, region_w), dtype=bool)
    inner_area[inner_y0:inner_y1, inner_x0:inner_x1] = True
    search_margin = max(1, min(2, dilate_px // 2 or 1))
    sx0 = max(0, inner_x0 - search_margin)
    sy0 = max(0, inner_y0 - search_margin)
    sx1 = min(region_w, inner_x1 + search_margin)
    sy1 = min(region_h, inner_y1 + search_margin)
    search_mask[sy0:sy1, sx0:sx1] = 255

    inner_pixels = region[inner_y0:inner_y1, inner_x0:inner_x1].reshape(-1, 3)
    fallback_color = np.array(block_color, dtype=np.float32)
    bg_color, text_color = _estimate_text_color(inner_pixels, fallback_color)

    region_float = region.astype(np.float32)
    bg_dist = np.linalg.norm(region_float - bg_color, axis=2)
    text_dist = np.linalg.norm(region_float - text_color, axis=2)
    color_separation = float(np.linalg.norm(text_color - bg_color))
    color_margin = max(6.0, min(24.0, color_separation * 0.2))
    color_mask = text_dist + color_margin < bg_dist

    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    blur_size = max(3, min(31, ((max(word.height, 6) // 2) * 2) + 1))
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    contrast = cv2.absdiff(gray, blurred)
    otsu_threshold, contrast_mask = cv2.threshold(
        contrast,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    contrast_floor = max(8, int(otsu_threshold))
    contrast_mask = contrast >= contrast_floor

    search_area = search_mask > 0
    candidate = search_area & contrast_mask & color_mask
    word_area = max(1, word.width * word.height)

    if candidate.sum() < max(4, int(word_area * 0.02)):
        search_bg_dist = bg_dist[search_area]
        search_contrast = contrast[search_area]
        bg_threshold = max(10.0, np.percentile(search_bg_dist, 65)) if len(search_bg_dist) else 10.0
        contrast_threshold = max(8.0, np.percentile(search_contrast, 55)) if len(search_contrast) else 8.0
        candidate = search_area & (bg_dist >= bg_threshold) & (contrast >= contrast_threshold)

    inner_bg_dist = bg_dist[inner_area]
    inner_contrast = contrast[inner_area]
    if len(inner_bg_dist) and len(inner_contrast):
        fringe_bg_threshold = max(8.0, np.percentile(inner_bg_dist, 55))
        fringe_contrast_threshold = max(6.0, np.percentile(inner_contrast, 45))
        fringe_candidate = inner_area & (bg_dist >= fringe_bg_threshold) & (contrast >= fringe_contrast_threshold)

        if color_separation >= 12.0:
            inner_text_dist = text_dist[inner_area]
            if len(inner_text_dist):
                fringe_text_threshold = np.percentile(inner_text_dist, 45)
                fringe_candidate &= (text_dist <= fringe_text_threshold) | (bg_dist >= fringe_bg_threshold + 4.0)

        candidate |= fringe_candidate

    candidate = _filter_components(candidate, word_area)
    candidate_u8 = (candidate.astype(np.uint8)) * 255

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    candidate_u8 = cv2.morphologyEx(candidate_u8, cv2.MORPH_CLOSE, close_kernel)

    dilate_radius = max(1, min(3, (dilate_px + 1) // 2))
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (dilate_radius * 2 + 1, dilate_radius * 2 + 1),
    )
    candidate_u8 = cv2.dilate(candidate_u8, dilate_kernel)

    return slice(y0, y1), slice(x0, x1), candidate_u8


def _build_mask(
    img_width: int,
    img_height: int,
    blocks: list[TextBlock],
    dilate_px: int = 4,
    img_arr: np.ndarray | None = None,
) -> Image.Image:
    """Create a binary mask covering detected text.

    When the source image is provided, the mask is refined inside each OCR word
    box so inpainting targets glyph-shaped pixels instead of the full word
    rectangle. This keeps neighboring panel colors from bleeding together.
    """
    if img_arr is None:
        return _build_rectangular_mask(img_width, img_height, blocks, dilate_px)

    try:
        mask_arr = np.zeros((img_height, img_width), dtype=np.uint8)
        for block in blocks:
            for word in block.words:
                ys, xs, word_mask = _build_word_mask(img_arr, word, block.color, dilate_px)
                existing = mask_arr[ys, xs]
                mask_arr[ys, xs] = np.maximum(existing, word_mask)
        return Image.fromarray(mask_arr, mode="L")
    except ImportError:
        return _build_rectangular_mask(img_width, img_height, blocks, dilate_px)


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
    try:
        from scipy import ndimage  # type: ignore

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

    img_arr = np.array(img)
    mask = _build_mask(w, h, blocks, dilate_px=dilate_px, img_arr=img_arr)

    try:
        mask_arr = np.array(mask)
        result_arr = _inpaint_opencv(img_arr, mask_arr, radius=inpaint_radius)
        result = Image.fromarray(result_arr)
    except ImportError:
        result = _inpaint_pil_fallback(img, mask)

    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_clean.png"

    result.save(str(output_path))
    return output_path
