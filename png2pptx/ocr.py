"""Tesseract OCR extraction — returns word-level bounding boxes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image

from .models import WordBox

_AGGRESSIVE_COMMON_SHORT_WORDS = {
    "a", "ai", "al", "an", "as", "at", "be", "by", "do", "go", "he",
    "hi", "i", "if", "in", "is", "it", "low", "max", "min", "ml", "no",
    "of", "ok", "on", "or", "so", "to", "ui", "up", "us", "ux", "we",
    "xai",
}
_AI_CONTEXT_WORDS = {
    "accuracy", "adaptation", "ai", "data", "deployed", "deployment",
    "deployments", "drift", "explainable", "fairness", "model", "models",
    "monitoring", "prediction", "predictions", "trained", "trustworthy",
    "visual", "visualization", "visualizations",
}


@dataclass
class _LocalOcrRegion:
    lines: list[list[WordBox]] = field(default_factory=list)
    x: int = 0
    y: int = 0
    right: int = 0
    bottom: int = 0
    median_height: float = 0.0

    @classmethod
    def from_line(cls, line: list[WordBox]) -> "_LocalOcrRegion":
        x, y, right, bottom = _line_bounds(line)
        return cls(
            lines=[sorted(line, key=lambda word: word.x)],
            x=x,
            y=y,
            right=right,
            bottom=bottom,
            median_height=float(_line_median_height(line)),
        )

    @property
    def width(self) -> int:
        return self.right - self.x

    @property
    def height(self) -> int:
        return self.bottom - self.y

    @property
    def text_char_count(self) -> int:
        return sum(len(_normalize_text(word.text)) for line in self.lines for word in line)

    def add_line(self, line: list[WordBox]) -> None:
        sorted_line = sorted(line, key=lambda word: word.x)
        x, y, right, bottom = _line_bounds(sorted_line)
        self.lines.append(sorted_line)
        self.x = min(self.x, x)
        self.y = min(self.y, y)
        self.right = max(self.right, right)
        self.bottom = max(self.bottom, bottom)
        heights = sorted(_line_median_height(region_line) for region_line in self.lines)
        self.median_height = float(heights[len(heights) // 2])

    def crop_box(self, image_width: int, image_height: int) -> tuple[int, int, int, int]:
        pad_x = max(20, int(round(self.median_height * 1.4)))
        pad_y = max(12, int(round(self.median_height * 0.9)))
        return (
            max(0, self.x - pad_x),
            max(0, self.y - pad_y),
            min(image_width, self.right + pad_x),
            min(image_height, self.bottom + pad_y),
        )


@dataclass
class _WordLineBucket:
    words: list[WordBox] = field(default_factory=list)
    center_y: float = 0.0
    median_height: float = 0.0


def extract_words(
    image_path: str | Path,
    confidence_threshold: float = 40.0,
    lang: str = "eng",
    ocr_mode: str = "fast",
) -> tuple[list[WordBox], int, int]:
    """Run Tesseract OCR and return words with bounding boxes.

    Returns:
        Tuple of (list of WordBox, image_width, image_height).
    """
    img = Image.open(image_path).convert("RGB")
    img_width, img_height = img.size

    if ocr_mode == "fast":
        words = _run_ocr_pass(
            img,
            lang=lang,
            confidence_threshold=confidence_threshold,
            config="--oem 3 --psm 6",
        )
    elif ocr_mode == "aggressive":
        words = _extract_words_aggressive(
            img,
            lang=lang,
            confidence_threshold=confidence_threshold,
        )
    else:
        raise ValueError(f"Unsupported OCR mode: {ocr_mode}")

    return words, img_width, img_height


def _extract_words_aggressive(
    img: Image.Image,
    lang: str,
    confidence_threshold: float,
) -> list[WordBox]:
    """Run multiple OCR passes on enhanced image variants and merge results."""
    pass_results: list[list[WordBox]] = []

    pass_results.append(
        _run_ocr_pass(
            img,
            lang=lang,
            confidence_threshold=confidence_threshold,
            config="--oem 3 --psm 6",
        )
    )

    for variant_image, config, scale in _build_aggressive_variants(img):
        pass_results.append(
            _run_ocr_pass(
                variant_image,
                lang=lang,
                confidence_threshold=confidence_threshold,
                config=config,
                scale=scale,
            )
        )

    merged = _reindex_words_by_geometry(_deduplicate_words(_merge_aggressive_passes(pass_results)))
    refined = _refine_with_local_crops(
        img,
        merged,
        lang=lang,
        confidence_threshold=confidence_threshold,
    )
    return _normalize_ai_confusions(refined)


def _build_aggressive_variants(
    img: Image.Image,
) -> list[tuple[Image.Image, str, float]]:
    """Build enhanced image variants for aggressive OCR."""
    rgb = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    scale = 2.0
    upscaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    contrast = cv2.convertScaleAbs(upscaled, alpha=1.6, beta=0)
    blurred = cv2.GaussianBlur(contrast, (0, 0), 1.2)
    sharpened = cv2.addWeighted(contrast, 1.5, blurred, -0.5, 0)
    otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    return [
        (Image.fromarray(sharpened), "--oem 3 --psm 6", scale),
        (Image.fromarray(otsu), "--oem 3 --psm 6", scale),
        (Image.fromarray(otsu), "--oem 3 --psm 11", scale),
        (Image.fromarray(adaptive), "--oem 3 --psm 11", scale),
    ]


def _refine_with_local_crops(
    img: Image.Image,
    words: list[WordBox],
    lang: str,
    confidence_threshold: float,
) -> list[WordBox]:
    refined_words = _reindex_words_by_geometry(words)
    image_width, image_height = img.size

    for region in _build_local_refinement_regions(refined_words, image_width, image_height):
        local_words = _run_local_crop_ocr(
            img,
            region,
            lang=lang,
            confidence_threshold=confidence_threshold,
        )
        if not local_words:
            continue
        refined_words = _replace_region_lines(refined_words, local_words, region)

    return _reindex_words_by_geometry(
        _deduplicate_words(refined_words),
        tolerance_factor=0.35,
        max_tolerance=16.0,
    )


def _build_local_refinement_regions(
    words: list[WordBox],
    image_width: int,
    image_height: int,
) -> list[_LocalOcrRegion]:
    del image_width
    from .layout import group_into_blocks

    regions: list[_LocalOcrRegion] = []
    line_candidates = [sorted(block.words, key=lambda word: word.x) for block in group_into_blocks(words)]
    for line in line_candidates:
        best_region: _LocalOcrRegion | None = None
        best_score = 0.0

        for region in regions:
            score = _region_line_match_score(region, line)
            if score > best_score:
                best_region = region
                best_score = score

        if best_region is None or best_score < 0.6:
            regions.append(_LocalOcrRegion.from_line(line))
            continue

        best_region.add_line(line)

    max_region_height = max(140, min(int(image_height * 0.24), 320))
    candidates = [
        region
        for region in regions
        if len(region.lines) >= 2
        and region.text_char_count >= 18
        and region.width >= 160
        and region.height <= max_region_height
        and 16.0 <= region.median_height <= 60.0
        and _region_should_run_local_ocr(region, image_height)
    ]
    candidates.sort(key=lambda region: (len(region.lines), region.text_char_count), reverse=True)
    return candidates[:12]


def _region_should_run_local_ocr(region: _LocalOcrRegion, image_height: int) -> bool:
    top_paragraph = (
        region.y <= image_height * 0.14
        and len(region.lines) <= 3
        and region.text_char_count >= 40
    )
    return top_paragraph or _region_has_same_row_split(region)


def _region_has_same_row_split(region: _LocalOcrRegion) -> bool:
    sorted_lines = sorted(region.lines, key=lambda line: (min(word.y for word in line), min(word.x for word in line)))
    for index, line in enumerate(sorted_lines):
        line_x, line_y, line_right, line_bottom = _line_bounds(line)
        for other in sorted_lines[index + 1:]:
            other_x, other_y, other_right, other_bottom = _line_bounds(other)
            if abs(line_y - other_y) > region.median_height * 0.35 + 4.0:
                continue

            horizontal_gap = _range_gap(line_x, line_right, other_x, other_right)
            if horizontal_gap >= region.median_height * 1.8 + 24.0:
                return True

            vertical_overlap = _range_overlap_on_smaller(line_y, line_bottom, other_y, other_bottom)
            if vertical_overlap >= 0.55 and horizontal_gap >= region.median_height + 18.0:
                return True

    return False


def _group_words_by_geometry(
    words: list[WordBox],
    tolerance_factor: float = 0.6,
    max_tolerance: float | None = None,
) -> list[list[WordBox]]:
    if not words:
        return []

    grouped: dict[int, list[WordBox]] = {}
    for word in _reindex_words_by_geometry(
        sorted(words, key=lambda item: (item.y, item.x)),
        tolerance_factor=tolerance_factor,
        max_tolerance=max_tolerance,
    ):
        grouped.setdefault(word.block_num, []).append(word)

    return [
        sorted(line_words, key=lambda item: item.x)
        for _line_num, line_words in sorted(grouped.items())
    ]


def _region_line_match_score(region: _LocalOcrRegion, line: list[WordBox]) -> float:
    x, y, right, bottom = _line_bounds(line)
    line_height = float(max(1, bottom - y))
    vertical_gap = max(0.0, float(y - region.bottom))
    vertical_limit = max(region.median_height, line_height) * 0.8 + 10.0
    if vertical_gap > vertical_limit:
        return 0.0

    horizontal_overlap = _range_overlap_on_smaller(region.x, region.right, x, right)
    horizontal_gap = _range_gap(region.x, region.right, x, right)
    horizontal_limit = max(region.median_height, line_height) * 2.2 + 24.0
    if horizontal_overlap < 0.18 and horizontal_gap > horizontal_limit:
        return 0.0

    vertical_score = 1.0 if vertical_gap <= 0.0 else max(0.0, 1.0 - (vertical_gap / max(vertical_limit, 1.0)))
    horizontal_score = horizontal_overlap
    if horizontal_overlap <= 0.0:
        width_limit = max(region.width, right - x) * 0.45 + 30.0
        horizontal_score = max(0.0, 1.0 - (horizontal_gap / max(width_limit, 1.0)))

    return vertical_score + horizontal_score


def _run_local_crop_ocr(
    img: Image.Image,
    region: _LocalOcrRegion,
    lang: str,
    confidence_threshold: float,
) -> list[WordBox]:
    x0, y0, x1, y1 = region.crop_box(*img.size)
    crop = np.array(img.crop((x0, y0, x1, y1)).convert("RGB"))
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    scale = 2.5 if region.median_height < 34.0 else 2.0
    upscaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    contrast = cv2.convertScaleAbs(upscaled, alpha=1.8, beta=0)
    blurred = cv2.GaussianBlur(contrast, (0, 0), 1.0)
    sharpened = cv2.addWeighted(contrast, 1.6, blurred, -0.6, 0)
    otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    local_passes = [
        _run_ocr_pass(
            Image.fromarray(sharpened),
            lang=lang,
            confidence_threshold=confidence_threshold,
            config="--oem 3 --psm 4",
            scale=scale,
            x_offset=x0,
            y_offset=y0,
        ),
        _run_ocr_pass(
            Image.fromarray(sharpened),
            lang=lang,
            confidence_threshold=confidence_threshold,
            config="--oem 3 --psm 6",
            scale=scale,
            x_offset=x0,
            y_offset=y0,
        ),
        _run_ocr_pass(
            Image.fromarray(otsu),
            lang=lang,
            confidence_threshold=confidence_threshold,
            config="--oem 3 --psm 4",
            scale=scale,
            x_offset=x0,
            y_offset=y0,
        ),
    ]

    return _deduplicate_words(_merge_aggressive_passes(local_passes))


def _replace_region_lines(
    words: list[WordBox],
    local_words: list[WordBox],
    region: _LocalOcrRegion,
) -> list[WordBox]:
    current_words = _reindex_words_by_geometry(words)
    inside_region = [word for word in current_words if _word_center_in_region(word, region)]
    if not inside_region:
        return current_words

    outside_region = [word for word in current_words if not _word_center_in_region(word, region)]
    existing_lines = _group_words_by_geometry(inside_region, tolerance_factor=0.35, max_tolerance=16.0)
    candidate_lines = _group_words_by_geometry(
        [word for word in local_words if _word_center_in_region(word, region)]
        ,
        tolerance_factor=0.35,
        max_tolerance=16.0,
    )
    if not candidate_lines:
        return current_words

    if _region_has_same_row_split(region) and _should_replace_region(existing_lines, candidate_lines):
        return _reindex_words_by_geometry(
            sorted(outside_region + [word for line in candidate_lines for word in line], key=lambda word: (word.y, word.x)),
            tolerance_factor=0.35,
            max_tolerance=16.0,
        )

    replaced_line_indexes: set[int] = set()
    pruned_word_ids: set[int] = set()
    accepted_candidates: list[WordBox] = []

    for candidate_line in candidate_lines:
        match_index = _best_matching_line_index(existing_lines, candidate_line)
        if match_index is None:
            if _should_add_line(candidate_line):
                accepted_candidates.extend(candidate_line)
            continue

        if match_index in replaced_line_indexes:
            continue

        if _should_replace_line(existing_lines[match_index], candidate_line):
            replaced_line_indexes.add(match_index)
            accepted_candidates.extend(candidate_line)
            for other_index, existing_line in enumerate(existing_lines):
                if other_index == match_index:
                    continue
                for word in existing_line:
                    if any(_candidate_region_score(candidate_word, word) >= 0.3 for candidate_word in candidate_line):
                        pruned_word_ids.add(id(word))

    kept_region_words = [
        word
        for index, existing_line in enumerate(existing_lines)
        if index not in replaced_line_indexes
        for word in existing_line
        if id(word) not in pruned_word_ids
    ]
    return _reindex_words_by_geometry(
        sorted(outside_region + kept_region_words + accepted_candidates, key=lambda word: (word.y, word.x)),
        tolerance_factor=0.35,
        max_tolerance=16.0,
    )


def _best_matching_line_index(
    existing_lines: list[list[WordBox]],
    candidate_line: list[WordBox],
) -> int | None:
    best_index: int | None = None
    best_score = 0.0

    for index, existing_line in enumerate(existing_lines):
        score = _line_match_score(existing_line, candidate_line)
        if score > best_score:
            best_index = index
            best_score = score

    if best_score < 0.8:
        return None
    return best_index


def _line_match_score(existing_line: list[WordBox], candidate_line: list[WordBox]) -> float:
    ex_x, ex_y, ex_right, ex_bottom = _line_bounds(existing_line)
    cand_x, cand_y, cand_right, cand_bottom = _line_bounds(candidate_line)

    vertical_overlap = _range_overlap_on_smaller(ex_y, ex_bottom, cand_y, cand_bottom)
    line_height = max(ex_bottom - ex_y, cand_bottom - cand_y)
    centers_close = abs(((ex_y + ex_bottom) / 2) - ((cand_y + cand_bottom) / 2)) <= line_height * 0.85
    if vertical_overlap < 0.2 and not centers_close:
        return 0.0

    horizontal_overlap = _range_overlap_on_smaller(ex_x, ex_right, cand_x, cand_right)
    horizontal_gap = _range_gap(ex_x, ex_right, cand_x, cand_right)
    horizontal_limit = max(ex_right - ex_x, cand_right - cand_x) * 0.5 + 30.0
    if horizontal_overlap < 0.12 and horizontal_gap > horizontal_limit:
        return 0.0

    score = vertical_overlap * 2.0 + horizontal_overlap
    if horizontal_gap <= 0.0:
        score += 0.25
    return score


def _should_replace_line(existing_line: list[WordBox], candidate_line: list[WordBox]) -> bool:
    existing_score = _line_quality_score(existing_line)
    candidate_score = _line_quality_score(candidate_line)
    existing_chars = _line_char_count(existing_line)
    candidate_chars = _line_char_count(candidate_line)

    if candidate_score < 140.0:
        return False
    if candidate_score >= existing_score + 30.0:
        return True
    return candidate_chars > existing_chars and candidate_score >= existing_score * 1.08


def _should_add_line(candidate_line: list[WordBox]) -> bool:
    avg_confidence = sum(word.confidence for word in candidate_line) / len(candidate_line)
    return (
        len(candidate_line) >= 3
        and _line_char_count(candidate_line) >= 14
        and avg_confidence >= 88.0
        and _line_quality_score(candidate_line) >= 220.0
    )


def _should_replace_region(
    existing_lines: list[list[WordBox]],
    candidate_lines: list[list[WordBox]],
) -> bool:
    existing_score = sum(_line_quality_score(line) for line in existing_lines)
    candidate_score = sum(_line_quality_score(line) for line in candidate_lines)
    existing_chars = sum(_line_char_count(line) for line in existing_lines)
    candidate_chars = sum(_line_char_count(line) for line in candidate_lines)

    if len(candidate_lines) < len(existing_lines):
        return False
    if candidate_score >= existing_score + 40.0:
        return True
    return candidate_chars > existing_chars and candidate_score >= existing_score * 1.05


def _line_quality_score(words: list[WordBox]) -> float:
    return sum(_word_quality_score(word) for word in words)


def _word_quality_score(word: WordBox) -> float:
    normalized = _normalize_text(word.text)
    if not normalized:
        return -50.0

    score = word.confidence + min(len(normalized), 12) * 8.0
    if len(normalized) >= 4:
        score += 10.0
    if len(normalized) == 1 and normalized not in {"a", "i"} and not normalized.isdigit():
        score -= 35.0
    if len(normalized) <= 2 and normalized not in _AGGRESSIVE_COMMON_SHORT_WORDS and not normalized.isdigit():
        score -= 15.0
    if any(char in word.text for char in "\\|[]{}~"):
        score -= 30.0
    if len(normalized) >= 5 and len(set(normalized)) <= 2:
        score -= 25.0
    if word.confidence < 55.0 and len(normalized) <= 5:
        score -= 20.0
    return score


def _normalize_ai_confusions(words: list[WordBox]) -> list[WordBox]:
    normalized_words: list[WordBox] = []

    for line in _group_words_by_geometry(words):
        for index, word in enumerate(line):
            previous_normalized = _normalize_text(line[index - 1].text) if index > 0 else ""
            next_normalized = _normalize_text(line[index + 1].text) if index + 1 < len(line) else ""
            normalized_words.append(
                WordBox(
                    text=_normalize_ai_like_token(word.text, previous_normalized, next_normalized),
                    x=word.x,
                    y=word.y,
                    width=word.width,
                    height=word.height,
                    confidence=word.confidence,
                    line_num=word.line_num,
                    block_num=word.block_num,
                    par_num=word.par_num,
                )
            )

    return _reindex_words_by_geometry(normalized_words)


def _normalize_ai_like_token(
    text: str,
    previous_normalized: str = "",
    next_normalized: str = "",
) -> str:
    match = re.match(r"^([^A-Za-z0-9]*)([A-Za-z0-9]+)([^A-Za-z0-9]*)$", text)
    if not match:
        return text

    prefix, core, suffix = match.groups()
    replacement = core
    if core == "Al" and ({previous_normalized, next_normalized} & _AI_CONTEXT_WORDS):
        replacement = "AI"
    elif core.endswith("Al") and core[:-2] and any(char.isupper() or char.isdigit() for char in core[:-2]):
        replacement = f"{core[:-2]}AI"

    return f"{prefix}{replacement}{suffix}"


def _line_bounds(words: list[WordBox]) -> tuple[int, int, int, int]:
    return (
        min(word.x for word in words),
        min(word.y for word in words),
        max(word.right for word in words),
        max(word.bottom for word in words),
    )


def _line_median_height(words: list[WordBox]) -> int:
    heights = sorted(word.height for word in words)
    return heights[len(heights) // 2]


def _line_char_count(words: list[WordBox]) -> int:
    return sum(len(_normalize_text(word.text)) for word in words)


def _word_center_in_region(word: WordBox, region: _LocalOcrRegion) -> bool:
    center_x = _center_x(word)
    return region.x <= center_x <= region.right and region.y <= word.center_y <= region.bottom


def _range_overlap_on_smaller(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    overlap = max(0, min(end_a, end_b) - max(start_a, start_b))
    smaller = min(end_a - start_a, end_b - start_b)
    if smaller <= 0:
        return 0.0
    return float(overlap) / float(smaller)


def _range_gap(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    if end_a < start_b:
        return float(start_b - end_a)
    if end_b < start_a:
        return float(start_a - end_b)
    return 0.0


def _run_ocr_pass(
    img: Image.Image,
    lang: str,
    confidence_threshold: float,
    config: str = "",
    scale: float = 1.0,
    x_offset: int = 0,
    y_offset: int = 0,
) -> list[WordBox]:
    """Run a single Tesseract OCR pass and map boxes back to original pixels."""
    import pytesseract

    data = pytesseract.image_to_data(
        img,
        lang=lang,
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    words: list[WordBox] = []
    n_items = len(data["text"])

    for i in range(n_items):
        text = data["text"][i].strip()
        if not text:
            continue

        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            continue
        if conf < confidence_threshold:
            continue

        words.append(
            WordBox(
                text=text,
                x=int(round(float(data["left"][i]) / scale)) + x_offset,
                y=int(round(float(data["top"][i]) / scale)) + y_offset,
                width=max(1, int(round(float(data["width"][i]) / scale))),
                height=max(1, int(round(float(data["height"][i]) / scale))),
                confidence=conf,
                line_num=int(data["line_num"][i]),
                block_num=int(data["block_num"][i]),
                par_num=int(data["par_num"][i]),
            )
        )

    return words


def _deduplicate_words(words: list[WordBox]) -> list[WordBox]:
    """Remove overlapping duplicate detections, keeping the strongest candidate."""
    ranked = sorted(
        words,
        key=lambda word: (
            word.confidence,
            len(_normalize_text(word.text)),
            word.width * word.height,
        ),
        reverse=True,
    )

    kept: list[WordBox] = []
    for candidate in ranked:
        if any(_same_word_region(candidate, existing) for existing in kept):
            continue
        kept.append(candidate)

    return sorted(kept, key=lambda word: (word.y, word.x))


def _merge_aggressive_passes(pass_results: list[list[WordBox]]) -> list[WordBox]:
    """Merge multiple OCR passes, preferring consensus and discarding likely artifacts."""
    clusters: list[list[tuple[int, WordBox]]] = []

    for pass_id, words in enumerate(pass_results):
        for word in sorted(words, key=lambda item: (item.center_y, item.x)):
            best_cluster: list[tuple[int, WordBox]] | None = None
            best_score = 0.0

            for cluster in clusters:
                score = max(
                    _candidate_region_score(word, existing_word)
                    for _existing_pass_id, existing_word in cluster
                )
                if score > best_score:
                    best_cluster = cluster
                    best_score = score

            if best_cluster is None or best_score < 0.25:
                clusters.append([(pass_id, word)])
                continue

            best_cluster.append((pass_id, word))

    merged: list[WordBox] = []
    for cluster in clusters:
        chosen = _select_cluster_candidate(cluster)
        if chosen is not None:
            merged.append(chosen)

    return sorted(merged, key=lambda word: (word.y, word.x))


def _select_cluster_candidate(cluster: list[tuple[int, WordBox]]) -> WordBox | None:
    grouped: dict[str, list[tuple[int, WordBox]]] = {}

    for pass_id, word in cluster:
        key = _normalize_text(word.text) or word.text.strip().lower()
        grouped.setdefault(key, []).append((pass_id, word))

    best_group: list[tuple[int, WordBox]] | None = None
    best_score: float | None = None

    for group in grouped.values():
        score = _candidate_group_score(group)
        if best_score is None or score > best_score:
            best_group = group
            best_score = score

    if best_group is None:
        return None

    chosen = max(
        (word for _pass_id, word in best_group),
        key=lambda word: (
            word.confidence,
            len(_normalize_text(word.text)),
            word.width * word.height,
        ),
    )

    if len({pass_id for pass_id, _word in best_group}) == 1 and _is_suspicious_candidate(chosen):
        return None

    return chosen


def _candidate_group_score(group: list[tuple[int, WordBox]]) -> float:
    pass_count = len({pass_id for pass_id, _word in group})
    confidences = [word.confidence for _pass_id, word in group]
    best_word = max(
        (word for _pass_id, word in group),
        key=lambda word: (
            word.confidence,
            len(_normalize_text(word.text)),
            word.width * word.height,
        ),
    )
    normalized = _normalize_text(best_word.text)

    score = pass_count * 100.0
    score += max(confidences)
    score += min(len(normalized), 12) * 2.0

    if _is_suspicious_candidate(best_word):
        score -= 35.0

    return score


def _is_suspicious_candidate(word: WordBox) -> bool:
    text = word.text.strip()
    normalized = _normalize_text(text)
    if not normalized:
        return True

    if _is_numeric_marker_text(text, normalized):
        return False

    if len(normalized) == 1:
        return normalized not in {"a", "i"}

    if len(normalized) <= 3 and text != normalized:
        return True

    if len(normalized) <= 3 and normalized not in _AGGRESSIVE_COMMON_SHORT_WORDS:
        if word.height <= 14 or word.width <= 14:
            return True
        if text.isupper() or (text[:1].isupper() and text[1:].islower()):
            return True

    if len(normalized) >= 5 and len(set(normalized)) <= 2:
        return True

    return False


def _same_word_region(a: WordBox, b: WordBox) -> bool:
    """Return True if two words likely describe the same visual word region."""
    if _box_iou(a, b) >= 0.45:
        return True

    normalized_a = _normalize_text(a.text)
    normalized_b = _normalize_text(b.text)
    if not normalized_a or normalized_a != normalized_b:
        return False

    center_x_close = abs((a.x + a.width / 2) - (b.x + b.width / 2)) <= max(a.width, b.width) * 0.4
    center_y_close = abs(a.center_y - b.center_y) <= max(a.height, b.height) * 0.6
    return center_x_close and center_y_close


def _candidate_region_score(a: WordBox, b: WordBox) -> float:
    iou = _box_iou(a, b)
    overlap = _box_overlap_on_smaller(a, b)
    if iou >= 0.25 or overlap >= 0.6:
        return max(iou, overlap)

    center_x_close = abs(_center_x(a) - _center_x(b)) <= max(a.width, b.width) * 0.35
    center_y_close = abs(a.center_y - b.center_y) <= max(a.height, b.height) * 0.75
    if not center_x_close or not center_y_close:
        return 0.0

    width_ratio = min(a.width, b.width) / max(a.width, b.width)
    height_ratio = min(a.height, b.height) / max(a.height, b.height)
    return min(width_ratio, height_ratio)


def _normalize_text(text: str) -> str:
    return "".join(char for char in text.lower() if char.isalnum())


def _is_numeric_marker_text(text: str, normalized: str) -> bool:
    return bool(normalized) and normalized.isdigit() and text.endswith((".", ")"))


def _box_iou(a: WordBox, b: WordBox) -> float:
    left = max(a.x, b.x)
    top = max(a.y, b.y)
    right = min(a.right, b.right)
    bottom = min(a.bottom, b.bottom)
    if right <= left or bottom <= top:
        return 0.0

    intersection = float((right - left) * (bottom - top))
    union = float(a.width * a.height + b.width * b.height) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _box_overlap_on_smaller(a: WordBox, b: WordBox) -> float:
    intersection = _box_intersection_area(a, b)
    if intersection <= 0.0:
        return 0.0

    smaller = float(min(a.width * a.height, b.width * b.height))
    if smaller <= 0.0:
        return 0.0

    return intersection / smaller


def _box_intersection_area(a: WordBox, b: WordBox) -> float:
    left = max(a.x, b.x)
    top = max(a.y, b.y)
    right = min(a.right, b.right)
    bottom = min(a.bottom, b.bottom)
    if right <= left or bottom <= top:
        return 0.0
    return float((right - left) * (bottom - top))


def _center_x(word: WordBox) -> float:
    return word.x + word.width / 2


def _reindex_words_by_geometry(
    words: list[WordBox],
    tolerance_factor: float = 0.6,
    max_tolerance: float | None = None,
) -> list[WordBox]:
    """Assign stable line IDs after merging OCR results from multiple passes."""
    if not words:
        return []

    lines: list[_WordLineBucket] = []
    for word in sorted(words, key=lambda item: (item.center_y, item.x)):
        best_line: _WordLineBucket | None = None
        best_distance: float | None = None

        for line in lines:
            line_center_y = line.center_y
            median_height = line.median_height
            tolerance = max(median_height, word.height) * tolerance_factor
            if max_tolerance is not None:
                tolerance = min(tolerance, max_tolerance)
            distance = abs(word.center_y - line_center_y)
            if distance <= tolerance and (best_distance is None or distance < best_distance):
                best_line = line
                best_distance = distance

        if best_line is None:
            lines.append(
                _WordLineBucket(
                    words=[word],
                    center_y=word.center_y,
                    median_height=float(word.height),
                )
            )
            continue

        line_words = best_line.words
        line_words.append(word)
        heights = sorted(item.height for item in line_words)
        mid = len(heights) // 2
        best_line.median_height = float(heights[mid])
        best_line.center_y = sum(item.center_y for item in line_words) / len(line_words)

    reindexed: list[WordBox] = []
    for line_index, line in enumerate(
        sorted(lines, key=lambda item: min(word.y for word in item.words)),
        start=1,
    ):
        line_words = sorted(line.words, key=lambda item: item.x)
        for word in line_words:
            reindexed.append(
                WordBox(
                    text=word.text,
                    x=word.x,
                    y=word.y,
                    width=word.width,
                    height=word.height,
                    confidence=word.confidence,
                    line_num=1,
                    block_num=line_index,
                    par_num=1,
                )
            )

    return reindexed
