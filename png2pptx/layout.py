"""Group OCR words into text blocks (lines and paragraphs)."""

from __future__ import annotations

import re

from .models import TextBlock, WordBox


def group_into_blocks(
    words: list[WordBox],
    merge_lines: bool = False,
    paragraph_gap_factor: float = 1.5,
    gap_split_factor: float = 2.4,
) -> list[TextBlock]:
    """Group words into text blocks using Tesseract's block/paragraph/line numbering.

    By default, each Tesseract line becomes its own TextBlock for precise
    positioning.  Set *merge_lines=True* to combine nearby lines into
    paragraph-level blocks (useful for reflowable documents, but bad for
    infographics with scattered text).

    Words on the same Tesseract line that have a large horizontal gap
    (> gap_split_factor × median word height) are split into separate blocks.
    This prevents cross-column merging in multi-column layouts.

    Returns:
        List of TextBlock, one per Tesseract line (or per paragraph if merging).
    """
    if not words:
        return []

    words = [word for word in words if not _is_noise_word(word)]
    if not words:
        return []

    # Key: (block_num, par_num, line_num) → list of words
    line_groups: dict[tuple[int, int, int], list[WordBox]] = {}
    for w in words:
        key = (w.block_num, w.par_num, w.line_num)
        line_groups.setdefault(key, []).append(w)

    blocks: list[TextBlock] = []
    for _key, line_words in sorted(line_groups.items()):
        if line_words:
            # Split lines with large horizontal gaps (cross-column text)
            sub_blocks = _split_wide_gaps(line_words, gap_split_factor)
            for sub_block in sub_blocks:
                cleaned = _clean_block_words(sub_block.words)
                if cleaned is not None:
                    blocks.append(cleaned)

    # Remove noise blocks (tiny artifacts, single punctuation, etc.)
    blocks = _filter_noise(blocks)

    if not merge_lines or len(blocks) <= 1:
        return blocks

    return _merge_nearby_blocks(blocks, paragraph_gap_factor)

_NOISE_RE = re.compile(r'^[\s_\-—–;:.,!?\'"()|\[\]{}]+$')
_SYMBOLIC_CHARS = set("|[]{}\\/@~")
_COMMON_SHORT_WORDS = {
    "a", "ai", "al", "an", "as", "at", "be", "by", "do", "go", "he",
    "i", "if", "in", "is", "it", "ml", "no", "of", "ok", "on", "or",
    "so", "to", "ui", "up", "us", "ux", "we",
}
_SHORT_STANDALONE_ALLOWLIST = {"data", "high", "low", "time"}


def _normalized_text(text: str) -> str:
    return "".join(char for char in text.lower() if char.isalnum())


def _is_number_marker(text: str, normalized: str) -> bool:
    return bool(normalized) and normalized.isdigit() and bool(re.fullmatch(r"\d+[.)]?", text))


def _contains_symbolic_noise(text: str) -> bool:
    return any(char in _SYMBOLIC_CHARS for char in text)


def _is_noise_word(word: WordBox) -> bool:
    text = word.text.strip()
    normalized = _normalized_text(text)
    if not text:
        return True

    if not normalized:
        return _contains_symbolic_noise(text) or word.width <= 3 or word.height <= 3

    if _contains_symbolic_noise(text):
        return normalized not in {"xai"} or len(normalized) <= 2

    if len(normalized) == 1:
        if _is_number_marker(text, normalized):
            return False
        if normalized in {"a", "i"}:
            return word.confidence < 80.0 or word.width <= max(4, int(word.height * 0.2))
        return True

    if len(normalized) == 2 and text.isupper() and normalized not in {"ai", "al", "ok"}:
        return True

    if len(normalized) == 2 and text != text.lower() and normalized not in _COMMON_SHORT_WORDS and normalized not in {"ai", "al", "ok"}:
        return True

    if len(normalized) == 2 and normalized not in _COMMON_SHORT_WORDS:
        if text != text.lower():
            return True
        if word.confidence < 80.0:
            return True
        if word.width > word.height * 2.6 or word.height > word.width * 3.0:
            return True

    if len(normalized) <= 3 and normalized not in _COMMON_SHORT_WORDS and word.confidence < 65.0:
        return True

    if len(normalized) >= 5 and len(set(normalized)) <= 2 and word.confidence < 70.0:
        return True

    if len(normalized) >= 3 and len(set(normalized)) == 1:
        return True

    return False


def _drop_inline_separator(word: WordBox) -> bool:
    text = word.text.strip()
    normalized = _normalized_text(text)
    if not normalized:
        return text not in {"-", "—", "–"}
    return "|" in text or "~" in text


def _is_edge_noise_word(word: WordBox) -> bool:
    normalized = _normalized_text(word.text.strip())
    if not normalized:
        return _drop_inline_separator(word)

    if (
        len(normalized) <= 3
        and word.text.strip() != normalized
        and not word.text.strip().isalpha()
        and not _is_number_marker(word.text.strip(), normalized)
    ):
        return True

    if len(normalized) <= 2 and normalized not in _COMMON_SHORT_WORDS and not _is_number_marker(word.text.strip(), normalized):
        return True

    if word.width <= max(4, int(word.height * 0.2)):
        return True

    return _is_noise_word(word)


def _is_weak_edge_word(word: WordBox) -> bool:
    text = word.text.strip()
    normalized = _normalized_text(text)
    if not normalized:
        return True

    if word.confidence < 55.0:
        return True

    if word.confidence < 65.0 and (len(normalized) <= 5 or text != normalized):
        return True

    if len(normalized) <= 2 and word.confidence < 80.0 and normalized not in _COMMON_SHORT_WORDS:
        return True

    return False


def _clean_block_words(words: list[WordBox]) -> TextBlock | None:
    kept = [word for word in sorted(words, key=lambda item: item.x) if not _drop_inline_separator(word)]
    kept = _drop_overlapping_noise_words(kept)
    while len(kept) > 1 and _is_edge_noise_word(kept[0]):
        kept = kept[1:]
    while len(kept) > 1 and _is_edge_noise_word(kept[-1]):
        kept = kept[:-1]
    while len(kept) > 1 and _is_weak_edge_word(kept[0]):
        kept = kept[1:]
    while len(kept) > 1 and _is_weak_edge_word(kept[-1]):
        kept = kept[:-1]
    if not kept:
        return None
    if len(kept) == 1 and _is_edge_noise_word(kept[0]):
        return None
    return TextBlock(words=kept)


def _drop_overlapping_noise_words(words: list[WordBox]) -> list[WordBox]:
    kept: list[WordBox] = []

    for word in words:
        if any(_should_drop_overlapping_word(word, other) for other in words if other is not word):
            continue
        kept.append(word)

    return kept


def _should_drop_overlapping_word(word: WordBox, other: WordBox) -> bool:
    normalized = _normalized_text(word.text.strip())
    other_normalized = _normalized_text(other.text.strip())
    if not normalized or not other_normalized or normalized == other_normalized:
        return False

    if _box_overlap_on_smaller(word, other) < 0.45:
        return False

    if word.confidence + 8.0 > other.confidence:
        return False

    if len(normalized) <= 3 and normalized not in _COMMON_SHORT_WORDS:
        return True

    if word.text.strip() != normalized:
        return True

    return len(normalized) < len(other_normalized)


def _filter_noise(blocks: list[TextBlock]) -> list[TextBlock]:
    """Remove blocks that are OCR noise — tiny artifacts, punctuation-only, etc."""
    filtered = []
    for b in blocks:
        # Skip blocks with very small pixel height (sub-character noise)
        if b.height < 8:
            continue
        # Skip tiny blocks (both dimensions small — icon/graphic artifacts)
        if b.width < 25 and b.height < 25:
            continue
        # Skip single-character blocks that aren't alphanumeric
        text = b.text.strip()
        if len(text) <= 1 and not text.isalnum():
            continue
        # Skip blocks that are only punctuation/whitespace/underscores
        if _NOISE_RE.match(text):
            continue
        # Skip blocks where estimated font is unreasonably large (graphic artifacts)
        if b.estimated_font_size_px > 100:
            continue
        normalized = _normalized_text(text)
        avg_conf = sum(word.confidence for word in b.words) / len(b.words)
        if len(b.words) == 1:
            word = b.words[0]
            if avg_conf < 60.0 and (len(normalized) <= 6 or text != normalized):
                continue
            if (
                len(normalized) <= 3
                and normalized not in _COMMON_SHORT_WORDS
                and not _is_number_marker(text, normalized)
                and b.height <= 14
            ):
                continue
            if (
                len(normalized) <= 4
                and normalized not in _COMMON_SHORT_WORDS
                and normalized not in _SHORT_STANDALONE_ALLOWLIST
                and (text.isupper() or (text[:1].isalpha() and text[:1].isupper() and text[1:].islower()))
                and avg_conf < 97.0
            ):
                continue
            if (
                len(normalized) <= 3
                and text[:1].isalpha()
                and text != text.lower()
                and text != text.upper()
                and avg_conf < 90.0
            ):
                continue
        filtered.append(b)
    return filtered


def _split_wide_gaps(
    words: list[WordBox],
    gap_factor: float,
) -> list[TextBlock]:
    """Split a single Tesseract line into multiple blocks if there are large gaps."""
    if len(words) <= 1:
        return [TextBlock(words=words)]

    sorted_words = sorted(words, key=lambda w: w.x)
    median_h = sorted(w.height for w in sorted_words)[len(sorted_words) // 2]
    gaps = [max(0, current.x - previous.right) for previous, current in zip(sorted_words, sorted_words[1:])]
    positive_gaps = [gap for gap in gaps if gap > 0]
    median_gap = 0 if not positive_gaps else sorted(positive_gaps)[len(positive_gaps) // 2]
    threshold = max(median_h * gap_factor, median_gap * 4)

    groups: list[list[WordBox]] = [[sorted_words[0]]]
    for w in sorted_words[1:]:
        prev = groups[-1][-1]
        gap = w.x - prev.right
        if gap > threshold:
            groups.append([w])
        else:
            groups[-1].append(w)

    return [TextBlock(words=g) for g in groups]


def _box_overlap_on_smaller(a: WordBox, b: WordBox) -> float:
    left = max(a.x, b.x)
    top = max(a.y, b.y)
    right = min(a.right, b.right)
    bottom = min(a.bottom, b.bottom)
    if right <= left or bottom <= top:
        return 0.0

    intersection = float((right - left) * (bottom - top))
    smaller = float(min(a.width * a.height, b.width * b.height))
    if smaller <= 0.0:
        return 0.0

    return intersection / smaller


def _merge_nearby_blocks(
    blocks: list[TextBlock],
    paragraph_gap_factor: float,
) -> list[TextBlock]:
    """Merge text blocks that are close vertically and horizontally aligned."""
    if not blocks:
        return blocks

    # Sort blocks top-to-bottom by their y position
    sorted_blocks = sorted(blocks, key=lambda b: b.y)
    merged: list[TextBlock] = [sorted_blocks[0]]

    for current in sorted_blocks[1:]:
        prev = merged[-1]
        prev_line_height = prev.estimated_font_size_px
        vertical_gap = current.y - prev.bottom

        # Check horizontal overlap — blocks should be roughly aligned
        overlap_left = max(prev.x, current.x)
        overlap_right = min(prev.right, current.right)
        has_horizontal_overlap = overlap_right > overlap_left

        if (
            has_horizontal_overlap
            and vertical_gap < prev_line_height * paragraph_gap_factor
            and vertical_gap >= 0
        ):
            # Merge into previous block
            merged[-1] = TextBlock(words=prev.words + current.words)
        else:
            merged.append(current)

    return merged
