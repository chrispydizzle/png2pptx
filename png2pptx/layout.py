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
    blocks = _drop_orphan_trailing_fragments(blocks)

    if not merge_lines or len(blocks) <= 1:
        return blocks

    return _merge_nearby_blocks(blocks, paragraph_gap_factor)

_NOISE_RE = re.compile(r'^[\s_\-—–;:.,!?\'"()|\[\]{}]+$')
_SYMBOLIC_CHARS = set("|[]{}\\/@~")
_INLINE_PUNCTUATION_KEEP = {"&", "+", "%"}
_COMMON_SHORT_WORDS = {
    "a", "ai", "al", "an", "as", "at", "be", "by", "do", "go", "he",
    "i", "if", "in", "is", "it", "ml", "no", "of", "ok", "on", "or",
    "so", "to", "ui", "up", "us", "ux", "we",
}
_COMMON_SHORT_UNITS = {"cm", "db", "ft", "gb", "hz", "kb", "kg", "km", "mb", "mhz", "mm", "ms", "tb"}
# When a 2-letter word is in ALL CAPS, it's much more likely to be
# legitimate (a heading word) than noise. The allowlist mirrors common
# real English/abbreviation tokens that show up in headings.
_ALLOWED_ALL_CAPS_SHORT_WORDS = {
    "ai", "al", "an", "as", "at", "be", "by", "do", "go", "he", "if",
    "in", "is", "it", "km", "ml", "no", "of", "ok", "on", "or", "so",
    "to", "ui", "us", "ux", "we",
}
_SHORT_STANDALONE_ALLOWLIST = {"avg", "data", "high", "low", "time"}
_ROMAN_NUMERAL_MARKER_RE = re.compile(r"^(?:[IVXLCM]{1,5}|[ivxlcm]{1,5})[.)]$")
# Tesseract regularly mis-reads `II.`/`III.`/`IIV.` etc. as `Il.`/`Ill.`/`IlV.`
# (capital I followed by lowercase L). Recognise the body if every character
# is a Roman numeral OR a lowercase `l`, and the trailing punctuation is `.`/`)`.
_ROMAN_NUMERAL_OCR_CONFUSION_RE = re.compile(r"^(?=.*[Ii])[IiVvXxLlCcMm]{1,5}[.)]$")


def _normalized_text(text: str) -> str:
    return "".join(char for char in text.lower() if char.isalnum())


def _is_number_marker(text: str, normalized: str) -> bool:
    """A digit followed by `.` or `)` (e.g. `1.`, `2)`) used as a list marker."""
    return bool(normalized) and normalized.isdigit() and bool(re.fullmatch(r"\d+[.)]", text))


def _is_roman_marker(text: str) -> bool:
    """A short Roman numeral followed by `.` or `)` (e.g. `I.`, `IV.`, `iii)`).

    Also accepts common Tesseract OCR confusions where lowercase `l` was read
    instead of `I` (e.g. `Il.` -> `II.`, `Ill.` -> `III.`).
    """
    if _ROMAN_NUMERAL_MARKER_RE.fullmatch(text):
        return True
    if not _ROMAN_NUMERAL_OCR_CONFUSION_RE.fullmatch(text):
        return False
    # Body must contain at least one I/i so we don't false-positive on plain
    # `L.` / `LL.` (which the strict regex above already accepts as Roman
    # numerals if all uppercase). Substitute lowercase l -> I and re-validate.
    body = text[:-1]
    promoted = body.replace("l", "I").replace("L", "I") if any(c in "Ii" for c in body) else body
    return bool(_ROMAN_NUMERAL_MARKER_RE.fullmatch(promoted + text[-1]))


def normalize_roman_marker_text(text: str) -> str:
    """If *text* is a Roman-numeral list marker (possibly with `l`->`I` OCR
    confusion), return the normalised uppercase form. Otherwise return *text*
    unchanged.
    """
    if not text:
        return text
    if _ROMAN_NUMERAL_MARKER_RE.fullmatch(text):
        return text.upper()
    if _ROMAN_NUMERAL_OCR_CONFUSION_RE.fullmatch(text):
        body = text[:-1].replace("l", "I").replace("L", "I")
        return body.upper() + text[-1]
    return text


def _is_list_marker(text: str, normalized: str) -> bool:
    return _is_number_marker(text, normalized) or _is_roman_marker(text)


def _is_numeric_fragment(text: str, normalized: str) -> bool:
    if not normalized or not normalized.isdigit():
        return False

    if not re.fullmatch(r"[$€£]?\d[\d,]*(?:[.\-–—:]\d[\d,]*)*[-–—.]?[)]?", text):
        return False

    # A bare single digit with no separator is too ambiguous to keep — it's
    # almost always OCR noise from gridlines, ticks, or stray pixels.
    if len(normalized) == 1 and re.fullmatch(r"\d", text):
        return False

    return True


def _is_allowed_short_token(normalized: str) -> bool:
    return normalized in _COMMON_SHORT_WORDS or normalized in _COMMON_SHORT_UNITS


def _contains_symbolic_noise(text: str) -> bool:
    return any(char in _SYMBOLIC_CHARS for char in text)


def _is_noise_word(word: WordBox) -> bool:
    text = word.text.strip()
    normalized = _normalized_text(text)
    if not text:
        return True

    if _is_list_marker(text, normalized):
        return False

    if not normalized:
        if text in _INLINE_PUNCTUATION_KEEP:
            return False
        return _contains_symbolic_noise(text) or word.width <= 3 or word.height <= 3

    if _contains_symbolic_noise(text):
        return normalized not in {"xai"} or len(normalized) <= 2

    if len(normalized) == 1:
        if _is_number_marker(text, normalized) or _is_numeric_fragment(text, normalized):
            return False
        if normalized in {"a", "i"}:
            return word.confidence < 80.0 or word.width <= max(4, int(word.height * 0.2))
        return True

    if len(normalized) == 2 and text.isupper():
        return normalized not in _ALLOWED_ALL_CAPS_SHORT_WORDS

    if len(normalized) == 2 and text != text.lower() and not _is_allowed_short_token(normalized) and normalized not in {"ai", "al", "ok"}:
        return True

    if len(normalized) == 2 and not _is_allowed_short_token(normalized):
        if text != text.lower():
            return True
        if word.confidence < 80.0:
            return True
        if word.width > word.height * 2.6 or word.height > word.width * 3.0:
            return True

    if len(normalized) <= 3 and not _is_allowed_short_token(normalized) and word.confidence < 65.0 and not _is_numeric_fragment(text, normalized):
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
        if text in _INLINE_PUNCTUATION_KEEP:
            return False
        return text not in {"-", "—", "–"}
    return "|" in text or "~" in text


def _is_edge_noise_word(word: WordBox) -> bool:
    text = word.text.strip()
    normalized = _normalized_text(text)

    if _is_list_marker(text, normalized):
        return False

    if not normalized:
        return _drop_inline_separator(word)

    if (
        len(normalized) <= 3
        and text != normalized
        and not text.isalpha()
        and not _is_number_marker(text, normalized)
        and not _is_numeric_fragment(text, normalized)
        and not _is_roman_marker(text)
    ):
        return True

    if (
        len(normalized) <= 2
        and not _is_allowed_short_token(normalized)
        and not _is_number_marker(text, normalized)
        and not _is_numeric_fragment(text, normalized)
    ):
        return True

    if word.width <= max(4, int(word.height * 0.2)):
        return True

    return _is_noise_word(word)


def _is_weak_edge_word(word: WordBox) -> bool:
    text = word.text.strip()
    normalized = _normalized_text(text)
    if not normalized:
        if text in _INLINE_PUNCTUATION_KEEP:
            return False
        return True

    if _is_list_marker(text, normalized):
        return False

    if word.confidence < 55.0:
        return True

    if word.confidence < 65.0 and (len(normalized) <= 5 or text != normalized):
        return True

    if len(normalized) <= 2 and word.confidence < 80.0 and not _is_allowed_short_token(normalized):
        numeric_fragment = _is_numeric_fragment(text, normalized)
        if not numeric_fragment or (len(normalized) <= 1 and word.confidence < 75.0):
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
    kept = _drop_height_outlier_edges(kept)
    if not kept:
        return None
    if len(kept) == 1 and _is_edge_noise_word(kept[0]):
        return None
    for word in kept:
        normalised = normalize_roman_marker_text(word.text.strip())
        if normalised != word.text:
            word.text = normalised
    return TextBlock(words=kept)


def _drop_height_outlier_edges(words: list[WordBox]) -> list[WordBox]:
    """Trim leading/trailing words whose height is dramatically smaller than the
    rest of the line.  This catches stray OCR prefixes/suffixes (e.g. a small
    `we` glued to the start of an otherwise large title block).

    Only triggers when:
      * the line has at least 3 words (so single-word lines never lose their
        only word),
      * the candidate is a short common-word token (`we`, `of`, ...) or a
        single character — never a long word,
      * the candidate's height is < 0.6x the median height of the remaining
        words, and
      * the candidate's confidence is < 90.
    """
    if len(words) < 3:
        return words

    def median_height(seq: list[WordBox]) -> float:
        heights = sorted(w.height for w in seq)
        return float(heights[len(heights) // 2])

    def is_height_outlier(word: WordBox, others: list[WordBox]) -> bool:
        normalized = _normalized_text(word.text.strip())
        if not normalized:
            return False
        if len(normalized) > 3 and normalized not in _COMMON_SHORT_WORDS:
            return False
        if word.confidence >= 90.0:
            return False
        median = median_height(others)
        if median <= 0:
            return False
        return word.height < median * 0.6

    def is_case_mismatch_lead(word: WordBox, others: list[WordBox]) -> bool:
        """Drop a leading/trailing short lowercase word when the rest of the
        block is otherwise all-uppercase. This catches stray prefixes like
        `we UNDERSTANDING FROG MATING HABITS:` where Tesseract glued a tiny
        decorative scrap onto the front of a heading.
        """
        text = word.text.strip()
        normalized = _normalized_text(text)
        if not normalized:
            return False
        # Only consider short common-word tokens — never a long word, never
        # something with digits.
        if len(normalized) > 3:
            return False
        if normalized not in _COMMON_SHORT_WORDS:
            return False
        if not text.isalpha() or text != text.lower():
            return False
        if word.confidence >= 90.0:
            return False
        # The remaining words must be predominantly uppercase, so a lowercase
        # leader genuinely doesn't fit.
        upper_words = [w for w in others if any(c.isupper() for c in w.text) and w.text == w.text.upper()]
        if len(upper_words) < max(2, int(len(others) * 0.6)):
            return False
        return True

    # Trim from front
    while len(words) >= 3 and (
        is_height_outlier(words[0], words[1:])
        or is_case_mismatch_lead(words[0], words[1:])
    ):
        words = words[1:]
    # Trim from back
    while len(words) >= 3 and (
        is_height_outlier(words[-1], words[:-1])
        or is_case_mismatch_lead(words[-1], words[:-1])
    ):
        words = words[:-1]
    return words


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
        text = b.text.strip()
        normalized = _normalized_text(text)
        # Skip blocks with very small pixel height (sub-character noise)
        if b.height < 8:
            continue
        # Skip tiny blocks (both dimensions small — icon/graphic artifacts)
        numeric_fragment = _is_numeric_fragment(text, normalized)
        avg_conf = sum(word.confidence for word in b.words) / len(b.words)
        if (
            b.width < 25
            and b.height < 25
            and not (numeric_fragment and (len(normalized) > 1 or avg_conf >= 75.0))
        ):
            continue
        # Skip single-character blocks that aren't alphanumeric
        if len(text) <= 1 and not text.isalnum():
            continue
        # Single-token blocks that hold only inline punctuation we want to
        # keep (e.g. a standalone `&` or `+`) survive even though _NOISE_RE
        # would otherwise drop them.
        if not _NOISE_RE.match(text):
            pass
        elif text in _INLINE_PUNCTUATION_KEEP:
            pass
        else:
            continue
        # Skip blocks where estimated font is unreasonably large (graphic artifacts)
        if b.estimated_font_size_px > 100:
            continue
        if len(b.words) == 1:
            word = b.words[0]
            if _is_list_marker(text, normalized):
                filtered.append(b)
                continue
            if avg_conf < 60.0 and (len(normalized) <= 6 or text != normalized):
                continue
            if (
                len(normalized) <= 3
                and not _is_allowed_short_token(normalized)
                and not _is_number_marker(text, normalized)
                and not _is_numeric_fragment(text, normalized)
                and b.height <= 14
            ):
                continue
            if (
                len(normalized) <= 4
                and not _is_allowed_short_token(normalized)
                and normalized not in _SHORT_STANDALONE_ALLOWLIST
                and not _is_numeric_fragment(text, normalized)
                and not any(char.isdigit() for char in normalized)
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


def _drop_orphan_trailing_fragments(blocks: list[TextBlock]) -> list[TextBlock]:
    filtered: list[TextBlock] = []

    for block in blocks:
        if _is_orphan_trailing_fragment(block, blocks):
            continue
        filtered.append(block)

    return filtered


def _is_orphan_trailing_fragment(block: TextBlock, blocks: list[TextBlock]) -> bool:
    if len(block.words) != 1:
        return False

    word = block.words[0]
    text = word.text.strip()
    normalized = _normalized_text(text)
    if len(normalized) < 3 or len(normalized) > 5:
        return False
    if text != text.lower() or not normalized.isalpha():
        return False
    if word.confidence >= 75.0:
        return False
    if block.width > max(70, int(block.height * 4.0)):
        return False

    for other in blocks:
        if other is block:
            continue
        if _range_overlap_on_smaller(block.y, block.bottom, other.y, other.bottom) < 0.55:
            continue
        if block.x <= other.right:
            continue

        horizontal_gap = block.x - other.right
        if horizontal_gap >= max(block.height, other.height) * 4.0 + 36.0:
            return True

    return False


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
    local_gaps = [gap for gap in positive_gaps if gap <= median_h * 4.0]
    reference_gap = median_gap if not local_gaps else sorted(local_gaps)[len(local_gaps) // 2]
    threshold = max(median_h * gap_factor, reference_gap * 3.0)

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


def _range_overlap_on_smaller(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    overlap = max(0, min(end_a, end_b) - max(start_a, start_b))
    smaller = min(end_a - start_a, end_b - start_b)
    if smaller <= 0:
        return 0.0

    return float(overlap) / float(smaller)


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
