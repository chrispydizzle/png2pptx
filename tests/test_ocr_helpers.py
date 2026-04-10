"""Tests for OCR helper logic that does not require Tesseract."""

from png2pptx.models import WordBox
from png2pptx.ocr import _deduplicate_words, _merge_aggressive_passes, _reindex_words_by_geometry


def test_deduplicate_words_keeps_highest_confidence_overlap():
    words = [
        WordBox(text="Hello", x=10, y=10, width=40, height=15, confidence=55.0),
        WordBox(text="Hello", x=11, y=10, width=40, height=15, confidence=88.0),
        WordBox(text="World", x=70, y=10, width=45, height=15, confidence=90.0),
    ]

    deduped = _deduplicate_words(words)

    assert len(deduped) == 2
    assert any(word.text == "Hello" and word.confidence == 88.0 for word in deduped)
    assert any(word.text == "World" for word in deduped)


def test_reindex_words_by_geometry_groups_same_line():
    words = [
        WordBox(text="Hello", x=10, y=10, width=40, height=15, confidence=80.0, block_num=1, par_num=1, line_num=1),
        WordBox(text="World", x=60, y=12, width=45, height=15, confidence=82.0, block_num=7, par_num=3, line_num=2),
        WordBox(text="Below", x=10, y=50, width=38, height=15, confidence=84.0, block_num=9, par_num=1, line_num=4),
    ]

    reindexed = _reindex_words_by_geometry(words)

    assert reindexed[0].block_num == reindexed[1].block_num
    assert reindexed[2].block_num != reindexed[0].block_num


def test_merge_aggressive_passes_prefers_consensus_and_drops_noise():
    pass_results = [
        [
            WordBox(text="QOO)", x=10, y=10, width=100, height=40, confidence=74.0),
            WordBox(text="faimes", x=200, y=10, width=110, height=32, confidence=76.0),
        ],
        [
            WordBox(text="fairness", x=202, y=10, width=120, height=32, confidence=95.0),
        ],
        [
            WordBox(text="fairness", x=204, y=12, width=118, height=30, confidence=93.0),
        ],
    ]

    merged = _merge_aggressive_passes(pass_results)

    assert [word.text for word in merged] == ["fairness"]
