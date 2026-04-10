"""Tests for the layout module."""

from png2pptx.layout import group_into_blocks
from png2pptx.models import WordBox


def _make_word(text, x, y, w=50, h=20, block=1, par=1, line=1, confidence=95.0):
    return WordBox(
        text=text, x=x, y=y, width=w, height=h,
        confidence=confidence, line_num=line, block_num=block, par_num=par,
    )


def test_empty_input():
    assert group_into_blocks([]) == []


def test_single_word():
    words = [_make_word("hello", 10, 10)]
    blocks = group_into_blocks(words)
    assert len(blocks) == 1
    assert blocks[0].text == "hello"


def test_same_line_grouped():
    """Words on the same Tesseract line should end up in one block."""
    words = [
        _make_word("hello", 10, 10, line=1, block=1, par=1),
        _make_word("world", 70, 10, line=1, block=1, par=1),
    ]
    blocks = group_into_blocks(words)
    assert len(blocks) == 1
    assert "hello" in blocks[0].text
    assert "world" in blocks[0].text


def test_same_row_cross_panel_gap_splits_into_two_blocks():
    words = [
        _make_word("After", 20, 10, w=70, h=40),
        _make_word("deployment,", 110, 10, w=140, h=40),
        _make_word("accuracy", 270, 10, w=110, h=40),
        _make_word("alone", 400, 10, w=80, h=40),
        _make_word("Release", 620, 10, w=120, h=40),
        _make_word("the", 760, 10, w=45, h=40),
        _make_word("model", 825, 10, w=95, h=40),
    ]

    blocks = group_into_blocks(words)

    assert len(blocks) == 2
    assert blocks[0].text == "After deployment, accuracy alone"
    assert blocks[1].text == "Release the model"


def test_different_lines_separate():
    """Words on different lines with large gap should stay separate."""
    words = [
        _make_word("top", 10, 10, line=1, block=1, par=1),
        _make_word("bottom", 10, 200, line=2, block=2, par=1),
    ]
    blocks = group_into_blocks(words)
    assert len(blocks) == 2


def test_nearby_lines_not_merged_by_default():
    """By default (merge_lines=False), nearby lines stay separate."""
    words = [
        _make_word("line1", 10, 10, h=20, line=1, block=1, par=1),
        _make_word("line2", 10, 32, h=20, line=2, block=1, par=1),
    ]
    blocks = group_into_blocks(words)
    assert len(blocks) == 2


def test_nearby_lines_merged_when_enabled():
    """With merge_lines=True, close lines with horizontal overlap merge."""
    words = [
        _make_word("line1", 10, 10, h=20, line=1, block=1, par=1),
        _make_word("line2", 10, 32, h=20, line=2, block=1, par=1),
    ]
    blocks = group_into_blocks(words, merge_lines=True)
    # Gap is 2px (32 - 30), line height is 20px, so gap < 1.5 * 20 → merged
    assert len(blocks) == 1


def test_text_property_sorts_left_to_right():
    """TextBlock.text should join words left-to-right."""
    words = [
        _make_word("world", 70, 10, line=1, block=1, par=1),
        _make_word("hello", 10, 10, line=1, block=1, par=1),
    ]
    blocks = group_into_blocks(words)
    assert blocks[0].text == "hello world"


def test_group_into_blocks_drops_bracketed_icon_word():
    words = [
        _make_word("[OK]", 10, 10, w=40, h=20),
        _make_word("Moving", 70, 10, w=70, h=20),
        _make_word("to", 150, 10, w=25, h=20),
        _make_word("production", 185, 10, w=95, h=20),
    ]

    blocks = group_into_blocks(words)

    assert len(blocks) == 1
    assert blocks[0].text == "Moving to production"


def test_group_into_blocks_removes_separator_word_inside_line():
    words = [
        _make_word("Accuracy", 10, 10, w=90, h=22),
        _make_word("|", 120, 10, w=4, h=40),
    ]

    blocks = group_into_blocks(words)

    assert len(blocks) == 1
    assert blocks[0].text == "Accuracy"


def test_group_into_blocks_drops_short_unknown_noise_block():
    words = [
        _make_word("Fe", 10, 10, w=60, h=22, confidence=59.0),
    ]

    blocks = group_into_blocks(words)

    assert blocks == []


def test_group_into_blocks_drops_mixed_case_short_noise_block():
    words = [
        _make_word("iN", 10, 10, w=34, h=39, confidence=45.0),
        _make_word("i", 54, 10, w=12, h=39, confidence=57.0),
    ]

    blocks = group_into_blocks(words)

    assert blocks == []


def test_group_into_blocks_trims_low_confidence_edge_words():
    words = [
        _make_word("ston", 10, 10, w=42, h=20, confidence=49.0),
        _make_word("ROBUST,", 62, 10, w=90, h=20, confidence=95.0),
        _make_word("FAIR", 162, 10, w=55, h=20, confidence=97.0),
    ]

    blocks = group_into_blocks(words)

    assert len(blocks) == 1
    assert blocks[0].text == "ROBUST, FAIR"


def test_group_into_blocks_drops_low_confidence_punctuated_standalone():
    words = [
        _make_word("liven.”", 10, 10, w=80, h=30, confidence=45.0),
    ]

    blocks = group_into_blocks(words)

    assert blocks == []


def test_group_into_blocks_keeps_numbered_step_marker():
    words = [
        _make_word("1.", 10, 10, w=22, h=24),
        _make_word("Deploy", 42, 10, w=70, h=24),
        _make_word("the", 122, 10, w=30, h=24),
        _make_word("Model", 162, 10, w=65, h=24),
    ]

    blocks = group_into_blocks(words)

    assert len(blocks) == 1
    assert blocks[0].text == "1. Deploy the Model"


def test_group_into_blocks_drops_punctuated_icon_fragment_at_block_edge():
    words = [
        _make_word("QOO)", 10, 10, w=110, h=44, confidence=74.0),
        _make_word("Who", 180, 10, w=90, h=44),
        _make_word("is", 285, 10, w=30, h=32),
        _make_word("affected.", 330, 10, w=180, h=44),
    ]

    blocks = group_into_blocks(words)

    assert len(blocks) == 1
    assert blocks[0].text == "Who is affected."


def test_group_into_blocks_drops_overlapping_short_noise_word():
    words = [
        _make_word("why", 10, 10, w=50, h=32, confidence=90.0),
        _make_word("it", 70, 10, w=20, h=28, confidence=95.0),
        _make_word("changed,", 100, 10, w=110, h=32, confidence=96.0),
        _make_word("wno", 220, 10, w=40, h=28, confidence=87.0),
        _make_word("who", 236, 10, w=60, h=31, confidence=96.0),
        _make_word("is", 308, 10, w=22, h=22, confidence=96.0),
        _make_word("affected", 340, 10, w=120, h=31, confidence=96.0),
    ]

    blocks = group_into_blocks(words)

    assert len(blocks) == 1
    assert blocks[0].text == "why it changed, who is affected"


def test_group_into_blocks_drops_tiny_short_noise_block():
    words = [
        _make_word("BOS", 10, 10, w=87, h=12, confidence=77.0),
    ]

    blocks = group_into_blocks(words)

    assert blocks == []


def test_group_into_blocks_drops_short_titlecase_noise_block():
    words = [
        _make_word("Rad", 10, 10, w=36, h=20, confidence=96.0),
    ]

    blocks = group_into_blocks(words)

    assert blocks == []


def test_group_into_blocks_keeps_allowed_short_standalone_label():
    words = [
        _make_word("Low", 10, 10, w=42, h=20, confidence=96.0),
    ]

    blocks = group_into_blocks(words)

    assert len(blocks) == 1
    assert blocks[0].text == "Low"


def test_group_into_blocks_keeps_titlecase_common_short_word():
    words = [
        _make_word("It", 10, 10, w=22, h=28, confidence=96.0),
        _make_word("helps", 44, 10, w=60, h=28, confidence=96.0),
    ]

    blocks = group_into_blocks(words)

    assert len(blocks) == 1
    assert blocks[0].text == "It helps"
