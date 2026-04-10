from png2pptx.layout import group_into_blocks
from png2pptx.models import WordBox
from png2pptx.ocr import _LocalOcrRegion, _normalize_ai_confusions, _replace_region_lines


def _make_word(text, x, y, w=60, h=28, confidence=96.0):
    return WordBox(
        text=text,
        x=x,
        y=y,
        width=w,
        height=h,
        confidence=confidence,
        line_num=1,
        block_num=1,
        par_num=1,
    )


def _make_region(*lines):
    region = _LocalOcrRegion.from_line(list(lines[0]))
    for line in lines[1:]:
        region.add_line(list(line))
    return region


def _block_texts(words):
    return [block.text for block in group_into_blocks(words)]


def test_normalize_ai_confusions_uses_context():
    words = [
        _make_word("Explainable", 10, 10, w=120),
        _make_word("Al", 140, 10, w=26),
        _make_word("Model", 176, 10, w=70),
        _make_word("VIS4Al", 10, 52, w=80),
        _make_word("supports", 100, 52, w=90),
        _make_word("Al", 10, 94, w=24),
        _make_word("arrived", 44, 94, w=70),
    ]

    normalized = _normalize_ai_confusions(words)
    texts = [word.text for word in sorted(normalized, key=lambda word: (word.y, word.x))]

    assert texts[:3] == ["Explainable", "AI", "Model"]
    assert texts[3] == "VIS4AI"
    assert texts[5] == "Al"


def test_replace_region_lines_prefers_stronger_local_candidate():
    existing = [
        _make_word("Identify", 10, 10, w=90),
        _make_word("concept", 110, 10, w=85),
        _make_word("drift,", 205, 10, w=65),
        _make_word("adversarial", 280, 10, w=130),
        _make_word("changing", 10, 50, w=95),
        _make_word("and", 116, 50, w=38),
        _make_word("issues", 292, 50, w=72),
        _make_word("in", 374, 50, w=24),
        _make_word("conditions", 150, 90, w=128),
    ]
    local = [
        _make_word("behavior,", 10, 50, w=102),
        _make_word("and", 122, 50, w=38),
        _make_word("fairness", 170, 50, w=92),
        _make_word("issues", 272, 50, w=72),
        _make_word("in", 354, 50, w=24),
        _make_word("changing", 10, 90, w=95),
        _make_word("conditions.", 116, 90, w=138),
    ]

    region = _make_region(existing[:4], existing[4:8], existing[8:])
    refined = _replace_region_lines(existing, local, region)
    texts = _block_texts(refined)

    assert "behavior, and fairness issues in" in texts
    assert "changing conditions." in texts
    assert "changing and issues in" not in texts


def test_replace_region_lines_keeps_stronger_existing_line():
    existing = [
        _make_word("Model", 10, 10, w=62),
        _make_word("deployment", 82, 10, w=120),
        _make_word("is", 212, 10, w=24),
        _make_word("not", 246, 10, w=34),
        _make_word("a", 290, 10, w=16),
        _make_word("final", 316, 10, w=48),
        _make_word("handoff.", 374, 10, w=82),
        _make_word("is", 468, 10, w=24),
        _make_word("an", 502, 10, w=30),
        _make_word("ongoing", 542, 10, w=82),
        _make_word("cycle", 634, 10, w=58),
        _make_word("of", 702, 10, w=22),
        _make_word("explanation,", 734, 10, w=118),
        _make_word("monitoring,", 10, 52, w=104),
        _make_word("review,", 124, 52, w=76),
        _make_word("and", 210, 52, w=38),
        _make_word("improvement", 258, 52, w=118),
        _make_word("in", 386, 52, w=24),
        _make_word("a", 420, 52, w=16),
        _make_word("live", 446, 52, w=40),
        _make_word("production", 496, 52, w=106),
        _make_word("environment.", 612, 52, w=124),
    ]
    local = [
        _make_word("Model", 10, 10, w=62),
        _make_word("deployment", 82, 10, w=120),
        _make_word("is", 212, 10, w=24),
        _make_word("not", 246, 10, w=34),
        _make_word("a", 290, 10, w=16),
        _make_word("final", 316, 10, w=48),
        _make_word("handoff.", 374, 10, w=82),
        _make_word("It", 468, 10, w=22),
        _make_word("is", 500, 10, w=24),
        _make_word("an", 534, 10, w=30),
        _make_word("ongoing", 574, 10, w=82),
        _make_word("cycle", 666, 10, w=58),
        _make_word("of", 734, 10, w=22),
        _make_word("explanation,", 766, 10, w=118),
        _make_word("moanitnrinn", 10, 52, w=110, confidence=61.0),
        _make_word("raviaw", 130, 52, w=70, confidence=58.0),
        _make_word("and", 210, 52, w=38, confidence=88.0),
        _make_word("imnrnvamant", 258, 52, w=122, confidence=57.0),
        _make_word("in", 390, 52, w=24, confidence=95.0),
        _make_word("4", 424, 52, w=16, h=18, confidence=82.0),
        _make_word("liva", 450, 52, w=44, confidence=84.0),
    ]

    region = _make_region(existing[:13], existing[13:])
    refined = _replace_region_lines(existing, local, region)
    texts = _block_texts(refined)

    assert "Model deployment is not a final handoff. It is an ongoing cycle of explanation," in texts
    assert "monitoring, review, and improvement in a live production environment." in texts
    assert not any("raviaw" in text for text in texts)
