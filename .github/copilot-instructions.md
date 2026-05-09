# Copilot Instructions

## Commands

```bash
# Install (editable + dev deps)
pip install -e ".[dev]"

# Run all tests
pytest -q

# Run a single test
pytest tests/test_layout.py::test_empty_input -q

# Smoke test
png2pptx convert examples/sample_input.png -o sample_output.pptx
```

Tesseract OCR must be installed separately (not a pip package). On Windows, add it to `PATH` or set `$env:path="C:\Program Files\Tesseract-OCR;$env:path"`.

## Architecture

The pipeline is a linear sequence of transformations, one pass per input PNG:

```
PNG file
  → ocr.py         extract_words()       → list[WordBox]
  → layout.py      group_into_blocks()   → list[TextBlock]
  → styles.py      extract_text_colors() → list[TextBlock] (colors set)
  → inpaint.py     remove_text()         → cleaned PNG (temp file)
  → pptx_builder.py build_pptx()         → .pptx file
```

`cli.py` orchestrates all steps. `models.py` defines the shared data structures that flow between every stage.

### Data model (`models.py`)

- **`WordBox`** — one word from Tesseract with pixel bounding box (`x`, `y`, `width`, `height`), `confidence`, and Tesseract's `block_num`/`par_num`/`line_num` grouping keys.
- **`TextBlock`** — a logical text line made of one or more `WordBox`es. Computes `text`, bounds, and `estimated_font_size_px` as computed properties. Holds the sampled RGB `color`.
- **`SlideData`** — everything needed for one PPTX slide: image path, pixel dimensions, and the list of `TextBlock`s.

### Key module details

**`layout.py`** — The most complex module. Groups words by Tesseract's `(block_num, par_num, line_num)` triplet, then applies extensive noise filtering:
- Words are pre-filtered by `_is_noise_word()` before grouping.
- Lines with large horizontal gaps are split into separate blocks via `_split_wide_gaps()` (handles multi-column infographics).
- After grouping, `_clean_block_words()` trims low-confidence or noisy edge words.
- `_filter_noise()` removes entire blocks that are OCR artifacts.
- `merge_lines=False` by default — each Tesseract line becomes its own `TextBlock`. This is intentional for infographics where text is scattered; don't change this default.

**`pptx_builder.py`** — Slide dimensions are computed from the image aspect ratio, with the long edge fixed at `SLIDE_LONG_EDGE_EMU` (≈13.33 inches). All text boxes use zero margins, `word_wrap=False`, and transparent fill. Font size is determined by binary-search fitting (`_fit_font_size_px`) using `Pillow`'s `ImageFont` with `lru_cache` for performance. The hardcoded font family is `Calibri` (`_FONT_FAMILY`).

**`inpaint.py`** — Tries OpenCV (`cv2.inpaint` with Telea algorithm) first; falls back to a PIL median-color fill. The mask is built per-word using local color/contrast analysis (`_build_word_mask`) rather than simple rectangles, to avoid bleeding colors across adjacent panels. `cv2` is imported inside functions to allow graceful fallback.

**`ocr.py`** — Two OCR modes: `fast` (single Tesseract pass) and `aggressive` (multiple passes with different configs, results merged). Coordinates are always in source-image pixels.

## Quality improvement loop

The `examples/` folder contains three named test images (`sample_input`, `Infographic`, `CommandCenter`), each with a set of versioned artifacts:

| File | Purpose |
|---|---|
| `<name>.png` | Source input |
| `<name>_baseline_clean.png` | Gold-standard inpainted background (do not overwrite) |
| `<name>_current_clean.png` | Output from the current code — regenerate to check regressions |
| `<name>_compare.png` | Side-by-side visual diff of baseline vs current |

### Regenerate `_current_clean.png` for all examples

Preferred repo command:

```bash
png2pptx quality-loop --examples-dir examples --output-dir quality_output
```

This writes fresh `_current_clean` images, per-example PPTX outputs, overlay review images, and `quality_output/summary.json`.

If you need the lower-level module path explicitly:

```python
# Run from the repo root
from pathlib import Path
from png2pptx.ocr import extract_words
from png2pptx.layout import group_into_blocks
from png2pptx.styles import extract_text_colors
from png2pptx.inpaint import remove_text

for name in ["sample_input", "Infographic", "CommandCenter"]:
    path = Path(f"examples/{name}.png")
    words, w, h = extract_words(path, confidence_threshold=40.0, ocr_mode="aggressive")
    blocks = group_into_blocks(words)
    blocks = extract_text_colors(path, blocks)
    remove_text(path, blocks, output_path=Path(f"examples/{name}_current_clean.png"))
    print(f"Done: {name}")
```

Then open `_current_clean.png` and `_baseline_clean.png` side by side to spot regressions in inpainting quality.

To also check PPTX output (text placement, font sizes, colors):

```bash
png2pptx convert examples/sample_input.png -o examples/sample_input.pptx
png2pptx convert examples/Infographic.png -o examples/Infographic.pptx
png2pptx convert examples/CommandCenter.png -o examples/CommandCenter.pptx
```

### Diagnostic tools (in `.artifacts/`)

**`diag.py`** — Prints every detected text block with its pixel position, estimated font size, sampled color, and text content. Run from the `.artifacts/` directory:

```bash
cd .artifacts
python diag.py
```

**`debug_overlay.py`** — Renders text-box bounding boxes from a PPTX back onto the original image as a red-outline overlay, saved to `debug_overlay.png`. Useful for diagnosing misaligned or missing text boxes:

```bash
cd .artifacts
python debug_overlay.py ../examples/Infographic.pptx
```

## Conventions

- All modules use `from __future__ import annotations`.
- All domain objects are `@dataclass`es in `models.py`; no domain logic lives outside of `models.py`, `layout.py`, `styles.py`, `inpaint.py`, or `pptx_builder.py`.
- Progress/diagnostic output goes to **stderr** (`click.echo(..., err=True)`), not stdout.
- Internal helper functions in `layout.py` are module-private (prefixed `_`). The noise-filtering heuristics are deliberate and fragile — check the existing tests before modifying thresholds.
- Tests use a `_make_word()` factory helper to construct `WordBox` instances with sensible defaults. Follow this pattern for new tests in `test_layout.py`.
- Do not commit generated files: `sample_output.pptx`, `build/`, `dist/`, `*.egg-info/`, or inpainted temp PNGs.
