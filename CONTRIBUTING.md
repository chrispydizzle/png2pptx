# Contributing to png2pptx

Thanks for taking an interest in `png2pptx`.

## Scope

The current public release is a **v0.1 beta**. Contributions that improve reliability, documentation, packaging, tests, and visually obvious OCR/PPTX issues are the best fit right now.

If you want to propose a larger change, open an issue first so the direction is clear before you invest time in a big PR.

## Development setup

1. Install **Python 3.10+**
2. Install **Tesseract OCR**
3. Install the project in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

## Run the test suite

```bash
pytest -q
```

## Useful smoke test

```bash
png2pptx convert examples/sample_input.png -o sample_output.pptx
```

## Pull request checklist

- keep changes scoped to the problem you are solving
- update docs when behavior or defaults change
- add or update tests when behavior changes
- do not commit local environment files, build outputs, or one-off debug artifacts
- call out any tradeoffs or limitations in the PR description

## Reporting OCR quality issues

When filing OCR/layout bugs, include:

- operating system
- Python version
- Tesseract version
- the command you ran
- what you expected vs what actually happened
- a safe-to-share input image or screenshot if possible
