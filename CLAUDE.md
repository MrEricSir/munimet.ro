# Claude Code Instructions

## Key Documentation

- **[README.md](README.md)** — Project overview, quick start, development workflow
- **[docs/detection-fix-workflow.md](docs/detection-fix-workflow.md)** — How to investigate detection failures, add test images, fix OCR/symbol bugs, and iterate efficiently
- **[SETUP.md](SETUP.md)** — Installation instructions for macOS, Linux, Windows
- **[CONFIGURATION.md](CONFIGURATION.md)** — All configurable constants (also see `lib/config.py`)
- **[deploy/README.md](deploy/README.md)** — Local and cloud deployment
- **[api/README.md](api/README.md)** — API endpoints

## Environment

- Python venv at `.venv/` — always run with `.venv/bin/python` or `.venv/bin/pytest`
- Tests: `pytest tests/test_*.py` — the train detection tests were baselined in Docker; ~14 failures are expected on macOS due to Tesseract version differences

## Key Commands

```bash
# Run all tests
.venv/bin/pytest tests/

# Run a single image's tests (much faster — use this first)
.venv/bin/pytest tests/test_train_detection.py -k "snapshot_name_stem"

# Fast OCR unit tests only (no images, no Tesseract required)
.venv/bin/pytest tests/test_train_detection.py::TestOCRExtraction -v

# Inspect raw OCR at a specific x position in an image
.venv/bin/python scripts/inspect-ocr.py tests/images/foo.jpg 804

# Print current detections for all baseline images (use to draft new baselines)
.venv/bin/python tests/test_train_detection.py

# Start diagnostic viewer (interactive web UI for detection analysis)
.venv/bin/python scripts/diagnostic_viewer.py
.venv/bin/python scripts/diagnostic_viewer.py path/to/image.jpg
```

## Key Files

| File | Purpose |
|---|---|
| `lib/train_detector.py` | Train detection: `detect_trains()`, `_detect_by_ocr()`, `_detect_symbols()`, `_extract_train_ids()` |
| `lib/detection.py` | System status: `detect_system_status()`, `calculate_system_status()` |
| `lib/station_detector.py` | Station position auto-detection |
| `lib/config.py` | All thresholds and constants |
| `tests/baseline_trains.json` | Expected train detections per image |
| `tests/baseline_delay_summaries.json` | Expected delay summaries per image |
| `tests/test_system_status.py` | System status tests; `KNOWN_STATUSES` list |
| `scripts/diagnostic_viewer.py` | Interactive web UI for detection diagnostics |
| `scripts/inspect-ocr.py` | Quick OCR inspection for a specific image + x position |

## Detection Fix Workflow (Summary)

When adding a new test image or fixing a detection bug:

1. **Isolate first** — run just the new image's test, not the full suite
2. **Inspect OCR** — use `inspect-ocr.py` to see raw Tesseract output before reasoning about fixes
3. **Add a unit test** — add a case to `TestOCRExtraction` before modifying `_extract_train_ids`
4. **Draft the baseline** — run `python tests/test_train_detection.py` to get current detections; use `ocr_override` when OCR systematically misreads an ID
5. **Verify no regressions** — run the full suite once at the end; expect ~14 pre-existing failures on macOS

Full details: [docs/detection-fix-workflow.md](docs/detection-fix-workflow.md)
