# ALPR — Automatic License Plate Recognition

Detects license plates in video, reads them, validates them against Indian and German plate
grammars, and logs **one deduplicated row per vehicle** to an Excel workbook.

> **Status:** Phase 0 of 9 — foundation. See [ROADMAP.md](ROADMAP.md) for the full plan.

## Pipeline

```
video/camera ──▶ FrameSource ──▶ Detector ──▶ Tracker ──▶ per-track crop buffer
                 (file/cam/rtsp)   (YOLO)     (ByteTrack)          │
                                                                   ▼
   Excel log ◀── Deduplicator ◀── Validator ◀── Voter ◀────────── OCR
   (openpyxl)    (cooldown)       (IN / DE)   (multi-frame)   (PaddleOCR)
```

Results are emitted **per track, not per frame** — a vehicle visible for 40 frames produces one
row, voted across all 40 reads, not 40 rows of varying quality.

## Where it runs

GPU work runs on **Google Colab (T4)**; the repo is cloned into the runtime and installed with
`pip install -e ".[gpu]"`. Trained weights and the dataset live on the Hugging Face Hub, so
nothing large is ever committed and no Drive mount is needed.

Live webcam and RTSP modes (Phase 9) are the exception — Colab cannot reach a local camera or a
LAN stream, so those run locally on Apple Silicon via Metal/MPS.

## Local setup

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
```

Check the environment:

```bash
alpr env
```

Run the tests:

```bash
pytest
```

Tests needing a GPU, trained weights, or a camera are marked and skipped by default. Run them
explicitly with `pytest -m gpu`, `-m weights`, or `-m camera`.

## Layout

| Path | Contents |
|---|---|
| `src/alpr/` | The package — all logic worth testing lives here |
| `notebooks/` | Thin Colab drivers; no business logic |
| `tests/` | pytest suite, runs in CI without a GPU |
| `configs/` | Training and pipeline configs, committed for reproducibility |
| `data/` | Local scratch only — real data lives on the Hub, and is gitignored |

## License

See [LICENSE](LICENSE).
