# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Batman-Vision: a real-time webcam object detection/tracking prototype built on Ultralytics YOLOE. It tracks objects across frames, buffers the best crops per track, and hands off "finalized" tracks to a background worker that deduplicates against previously seen objects (via YOLOE embeddings) and tags new ones by calling a VLM (through NVIDIA's OpenAI-compatible API) for object name/tags/OCR text. Results are persisted to SQLite with FTS5 full-text search.

There is no package manifest (no requirements.txt/pyproject.toml) — dependencies are whatever's installed in `.venv`. If a script needs a new dependency, install it into `.venv` and mention the addition, since nothing else declares it.

## Environment & running things

- Python 3.11 venv at `.venv/`. Activate with `source .venv/bin/activate` or invoke `.venv/bin/python3` directly.
- `.env` holds `NVIDIA_API_KEY`, loaded via `python-dotenv`. Required for anything that calls the tagging worker's VLM API.
- Model weights (`models/*.pt`) and object embeddings (`embeddings/*.pt`) are gitignored and downloaded/generated locally, not committed.
- `python download_models.py` fetches YOLOE checkpoints (`yoloe-11s-seg-pf.pt`, `yoloe-26s-seg-pf.pt`, `yoloe-26l-seg-pf.pt`) from the ultralytics GitHub release into `models/`. It skips files that already exist.
- `objects.db` (SQLite) and `captures/` (saved crop JPEGs) are the runtime data stores, both gitignored, both recreated by `db.init_db()`.

## Running tests / scripts

There's no test runner config — tests are plain `unittest`/manual scripts run directly:

- `python tests/test_db.py` — unit tests for `db.py` (init, insert, update, FTS5 search, delete-trigger propagation). Isolates itself via `BATMAN_DB_PATH`/`BATMAN_CAPTURES_DIR` env vars pointed at `tests/test_*` paths, and cleans up after itself.
- `python tests/test_tagging_worker.py` — end-to-end manual test of the tagging worker (`yoloe_test.tagging_worker_func`) against synthetic crops, including a deliberate API-failure case (bad base_url) to verify `pending -> failed` transitions. Requires a real YOLOE model file in `models/` and a real `NVIDIA_API_KEY` (makes live network calls).
- `python tests/smoke_test.py` — checks `torch.backends.mps.is_available()` and opens the webcam to confirm capture works. Interactive; quit with `q`.
- `python tests/yoloe_test.py` — the main live application (see below). Interactive; quit with `q`.

There is no mocking of the VLM API in tests — `test_tagging_worker.py` hits the real NVIDIA endpoint, so it costs real API calls and requires network access.

## Architecture

**`tests/yoloe_test.py`** is the core of the system (despite living in `tests/`) and contains most of the logic:

1. **Main loop (`main()`)** — opens the webcam, runs `YOLOE.track()` per frame with a custom tracker config (`tests/custom_bytetrack.yaml`, tuned with a longer track buffer and dual confidence thresholds), and draws the HUD overlay (boxes, labels, FPS).
2. **Per-track crop buffering (`TrackBuffer`)** — for each active track ID, keeps up to 6 crops, deduplicating spatially-similar crops (via IoU) by keeping the higher-scoring one and evicting the worst crop when full. A crop is only considered if it clears three thresholds: `MIN_CONFIDENCE`, `MIN_BBOX_SIZE`, `MIN_SHARPNESS` (Laplacian variance blur check). Crop quality score = weighted sum of confidence, bbox size, and sharpness (`compute_combined_score`).
3. **Track lifecycle finalization** — a track with no detections for >1.5s is "finalized." It's accepted (pushed to `finalized_queue`) only if it was tracked ≥1.0s and has ≥2 buffered crops; otherwise discarded as noise. Any tracks still alive when the loop exits are force-finalized the same way.
4. **Background tagging worker (`tagging_worker_func`)**, running in its own thread, consumes `finalized_queue`:
   - Computes a YOLOE embedding (CPU) for the best crop.
   - De-duplication: cosine-similarity-compares against embeddings of all `status='tagged'` DB rows; if similarity ≥ `DEDUP_SIMILARITY_THRESHOLD` (0.9), treats it as a re-sighting (`db.update_object_re_sighting`) and skips the VLM call entirely.
   - Otherwise inserts a `pending` row, then calls the VLM (`execute_tagging_with_retries`) with all buffered crops as base64 image blocks, expecting strict JSON (`object_name`, `tags`, `ocr_text`, `confidence`).
   - VLM call tries `qwen/qwen3.5-397b-a17b` first; on failure/timeout/low-confidence, falls back to `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` with `tenacity` retries (3 attempts, exponential backoff) for connection/timeout/rate-limit errors, plus one manual retry on a "low confidence" result.
   - On success: `db.update_object_result(..., status='tagged', embedding=...)`. On any exception: `status='failed'`.
   - Status transitions are always `None -> pending -> {tagged, failed}`, logged to console at each step.

**`db.py`** is the persistence layer, independent of the tracking/tagging logic:
   - Single `objects` table (`track_id` PK, timestamps, `crop_paths` as JSON, `tags`/`ocr_text` nullable, `status` CHECK'd to `pending|tagged|failed`, `confidence`, `embedding` BLOB of float32 bytes).
   - An FTS5 virtual table `objects_fts` kept in sync via SQL triggers (insert/update-of-tags-or-ocr/delete) — never written to directly from Python.
   - `DB_PATH`/`CAPTURES_DIR` are overridable via `BATMAN_DB_PATH`/`BATMAN_CAPTURES_DIR` env vars, which is how tests isolate themselves from the real `objects.db`/`captures/`.
   - Crop paths are stored relative to the project root (via `os.path.relpath`) for portability.

**`download_models.py`** is a standalone stdlib-only script (no dependency on the rest of the codebase) for fetching model weights.

## Working in this codebase

- Changes to tracking/buffering/finalization logic live in `tests/yoloe_test.py`; changes to persistence/search live in `db.py`. The two are only coupled through the `db` module's public functions.
- When touching thresholds (`MIN_SHARPNESS`, `MIN_BBOX_SIZE`, `MIN_CONFIDENCE`, `DEDUP_SIMILARITY_THRESHOLD`, tracker YAML) understand these were tuned empirically against live webcam behavior — verify changes against the actual webcam loop (`tests/yoloe_test.py` main or `tests/smoke_test.py`), not just unit tests, since `test_db.py` doesn't exercise the CV pipeline at all.
- `device='mps'` is hardcoded for the live tracking loop's happy path (falls back to `'cpu'` only if MPS is unavailable) — this project targets Apple Silicon.
