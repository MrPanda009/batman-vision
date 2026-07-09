# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Batman-Vision: a real-time webcam object detection/tracking prototype built on Ultralytics YOLOE. It tracks objects across frames, buffers the best crops per track, and hands off "finalized" tracks to a background worker that deduplicates against previously seen objects (via YOLOE embeddings) and tags new ones by calling a VLM (through NVIDIA's OpenAI-compatible API) for object name/tags/OCR text. Results are persisted to SQLite with FTS5 full-text search.

The tracking/tagging pipeline exists in two forms that share the same logic but aren't wired together:
- **`main.py`** — a FastAPI server that runs the pipeline as an on-demand background thread, controlled via HTTP endpoints, and serves an MJPEG HUD stream + JSON object/status APIs to the `../frontend` Next.js dashboard. This is the one the frontend talks to.
- **`tests/yoloe_test.py`** — the original standalone script: same tracking/buffering/tagging logic, but runs directly against the webcam with an OpenCV display window (no HTTP, no server). Useful for local iteration without spinning up the frontend.

Changes to core detection/tracking/tagging logic (thresholds, VLM fallback chain, dedup) generally need to be made in **both** files, since `main.py` doesn't import from `tests/yoloe_test.py` — it duplicates the same functions (`TrackBuffer`, `compute_combined_score`, `execute_tagging_with_retries`, `tagging_worker_func`, etc.) adapted for threaded/server use.

There is no package manifest (no requirements.txt/pyproject.toml) — dependencies are whatever's installed in `.venv`. If a script needs a new dependency, install it into `.venv` and mention the addition, since nothing else declares it.

## Environment & running things

- Python 3.11 venv at `.venv/`. Activate with `source .venv/bin/activate` or invoke `.venv/bin/python3` directly.
- `.env` holds `NVIDIA_API_KEY`, loaded via `python-dotenv`. Required for anything that calls the tagging worker's VLM API.
- Model weights (`models/*.pt`) and object embeddings (`embeddings/*.pt`) are gitignored and downloaded/generated locally, not committed.
- `python download_models.py` fetches YOLOE checkpoints (`yoloe-11s-seg-pf.pt`, `yoloe-26s-seg-pf.pt`, `yoloe-26l-seg-pf.pt`) from the ultralytics GitHub release into `models/`. It skips files that already exist.
- `objects.db` (SQLite) and `captures/` (saved crop JPEGs) are the runtime data stores, both gitignored, both recreated by `db.init_db()`.
- `main.py` hardcodes `models/yoloe-26l-seg-pf.pt` as the model for both the capture loop and the embedding worker — that specific checkpoint must exist locally (via `download_models.py`) for the server to start the pipeline successfully.
- `uvicorn main:app --reload` (or `--host 0.0.0.0 --port 8000`) runs the API server. CORS is hardcoded to allow `http://localhost:3000` / `http://127.0.0.1:3000` (the Next.js dev server) — update `main.py`'s `CORSMiddleware` config if the frontend origin changes.

## Running tests / scripts

There's no test runner config — tests are plain `unittest`/manual scripts run directly:

- `python tests/test_db.py` — unit tests for `db.py` (init, insert, update, FTS5 search, delete-trigger propagation). Isolates itself via `BATMAN_DB_PATH`/`BATMAN_CAPTURES_DIR` env vars pointed at `tests/test_*` paths, and cleans up after itself.
- `python tests/test_tagging_worker.py` — end-to-end manual test of the tagging worker (`yoloe_test.tagging_worker_func`) against synthetic crops, including a deliberate API-failure case (bad base_url) to verify `pending -> failed` transitions. Requires a real YOLOE model file in `models/` and a real `NVIDIA_API_KEY` (makes live network calls).
- `python tests/smoke_test.py` — checks `torch.backends.mps.is_available()` and opens the webcam to confirm capture works. Interactive; quit with `q`.
- `python tests/yoloe_test.py` — the standalone live application with an OpenCV display window (see below). Interactive; quit with `q`.
- `python tests/seed_recall_db.py` — populates an isolated `recall_test.db`/`recall_captures/` (via `BATMAN_DB_PATH`/`BATMAN_CAPTURES_DIR`) with a few fake `tagged` objects, for exercising `recall.py` without running the live pipeline.

There is no mocking of the VLM API in tests — `test_tagging_worker.py` hits the real NVIDIA endpoint, so it costs real API calls and requires network access.

`uvicorn main:app` is the server entry point (not a test), started separately — see "Environment & running things" above.

`python recall.py "<query>"` (or with no args, prompts interactively) is a standalone CLI: sends the query to an NVIDIA model to extract search keywords, runs them through `db.search_objects()` (FTS5), and prints ranked results. Requires `NVIDIA_API_KEY` and a populated `objects.db`.

## Architecture

**`main.py`** is a FastAPI app that wraps the same pipeline logic behind an HTTP API for the `../frontend` dashboard:
- `POST /pipeline/start` / `POST /pipeline/stop` — start/stop the capture thread (`capture_loop_func`, a threaded variant of the `tests/yoloe_test.py` main loop) on demand; `GET /pipeline/status` reports whether it's alive.
- `GET /video_feed` — MJPEG stream of the latest HUD-annotated frame (`latest_frame_jpeg`, guarded by `frame_lock`); yields a static "Pipeline Stopped" placeholder frame when idle.
- `GET /api/objects` / `GET /api/stats` — JSON reads of the `objects` table (full rows / status counts).
- `POST /api/clear` — wipes the `objects` table and deletes all files in `captures/`; rejected with 400 while the pipeline is active.
- `/captures` is mounted as a static file directory so the frontend can load crop JPEGs directly by relative path.
- The tagging worker thread (`tagging_worker_func`) is started once at import time (daemon thread, not tied to pipeline start/stop) and runs for the lifetime of the server; the capture thread is started/stopped per-request via the `/pipeline/*` endpoints.
- On `shutdown` (FastAPI lifecycle event), signals `stop_event`, joins the capture thread, then sends `None` into `finalized_queue` to stop the worker thread.

**`recall.py`** is a standalone CLI for querying already-tagged objects: takes a natural-language query, calls an NVIDIA model (`nemotron-3-nano-30b-a3b`, thinking disabled) to extract 2-5 search keywords, runs them as an FTS5 `OR` query via `db.search_objects()`, then re-ranks results in Python by how many extracted keywords actually appear in each object's tags/OCR text.

**`tests/yoloe_test.py`** contains the original, standalone version of the same tracking/tagging logic (see `main.py` above for the server variant):

1. **Main loop (`main()`)** — opens the webcam, runs `YOLOE.track()` per frame with a custom tracker config (`tests/custom_bytetrack.yaml`, tuned with a longer track buffer and dual confidence thresholds), and draws the HUD overlay (boxes, labels, FPS).
2. **Per-track crop buffering (`TrackBuffer`)** — for each active track ID, keeps up to 6 crops, deduplicating spatially-similar crops (via IoU) by keeping the higher-scoring one and evicting the worst crop when full. A crop is only considered if it clears three thresholds: `MIN_CONFIDENCE`, `MIN_BBOX_SIZE`, `MIN_SHARPNESS` (Laplacian variance blur check). Crop quality score = weighted sum of confidence, bbox size, and sharpness (`compute_combined_score`).
3. **Track lifecycle finalization** — a track with no detections for >1.5s is "finalized." It's accepted (pushed to `finalized_queue`) only if it was tracked ≥1.0s and has ≥2 buffered crops; otherwise discarded as noise. Any tracks still alive when the loop exits are force-finalized the same way.
4. **Background tagging worker (`tagging_worker_func`)**, running in its own thread, consumes `finalized_queue`:
   - Computes a YOLOE embedding (CPU) for the best crop.
   - De-duplication: cosine-similarity-compares against embeddings of all `status='tagged'` DB rows; if similarity ≥ `DEDUP_SIMILARITY_THRESHOLD` (0.9), treats it as a re-sighting (`db.update_object_re_sighting`) and skips the VLM call entirely.
   - Otherwise inserts a `pending` row, then calls the VLM (`execute_tagging_with_retries`) with all buffered crops as base64 image blocks, expecting strict JSON (`object_name`, `tags`, `ocr_text`, `confidence`).
   - Three-level VLM fallback chain:
     1. `nvidia/llama-3.1-nemotron-nano-vl-8b-v1` (primary VL) — 20s timeout, wrapped in its own `tenacity` retry (3 attempts, exponential backoff) for connection/timeout/rate-limit/server errors, plus one manual retry. Treated as failed if it returns `confidence: "low"`.
     2. `qwen/qwen3.5-397b-a17b` (first fallback) — 15s/60s timeout; treated as failed if it returns `confidence: "low"`.
     3. `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` (reasoning fallback) — 15s/180s timeout, reasoning enabled.
   - On success: `db.update_object_result(..., status='tagged', embedding=...)`. On any exception (all three VLMs exhausted): `status='failed'`.
   - Status transitions are always `None -> pending -> {tagged, failed}`, logged to console at each step.

**`db.py`** is the persistence layer, independent of the tracking/tagging logic:
   - Single `objects` table (`track_id` PK, timestamps, `crop_paths` as JSON, `tags`/`ocr_text` nullable, `status` CHECK'd to `pending|tagged|failed`, `confidence`, `embedding` BLOB of float32 bytes).
   - An FTS5 virtual table `objects_fts` kept in sync via SQL triggers (insert/update-of-tags-or-ocr/delete) — never written to directly from Python.
   - `DB_PATH`/`CAPTURES_DIR` are overridable via `BATMAN_DB_PATH`/`BATMAN_CAPTURES_DIR` env vars, which is how tests isolate themselves from the real `objects.db`/`captures/`.
   - Crop paths are stored relative to the project root (via `os.path.relpath`) for portability.

**`download_models.py`** is a standalone stdlib-only script (no dependency on the rest of the codebase) for fetching model weights.

## Working in this codebase

- Changes to tracking/buffering/finalization/tagging logic must be made in both `main.py` (server path, used by the frontend) and `tests/yoloe_test.py` (standalone path) — they're duplicated, not shared, so a fix in one silently doesn't apply to the other. Changes to persistence/search live in `db.py` only, and both consumers are only coupled to it through `db`'s public functions.
- When touching thresholds (`MIN_SHARPNESS`, `MIN_BBOX_SIZE`, `MIN_CONFIDENCE`, `DEDUP_SIMILARITY_THRESHOLD`, tracker YAML) understand these were tuned empirically against live webcam behavior — verify changes against the actual webcam loop (`tests/yoloe_test.py` main, `python -m uvicorn main:app` + frontend, or `tests/smoke_test.py`), not just unit tests, since `test_db.py` doesn't exercise the CV pipeline at all.
- `device='mps'` is hardcoded for the live tracking loop's happy path (falls back to `'cpu'` only if MPS is unavailable) — this project targets Apple Silicon.
- `../frontend` is a Next.js dashboard that talks to `main.py` over HTTP (`NEXT_PUBLIC_API_URL`, default `http://localhost:8000`) — see `frontend/CLAUDE.md` for its side.
