@AGENTS.md

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

The dashboard/control UI for Batman-Vision, a real-time webcam object detection & tagging system. This is a Next.js (App Router) single-page dashboard — there's effectively one screen (`app/page.tsx`, `'use client'`) that:

- Starts/stops the Python pipeline on the backend (`POST /pipeline/start`, `/pipeline/stop`).
- Renders the live HUD-annotated webcam feed as an `<img>` pointed at the backend's MJPEG stream (`GET /video_feed`).
- Polls `GET /pipeline/status`, `GET /api/stats`, and `GET /api/objects` every 1.5s (`setInterval` in `useEffect`) to keep pipeline state, tagged/pending/failed counts, and the object gallery in sync.
- Lets you clear the database/crops (`POST /api/clear`, blocked while the pipeline is active) and view all buffered crops for a track in a modal.

There is no separate backend logic here — all detection/tracking/tagging happens in `../backend` (a FastAPI server, see `../backend/CLAUDE.md`). This app is a thin client over that HTTP API; it holds no persistent state of its own beyond React component state re-derived from polling.

## Backend connection

- `NEXT_PUBLIC_API_URL` (see `.env.local`) sets the backend base URL, default `http://localhost:8000`. Every fetch in `app/page.tsx` is built off the `BACKEND_URL` constant derived from this env var — there's no other config surface for it.
- The backend must be running (`uvicorn main:app`) with CORS allowing this app's origin (`http://localhost:3000` by default, hardcoded backend-side) for any of this to work; a cold backend just means failed fetches, caught and surfaced as `statusMessage` errors, not crashes.
- Crop images are loaded directly from the backend's `/captures` static mount (`${BACKEND_URL}/${crop_path}`), not proxied through Next.js.

## Working in this codebase

- Nearly everything lives in `app/page.tsx` — there's no component decomposition yet. If you're adding a feature, consider whether it's worth extracting a component, but don't force a refactor unrelated to the task.
- Styling is Tailwind CSS v4 utility classes only (no CSS modules, no component library) with a dark, monospace, "HUD" aesthetic (`zinc`/`cyan`/`emerald`/`rose`/`amber` palette, `font-mono` throughout). Match this style for any new UI.
- Object `status` values (`pending` | `tagged` | `failed`) map 1:1 to backend statuses in `../backend/db.py` — don't introduce new status strings on the frontend without adding them backend-side first.
