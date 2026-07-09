# Batman Vision — Orchestrator Dashboard

The control UI for [Batman-Vision](../backend/README.md): a single-page Next.js dashboard that starts/stops the live webcam tracking pipeline, shows the HUD-annotated video feed, and browses tagged objects as they come in. It's a thin client over the `../backend` FastAPI server — all detection, tracking, and VLM tagging happens there.

## Setup

```bash
npm install
```

Create `.env.local` (already present with a sensible default) to point at the backend:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Start the backend first (`uvicorn main:app` from `../backend`) — the dashboard polls it continuously and will show connection errors if it isn't running.

## Running the dev server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The backend must allow this origin via CORS (hardcoded to `localhost:3000`/`127.0.0.1:3000` in `../backend/main.py`).

## What it does

- **Pipeline control** — `START PIPELINE` / `STOP PIPELINE` buttons call the backend's `/pipeline/start` and `/pipeline/stop` endpoints.
- **Live feed** — while the pipeline is active, streams the backend's MJPEG HUD feed (`/video_feed`) directly in an `<img>` tag.
- **Stats & object gallery** — polls `/api/stats` and `/api/objects` every 1.5s to show tagged/pending/failed counts and a live-updating grid of detected objects (tags, OCR text, crop thumbnails). Clicking an object opens a modal with all buffered crops for that track.
- **Clear database** — wipes all objects and crop files via `/api/clear` (only available while the pipeline is stopped).

## Other commands

```bash
npm run build   # production build
npm run start   # serve the production build
npm run lint    # eslint
```

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.
