# GUI Architecture — Technology and Design Decisions

## Framework choice: FastAPI + HTML/JS (not Streamlit)

The original skeleton used **Streamlit**. After evaluating against the wireframe spec, the team
switched to **FastAPI serving a static HTML/JS frontend**.

### Why not Streamlit

| Requirement | Streamlit | FastAPI + HTML |
|---|---|---|
| Smooth frame playback (Play/Stop/slider) | No — every widget triggers full Python re-run | Yes — pure JS `setInterval`, zero backend calls |
| Bounding box overlay on frames | Server-side PIL draw + re-upload per frame | CSS `position:absolute` divs, instant |
| Custom sidebar with icons and active states | Fragile CSS injection via `unsafe_allow_html` | Native CSS, full control |
| Confidence threshold adjusts overlay live | Full re-run required | Filter detections in JS, no round-trip |
| Tab/screen navigation without page reload | Not possible natively | Native JS |

The core problem is that Streamlit re-executes the entire Python script on every widget interaction.
For a frame viewer with a playback slider that needs to update at 12 fps, this is unusable.

### Why FastAPI + HTML

The wireframe prototype (`docs/HybridVision_GUI_Prototype.html`) is already a fully functional
frontend — navigation, playback, bbox overlay, confidence slider, all working in pure HTML/JS.
It just needs real data wired in via `fetch()` calls. FastAPI provides those endpoints.

The e2vid inference service already uses FastAPI, so the team knows the stack.

---

## Architecture

```
Browser
  │
  │  GET /          → serves static/index.html
  │  GET /api/*     → JSON data endpoints
  │  GET /frames/*  → JPEG frame files (future)
  ▼
services/web  (FastAPI :8080)
  │
  ├── GET /api/admin     → polls /health on each service, scans data dirs
  ├── GET /api/kpis      → reads data/kpis/*.json          [planned]
  ├── GET /api/sequences → lists FRED sequences             [planned]
  ├── GET /api/detect    → proxies to e2vid :8001 /detect   [planned]
  └── GET /frames/{seq}/{frame} → serves JPEG from /data/recon  [planned]
  │
  ├── services/e2vid      (FastAPI :8001)  ← YOLO inference + health
  ├── services/hypere2vid (FastAPI :8002)  ← not yet implemented
  └── services/fusion     (FastAPI :8003)  ← not yet implemented
```

All backend service calls (to :8001, :8002, :8003) go through the web service — the browser
never calls them directly. This avoids CORS issues and keeps Docker internal hostnames hidden.

---

## Screen implementation status

| Screen | Status | Notes |
|---|---|---|
| Admin | **Done** | Live data from `/api/admin` |
| KPIs | Planned | Read `data/kpis/train_yolo.json` at runtime |
| Upload | Planned | Browser lists `/data/fred/sequence_*` |
| Reconstruction | Planned | Serve frames from `/data/recon` via `/frames/` endpoint |
| Detection | Planned | Call `/api/detect`, render bboxes client-side with Canvas |
| Comparison | Planned | Same as Detection, three panels synced via shared JS slider |

---

## Key implementation decisions

### Admin screen — live polling
Service health is fetched from each service's `/health` endpoint at page load (2 s timeout).
The `weights_loaded` flag in the response drives the green/amber/grey status dot.
Restart buttons are present in the UI but disabled — implementing them requires mounting the
Docker socket into the web container, deferred to v2.

### Detection overlay — client-side rendering
Bounding boxes are rendered as CSS `position:absolute` divs over the frame `<img>` tag.
The confidence threshold slider filters the JS detections array in place — no backend call.
This matches what the wireframe prototype already does and gives instant response.

### Pipeline "Run" button — SSE progress stream
Running reconstruction or detection on a full sequence takes minutes. The Run button will POST
to `/api/run` which returns a `text/event-stream` SSE response. The browser reads progress lines
with `EventSource` and updates a progress bar. FastAPI's `StreamingResponse` supports this natively.

### Frame serving
Reconstructed frames are JPEG files at `/data/recon/{seq}/reconstruction_e2vid/frame_{N:06d}.jpg`.
The web service exposes them at `/frames/{seq}/{n}` — a thin wrapper that calls
`FileResponse(RECON_ROOT / seq / "reconstruction_e2vid" / f"frame_{n:06d}.jpg")`.
The playback slider updates `<img src="/frames/{seq}/{frame}">` directly.

### KPIs — live from JSON
The spec notes KPIs can be static or live. We use live: the `/api/kpis` endpoint reads
`data/kpis/train_yolo.json` and the per-sequence `reconstruct_sequence_*.json` files at request
time. No hardcoded numbers in the HTML.

---

## Data paths (Docker mounts)

| Path in container | Contents | Access |
|---|---|---|
| `/data/fred` | Raw FRED sequences (`sequence_*/coordinates.txt`) | Read-only |
| `/data/recon` | Reconstructed frames + detection JSON cache | Read-write |
| `/app/weights` | YOLO weights (`yolo_e2vid.pt`) | Read-only |

Configured via environment variables `FRED_DATA_PATH`, `RECON_DATA_PATH`, `WEIGHTS_PATH`.
