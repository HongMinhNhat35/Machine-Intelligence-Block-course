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
  │  GET /              → serves static/index.html (Cache-Control: no-store)
  │  GET /api/*         → JSON data endpoints
  │  GET /frames/*      → JPEG/PNG frame files
  ▼
services/web  (FastAPI :8080)
  │
  ├── GET  /api/sequences              → lists sequences from FRED + recon dirs
  ├── GET  /api/kpis                   → reads /app/kpis/*.json (no-store)
  ├── GET  /api/detections/{seq}?model → reads detections_{model}.json cache
  ├── POST /api/detect                 → SSE stream, proxies to model service
  ├── GET  /frames/{seq}/{n}?model     → serves frame from reconstruction_{model}/
  └── GET  /api/admin                  → polls /health on each service
  │
  ├── services/e2vid      (FastAPI :8001)  ← YOLO inference, health, weights loaded
  ├── services/hypere2vid (FastAPI :8002)  ← implemented; no weights/recon data yet
  └── services/fusion     (FastAPI :8003)  ← not yet implemented
```

All backend service calls (to :8001, :8002, :8003) go through the web service — the browser
never calls them directly. This avoids CORS issues and keeps Docker internal hostnames hidden.

---

## Screen implementation status

| Screen | Status | Notes |
|---|---|---|
| Upload | **Done** | Lists sequences from FRED + recon dirs; shows events/coords/frame status |
| Reconstruction | **Done** | Serves E2VID frames; HyperE2VID side loads when reconstruction available |
| Detection | **Done** | SSE-streamed detection run; bbox overlay; confidence slider; model selector |
| Comparison | **Done** | E2VID vs HyperE2VID side-by-side; KPI metrics per model; split selector |
| KPIs | **Done** | Three tables (detection, reconstruction, training) from `/app/kpis/*.json` |
| Admin | **Done** | Live health from `/health` on each service; data paths; model settings |

---

## Key implementation decisions

### Admin screen — live polling
Service health is fetched from each service's `/health` endpoint at page load (2 s timeout).
The `weights_loaded` flag in the response drives the green/amber/grey status dot.

### Detection overlay — client-side rendering
Bounding boxes are rendered as CSS `position:absolute` divs over the frame `<img>` tag.
The confidence threshold slider filters the JS detections array in place — no backend call.

### Pipeline "Run" button — SSE progress stream
Running detection on a full sequence takes minutes on CPU. The Run button POSTs to `/api/detect`
which returns a `text/event-stream` SSE response. The browser reads progress lines and updates
a status line. FastAPI's `StreamingResponse` supports this natively.

### Frame serving — model-aware
Reconstructed frames are served at `/frames/{seq}/{n}?model=e2vid` (default: `e2vid`).
The `model` parameter selects the reconstruction directory:
`RECON_ROOT/{seq}/reconstruction_{model}/frame_{n:06d}.{jpg|png}`.

### Detection routing — model-aware
`/api/detect` (POST) and `/api/detections/{seq}` (GET) both accept a `model` query parameter.
The detect endpoint routes to the appropriate service URL:
- `e2vid` → `:8001`
- `hypere2vid` → `:8002`
- `fusion` → `:8003`

Detection results are cached per-model as `detections_{model}.json` in the sequence directory.

### KPIs — live from JSON files
The `/api/kpis` endpoint scans `/app/kpis/*.json` at request time and returns all entries sorted
by filename (natural sort). KPI files are baked into the web image at build time from
`services/web/kpis/`. Adding a new run means adding a new JSON file and rebuilding the image.
The endpoint sets `Cache-Control: no-store` to prevent stale results in the browser.

### Sidebar mAP50 — populated at page load
On page load the browser fetches `/api/kpis`, picks the latest e2vid run, and populates the
sidebar `mAP50` field. This keeps the sidebar current without navigating to the KPIs tab.

---

## Data paths (Docker mounts)

| Path in container | Contents | Access |
|---|---|---|
| `/data/fred` | Raw FRED sequences (`sequence_*/coordinates.txt`) | Read-only |
| `/data/recon` | Reconstructed frames + detection JSON cache | Read-write |
| `/app/kpis` | KPI JSON files (baked into image) | Read-only |

Configured via environment variables `FRED_DATA_PATH`, `RECON_DATA_PATH`.
`KPIS_PATH` defaults to `/app/kpis` (no mount needed — files are baked into the image).
