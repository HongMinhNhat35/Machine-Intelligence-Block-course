import asyncio
import json
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Hybrid Vision System")

E2VID_URL      = os.getenv("E2VID_URL",      "http://e2vid:8001")
HYPERE2VID_URL = os.getenv("HYPERE2VID_URL", "http://hypere2vid:8002")
FUSION_URL     = os.getenv("FUSION_URL",     "http://fusion:8003")
FRED_ROOT      = Path(os.getenv("FRED_DATA_PATH",  "/data/fred"))
RECON_ROOT     = Path(os.getenv("RECON_DATA_PATH", "/data/recon"))
WEIGHTS_ROOT   = Path(os.getenv("WEIGHTS_PATH",    "/app/weights"))

STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC / "index.html"))


# ── Sequences ──────────────────────────────────────────────────────────────────

@app.get("/api/sequences")
def get_sequences():
    seq_ids: set[str] = set()
    if FRED_ROOT.exists():
        seq_ids.update(p.name for p in FRED_ROOT.glob("sequence_*") if p.is_dir())
    if RECON_ROOT.exists():
        seq_ids.update(p.name for p in RECON_ROOT.glob("sequence_*") if p.is_dir())

    result = []
    for seq_id in sorted(seq_ids):
        fred_dir  = FRED_ROOT  / seq_id
        recon_dir = RECON_ROOT / seq_id
        e2vid_dir = recon_dir / "reconstruction_e2vid"

        frames = list(e2vid_dir.glob("frame_*")) if e2vid_dir.exists() else []

        # events.zip / events.h5 land in RECON_ROOT after preprocessing;
        # raw FRED extracts put the events in FRED_ROOT/seq/Event/ instead.
        has_events_zip = (recon_dir / "events.zip").exists()
        has_events_h5  = (recon_dir / "events.h5").exists()
        has_raw_events = (fred_dir / "Event").is_dir()

        result.append({
            "id":              seq_id,
            "has_events_zip":  has_events_zip,
            "has_events_h5":   has_events_h5,
            "has_raw_events":  has_raw_events,
            "has_coordinates": (fred_dir / "coordinates.txt").exists(),
            "e2vid_done":      len(frames) > 0,
            "frame_count":     len(frames),
        })
    return result


# ── Run pipeline (SSE) ────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    sequence_id: str
    steps: list[str]   # e.g. ["detect_e2vid"]


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


async def _run_stream(sequence_id: str, steps: list[str]):
    yield _sse("log", f"Starting pipeline for {sequence_id}")

    if "detect_e2vid" in steps:
        e2vid_dir = RECON_ROOT / sequence_id / "reconstruction_e2vid"
        if not e2vid_dir.exists() or not any(e2vid_dir.glob("frame_*")):
            yield _sse("error", "e2vid frames not found — run reconstruction first via scripts/reconstruct.py")
            yield _sse("done", "failed")
            return

        yield _sse("log", "Running YOLO detection via e2vid service…")
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                r = await client.post(
                    f"{E2VID_URL}/detect",
                    json={"sequence_id": sequence_id},
                )
            if r.status_code == 200:
                data = r.json()
                n = len(data.get("detections", []))
                yield _sse("log", f"Detection complete — {n} detections across {data.get('frame_end', '?')} frames")
                yield _sse("result", json.dumps({"detections": n}))
            else:
                yield _sse("error", f"e2vid service error {r.status_code}: {r.text[:200]}")
                yield _sse("done", "failed")
                return
        except Exception as exc:
            yield _sse("error", f"Could not reach e2vid service: {exc}")
            yield _sse("done", "failed")
            return

    yield _sse("done", "ok")


@app.post("/api/run")
async def run_pipeline(req: RunRequest):
    return StreamingResponse(
        _run_stream(req.sequence_id, req.steps),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Admin ──────────────────────────────────────────────────────────────────────

@app.get("/api/admin")
async def admin():
    services = []
    for name, url, port in [
        ("e2vid",      E2VID_URL,      8001),
        ("hypere2vid", HYPERE2VID_URL, 8002),
        ("fusion",     FUSION_URL,     8003),
    ]:
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{url}/health", timeout=2.0)
            data = r.json()
            if data.get("weights_loaded"):
                status, detail = "ok", "weights loaded"
            else:
                status, detail = "warn", "weights missing"
        except Exception:
            status, detail = "error", "unreachable"
        services.append({"name": name, "port": port, "status": status, "detail": detail})

    sequences = []
    if FRED_ROOT.exists():
        for seq in sorted(FRED_ROOT.glob("sequence_*")):
            e2vid_dir = RECON_ROOT / seq.name / "reconstruction_e2vid"
            hyper_dir = RECON_ROOT / seq.name / "reconstruction_hypere2vid"
            sequences.append({
                "id":         seq.name,
                "e2vid":      e2vid_dir.exists() and any(e2vid_dir.iterdir()),
                "hypere2vid": hyper_dir.exists() and any(hyper_dir.iterdir()),
            })

    return {
        "services": services,
        "paths": {
            "FRED_DATA_PATH":  str(FRED_ROOT),
            "RECON_DATA_PATH": str(RECON_ROOT),
        },
        "sequences": sequences,
        "model_settings": {
            "dataset_image_size":  "1280 × 720 px",
            "yolo_inference_size": "640 × 640 px",
            "e2vid_weights":       str(WEIGHTS_ROOT / "yolo_e2vid.pt"),
            "hypere2vid_weights":  str(WEIGHTS_ROOT / "hypere2vid_best.pt"),
            "fusion_weights":      str(WEIGHTS_ROOT / "fusion_best.pt"),
        },
    }
