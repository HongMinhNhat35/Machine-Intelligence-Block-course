import asyncio
import json
import os
import re
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel  # noqa: F401 (used by DetectRequest)


def _nat_key(s: str) -> list:
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", s)]

app = FastAPI(title="Hybrid Vision System")

E2VID_URL      = os.getenv("E2VID_URL",      "http://e2vid:8001")
HYPERE2VID_URL = os.getenv("HYPERE2VID_URL", "http://hypere2vid:8002")
FUSION_URL     = os.getenv("FUSION_URL",     "http://fusion:8003")
FRED_ROOT      = Path(os.getenv("FRED_DATA_PATH",  "/data/fred"))
RECON_ROOT     = Path(os.getenv("RECON_DATA_PATH", "/data/recon"))
WEIGHTS_ROOT   = Path(os.getenv("WEIGHTS_PATH",    "/app/weights"))
KPIS_ROOT      = Path(os.getenv("KPIS_PATH",       "/app/kpis"))

STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/")
def index():
    return FileResponse(
        str(STATIC / "index.html"),
        headers={"Cache-Control": "no-store"},
    )


# ── Sequences ──────────────────────────────────────────────────────────────────

@app.get("/api/sequences")
def get_sequences():
    seq_ids: set[str] = set()
    if FRED_ROOT.exists():
        seq_ids.update(p.name for p in FRED_ROOT.glob("sequence_*") if p.is_dir())
    if RECON_ROOT.exists():
        seq_ids.update(p.name for p in RECON_ROOT.glob("sequence_*") if p.is_dir())

    result = []
    for seq_id in sorted(seq_ids, key=_nat_key):
        fred_dir  = FRED_ROOT  / seq_id
        recon_dir = RECON_ROOT / seq_id
        e2vid_dir = recon_dir / "reconstruction_e2vid"
        hyper_dir = recon_dir / "reconstruction_hypere2vid"

        frames       = list(e2vid_dir.glob("frame_*")) if e2vid_dir.exists() else []
        hyper_frames = list(hyper_dir.glob("frame_*")) if hyper_dir.exists() else []

        # events.zip / events.h5 land in RECON_ROOT after preprocessing;
        # raw FRED extracts put the events in FRED_ROOT/seq/Event/ instead.
        has_events_zip = (recon_dir / "events.zip").exists()
        has_events_h5  = (recon_dir / "events.h5").exists()
        has_raw_events = (fred_dir / "Event").is_dir()

        result.append({
            "id":                seq_id,
            "has_events_zip":    has_events_zip,
            "has_events_h5":     has_events_h5,
            "has_raw_events":    has_raw_events,
            "has_coordinates":   (fred_dir / "coordinates.txt").exists(),
            "e2vid_done":        len(frames) > 0,
            "frame_count":       len(frames),
            "hypere2vid_done":   len(hyper_frames) > 0,
            "hyper_frame_count": len(hyper_frames),
        })
    return result


# ── Run pipeline (SSE) ────────────────────────────────────────────────────────

@app.get("/frames/{seq}/{n}")
def get_frame(seq: str, n: int, model: str = "e2vid"):
    recon_dir = f"reconstruction_{model}"
    for ext in ("jpg", "png"):
        p = RECON_ROOT / seq / recon_dir / f"frame_{n:06d}.{ext}"
        if p.exists():
            return FileResponse(str(p), media_type=f"image/{ext}")
    raise HTTPException(status_code=404, detail=f"Frame {n} not found for {seq} ({model})")


@app.get("/api/kpis")
def get_kpis():
    from fastapi.responses import JSONResponse
    if not KPIS_ROOT.exists():
        return JSONResponse([], headers={"Cache-Control": "no-store"})
    results = []
    for f in sorted(KPIS_ROOT.glob("*.json"), key=lambda p: _nat_key(p.name)):
        try:
            results.append(json.loads(f.read_text()))
        except Exception:
            pass
    return JSONResponse(results, headers={"Cache-Control": "no-store"})


@app.get("/api/detections/{seq_id}")
def get_detections(seq_id: str, model: str = "e2vid"):
    cache = RECON_ROOT / seq_id / f"detections_{model}.json"
    if not cache.exists():
        raise HTTPException(status_code=404, detail="No detections cached for this sequence")
    data = json.loads(cache.read_text())
    return {"sequence_id": seq_id, "model": model, "detections": data.get("detections", [])}


@app.get("/api/detect_frame/{seq_id}/{n}")
async def detect_frame_live(seq_id: str, n: int, model: str = "e2vid"):
    service_url = {
        "e2vid":      E2VID_URL,
        "hypere2vid": HYPERE2VID_URL,
    }.get(model, E2VID_URL)
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            r = await client.get(f"{service_url}/detect_frame", params={"sequence_id": seq_id, "frame_n": n})
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=exc.response.status_code, detail="Detection service error")
        except Exception:
            raise HTTPException(status_code=503, detail="Detection service unavailable")


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


async def _detect_stream(sequence_id: str, frame_count: int, model: str = "e2vid"):
    yield _sse("log", f"Starting detection for {sequence_id} ({frame_count} frames)…")
    recon_dir = RECON_ROOT / sequence_id / f"reconstruction_{model}"
    if not recon_dir.exists() or not any(recon_dir.glob("frame_*")):
        yield _sse("error", "No reconstructed frames found — reconstruction must be run first")
        yield _sse("done", "failed")
        return
    service_url = {
        "e2vid":      E2VID_URL,
        "hypere2vid": HYPERE2VID_URL,
        "fusion":     FUSION_URL,
    }.get(model, E2VID_URL)
    try:
        async with httpx.AsyncClient(timeout=3600.0) as client:
            async with client.stream(
                "POST",
                f"{service_url}/detect",
                json={"sequence_id": sequence_id, "frame_start": 0, "frame_end": frame_count},
            ) as r:
                event_type: str | None = None
                async for line in r.aiter_lines():
                    if line.startswith("event: "):
                        event_type = line[7:].strip()
                    elif line.startswith("data: "):
                        try:
                            payload = json.loads(line[6:].strip())
                        except Exception:
                            continue
                        if event_type == "progress":
                            if payload.get("cached"):
                                yield _sse("log", "Loading from cache…")
                            else:
                                frame = payload.get("frame", 0)
                                total = payload.get("total", frame_count) or frame_count
                                pct   = int(frame / total * 100) if total else 0
                                yield _sse("log", f"Processing… {frame}/{total} frames ({pct}%)")
                        elif event_type == "done":
                            n = payload.get("n", 0)
                            yield _sse("log", f"Done — {n} detections found")
                            yield _sse("result", json.dumps({"detections": n}))
                            yield _sse("done", "ok")
                        event_type = None
    except Exception as exc:
        yield _sse("error", f"Could not reach {model} service: {exc}")
        yield _sse("done", "failed")


class DetectRequest(BaseModel):
    sequence_id: str
    frame_count: int
    model: str = "e2vid"


@app.post("/api/detect")
async def detect(req: DetectRequest):
    return StreamingResponse(
        _detect_stream(req.sequence_id, req.frame_count, req.model),
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
        for seq in sorted(FRED_ROOT.glob("sequence_*"), key=lambda p: _nat_key(p.name)):
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
