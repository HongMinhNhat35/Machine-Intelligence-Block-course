import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

MODEL_NAME = "fusion"
RECON_ROOT = Path("/data/recon")

app = FastAPI()


class DetectRequest(BaseModel):
    sequence_id: str
    frame_start: int = 0
    frame_end: int = 9999


@app.get("/health")
def health():
    from run_pipeline import weights_available, WEIGHTS_ROOT
    ok = weights_available()
    return {
        "status": "ok" if ok else "warn",
        "model": MODEL_NAME,
        "weights_loaded": ok,
        "weights": {
            "fusion_rgb":   str(WEIGHTS_ROOT / "fusion_rgb.pt"),
            "fusion_event": str(WEIGHTS_ROOT / "fusion_event.pt"),
        },
    }


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


@app.post("/detect")
def detect(req: DetectRequest):
    cache_path = RECON_ROOT / req.sequence_id / "detections_fusion.json"

    def _stream():
        if cache_path.exists():
            data = json.loads(cache_path.read_text())
            n = len(data.get("detections", []))
            yield _sse("progress", json.dumps({"frame": n, "total": n, "cached": True}))
            yield _sse("done", json.dumps({"n": n}))
            return

        from run_pipeline import _get_frame_files, process_frame, weights_available
        if not weights_available():
            yield _sse("error", "Model weights not found — place fusion_rgb.pt and fusion_event.pt in /app/weights/")
            yield _sse("done", json.dumps({"n": 0}))
            return

        try:
            rgb_files, event_files = _get_frame_files(req.sequence_id)
        except Exception as e:
            yield _sse("error", f"Failed to list frame files: {e}")
            yield _sse("done", json.dumps({"n": 0}))
            return

        n_total = min(len(rgb_files), len(event_files), req.frame_end)
        if n_total == 0:
            yield _sse("error", "No paired RGB/event frames found for this sequence")
            yield _sse("done", json.dumps({"n": 0}))
            return

        detections = []
        try:
            for i in range(req.frame_start, n_total):
                fused = process_frame(str(rgb_files[i]), str(event_files[i]))
                for det in fused:
                    x1, y1, x2, y2 = det["box"]
                    detections.append({
                        "frame": i,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "confidence": det["confidence"],
                        "class": "drone",
                        "source": det.get("source", "fusion"),
                    })
                if i % 32 == 0 or i == n_total - 1:
                    yield _sse("progress", json.dumps({
                        "frame": i + 1,
                        "total": n_total,
                        "cached": False,
                    }))
        except Exception as e:
            yield _sse("error", str(e))
            yield _sse("done", json.dumps({"n": len(detections)}))
            return

        result = {
            "sequence_id": req.sequence_id,
            "model": MODEL_NAME,
            "cached": False,
            "detections": detections,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result, indent=2))
        yield _sse("done", json.dumps({"n": len(detections)}))

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
