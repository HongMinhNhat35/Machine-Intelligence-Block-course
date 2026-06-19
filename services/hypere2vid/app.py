import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

MODEL_NAME   = "hypere2vid"
WEIGHTS_PATH = Path("/app/weights/hypere2vid_best.pt")
RECON_ROOT   = Path("/data/recon")

app = FastAPI()

_model = None

def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(str(WEIGHTS_PATH))
    return _model


@app.on_event("startup")
def startup():
    if WEIGHTS_PATH.exists():
        get_model()


@app.get("/health")
def health():
    return {
        "status":         "ok",
        "model":          MODEL_NAME,
        "weights":        WEIGHTS_PATH.name,
        "weights_loaded": WEIGHTS_PATH.exists(),
    }


class DetectRequest(BaseModel):
    sequence_id: str
    frame_start: int = 0
    frame_end:   int = 3300


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


@app.post("/detect")
def detect(req: DetectRequest):
    cache_path = RECON_ROOT / req.sequence_id / "detections_hypere2vid.json"

    frames_dir = RECON_ROOT / req.sequence_id / "reconstruction_hypere2vid"
    if not frames_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Reconstructed frames not found at {frames_dir}.",
        )
    if not WEIGHTS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model weights not found at {WEIGHTS_PATH}.",
        )

    all_frames = sorted(frames_dir.glob("frame_*.png")) + sorted(frames_dir.glob("frame_*.jpg"))
    frame_files = [
        f for f in all_frames
        if req.frame_start <= int(f.stem.split("_")[-1]) <= req.frame_end
    ]

    BATCH = 32
    REPORT_EVERY = 10

    def generate():
        model      = get_model()
        detections = []
        total      = len(frame_files)

        for batch_start in range(0, total, BATCH):
            batch_paths = frame_files[batch_start: batch_start + BATCH]
            results = model(
                [str(p) for p in batch_paths],
                verbose=False,
                conf=0.1,
            )
            for frame_path, result in zip(batch_paths, results):
                frame_num = int(frame_path.stem.split("_")[-1])
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        "frame":      frame_num,
                        "bbox":       [x1, y1, x2 - x1, y2 - y1],
                        "confidence": float(box.conf[0]),
                        "class":      "drone",
                    })

            batch_idx   = batch_start // BATCH
            done_frames = min(batch_start + BATCH, total)
            if batch_idx % REPORT_EVERY == 0 or done_frames == total:
                yield _sse("progress", json.dumps({
                    "frame":  done_frames,
                    "total":  total,
                    "cached": False,
                }))

        response = {
            "sequence_id": req.sequence_id,
            "model":       MODEL_NAME,
            "detections":  detections,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(response, indent=2))
        yield _sse("done", json.dumps({"n": len(detections)}))

    return StreamingResponse(generate(), media_type="text/event-stream")
