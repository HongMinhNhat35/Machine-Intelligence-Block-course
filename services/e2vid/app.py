import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_NAME = "e2vid"
WEIGHTS_PATH = Path("/app/weights/yolo_e2vid.pt")
RECON_ROOT = Path("/data/recon")

app = FastAPI()


class DetectRequest(BaseModel):
    sequence_id: str
    frame_start: int = 0
    frame_end: int = 3300


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "weights": WEIGHTS_PATH.name,
        "weights_loaded": WEIGHTS_PATH.exists(),
    }


@app.post("/detect")
def detect(req: DetectRequest):
    cache_path = RECON_ROOT / req.sequence_id / f"detections_e2vid_{req.frame_start}_{req.frame_end}.json"

    if cache_path.exists():
        return json.loads(cache_path.read_text())

    frames_dir = RECON_ROOT / req.sequence_id / "reconstruction_e2vid"
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

    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS_PATH))

    all_frames = sorted(frames_dir.glob("frame_*.png")) + sorted(frames_dir.glob("frame_*.jpg"))
    frame_files = [
        f for f in all_frames
        if req.frame_start <= int(f.stem.split("_")[-1]) <= req.frame_end
    ]

    detections = []
    for frame_path in frame_files:
        frame_num = int(frame_path.stem.split("_")[-1])
        results = model(str(frame_path), verbose=False, conf=0.1)
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "frame": frame_num,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "confidence": float(box.conf[0]),
                "class": "drone",
            })

    response = {
        "sequence_id": req.sequence_id,
        "model": MODEL_NAME,
        "frame_start": req.frame_start,
        "frame_end": req.frame_end,
        "detections": detections,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(response, indent=2))

    return response
