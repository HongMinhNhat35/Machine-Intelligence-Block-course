import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_NAME = "hypere2vid"
WEIGHTS_PATH = Path("/app/weights/yolo_hypere2vid.pt")
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
    cache_path = RECON_ROOT / req.sequence_id / "detections_hypere2vid.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    # Teammate 1 implements inference here
    raise HTTPException(status_code=501, detail="HyperE2VID inference not yet implemented.")
