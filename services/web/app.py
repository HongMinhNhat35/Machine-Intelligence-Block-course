import os
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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
