# e2vid Service

FastAPI microservice that runs YOLOv8 drone detection over pre-reconstructed E2VID frames.

Part of the Hybrid Vision System — sits behind the `web` service and is not exposed directly to the host.

---

## Responsibilities

This service does **not** perform E2VID reconstruction at runtime. Reconstruction is an offline step (see `scripts/reconstruct.py` and `docs/e2vid_pipeline.md`). The service expects frames to already exist on disk and runs YOLO inference over them on demand.

---

## API

### `GET /health`

Returns service status and whether the model weights file is present.

```json
{
  "status": "ok",
  "model": "e2vid",
  "weights": "yolo_e2vid.pt",
  "weights_loaded": true
}
```

### `POST /detect`

Runs YOLO inference over a range of reconstructed frames for a given sequence.
Results are cached to disk after the first call — subsequent calls for the same sequence return instantly.

**Request:**

```json
{
  "sequence_id": "sequence_0",
  "frame_start": 0,
  "frame_end": 3300
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `sequence_id` | string | required | matches a subdirectory under `/data/recon/` |
| `frame_start` | int | `0` | first frame index to process (inclusive) |
| `frame_end` | int | `3300` | last frame index to process (inclusive) |

**Response:**

```json
{
  "sequence_id": "sequence_0",
  "model": "e2vid",
  "detections": [
    {
      "frame": 102,
      "bbox": [975.9, 386.9, 27.5, 25.1],
      "confidence": 0.645,
      "class": "drone"
    }
  ]
}
```

`bbox` is `[x, y, width, height]` in pixels (top-left origin, 1280×720 coordinate space).
Frames with no detection above the model threshold are omitted from the list.

**Error responses:**

| Code | Cause |
|---|---|
| 404 | `reconstruction_e2vid/` directory not found for the given sequence |
| 503 | Model weights file missing from `/app/weights/` |

---

## Caching

After the first `/detect` call for a sequence, results are written to:

```
/data/recon/<sequence_id>/detections_e2vid.json
```

Subsequent calls return this file directly without re-running inference.
To force re-inference, delete the cache file and call `/detect` again.

---

## Detection threshold

The default YOLO confidence threshold is 0.25 (ultralytics default).
Based on validation against the FRED dataset, a threshold of **0.35** eliminates observed
false positives while retaining all true detections.
This can be tuned by passing `conf=0.35` to `model()` in `app.py`.

---

## Paths inside the container

| Path | Contents |
|---|---|
| `/app/weights/yolo_e2vid.pt` | YOLOv8n weights, trained on FRED e2vid frames (epoch 80, mAP50=0.874) |
| `/data/recon/<sequence_id>/reconstruction_e2vid/` | input frames (mounted from host `data/processed/`) |
| `/data/recon/<sequence_id>/detections_e2vid.json` | cached detection output |

---

## Weights

Weights are baked into the Docker image at build time (`COPY weights/ /app/weights/`).
To update weights after retraining:

```bash
# 1. Place new weights
cp data/yolo_runs/e2vid/weights/best.pt services/e2vid/weights/yolo_e2vid.pt

# 2. Rebuild and restart
docker compose build e2vid
docker compose up -d e2vid
```

---

## Dependencies

| Package | Version | Notes |
|---|---|---|
| fastapi | 0.111.0 | HTTP framework |
| uvicorn | 0.30.1 | ASGI server |
| ultralytics | 8.4.54 | YOLOv8 inference |
| torch | 2.3.0+cpu | CPU-only build (inference only, no GPU needed) |
| opencv-python-headless | latest | OpenCV without X11 (required in headless container) |
