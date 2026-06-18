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
Returns a **Server-Sent Events (SSE)** stream — not a single JSON response.
Results are cached to disk after completion; subsequent calls for the same sequence return instantly from cache.

**Request:**

```json
{
  "sequence_id": "sequence_85",
  "frame_start": 0,
  "frame_end": 6777
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `sequence_id` | string | required | matches a subdirectory under `/data/recon/` |
| `frame_start` | int | `0` | first frame index to process (inclusive) |
| `frame_end` | int | `3300` | last frame index to process (inclusive) |

**Response — SSE stream (`text/event-stream`):**

Progress events during inference:
```
event: progress
data: {"frame": 320, "total": 6777, "cached": false}

event: progress
data: {"frame": 640, "total": 6777, "cached": false}
```

When loading from cache:
```
event: progress
data: {"frame": 0, "total": 6777, "cached": true}
```

Final event on completion:
```
event: done
data: {"n": 9188}
```

`n` is the total number of detections found across all frames.

The full detection data is written to the cache file (see [Caching](#caching)) — it is not streamed.

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

Subsequent calls return cached events directly without re-running inference.
To force re-inference, delete the cache file and call `/detect` again (the `web` service's
**Run Detection** button always deletes the cache before calling this endpoint).

---

## Detection threshold

The model runs at confidence threshold **0.1** — lower than the ultralytics default of 0.25.
This stores all borderline detections in the cache. The GUI confidence slider then filters
what is displayed without any server call, giving instant visual feedback.

---

## Paths inside the container

| Path | Contents |
|---|---|
| `/app/weights/yolo_e2vid.pt` | YOLOv8s weights, Run 5, trained on FRED e2vid frames (mAP50=79.2%, best at epoch 5) |
| `/data/recon/<sequence_id>/reconstruction_e2vid/` | input frames (mounted from host) |
| `/data/recon/<sequence_id>/detections_e2vid.json` | cached detection output |

---

## Weights

Weights are baked into the Docker image at build time (`COPY weights/ /app/weights/`).
The image is published to GHCR. To update weights after retraining:

```bash
# 1. Place new weights
cp data/yolo_runs/e2vid/weights/best.pt services/e2vid/weights/yolo_e2vid.pt

# 2. Build and push to GHCR
docker build -t ghcr.io/gennepy/ami-e2vid:latest services/e2vid/
gh auth token | docker login ghcr.io -u gennepy --password-stdin
docker push ghcr.io/gennepy/ami-e2vid:latest

# 3. Teammates pull the update
docker compose pull e2vid && docker compose up -d e2vid
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
