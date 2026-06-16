# Group 1 — Hybrid Vision System: Quickstart

Standalone deployment guide for the AMI drone-detection GUI.  
No Python or ML setup required — everything runs inside Docker.

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with Docker Compose v2
- Git
- FRED dataset sequences extracted on your machine
- A writable directory for reconstruction output (can be empty)

---

## 1. Clone the repository

```bash
git clone https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course.git \
  --branch gui --single-branch ami-gui
cd ami-gui
```

---

## 2. Configure paths

```bash
cp .env.example .env
```

Open `.env` and set the two paths:

| Variable | Description |
|---|---|
| `FRED_DATA_PATH` | Directory containing the extracted FRED sequences (`sequence_84/`, `sequence_127/`, …). Each folder must contain `Event/`, `RGB/`, and `coordinates.txt`. |
| `RECON_DATA_PATH` | Writable directory for reconstructed frames and detection results. Can be empty — the pipeline writes here as it runs. |

Example:

```env
FRED_DATA_PATH=/data/fred/sequences
RECON_DATA_PATH=/data/recon
```

---

## 3. Start

```bash
docker compose up -d
```

Docker pulls the pre-built images from GHCR on first run (~1.5 GB total). Once all services are healthy, open:

**http://localhost:8080**

---

## 4. Stop

```bash
docker compose down
```

Data written to `RECON_DATA_PATH` (reconstructed frames, detection cache) is preserved on your host — it survives container restarts.

---

## What you can do in the GUI

| Tab | What it does |
|---|---|
| **Upload** | Select a FRED sequence. The sidebar shows frame count and cached detection count. |
| **Reconstruction** | Browse reconstructed e2vid frames with playback controls. |
| **Detection** | View YOLO bounding-box overlay. Adjust confidence threshold with the sidebar slider. Click **Run Detection** to run inference (takes several minutes on CPU; result is cached). |
| **Comparison** | Side-by-side view of all three pipelines (HyperE2VID and Late Fusion show "not yet available"). |
| **KPIs** | Detection accuracy and training metrics table. |
| **Admin** | Live health status of all backend services. |

---

## Notes

- Detection is cached in `RECON_DATA_PATH/<sequence_id>/detections_e2vid.json`. Subsequent visits load instantly from cache. Click **Run Detection** to overwrite (e.g. after new model weights).
- The confidence slider is a display filter only — it does not change what is stored in the cache. The cache always holds all detections at confidence ≥ 0.1.
- Reconstruction (event → frames) is not triggered from the GUI. Frames are pre-generated on Kaggle and must already be present in `RECON_DATA_PATH`.
