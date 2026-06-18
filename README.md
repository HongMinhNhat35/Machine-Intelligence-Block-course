# Hybrid Vision System — GUI

Web interface for the AMI drone detection pipeline: browse FRED sequences, run YOLO detection on reconstructed e2vid frames, and inspect results.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) (v2)
- Extracted FRED dataset sequences on your machine
- A writable directory for reconstruction output

## Setup

**1. Create your `.env` file**

```bash
cp .env.example .env
```

Open `.env` and set two paths:

| Variable | What to put there |
|---|---|
| `FRED_DATA_PATH` | Directory containing your extracted FRED sequences (`sequence_84/`, `sequence_127/`, …). Each folder must have `Event/`, `RGB/`, `coordinates.txt` inside. |
| `RECON_DATA_PATH` | Any writable directory. Reconstructed frames and detection results are written here. Can be empty to start. |

**2. Start the services**

```bash
docker compose up -d
```

Docker will pull the pre-built images on first run (~500 MB). Then open **http://localhost:8080**.

**3. Stop**

```bash
docker compose down
```

---

## Using the interface

### Upload tab
Browse your available sequences. The UI shows which files were found (events, annotations, reconstructed frames). Select a sequence to activate all other tabs.

### Reconstruction tab
Frame viewer with playback for the reconstructed e2vid output.

### Detection tab
Reconstructed frame alongside YOLO bounding box overlay. Confidence threshold slider filters detections client-side. Model selector routes detection to e2vid (:8001), hypere2vid (:8002), or fusion (:8003).

### Comparison tab
Side-by-side E2VID vs HyperE2VID view with bounding boxes and KPI metrics. Late Fusion column not yet available.

### KPIs tab
Detection accuracy and compute metrics table.

### Admin tab
Live service health, data volume status, and model settings.

---

## Running reconstruction

Reconstruction is currently a script-based step, not triggered from the GUI. Run it on a machine with OpenEB installed:

```bash
python scripts/reconstruct.py \
  --zip_path  /path/to/sequence_N/events.zip \
  --out_dir   /path/to/recon/output/sequence_N/reconstruction_e2vid \
  --work_dir  /tmp/rpg_e2vid
```

Once frames are written to `RECON_DATA_PATH`, the GUI will show the sequence as reconstructed and enable detection.

---

## Services

| Service | Port | Description |
|---|---|---|
| `web` | 8080 | FastAPI + static HTML/JS frontend |
| `e2vid` | 8001 (internal) | YOLO inference on e2vid frames. Weights baked into image. |
| `hypere2vid` | 8002 (internal) | Implemented; no weights or reconstruction data yet |
| `fusion` | 8003 (internal) | Not yet implemented |
