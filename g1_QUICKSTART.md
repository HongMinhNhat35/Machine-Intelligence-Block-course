# Group 1 — Hybrid Vision System: Quickstart

Standalone deployment guide for the AMI drone-detection GUI.  
This guide installs a fully functional demonstration environment with precomputed reconstructions, cached detections and evaluation results.  
No Python, no Git, no build step — just Docker.

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with Docker Compose v2
- ~4 GB free disk space (E2VID + HyperE2VID frames)

---

## Quick install (recommended)

Download and run the installer script:

```bash
curl -O https://raw.githubusercontent.com/HongMinhNhat35/Machine-Intelligence-Block-course/gui/g1-install.sh
bash g1-install.sh
```

The script will ask two questions:
1. **Where to store the data** — default: `~/g1-ami-data`
2. **Path to your FRED raw dataset** — needed for Late Fusion; press Enter to skip for the basic demo

It then downloads ~1.5 GB of pre-computed E2VID frames (Run 9, test set: 8, 9, 12, 20, 21, and prototyping set: 84, 85, 124, 127, 201) and ~290 MB of HyperE2VID frames (Run 2, sequences 84, 85, 124, 127, 201), writes `docker-compose.yml` and `.env`, and pulls the Docker images.

When it finishes:

```bash
cd ~/g1-ami-data          # or the path you chose
docker compose up -d
```

Open **http://localhost:8080**

> **Warning:** only run `docker compose up -d` once. If the stack is already running, starting it again from a different directory will conflict on port 8080. Stop the existing stack first: `docker compose down`.

## Sequences with reconstruction data from e2vid and hypere2vid

Reconstructed frames of e2vid and hypere2vid are only available for the following sequences. Only when selecting one of those a detection or comparison is possible:

e2vid:
  - test set: 8, 9, 12, 20, 21
  - prototyping set: 84, 85, 124, 127, 201

HyperE2VID:
  - prototyping set: 84, 85, 124, 127, 201

---

## Manual installation (if the script fails)

### 1. Create the data directory

```bash
RECON=~/g1-ami-data      # absolute path — change if needed
mkdir -p $RECON
```

### 2. Download and extract pre-computed E2VID frames (Run 9)

```bash
BASE="https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v5.0"

# Test sequences
for seq in 8 9 12 20 21; do
  curl -L -o /tmp/sequence_${seq}.tar "$BASE/sequence_${seq}_e2vid_run9.tar"
  mkdir -p $RECON/sequence_${seq} && tar xf /tmp/sequence_${seq}.tar -C $RECON/sequence_${seq}/
done

# Prototyping sequences (HyperE2VID available for these)
for seq in 84 85 124 127 201; do
  curl -L -o /tmp/sequence_${seq}.tar "$BASE/sequence_${seq}_e2vid_run9.tar"
  mkdir -p $RECON/sequence_${seq} && tar xf /tmp/sequence_${seq}.tar -C $RECON/sequence_${seq}/
done
```

### 3. Download HyperE2VID frames (Run 2)

```bash
for seq in sequence_84 sequence_85 sequence_124 sequence_127 sequence_201; do
  curl -L -o /tmp/${seq}_hypere2vid.tar \
    https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v3.0/${seq}_hypere2vid.tar
  tar xf /tmp/${seq}_hypere2vid.tar -C $RECON/$seq/
done
```

### 4. Download pre-cached detections

```bash
# E2VID — all 10 demo sequences
curl -L -o /tmp/detections_e2vid_demo.tar.gz \
  https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v5.0/detections_e2vid_demo.tar.gz
tar xzf /tmp/detections_e2vid_demo.tar.gz -C $RECON/

# HyperE2VID (Run 2)
curl -L -o /tmp/detections_hypere2vid_run2.tar.gz \
  https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v3.0/detections_hypere2vid_run2.tar.gz
tar xzf /tmp/detections_hypere2vid_run2.tar.gz -C $RECON/

# Late Fusion (Run 1)
curl -L -o /tmp/detections_fusion_run1.tar.gz \
  https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v4.0/detections_fusion_run1.tar.gz
tar xzf /tmp/detections_fusion_run1.tar.gz -C $RECON/
```

### 5. Download the compose file, GUI files, and write `.env`

> **Important:** Use an absolute path for `RECON` — Docker Compose does not expand `~` or `$HOME`.

```bash
BASE_RAW="https://raw.githubusercontent.com/HongMinhNhat35/Machine-Intelligence-Block-course/gui"

curl -fsSL -o $RECON/docker-compose.yml "$BASE_RAW/docker-compose.deploy.yml"

mkdir -p $RECON/static
for f in app-admin.js app-compare.js app-core.js app-detect.js app-recon.js app-upload.js app-utils.js index.html style.css; do
  curl -fsSL -o $RECON/static/$f "$BASE_RAW/services/web/static/$f"
done

mkdir -p $RECON/kpis
for f in e2vid_run2.json e2vid_run3.json e2vid_run4.json e2vid_run5.json e2vid_run6.json e2vid_run7.json e2vid_run8.json e2vid_run9.json e2vid_run10.json fusion_event_run1.json fusion_event_run2.json fusion_rgb_run1.json fusion_run1.json hypere2vid_run1.json hypere2vid_run2.json; do
  curl -fsSL -o $RECON/kpis/$f "$BASE_RAW/services/web/kpis/$f"
done

cat > $RECON/.env <<EOF
FRED_DATA_PATH=/path/to/fred/sequences    # set to your FRED dataset, or same as RECON_DATA_PATH
RECON_DATA_PATH=$RECON
EOF
```

### 6. Pull images and start

```bash
cd $RECON
docker compose pull
docker compose up -d
```

Open **http://localhost:8080**

---

## Stop

```bash
docker compose down
```

Data in `~/g1-ami-data` is preserved — it survives container restarts.

---

## What you can do in the GUI

| Tab | What it does |
|---|---|
| **Upload** | Select a FRED sequence. The sidebar shows frame count and cached detection count. |
| **Reconstruction** | Browse E2VID and HyperE2VID reconstructed frames side-by-side with playback controls. |
| **Detection** | View YOLO bounding-box overlay. Choose from five models with the bottom buttons: e2vid, HyperE2VID, RGB (FRED RGB frame + fusion detections), Event (raw event frame + fusion detections), Late Fusion (FRED RGB frame + fusion detections). Adjust confidence with the sidebar slider. Detection results are pre-cached — no inference needed. |
| **Comparison** | Three-column view: left column shows e2vid or HyperE2VID (toggle at bottom) — middle and right show FRED RGB frames with fusion detection overlay. Left column is the master time axis; middle and right columns are synchronized automatically using the best available synchronization method (timestamp-based synchronization with automatic fallback calibration). |
| **KPIs** | Summary table (best run per model) plus full detail tables for detection accuracy, reconstruction, and training metrics across all runs. |
| **Dataset** | Per-sequence notes: event counts, annotation timing, known reconstruction quirks. |
| **Admin** | Live health status of all backend services. |

---

## Notes

- Detection results are pre-cached (`detections_e2vid.json`, `detections_hypere2vid.json`, `detections_fusion.json` in each sequence folder). They load instantly. Use the **Run Detection** buttons on the Upload tab to rebuild a cache (e.g. after new model weights).
- The confidence slider is a display filter only — the cache always holds all detections at confidence ≥ 0.1.
- **Early frames (~first 250 frames per sequence):** YOLO bounding boxes may appear over a visually empty region. This is expected — the E2VID LSTM needs ~20 s to warm up from a cold start. The detections are geometrically correct (verified against ground truth, IoU 0.73–0.92); the drone is present but the reconstruction quality is too low for human visibility.
- **Partial detection caches:** If a precompute run was interrupted, the cache may cover only part of the sequence. The Upload screen shows the cached detection count — if it looks low relative to the frame count, click **Run Detection** again to rebuild the cache from scratch. Detection and KPI metrics in the KPIs tab are computed over the full run, not the partial cache.
- Reconstruction (event → frames) is not triggered from the GUI. Frames are pre-generated and included in the downloaded data.
- The FRED raw dataset (`FRED_DATA_PATH`) is only needed for Late Fusion and the Event/RGB detection modes. The e2vid pipeline and comparison view work without it.
