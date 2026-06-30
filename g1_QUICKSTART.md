# Group 1 — Hybrid Vision System: Quickstart

Standalone deployment guide for the AMI drone-detection GUI.  
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

It then downloads ~1.5 GB of pre-computed E2VID frames (Run 7, events_per_pixel=0.1) and optionally ~485 MB of HyperE2VID frames (Run 1, events_per_frame=46080), writes `docker-compose.yml` and `.env`, and pulls the Docker images.

When it finishes:

```bash
cd ~/g1-ami-data          # or the path you chose
docker compose up -d
```

Open **http://localhost:8080**

> **Warning:** only run `docker compose up -d` once. If the stack is already running, starting it again from a different directory will conflict on port 8080. Stop the existing stack first: `docker compose down`.

---

## Manual installation (if the script fails)

### 1. Create the data directory

```bash
RECON=~/g1-ami-data      # absolute path — change if needed
mkdir -p $RECON
```

### 2. Download and extract pre-computed reconstruction frames

```bash
# sequence_84 (128 MB)
curl -L -o /tmp/sequence_84.tar \
  https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v2.0/sequence_84.tar
mkdir -p $RECON/sequence_84 && tar xf /tmp/sequence_84.tar -C $RECON/sequence_84/

# sequence_85 (83 MB)
curl -L -o /tmp/sequence_85.tar \
  https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v2.0/sequence_85.tar
mkdir -p $RECON/sequence_85 && tar xf /tmp/sequence_85.tar -C $RECON/sequence_85/

# sequence_124 (668 MB)
curl -L -o /tmp/sequence_124.tar \
  https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v2.0/sequence_124.tar
mkdir -p $RECON/sequence_124 && tar xf /tmp/sequence_124.tar -C $RECON/sequence_124/

# sequence_127 (242 MB)
curl -L -o /tmp/sequence_127.tar \
  https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v2.0/sequence_127.tar
mkdir -p $RECON/sequence_127 && tar xf /tmp/sequence_127.tar -C $RECON/sequence_127/

# sequence_201 (326 MB)
curl -L -o /tmp/sequence_201.tar \
  https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v2.0/sequence_201.tar
mkdir -p $RECON/sequence_201 && tar xf /tmp/sequence_201.tar -C $RECON/sequence_201/
```

### 3. Download HyperE2VID frames (optional, ~485 MB)

```bash
for seq in sequence_84 sequence_85 sequence_127 sequence_201; do
  curl -L -o /tmp/${seq}_hypere2vid.tar \
    https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v2.0/${seq}_hypere2vid.tar
  tar xf /tmp/${seq}_hypere2vid.tar -C $RECON/$seq/
done
```

### 4. Download pre-cached detections

```bash
curl -L -o /tmp/detections_e2vid_run7.tar.gz \
  https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v2.0/detections_e2vid_run7.tar.gz
tar xzf /tmp/detections_e2vid_run7.tar.gz -C $RECON/
```

### 5. Download the compose file and write `.env`

> **Important:** Use an absolute path for `RECON` — Docker Compose does not expand `~` or `$HOME`.

```bash
curl -fsSL -o $RECON/docker-compose.yml \
  https://raw.githubusercontent.com/HongMinhNhat35/Machine-Intelligence-Block-course/gui/docker-compose.yml

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
| **Detection** | View YOLO bounding-box overlay for E2VID or HyperE2VID frames. Switch model with the radio buttons. Adjust confidence threshold with the sidebar slider. Detection results are pre-cached — no need to run inference. |
| **Comparison** | Side-by-side E2VID vs HyperE2VID view with bounding boxes and model KPIs. Late Fusion column not yet available. |
| **KPIs** | Detection accuracy and training metrics table across all runs. |
| **Dataset** | Per-sequence notes: event counts, annotation timing, known reconstruction quirks. |
| **Admin** | Live health status of all backend services. |

---

## Notes

- Detection results are pre-cached (`detections_e2vid.json` in each sequence folder). They load instantly. Click **Run Detection** only to re-run inference (e.g. after new model weights).
- The confidence slider is a display filter only — the cache always holds all detections at confidence ≥ 0.1.
- **Early frames (~first 250 frames per sequence):** YOLO bounding boxes may appear over a visually empty region. This is expected — the E2VID LSTM needs ~20 s to warm up from a cold start. The detections are geometrically correct (verified against ground truth, IoU 0.73–0.92); the drone is present but the reconstruction quality is too low for human visibility.
- **Partial detection caches:** If a precompute run was interrupted, the cache may cover only part of the sequence. The Upload screen shows the cached detection count — if it looks low relative to the frame count, click **Run Detection** again to rebuild the cache from scratch. Detection and KPI metrics in the KPIs tab are computed over the full run, not the partial cache.
- Reconstruction (event → frames) is not triggered from the GUI. Frames are pre-generated and included in the downloaded data.
- The FRED raw dataset (`FRED_DATA_PATH`) is only needed for Late Fusion. The e2vid pipeline and comparison view work without it.
