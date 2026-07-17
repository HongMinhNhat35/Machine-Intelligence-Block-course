# Group 1 — Hybrid Vision System: Quickstart

Standalone deployment guide for the AMI drone-detection GUI.  
This guide installs a fully functional demonstration environment with precomputed reconstructions, cached detections and evaluation results.  
No Python, no Git, no build step — just Docker.

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with Docker Compose v2
- ~4 GB free disk space (E2VID + HyperE2VID frames)
- An LRZ account with a **Personal Access Token** (`read_registry`, `read_api`, `read_repository` scopes)

Create a token at: https://gitlab.lrz.de/-/user_settings/personal_access_tokens

## Quick install (recommended)

Set your token as an environment variable first — the installer will pick it up automatically so you won't be prompted again:

```bash
export GL_TOKEN=<your-token>
curl --header "PRIVATE-TOKEN: $GL_TOKEN" -o g1-install.sh "https://gitlab.lrz.de/api/v4/projects/269843/repository/files/g1-install.sh/raw?ref=gui"
bash g1-install.sh
```

The script will ask:
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

### 2. Download reconstruction frames and detection caches

```bash
TOKEN=YOUR_TOKEN
PKG="https://gitlab.lrz.de/api/v4/projects/269843/packages/generic/data/1.0"

# E2VID (Run 9) + HyperE2VID (Run 2) — all sequences (~1.6 GB)
curl -L --progress-bar --header "PRIVATE-TOKEN: $TOKEN" \
  -o /tmp/recon_frames.tar.gz "$PKG/recon_frames.tar.gz"
tar xzf /tmp/recon_frames.tar.gz -C $RECON/
rm /tmp/recon_frames.tar.gz

# Detection caches — all models, all demo sequences (~5 MB)
curl -L --progress-bar --header "PRIVATE-TOKEN: $TOKEN" \
  -o /tmp/detections_all.tar.gz "$PKG/detections_all_sequences.tar.gz"
tar xzf /tmp/detections_all.tar.gz -C $RECON/
rm /tmp/detections_all.tar.gz
```

### 3. Download the compose file, GUI files, and write `.env`

> **Important:** Use an absolute path for `RECON` — Docker Compose does not expand `~` or `$HOME`.

```bash
TOKEN=YOUR_TOKEN
GL_RAW="https://gitlab.lrz.de/ldv/teaching/ami/ami2026/group01/-/raw/gui"

curl -fsSL --header "PRIVATE-TOKEN: $TOKEN" -o $RECON/docker-compose.yml "$GL_RAW/docker-compose.deploy.yml"

mkdir -p $RECON/static
for f in app-admin.js app-compare.js app-core.js app-detect.js app-recon.js app-upload.js app-utils.js index.html style.css gui_user_guide.html fusion_viz.jpeg; do
  curl -fsSL --header "PRIVATE-TOKEN: $TOKEN" -o $RECON/static/$f "$GL_RAW/services/web/static/$f"
done

mkdir -p $RECON/kpis
for f in e2vid_run2.json e2vid_run3.json e2vid_run4.json e2vid_run5.json e2vid_run6.json e2vid_run7.json e2vid_run8.json e2vid_run9.json e2vid_run10.json fusion_event_run1.json fusion_event_run2.json fusion_rgb_run1.json fusion_rgb_run2.json fusion_run1.json fusion_run2.json hypere2vid_run1.json hypere2vid_run2.json; do
  curl -fsSL --header "PRIVATE-TOKEN: $TOKEN" -o $RECON/kpis/$f "$GL_RAW/services/web/kpis/$f"
done

cat > $RECON/.env <<EOF
FRED_DATA_PATH=/path/to/fred/sequences    # set to your FRED dataset, or same as RECON_DATA_PATH
RECON_DATA_PATH=$RECON
EOF
```

### 4. Pull images and start

```bash
echo YOUR_TOKEN | docker login gitlab.lrz.de:5005 --username YOUR_LRZ_USERNAME --password-stdin
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
