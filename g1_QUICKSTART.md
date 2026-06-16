# Group 1 — Hybrid Vision System: Quickstart

Standalone deployment guide for the AMI drone-detection GUI.  
No Python, no Git, no build step — just Docker.

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with Docker Compose v2
- ~20 GB free disk space

---

## Install

Download and run the installer:

```bash
curl -O https://raw.githubusercontent.com/HongMinhNhat35/Machine-Intelligence-Block-course/gui/g1-install.sh
bash g1-install.sh
```

The script will:
1. Ask where to store the data (default: `~/g1-ami-data`)
2. Ask where the FRED raw dataset is (press Enter if you don't have it — the demo works without it)
3. Download ~15 GB of pre-processed sequences from GitHub Releases
4. Download `docker-compose.yml` and write `.env`
5. Pull the Docker images

---

## Start

```bash
cd ~/g1-ami-data          # or whatever path you chose during install
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
| **Reconstruction** | Browse reconstructed e2vid frames with playback controls. |
| **Detection** | View YOLO bounding-box overlay. Adjust confidence threshold with the sidebar slider. Detection results are pre-cached — no need to run inference. |
| **Comparison** | Side-by-side view of all three pipelines (HyperE2VID and Late Fusion show "not yet available"). |
| **KPIs** | Detection accuracy and training metrics table. |
| **Admin** | Live health status of all backend services. |

---

## Notes

- Detection results are pre-cached in each sequence folder (`detections_e2vid.json`). They load instantly on first visit. Click **Run Detection** only if you want to re-run inference (e.g. after new model weights).
- The confidence slider is a display filter only — it does not change what is stored in the cache.
- Reconstruction (event → frames) is not triggered from the GUI. Frames are pre-generated and included in the downloaded data.
