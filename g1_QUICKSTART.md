# Group 1 — Hybrid Vision System: Quickstart

Standalone deployment guide for the AMI drone-detection GUI.  
No Python, no Git, no build step — just Docker.

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with Docker Compose v2
- ~20 GB free disk space

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

It then downloads ~15 GB of pre-processed sequences, writes `docker-compose.yml` and `.env`, and pulls the Docker images.

When it finishes:

```bash
cd ~/g1-ami-data          # or the path you chose
docker compose up -d
```

Open **http://localhost:8080**

---

## Manual installation (if the script fails)

### 1. Create the data directory

```bash
RECON=~/g1-ami-data      # absolute path — change if needed
mkdir -p $RECON
```

### 2. Download and extract the sequences

```bash
# sequence_85 (1.3 GB)
curl -L -o /tmp/sequence_85.tar \
  https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v1.0/sequence_85.tar
mkdir -p $RECON/sequence_85 && tar xf /tmp/sequence_85.tar -C $RECON/sequence_85/

# sequence_127 (2.1 GB — two parts)
curl -L -o /tmp/seq127.00 https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v1.0/sequence_127.tar.00
curl -L -o /tmp/seq127.01 https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v1.0/sequence_127.tar.01
mkdir -p $RECON/sequence_127 && cat /tmp/seq127.* | tar x -C $RECON/sequence_127/

# sequence_201 (2.1 GB — two parts)
curl -L -o /tmp/seq201.00 https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v1.0/sequence_201.tar.00
curl -L -o /tmp/seq201.01 https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v1.0/sequence_201.tar.01
mkdir -p $RECON/sequence_201 && cat /tmp/seq201.* | tar x -C $RECON/sequence_201/

# sequence_84 (2.3 GB — two parts)
curl -L -o /tmp/seq84.00 https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v1.0/sequence_84.tar.00
curl -L -o /tmp/seq84.01 https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v1.0/sequence_84.tar.01
mkdir -p $RECON/sequence_84 && cat /tmp/seq84.* | tar x -C $RECON/sequence_84/

# sequence_124 (6.7 GB — four parts)
for i in 00 01 02 03; do
  curl -L -o /tmp/seq124.$i https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v1.0/sequence_124.tar.$i
done
mkdir -p $RECON/sequence_124 && cat /tmp/seq124.* | tar x -C $RECON/sequence_124/

# HyperE2VID reconstructions — not yet available, uncomment when released
# curl -L -o /tmp/hypere2vid_sequence_85.tar \
#   https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v1.0/hypere2vid_sequence_85.tar
# mkdir -p $RECON/sequence_85 && tar xf /tmp/hypere2vid_sequence_85.tar -C $RECON/sequence_85/
# (repeat for sequence_84, sequence_127, sequence_201, sequence_124)
```

### 3. Download pre-cached detections

```bash
curl -L -o /tmp/detections_e2vid_run5.tar.gz \
  https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v1.0/detections_e2vid_run5.tar.gz
tar xzf /tmp/detections_e2vid_run5.tar.gz -C $RECON/
```

### 4. Download the compose file and write `.env`

> **Important:** Use an absolute path for `RECON` — Docker Compose does not expand `~` or `$HOME`.

```bash
curl -fsSL -o $RECON/docker-compose.yml \
  https://raw.githubusercontent.com/HongMinhNhat35/Machine-Intelligence-Block-course/gui/docker-compose.yml

cat > $RECON/.env <<EOF
FRED_DATA_PATH=/path/to/fred/sequences    # set to your FRED dataset, or same as RECON_DATA_PATH
RECON_DATA_PATH=$RECON
EOF
```

### 5. Pull images and start

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
| **Reconstruction** | Browse reconstructed e2vid frames with playback controls. |
| **Detection** | View YOLO bounding-box overlay. Adjust confidence threshold with the sidebar slider. Detection results are pre-cached — no need to run inference. |
| **Comparison** | Side-by-side E2VID vs HyperE2VID view with bounding boxes and model KPIs. Late Fusion column not yet available. |
| **KPIs** | Detection accuracy and training metrics table. |
| **Admin** | Live health status of all backend services. |

---

## Notes

- Detection results are pre-cached (`detections_e2vid.json` in each sequence folder). They load instantly. Click **Run Detection** only to re-run inference (e.g. after new model weights).
- The confidence slider is a display filter only — the cache always holds all detections at confidence ≥ 0.1.
- Reconstruction (event → frames) is not triggered from the GUI. Frames are pre-generated and included in the downloaded data.
- The FRED raw dataset (`FRED_DATA_PATH`) is only needed for Late Fusion. The e2vid pipeline and comparison view work without it.
