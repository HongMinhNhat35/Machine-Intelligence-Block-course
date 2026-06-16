# Group 1 — Hybrid Vision System: Quickstart

Standalone deployment guide for the AMI drone-detection GUI.  
No Python, no Git, no build step — just Docker.

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with Docker Compose v2
- ~20 GB free disk space for the pre-processed sequences

---

## 1. Download the pre-processed sequences

The reconstructed frames and detection results are provided as release assets on GitHub. Download and extract them into a single directory (e.g. `/data/recon`):

```bash
RECON=/data/recon   # change to wherever you want the data
mkdir -p $RECON

# sequence_85 (1.3 GB)
curl -L -o /tmp/sequence_85.tar \
  https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v1.0/sequence_85.tar
tar xf /tmp/sequence_85.tar -C $RECON/sequence_85/ --strip-components=0

# sequence_127 (2.1 GB — two parts)
curl -L -o /tmp/seq127.00 https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v1.0/sequence_127.tar.00
curl -L -o /tmp/seq127.01 https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v1.0/sequence_127.tar.01
cat /tmp/seq127.00 /tmp/seq127.01 | tar x -C $RECON/sequence_127/

# sequence_201 (2.1 GB — two parts)
curl -L -o /tmp/seq201.00 https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v1.0/sequence_201.tar.00
curl -L -o /tmp/seq201.01 https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v1.0/sequence_201.tar.01
cat /tmp/seq201.00 /tmp/seq201.01 | tar x -C $RECON/sequence_201/

# sequence_84 (2.3 GB — two parts)
curl -L -o /tmp/seq84.00 https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v1.0/sequence_84.tar.00
curl -L -o /tmp/seq84.01 https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v1.0/sequence_84.tar.01
cat /tmp/seq84.00 /tmp/seq84.01 | tar x -C $RECON/sequence_84/

# sequence_124 (6.7 GB — four parts)
for i in 00 01 02 03; do
  curl -L -o /tmp/seq124.$i https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v1.0/sequence_124.tar.$i
done
cat /tmp/seq124.* | tar x -C $RECON/sequence_124/
```

Set `RECON` to the same path you'll use in `.env` below.

---

## 2. Download the two config files

```bash
curl -O https://raw.githubusercontent.com/HongMinhNhat35/Machine-Intelligence-Block-course/gui/docker-compose.yml
curl -O https://raw.githubusercontent.com/HongMinhNhat35/Machine-Intelligence-Block-course/gui/.env.example
```

---

## 3. Configure paths

```bash
cp .env.example .env
```

Open `.env` and set the two paths:

| Variable | Description |
|---|---|
| `FRED_DATA_PATH` | Can be set to the same directory as `RECON_DATA_PATH` — the raw FRED files are not needed for the demo. |
| `RECON_DATA_PATH` | The directory where you extracted the sequences in step 1 (e.g. `/data/recon`). |

Example:

```env
FRED_DATA_PATH=/data/recon
RECON_DATA_PATH=/data/recon
```

> **Important:** Use absolute paths. Docker Compose does not expand `~` or `$HOME`, so `~/data/recon` will not work.

---

## 4. Start

```bash
docker compose up -d
```

Docker pulls the pre-built images from GHCR on first run (~1.5 GB total). Once all services are healthy, open:

**http://localhost:8080**

---

## 5. Stop

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
