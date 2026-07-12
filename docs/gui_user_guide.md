# Hybrid Vision System — GUI User Guide

## Overview

The GUI provides a browser-based interface to the AMI drone-detection pipeline. Three detection approaches are available:

| Model | What it does |
|-------|-------------|
| **E2VID + YOLO** | Converts the event stream to greyscale frames (E2VID), then runs YOLO detection. Reconstruction was run on Kaggle; frames are stored on disk. |
| **HyperE2VID + YOLO** | Same pipeline as E2VID but uses the HyperE2VID reconstructor (higher quality, lower fps). |
| **Late Fusion** | Runs two separate YOLO models — one trained on FRED RGB camera frames, one on FRED event frames — and merges their detections. Does **not** use reconstructed frames. |

The GUI talks to four backend services running in Docker:

| Service | Port | Role |
|---------|------|------|
| `web` | 8080 | Serves the GUI and proxies API calls |
| `e2vid` | 8001 | YOLO inference on E2VID reconstructed frames |
| `hypere2vid` | 8002 | YOLO inference on HyperE2VID reconstructed frames |
| `fusion` | 8003 | Dual-model YOLO inference on raw FRED frames |

---

## Navigation

The interface has a **tab bar** at the top and a **sidebar** on the left. The sidebar is always visible and shows live sequence info once a dataset is selected.

---

## Sidebar — Sequence Info

After selecting a sequence the sidebar shows stats for each model:

| Field | Meaning |
|-------|---------|
| **Frames** | Reconstructed frame count for that model |
| **Detections @0.1** | Cached detections at confidence ≥ 0.1 (display filter does not change this) |
| **mAP50** | Validation accuracy of the current weights |

---

## Upload

Select a sequence from the browser list. Sequences show colour-coded badges:
- Green `e2vid N` — E2VID reconstruction available, N frames
- Green `hyper N` — HyperE2VID reconstruction available
- Grey badge — frames not yet available for that model

After selecting, the file panel below confirms what data is present (events file, ground-truth coordinates).

### Detection cache — pre-compute buttons

Three buttons allow you to run and cache detections for the selected sequence:

| Button | Model | Prerequisite |
|--------|-------|-------------|
| **E2VID** | e2vid + YOLO | E2VID reconstruction frames on disk |
| **HyperE2VID** | HyperE2VID + YOLO | HyperE2VID reconstruction frames on disk |
| **Late Fusion** | fusion | FRED raw data mounted at `FRED_DATA_PATH`; model weights `fusion_rgb.pt` and `fusion_event.pt` present in the fusion container |

Progress is streamed live. When a run completes the sidebar detection count updates automatically.

---

## Reconstruction

Side-by-side player showing E2VID (left) and HyperE2VID (right) reconstructed frames for the selected sequence. The HyperE2VID panel shows a "no frames" placeholder for sequences without HyperE2VID data.

### Playback controls

| Control | Action |
|---------|--------|
| **Play / Pause** | Continuous playback at ~6 fps |
| **Stop** | Pause and return to frame 0 |
| **‹ ›** | Step one frame backward / forward |
| **Number input** | Jump directly to a frame number |
| **Slider** | Drag to scrub |

---

## Detection

Shows one model's detection overlay at a time. Switch models with the five buttons at the bottom.

### Model buttons

| Button | Source frames | Detection cache | Timeline colour |
|--------|--------------|-----------------|-----------------|
| **e2vid** | E2VID reconstruction | `detections_e2vid.json` | Blue |
| **HyperE2VID** | HyperE2VID reconstruction | `detections_hypere2vid.json` | Purple |
| **RGB** | FRED RGB frame | `detections_fusion.json` (all fusion detections) | Green |
| **Event** | E2VID reconstruction | `detections_fusion.json` (all fusion detections) | Teal |
| **Late Fusion** | FRED RGB frame | `detections_fusion.json` (all fusion detections) | Orange |

RGB and Event modes show the full fusion detection cache because separate per-source caches are not yet available. They differ only in the background frame shown.

### Left panel — source frame

Shows the raw reconstructed frame (or FRED RGB frame for RGB / Late Fusion mode).

### Right panel — detection overlay

Same frame with YOLO bounding boxes. The pill shows how many boxes are visible at the current confidence threshold.

### Confidence timeline

A strip below the frame panels shows the maximum detection confidence at each frame for the active model. The colour matches the model button (see table above). A red vertical line marks the current frame position. The strip updates in real time while scrubbing.

### Confidence threshold (sidebar)

Filters which cached detections are drawn. Never affects the cache itself — the cache always covers confidence ≥ 0.1.

---

## Caching

Detection is slow (minutes per sequence on CPU). Results are saved to
`RECON_DATA_PATH/sequence_XX/detections_{model}.json` and loaded instantly on subsequent visits.

| File | Model |
|------|-------|
| `detections_e2vid.json` | E2VID + YOLO |
| `detections_hypere2vid.json` | HyperE2VID + YOLO |
| `detections_fusion.json` | Late Fusion |

**Re-run triggers:** new model weights pushed; reconstruction frames updated.  
**Never needed:** confidence threshold changes; these are display-only.

---

## Compare v1

Original three-column side-by-side view kept for reference:

| Column | Source frames | Detection source |
|--------|--------------|-----------------|
| E2VID + YOLO | `reconstruction_e2vid/` | `detections_e2vid.json` |
| HyperE2VID + YOLO | `reconstruction_hypere2vid/` | `detections_hypere2vid.json` |
| Late Fusion | FRED RGB frames | `detections_fusion.json` |

Model KPI metrics (mAP@0.5, mAP@0.5:95, Precision, Recall) are shown below each column.

---

## Comparison

Three-column view where the **left column** is the master time axis:

| Column | Source frames | Detection source |
|--------|--------------|-----------------|
| Left (e2vid or HyperE2VID) | Reconstruction frames | Respective detection cache |
| Middle — RGB | FRED RGB frames | `detections_fusion.json` |
| Right — Late Fusion | FRED RGB frames | `detections_fusion.json` |

Toggle the left column between e2vid and HyperE2VID with the buttons at the bottom. Middle and right columns always show FRED RGB frames with fusion detections. The **next detection** button (crosshair icon) jumps to the next frame in the left column that has a detection.

### Detection trail

The **Trail** button toggles a motion trail overlay on all three columns. While active, the last 30 frames of bbox centre positions are drawn as fading gold dots on each panel, with connecting line segments between consecutive frames (nearest-neighbour match). Turning Trail off clears the buffer. The trail resets automatically on sequence change.

### Confidence timeline

A 48 px strip between the image panels and the playback controls shows the maximum detection confidence per frame for two models simultaneously:

| Line | Colour | Source |
|------|--------|--------|
| Left column model (e2vid or HyperE2VID) | Blue | `detections_e2vid.json` or `detections_hypere2vid.json` |
| Fusion model | Orange | `detections_fusion.json` |

The model name is shown as a small legend inside the canvas. A red vertical marker tracks the current frame and moves in real time while scrubbing. The horizontal gridline at 50% confidence is a visual reference.

### Frame synchronisation

The left column drives the E2VID frame index. The slider position maps to middle/right (FRED RGB) frames via **cluster-based linear calibration**:

1. Detection clusters (sustained bursts of ≥ 5 detections in any 10-frame window) are identified in both the e2vid and fusion detection caches.
2. Cluster starts are paired by rank (1st cluster in e2vid → 1st cluster in fusion, etc.).
3. A least-squares linear regression through all matched pairs gives `fusion_frame = slope × e2vid_frame + offset`.

This handles the e2vid burst-skip (where early e2vid frames have no detections and thus map differently from the ratio formula) and improves alignment throughout the sequence, not just at the endpoints. Falls back to a ratio formula when fewer than two clusters can be matched.

The same calibration applies to the HyperE2VID column when left mode is HyperE2VID.

---

## KPIs

Two tables loaded from `kpis/*.json`:

| Table | What it shows |
|-------|---------------|
| **Best results** | One row per model: the run with the highest canonical mAP@0.5. Includes total runtime (reconstruction + training). |
| **Details — Detection accuracy** | All runs: mAP@0.5, mAP@0.5:95, Precision, Recall, total runtime |
| **Details — Reconstruction** | All runs: sequences, events_per_frame, total frames, runtime, fps, GPU |
| **Details — Training** | All runs: dataset split, epochs, learning rate, batch size, GPU |

---

## Admin

Shows live health of all backend services:
- **Green** — healthy, weights loaded
- **Amber** — service running but weights file not found
- **Red** — service unreachable

Also lists data mount paths and weight file locations baked into each container. Check this tab first if detection is not working.

---

## Tips

- **Fusion button is greyed out or fails:** the fusion service either cannot find the FRED raw frames at `FRED_DATA_PATH` or the weight files are missing. Check the Admin tab.
- **Nothing shows in the overlay:** the confidence slider may be too high. Lower it to 0.10.
- **"Run cache first" in Detection (fusion model):** fusion does not support live frame-by-frame mode. Go to Upload, select the sequence, and click **Late Fusion** to build the cache.
- **Comparison fusion column blank:** no `detections_fusion.json` for this sequence yet. Build it from the Upload tab.
- **Fusion boxes look temporally offset:** the ratio mapping assumes both cameras cover exactly the same time window. If the FRED RGB recording started later or ended earlier than the event stream the boxes will drift. This is a known limitation of the ratio approximation.
- **"Reconstruction not available":** frames for that model are not on disk. Reconstruction runs on Kaggle, not in this GUI.
- **"Could not reach … service":** container is unhealthy. Restart: `docker compose down && docker compose up -d`.
- **mAP50 in sidebar shows —:** the KPI fetch failed. Refresh the page.
