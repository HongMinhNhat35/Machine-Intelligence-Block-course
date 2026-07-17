# Hybrid Vision System — GUI User Guide

## Overview

The GUI provides a browser-based interface to the AMI drone-detection pipeline. Three detection approaches are available:

| Model | What it does |
|-------|-------------|
| **E2VID + YOLO** | Converts the event stream to greyscale frames (E2VID), then runs YOLO detection. Reconstruction was run on Kaggle; frames are stored on disk. |
| **HyperE2VID + YOLO** | Same pipeline as E2VID but uses the HyperE2VID reconstructor. |
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

The interface has a **tab bar** at the top and a **sidebar** on the left. The sidebar is always visible and shows live sequence info once a dataset is selected. The available screens are: Home, Upload, Reconstruction, Detection, Comparison, KPIs, Fusion Methods, and Admin.

---

## Sidebar — Sequence Info

When on the Detection or Comparison screens, a **Confidence** slider appears near the top of the sidebar (above the sequence stats). This keeps it reachable on small displays regardless of how much sequence data is shown below it.

After selecting a sequence the sidebar shows stats for each model:

| Field | Meaning |
|-------|---------|
| **Frames** | Reconstructed frame count for that model |
| **Detections @0.1** | Total detections in the cache at confidence ≥ 0.1 |
| **mAP50** | Canonical test accuracy of the featured model weights |

---

## Upload

Select a sequence from the browser list. Sequences show colour-coded badges:
- Green `e2vid N` — E2VID reconstruction available, N frames
- Green `hyper N` — HyperE2VID reconstruction available
- Grey badge — frames not yet available for that model

An info box below the browser shows the **Proposed datasets** — the sequences selected for the demo:
- **Test set** (e2vid): 8, 9, 12, 20, 21
- **Prototyping set** (e2vid + HyperE2VID): 84, 85, **124**, **127**, **201**

### Detection cache — pre-compute buttons

Three buttons allow you to run and cache detections for the selected sequence. This is not strictly required as the detection and comparison screens also run on-the-fly inference, but cached mode is faster and smoother.
The GUI comes with pre-cached detections for all 10 demo sequences: test set (8, 9, 12, 20, 21) and prototyping set (84, 85, 124, 127, 201).

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
| **Slider** | Drag to scrub through the sequence |

---

## Detection

Shows one model's detection overlay at a time. Switch models with the five buttons at the bottom.

### Model buttons

| Button | Source frames | Detection cache | Timeline colour |
|--------|--------------|-----------------|-----------------|
| **e2vid** | E2VID reconstruction | `detections_e2vid.json` | Blue |
| **HyperE2VID** | HyperE2VID reconstruction | `detections_hypere2vid.json` | Purple |
| **RGB** | FRED RGB frames | `detections_fusion.json` (RGB detections only) | Green |
| **Event** | FRED raw event frames | `detections_fusion.json` (event detections only) | Teal |
| **Late Fusion** | FRED RGB frames | `detections_fusion.json` (all detections) | Orange |

### Left panel — source frame

Shows the raw reconstructed frame (or FRED RGB / event frame for fusion modes).

### Right panel — detection overlay

Same frame with YOLO bounding boxes. The pill shows how many boxes are visible at the current confidence threshold.

### Playback controls

| Control | Action |
|---------|--------|
| **Play / Pause** | Continuous playback at ~6 fps |
| **Stop** | Pause and return to frame 0 |
| **‹ ›** | Step one frame backward / forward |
| **⊹ (crosshair)** | Jump to the next frame that has a detection above the confidence threshold (cached mode only) |
| **Trail** | Toggle motion trail — draws the last 30 frames of bbox centre positions as fading dots |
| **Number input** | Jump directly to a frame number |
| **Slider** | Drag to scrub through the sequence |

### Confidence timeline

A strip below the frame panels shows the maximum detection confidence at each frame for the active model. The colour matches the model button (see table above). A red vertical line marks the current frame position. The strip updates in real time while scrubbing.

### Confidence threshold (sidebar)

Filters which cached detections are drawn. Never affects the cache itself — the cache always covers confidence ≥ 0.1.

---

## Comparison

Three-column view where the **left column** is the master time axis:

| Column | Source frames | Detection source |
|--------|--------------|-----------------|
| Left (e2vid or HyperE2VID) | Reconstruction frames | Respective detection cache |
| Middle — RGB | FRED RGB frames | `detections_fusion.json` |
| Right — Late Fusion | FRED RGB frames | `detections_fusion.json` |

Toggle the left column between e2vid and HyperE2VID with the buttons at the bottom. Middle and right columns always show FRED RGB frames with fusion detections.

### Playback controls

| Control | Action |
|---------|--------|
| **Play / Pause** | Continuous playback at ~6 fps |
| **Stop** | Pause and return to frame 0 |
| **‹ ›** | Step one frame backward / forward |
| **⊹ (crosshair)** | Jump to the next frame in the left column that has a detection above the confidence threshold |
| **Trail** | Toggle motion trail on all three columns — last 30 bbox centres drawn as fading gold dots with connecting segments |
| **Number input** | Jump directly to a frame number |
| **Slider** | Drag to scrub through the sequence |

### Detection trail

While active, the last 30 frames of bbox centre positions are drawn as fading gold dots on each panel, with connecting line segments between consecutive frames (nearest-neighbour match). Turning Trail off clears the buffer. The trail resets automatically on sequence change.

### Confidence timeline

A strip labelled **Detection confidence per frame** between the image panels and the playback controls shows the maximum detection confidence at each frame for three models simultaneously:

| Line | Colour | Source |
|------|--------|--------|
| Left column model (e2vid or HyperE2VID) | Blue | `detections_e2vid.json` or `detections_hypere2vid.json` |
| RGB detector | Green | `detections_fusion_rgb.json` |
| Late Fusion | Orange | `detections_fusion.json` |

Each line rises toward the top when the model detects a drone with high confidence, and drops to the bottom when there is no detection. The model names are shown as a small legend inside the canvas. A red vertical marker tracks the current frame and moves in real time while scrubbing. The horizontal gridline at 50% confidence is a visual reference.

### Frame synchronisation

The left column drives the E2VID frame index. The slider position maps to middle/right (FRED RGB) frames using the best available method:

1. **Timestamp sync (primary)** — if `/api/sync` returns exact event timestamps for the sequence, each e2vid frame is mapped to the closest RGB frame by timestamp. This is the most accurate method.
2. **Cluster-based linear calibration (fallback)** — when timestamp data is unavailable, slope and offset are computed from the detection caches:
   - **Slope**: ratio of the secondary stream's detection span (first sustained cluster start → last sustained cluster end) to the primary stream's detection span. Sustained clusters require ≥ 5 detections in any 10-frame window and span ≥ 30 frames, which filters out noise blips.
   - **Offset**: a ±300-frame cross-correlation search around the span anchor picks the offset that maximises the number of primary detection frames that land on a secondary detection frame.
3. **Ratio fallback** — if calibration also fails (too few detections or no clusters), falls back to a simple frame-count ratio.

The same calibration applies to the HyperE2VID column when left mode is HyperE2VID.

---

## KPIs

Loaded from `kpis/*.json`. The summary table shows the **deployed** run per model (the run marked `"deployed": true`), falling back to the `"featured": true` run, or the highest mAP@0.5 if neither is set.

Five jump buttons at the top of the screen scroll directly to each section:

| Button | Section |
|--------|---------|
| **Best results** | One row per model: the deployed/featured run. Includes total runtime (reconstruction + training). |
| **Details** | All runs sorted by model then run number: mAP@0.5, mAP@0.5:95, Precision, Recall, total runtime |
| **Reconstruction** | All runs sorted by model then run number: sequences, events_per_pixel, total frames, runtime, fps, GPU |
| **Training** | All runs sorted by model then run number: dataset split, epochs, learning rate, batch size, GPU |
| **Dataset analysis** | FRED sequences used for training and validation: duration, event count, frame count, first annotation offset, and per-sequence notes |

---

## Fusion Methods

Documents the late fusion evaluation — how the deployed fusion layer was developed and selected.

### Pipeline

Shows the two-branch architecture: event frames and RGB frames each pass through their own YOLO model, producing independent detections. The fusion layer merges these into a single output. YOLO weights are fixed; only the fusion layer parameters were tuned.

### Evaluation stages

Three cards document the progression from baseline to deployed:

| Stage | What it is |
|-------|-----------|
| **Stage 1 — Run 2 (WBF baseline)** | Weighted Box Fusion (WBF) with hand-picked confidence thresholds. The starting point. |
| **Stage 2 — Phase 1 grid search** | Systematic search over 1,250 WBF configurations (event_t, rgb_t, iou_t, alpha, weighted_box). Best config lifted mAP@0.5 from 50.7% to 57.3%. |
| **Stage 3 — Run 3 (deployed)** | 486 additional configurations adding temporal smoothing on top of the phase 1 best. Detections consistent across adjacent frames (IoU ≥ 0.2) receive a confidence boost (×1.3); isolated single-frame detections are decayed (×0.8). Final mAP@0.5: 58.3%. |

### Results overview

A compact table beside the pipeline diagram shows the three-step progression (Run 2 → Phase 1 best → Run 3) with mAP@0.5 and the gain over the baseline at a glance. The full comparison table with all metrics is shown below the stage cards.

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
- **Fusion boxes look temporally offset:** sync uses exact per-frame timestamps when available (all 10 demo sequences include `timestamps.txt`). If you add a custom sequence without `timestamps.txt`, sync falls back to cluster-based calibration which may drift slightly.
- **"Reconstruction not available":** frames for that model are not on disk. Reconstruction runs on Kaggle, not in this GUI.
- **"Could not reach … service":** container is unhealthy. Restart: `docker compose down && docker compose up -d`.
- **mAP50 in sidebar shows —:** the KPI fetch failed. Refresh the page.
