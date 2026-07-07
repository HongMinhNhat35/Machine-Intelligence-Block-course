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

Shows one model's detection overlay at a time. Switch models with the radio buttons.

### Model radio buttons

- **e2vid + YOLO** — Run 7, YOLOv8s, mAP@0.5 = 93.6%
- **HyperE2VID + YOLO** — Run 2, YOLOv8n, mAP@0.5 = 56.7%
- **Late fusion** — Run 1 (YOLOv8n × 2); shows E2VID frame as background with fusion bounding boxes overlaid. Falls back to "run cache first" when no cache exists — live frame-by-frame inference is not supported for fusion because it requires two separate models running simultaneously.

### Left panel — source frame

Shows the raw reconstructed frame. For the fusion model this is the E2VID frame (fusion has no separate reconstruction).

### Right panel — detection overlay

Same frame with YOLO bounding boxes. The pill shows how many boxes are visible at the current confidence threshold.

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

## Comparison

Three-column side-by-side view with synchronised playback:

| Column | Source frames | Detection source |
|--------|--------------|-----------------|
| E2VID + YOLO | `reconstruction_e2vid/` | `detections_e2vid.json` |
| HyperE2VID + YOLO | `reconstruction_hypere2vid/` | `detections_hypere2vid.json` |
| Late Fusion | E2VID frames (background only) | `detections_fusion.json` |

Model KPI metrics (mAP@0.5, mAP@0.5:95, Precision, Recall) are shown below each column. The **next detection** button (crosshair icon) jumps to the nearest frame where any model has a detection.

### Frame synchronisation

**E2VID and HyperE2VID** are produced from the same event stream at the same output rate, so frame N in both columns is exactly the same moment in time.

**Late Fusion** runs on FRED raw frames (the physical RGB camera and event camera), which are captured at a different rate than the E2VID reconstruction. The frame indices in `detections_fusion.json` are FRED frame indices, not E2VID frame indices.

To keep the three columns visually aligned, the GUI uses a **ratio mapping**:

```
fusion_frame = round(slider_position × N_fusion / N_e2vid)
```

where `N_fusion` is the total number of paired FRED frames (derived from the detection cache) and `N_e2vid` is the total number of E2VID frames. This gives approximate temporal alignment proportional to the sequence duration. It is not exact per-frame, but the error is small (typically less than one RGB frame period) when both cameras cover the same recording window.

The "next detection" jump button applies the inverse mapping so the slider always lands on an E2VID frame index.

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
