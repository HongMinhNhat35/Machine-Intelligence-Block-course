# Hybrid Vision System — GUI User Guide

## Overview

The GUI gives you a browser-based interface to the AMI drone-detection pipeline. The pipeline has two stages:

1. **Reconstruction** — event-camera data (`.h5`) is converted to greyscale frames using **E2VID**. This was run once on Kaggle and the frames are stored on disk.
2. **Detection** — reconstructed frames are passed through a **YOLOv8s** model (Run 5) trained on the FRED dataset. Results are cached to disk and visualised here.

The GUI talks to three backend services running in Docker:

| Service | Port | Role |
|---------|------|------|
| `web` | 8080 | Serves the GUI and proxies API calls |
| `e2vid` | 8001 | Runs YOLO inference on reconstructed frames |
| `hypere2vid` | 8002 | Implemented; no reconstruction data available yet |
| `fusion` | 8003 | Not yet implemented |

---

## Navigation

The interface has a **tab bar** at the top and a **sidebar** on the left. Both do the same thing — switching tabs changes the main panel. The sidebar is always visible and additionally shows live sequence info once a dataset is selected.

---

## Sidebar — Sequence Info

As soon as you select a sequence in **Upload**, the following appears in the sidebar and stays visible on every tab:

| Field | Meaning |
|-------|---------|
| **ID** | Sequence identifier (e.g. `sequence_85`) |
| **Frames** | Total number of reconstructed frames |
| **Detections @0.1** | Total detections stored in the cache at confidence ≥ 0.1 (all detections, before any display filter) |
| **mAP50** | Validation accuracy of the current YOLO weights (Run 5: 79.2%) |

---

## Upload

Select a sequence from the list. Sequences with reconstructed frames show a green **N frames** badge. Sequences without frames show **not reconstructed** — detection cannot be run on these.

After selecting, the detected files panel below the list confirms what data is available:
- **events.h5** — raw event stream (used as input to E2VID)
- **coordinates.txt** — ground-truth bounding-box annotations
- **e2vid reconstruction** — whether the reconstructed frames exist and how many

---

## Reconstruction

Shows the E2VID-reconstructed frames. A second panel to the right shows HyperE2VID frames when reconstruction data is available for that sequence (currently not available for any sequence).

### Playback controls

| Control | Action |
|---------|--------|
| **Play / Pause** | Start or pause continuous playback at ~6 fps |
| **Stop** | Pause and return to frame 0 |
| **‹ ›** | Step one frame backward or forward |
| **Number input** | Type any frame number and press Enter to jump directly |
| **Slider** | Drag to scrub through all frames |

---

## Detection

Runs YOLO drone detection on the reconstructed frames and overlays bounding boxes.

### Left panel — original frame
Shows the raw reconstructed frame as produced by E2VID.

### Right panel — detection overlay
Shows the same frame with YOLO bounding boxes drawn. Each box shows the class name and confidence score. The pill in the header shows how many boxes are visible at the current confidence threshold.

### Playback controls
Same as Reconstruction (Play, Stop, ‹ ›, number input, slider).

### Confidence threshold (sidebar)
The slider filters which stored detections are drawn. It does **not** affect what is stored in the cache — the cache always holds everything at confidence ≥ 0.1. Moving the slider immediately updates the overlay without any server call.

A value of **0.20** (default) is a good starting point. At 0.10 you will see many low-confidence false positives; at 0.40+ the display becomes much cleaner but may miss real detections.

### Model selection
Three radio buttons select which detection pipeline to use:
- **e2vid + YOLO** — fully functional; Run 5 weights (YOLOv8s, mAP50 = 79.2%)
- **HyperE2VID + YOLO** — service implemented; reconstruction data not yet available
- **Late fusion** — not yet implemented

### Run Detection
Triggers a full detection pass over all frames of the selected sequence. **This always runs from scratch and overwrites any existing cache.**

Progress is streamed live in the status line below the button:
```
Processing… 320/6777 frames (4%)
```

When finished, the overlay and sidebar detection count update automatically.

---

## Caching

Detection is slow (several minutes on CPU). Results are saved to disk so subsequent visits load instantly.

### How it works

- **First visit** to a sequence with no cache: the sidebar shows `Detections @0.1: —`. Navigate to Detection and click **Run Detection**.
- **Subsequent visits**: the cache is loaded automatically as soon as you switch to the Detection tab. No button press needed.
- **Force re-run** (e.g. after new model weights): click **Run Detection** again. It always overwrites the existing cache regardless of whether one exists.

### What is cached

One file per sequence per model: `detections_{model}.json` stored in
`RECON_DATA_PATH/sequence_XX/`. It contains every detection with confidence ≥ 0.1.
The confidence slider is a pure display filter and never invalidates the cache.

### When to re-run detection

- After training a new YOLO model and pushing new container images.
- If the E2VID reconstruction was updated (new frames added or frame count changed).
- Never needed for confidence threshold changes — the cache covers the full 0.1–1.0 range.

### Cache file sizes

Roughly 1–4 MB per sequence (7 sequences ≈ 13 MB total, 2.4 MB compressed).
Pre-cached detections are included in the release download (`detections_e2vid_run5.tar.gz`).

---

## Comparison

Side-by-side view of E2VID (left) and HyperE2VID (right) with synchronised playback and bounding box overlays. Model KPI metrics are shown below each panel.

- **E2VID column** — fully functional; uses cached `detections_e2vid.json`
- **HyperE2VID column** — column is wired up but frames and detections are not yet available (no reconstruction data)
- **Split selector** — toggle between Canonical and Challenging evaluation splits for the KPI metrics

---

## KPIs

Three tables summarising results from all training runs loaded from `kpis/`.

| Table | What it shows |
|-------|---------------|
| **Detection accuracy** | mAP@0.5, mAP@0.5:95, Precision, Recall on the validation set (sequence_127) per run |
| **Reconstruction** | Total frames reconstructed, runtime, GPU used |
| **Training** | Dataset split, epochs, learning rate, batch size, GPU, best epoch |

---

## Admin

Shows live status of all three backend services (green = healthy, amber = weights missing, red = unreachable) and lists data paths and model settings baked into the container.

Use this tab to verify the correct weight files are loaded before running detection.

---

## Tips

- **Nothing shows in the overlay** after detection finishes: the confidence slider may be set too high. Try lowering it to 0.10.
- **"Reconstruction not available"** message: the sequence has no reconstructed frames. Reconstruction is run on Kaggle, not in this GUI — contact the system administrator.
- **"Could not reach e2vid service"**: the e2vid container is unhealthy. Check the Admin tab and restart the Docker stack (`docker compose down && docker compose up -d`).
- **Detections count in sidebar seems high**: remember it counts all stored detections at conf ≥ 0.1 across all frames, including low-confidence ones. Use the confidence slider to filter.
- **mAP50 in sidebar shows —**: the KPI fetch failed. Try refreshing the page.
