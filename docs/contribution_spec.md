# Contribution Spec — HyperE2VID & Late Fusion

What needs to be delivered for integration into the AMI Hybrid Vision System.

Please send all files below to Klaus directly. He will integrate them into the services and containers.

---

## HyperE2VID sub-team

### 1. Reconstructed frames (large)

One folder per sequence, named `reconstruction_hypere2vid/`, placed alongside the existing e2vid folder:

```
RECON_DATA_PATH/
  sequence_84/
    reconstruction_e2vid/      ← already exists
    reconstruction_hypere2vid/ ← new
      frame_000000.jpg
      frame_000001.jpg
      ...
      timestamps.txt
  sequence_85/
    ...
```

Same format as e2vid: zero-padded JPEG frames + `timestamps.txt` (one timestamp in seconds per line, matching frame order).

Expected size: ~15 GB for all 5 sequences.

**Delivery:** tar per sequence (split at 1.9 GB if larger), e.g. `hypere2vid_sequence_127.tar.00`, `hypere2vid_sequence_127.tar.01`.

### 2. YOLO weights

File: `hypere2vid_best.pt`

Trained on HyperE2VID-reconstructed frames using `scripts/train_yolo.py` with `--recon_root` pointing to your `reconstruction_hypere2vid/` folders.

### 3. KPI JSON

File name: `hypere2vid_run1.json` (increment run number for subsequent runs).

The fields that the GUI reads are nested under `detection`, `training`, and `reconstruction`. Use the full structure below — fill in actual values from your run:

```json
{
  "schema_version": "1.0",
  "model": "hypere2vid",
  "detector": "YOLOv8s",
  "run_id": "run_1",
  "contributor": "Team HyperE2VID",
  "timestamp": "2026-01-01T00:00:00",
  "note": "Brief description of this run.",

  "reconstruction": {
    "sequences": ["sequence_84", "sequence_85", "sequence_124", "sequence_201", "sequence_127"],
    "total_frames": 0,
    "total_runtime_s": null,
    "avg_fps": null,
    "gpu": null
  },

  "training": {
    "train_sequences": ["sequence_84", "sequence_85", "sequence_124", "sequence_201"],
    "val_sequences": ["sequence_127"],
    "n_train_images": 0,
    "n_val_images": 0,
    "lr0": 0.01,
    "epochs_requested": 100,
    "epochs_completed": 0,
    "best_epoch": 0,
    "n_gpus": 1,
    "batch_per_gpu": 16,
    "effective_batch": 16,
    "imgsz": 640,
    "runtime_s": null,
    "gpu": null
  },

  "detection": {
    "canonical": {
      "map50": 0.0,
      "map50_95": 0.0,
      "precision": 0.0,
      "recall": 0.0
    },
    "challenging": {
      "map50": null,
      "map50_95": null,
      "precision": null,
      "recall": null
    }
  }
}
```

**Required fields:** `model`, `detection.canonical.map50/map50_95/precision/recall`. All others are shown in the KPI table but missing values display as `—`.

---

## Late Fusion sub-team

### 1. Model weights

File: `fusion_best.pt`

### 2. KPI JSON

File name: `fusion_run1.json`. Same structure as above, with `"model": "fusion"`.

### 3. Service implementation

File: `services/fusion/app.py`

---

## Notes on integration

**KPI naming convention:** The GUI reads all `*.json` files from the `kpis/` folder sorted by name. Use `e2vid_run4.json`, `hypere2vid_run1.json`, `fusion_run1.json` — the `model` field in the JSON is what the GUI uses to filter, not the filename.

**Integration checklist:**
- [x] hypere2vid: Frames follow the `reconstruction_hypere2vid/` naming convention
- [x] hypere2vid: `timestamps.txt` exists and line count matches frame count
- [x] KPI JSON includes `"model": "hypere2vid"` or `"model": "fusion"`
