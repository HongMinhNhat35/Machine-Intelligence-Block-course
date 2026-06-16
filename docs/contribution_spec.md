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

Use this structure — the `model` field is required for the GUI to pick it up:

```json
{
  "stage": "training",
  "model": "hypere2vid",
  "train_sequences": ["sequence_84", "sequence_85", "sequence_124", "sequence_201"],
  "val_sequences": ["sequence_127"],
  "map50": 0.0,
  "map50_95": 0.0,
  "precision": 0.0,
  "recall": 0.0,
  "epochs_completed": 0,
  "best_epoch": 0,
  "imgsz": 640,
  "timestamp": "2026-01-01T00:00:00"
}
```

---

## Late Fusion sub-team

### 1. Model weights

File: `fusion_best.pt`

### 2. KPI JSON

File name: `fusion_run1.json`. Same structure, `"model": "fusion"`.

### 3. Service implementation

File: `services/fusion/app.py`

---

## Notes for Klaus on integration

**Services (app.py):** The hypere2vid and fusion services follow the same API contract as `services/e2vid/app.py`. Once the weights arrive, implementing them requires no special knowledge — just copy `services/e2vid/app.py`, point the frame path at `reconstruction_hypere2vid/` and the weights at `hypere2vid_best.pt`. Late Fusion is more complex (depends on the fusion approach used).

**KPI naming convention:** The GUI reads all `*.json` files from the `kpis/` folder sorted by name. Use `e2vid_run4.json`, `hypere2vid_run1.json`, `fusion_run1.json` — the `model` field in the JSON is what the GUI uses to filter, not the filename.

**Integration checklist:**
- [ ] Frames follow the `reconstruction_hypere2vid/` naming convention
- [ ] `timestamps.txt` exists and line count matches frame count
- [ ] Weights file loads without error in the service container
- [ ] KPI JSON includes `"model": "hypere2vid"` or `"model": "fusion"`
