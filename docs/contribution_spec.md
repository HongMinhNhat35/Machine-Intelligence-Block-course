# Contribution Spec — HyperE2VID & Late Fusion

What each sub-team needs to deliver for integration into the AMI Hybrid Vision System.

---

## HyperE2VID sub-team

### 1. Reconstructed frames (large)

One folder per sequence, named `reconstruction_hypere2vid/`, placed alongside the existing e2vid folder:

```
RECON_DATA_PATH/
  sequence_84/
    reconstruction_e2vid/      ← already exists
    reconstruction_hypere2vid/ ← new, from your team
      frame_000000.jpg
      frame_000001.jpg
      ...
      timestamps.txt
  sequence_85/
    ...
```

Same format as e2vid: zero-padded JPEG frames + `timestamps.txt` (one timestamp in seconds per line, matching frame order).

Expected size: ~15 GB for all 5 sequences.

**Delivery:** tar per sequence (same split as e2vid if > 2 GB), uploaded as GitHub release assets. We will add them to the installer script.

### 2. YOLO weights

File: `hypere2vid_best.pt`  
Destination: `services/hypere2vid/weights/hypere2vid_best.pt`

Trained on HyperE2VID-reconstructed frames using the same YOLO training pipeline (`scripts/train_yolo.py`). Use `--recon_root` pointing to your `reconstruction_hypere2vid/` folders.

### 3. KPI JSON

One file in the same format as `data/kpis/train_yolo.json`. At minimum include:

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

Destination: `data/kpis/train_hypere2vid.json`

### 4. Service implementation

File: `services/hypere2vid/app.py`

Must implement the same API contract as `services/e2vid/app.py`:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Returns `{"status": "ok", "weights_loaded": true/false}` |
| `/detect` | POST | Accepts `{"sequence_id": "...", "frame_start": 0, "frame_end": N}`, returns SSE stream with `progress` and `done` events |

The detection cache must be written to:
```
RECON_DATA_PATH/<sequence_id>/detections_hypere2vid.json
```

---

## Late Fusion sub-team

### 1. Model weights

File: `fusion_best.pt`  
Destination: `services/fusion/weights/fusion_best.pt`

### 2. KPI JSON

Same format as above, saved to `data/kpis/train_fusion.json`. Include which input modalities were fused and any additional fields relevant to the fusion approach.

### 3. Service implementation

File: `services/fusion/app.py`

Same API contract as above (`/health`, `/detect`). Detection cache written to:
```
RECON_DATA_PATH/<sequence_id>/detections_fusion.json
```

---

## Integration checklist

Before handing off, verify:

- [ ] Frames follow the `reconstruction_hypere2vid/` naming convention
- [ ] `timestamps.txt` exists and has the same number of lines as frames
- [ ] Weights file loads without error in the service container
- [ ] `/health` returns `weights_loaded: true`
- [ ] `/detect` streams SSE events and writes the cache JSON on completion
- [ ] KPI JSON is valid and parseable by `GET /api/kpis`
