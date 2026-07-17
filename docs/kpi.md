# KPI Results

This directory contains one JSON file per training run. Results are displayed in the GUI's KPIs screen automatically after each image rebuild.

## Naming convention

```
{model}_{run_id}.json
```

Examples: `e2vid_run2.json`, `hypere2vid_run1.json`, `fusion_run1.json`

## How to contribute your results

1. Copy `e2vid_run2.json` as a starting point and rename it for your model/run.
2. Fill in all fields you have. Set unknown or not-yet-run fields to `null` — **do not omit fields**.
3. Hand the file to Klaus. It will be added to the next Docker image build and appear in the GUI automatically.

## Schema

```json
{
  "schema_version": "1.0",
  "model": "hypere2vid",          // reconstruction model: e2vid | hypere2vid | fusion
  "detector": "YOLOv8n",          // detection model used
  "run_id": "run_1",              // run_1, run_2, …
  "contributor": "Your Name",
  "timestamp": "2026-06-13T12:00:00",
  "note": "Short description.",

  "reconstruction": {
    "sequences": [...],           // list of reconstructed sequences
    "total_frames": null,         // total frames across all sequences
    "total_runtime_s": null,      // total wall-clock time in seconds
    "avg_fps": null,              // average frames per second
    "gpu": null                   // e.g. "Tesla T4"
  },

  "training": {
    "train_sequences": [...],     // sequences used for training
    "val_sequences": [...],       // sequences used for validation — use ["sequence_127"]
    "n_train_images": null,
    "n_val_images": null,
    "lr0": null,                  // initial learning rate
    "epochs_requested": null,
    "epochs_completed": null,     // may be less than requested due to early stopping
    "best_epoch": null,
    "n_gpus": null,
    "batch_per_gpu": null,
    "effective_batch": null,      // n_gpus × batch_per_gpu
    "imgsz": null,                // YOLO inference image size (e.g. 640)
    "runtime_s": null,
    "gpu": null                   // e.g. "Tesla T4 x2"
  },

  "detection": {
    "canonical": {
      "map50": null,
      "map50_95": null,
      "precision": null,
      "recall": null
    },
    "challenging": {              // set to null if not yet evaluated
      "map50": null,
      "map50_95": null,
      "precision": null,
      "recall": null
    }
  }
}
```

## Important notes

- **Challenging split:** leave as `null` if not yet evaluated — the GUI will show `—`.
- **effective_batch:** report `n_gpus × batch_per_gpu`, not just the per-GPU batch size. This affects comparability since larger effective batch changes training dynamics.
