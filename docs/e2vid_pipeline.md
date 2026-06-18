# E2VID Pipeline — End-to-End Guide

Drone detection from event-camera data using the FRED dataset, E2VID frame reconstruction, and YOLOv8 detection.

---

## Overview

```
FRED event data (.zip)
        │
        ▼
  reconstruct.py          ← rpg_e2vid, run on Colab T4 GPU
  (events → frames)
        │
        ▼
  train_yolo.py           ← YOLOv8n, run on Colab T4 GPU
  (frames + annotations → best.pt)
        │
        ▼
  sync_from_drive.sh      ← rclone, run locally
  (Drive → local)
        │
        ▼
  docker compose build    ← bakes weights into container
  (service ready)
```

---

## 1. Dataset

### Download FRED

FRED (Florence RGB-Event Drone) is available on [HuggingFace](https://huggingface.co/datasets/GabrieleMagrini/FRED).
The canonical split (`dataset_splits/canonical/`) defines which sequences belong to train and test.

### Team-agreed sequences (current)

The group agreed to use sequences 84, 85, 201, 124 (train) and 127 (val).

| Sequence | Split | Total events | Burst (<5s) | Kept (>5s) | Burst% | Duration | ~Frames | events.zip |
|---|---|---|---|---|---|---|---|---|
| sequence_84 | train | 142.6 M | 32.6 M | 110.0 M | 23% | 109.0 s | 11,935 | 648 MB |
| sequence_85 | train | 73.6 M | 11.2 M | 62.5 M | 15% | 108.3 s | 6,776 | 374 MB |
| sequence_201 | train | 123.2 M | 23.1 M | 100.1 M | 19% | 117.3 s | 10,860 | 604 MB |
| sequence_124 | train | 486.6 M | 182.1 M | 304.6 M | 37% | 108.8 s | ~33,046 | 1.2 GB |
| sequence_127 | test (val) | 241.1 M | 141.6 M | 99.5 M | 59% | 47.5 s | 10,794 | 555 MB |

All sequences have a noise burst in the first 5 seconds — correctly handled by `start_s = 5.0`.

**sequence_124 notes:**
- Drone is a **DJI Tello EDU** (large box-shaped quadcopter), different from the Betafpv air75 in the other sequences. This adds scale/appearance diversity to the training set but may also increase label noise if the model struggles to generalise across drone types.
- 2,744 annotations starting at 19.6 s — drone appears late, which means roughly 14 s of warmup frames with no labels.
- ~33,046 estimated frames — significantly more than any other single sequence. Ensure the YOLO dataset build does not over-represent this sequence (check class balance in `train.txt`).
- Reconstruction time on Kaggle T4: ~2.6 h for this sequence alone.

**sequence_127 note:** unusually high burst ratio (59%) and only 47.5 s of usable data after the skip
(vs 108–117 s for the train sequences). Still produces ~10,800 frames which is sufficient for validation.

**Reconstruction time estimate for the 5 team sequences (Kaggle T4, 3.5 fps):** ~73,000 frames total
→ ~6 hours reconstruction + ~1 hour training. Fits within Kaggle's 9-hour GPU limit.
Sequences 84/85/201/127 are already reconstructed — only sequence_124 needs to be run.
See `notebooks/kaggle_launcher.ipynb` for the restore-previous-recon workflow.

Download:

```bash
curl -L -o data/raw/zips/84.zip  "https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/main/train/84.zip"
curl -L -o data/raw/zips/85.zip  "https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/main/train/85.zip"
curl -L -o data/raw/zips/201.zip "https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/main/train/201.zip"
curl -L -o data/raw/zips/124.zip "https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/main/train/124.zip"
curl -L -o data/raw/zips/127.zip "https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/main/test/127.zip"
```

Then prepare:

```bash
bash scripts/prepare_sequences.sh 84 85 201 124 127
```

### Reference sequences (initial development, sequences 0–8)

> **Outdated** — used during initial development before the team agreed on a shared dataset.
> The reconstructed frames for sequence_0 and sequence_1 are still on disk and give a rough
> sense of the output sizes to expect from the team sequences.

| Sequence | Split | Events (after skip) | Frames reconstructed | Frames disk size | Source |
|---|---|---|---|---|---|
| sequence_0 | train | 8.0 M | 871 | 287 MB | HuggingFace `train/0.zip` |
| sequence_1 | train | 5.1 M | 555 | 164 MB | HuggingFace `train/1.zip` |
| sequence_2 | train | 7.7 M | — | — | HuggingFace `train/2.zip` |
| sequence_3 | train | 2.5 M | — | — | HuggingFace `train/3.zip` |
| sequence_8 | test (val) | 5.0 M | — | — | HuggingFace `test/8.zip` |

Scaling to the team sequences (40,365 frames combined): expect roughly **16–17 GB of PNG frames**
on disk (measured at ~0.41 MB/frame on Kaggle). With `--compress_jpeg` (see section 5) this
becomes **~2.5 GB of JPEG frames** — required to stay within Kaggle's 20 GB working directory.

```bash
curl -L -o data/raw/zips/0.zip "https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/main/train/0.zip"
curl -L -o data/raw/zips/1.zip "https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/main/train/1.zip"
curl -L -o data/raw/zips/2.zip "https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/main/train/2.zip"
curl -L -o data/raw/zips/3.zip "https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/main/train/3.zip"
curl -L -o data/raw/zips/8.zip "https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/main/test/8.zip"
bash scripts/prepare_sequences.sh 0 1 2 3 8
```

What it does for each sequence:
1. Extracts `events.hdf5` → `data/processed/sequence_N/events.h5`
2. Extracts `coordinates.txt` → `data/raw/sequence_N/coordinates.txt`
3. Verifies the HDF5 contains `CD/events` with fields `x, y, p, t`
4. Converts HDF5 → `data/processed/sequence_N/events.zip` using a chunked reader (5M events/batch)
   to avoid OOM on large sequences (84: ~130M events, 127: ~180M events)

> **Warning:** `events.zip` is NOT a plain zip of `events.h5`. It contains a custom `events.txt`
> binary format that rpg_e2vid reads directly. Always use `prepare_sequences.sh` (or
> `reconstruct.convert_h5_to_zip`) to create it — never zip `events.h5` manually.

Requires `h5py`:
```bash
pip install h5py
```

### Local layout

```
ami/
└── data/
    ├── raw/
    │   ├── sequence_84/coordinates.txt
    │   ├── sequence_85/coordinates.txt
    │   ├── sequence_201/coordinates.txt
    │   ├── sequence_124/coordinates.txt
    │   ├── sequence_127/coordinates.txt
    │   └── zips/
    │       ├── 84.zip   (1.1 GB)
    │       ├── 85.zip   (879 MB)
    │       ├── 201.zip  (1.1 GB)
    │       ├── 124.zip  (1.2 GB)
    │       └── 127.zip  (774 MB)
    └── processed/
        ├── sequence_84/events.zip
        ├── sequence_85/events.zip   (357 MB)
        ├── sequence_201/events.zip
        ├── sequence_124/events.zip  (1.2 GB)
        └── sequence_127/events.zip
```

### Noise burst

When a Prophesee event camera first powers on, the sensor pixels haven't settled to their reference
voltage yet. Almost every pixel fires repeatedly for the first few seconds, producing tens of millions
of events with no relation to the scene. This is the **noise burst**.

It is identified by comparing the event rate before and after the 5-second mark:

| Sequence | Rate in first 5s | Rate after 5s | Ratio |
|---|---|---|---|
| sequence_84 | 6.5 M ev/s | 1.0 M ev/s | 6.5× |
| sequence_85 | 2.2 M ev/s | 0.6 M ev/s | 3.7× |
| sequence_201 | 4.6 M ev/s | 0.9 M ev/s | 5.1× |
| sequence_127 | 28.3 M ev/s | 2.1 M ev/s | 13.5× |

The sudden drop is the signature. A burst of 2–28 M ev/s with no corresponding visual activity is
unambiguously noise. `start_s = 5.0` is a conservative threshold — the burst decays within 2–4s
for most Prophesee sensors, and it is safe for all sequences.

### Annotation start times

| Sequence | First annotation | Last annotation | Note |
|---|---|---|---|
| sequence_84 | 2.9 s | 114.0 s | starts inside noise burst — pre-skip annotations are lost |
| sequence_85 | 0.4 s | 113.3 s | starts inside noise burst — pre-skip annotations are lost |
| sequence_201 | 26.9 s | 122.3 s | drone appears 21.9 s after the noise skip |
| sequence_127 | 12.6 s | 52.5 s | drone appears 7.6 s after the noise skip |

For sequences 84 and 85, the drone is already visible during the noise burst. Those annotated frames
are correctly skipped — E2VID reconstruction during the burst produces noise, not usable images.

For sequences 201 and 127, frames between the 5s skip and the first annotation become **background
frames** in the YOLO dataset (empty label files). This is intentional: background frames teach the
model that most frames contain no drone, reducing false positives.

**Potential optimisation:** start reconstruction at `max(5.0, first_annotation − 5.0)` to give the
E2VID LSTM 5 seconds of warmup while skipping unnecessary background. For sequence_201 this would
save reconstructing ~17 seconds of pure background frames. Not yet implemented.

---

## 2. Preprocessing: Generate events.zip

`prepare_sequences.sh` (section 1) handles this step automatically. This section documents what
it does internally and why.

### What the conversion does

Each sequence's `events.h5` is converted to a zip containing a single `events.txt` in the format
rpg_e2vid expects:

```
1280 720
5.008000000 38 220 1
5.009000000 112 543 0
...
```

First line is `width height`. Each subsequent line is `timestamp_s x y polarity`.

Events before `start_s = 5.0` are skipped (noise burst). The conversion is done by
`convert_h5_to_zip()` in `reconstruct.py`.

### Technical notes

**Chunked reading** — the team sequences have 62–304 M events. Loading everything into RAM at once
(e.g. `events[:] `) would require 1–4 GB and crash on a typical laptop. The converter reads 5M
events at a time and streams output directly into the zip.

**ZIP64** — at ~22 bytes per event line, 110M+ events produce files larger than the 4 GB limit of
the standard zip32 format. ZIP64 extensions are enabled automatically.

**ECF codec** — the HDF5 files use Prophesee's ECF compression codec. This codec is
**proprietary and only ships with OpenEB / Metavision SDK** — `pip install hdf5plugin` does
**not** help here, because `hdf5plugin` only bundles open-source codecs (Blosc, LZ4, Zstd).

If you see `OSError: Can't synchronously read data (can't find plugin)`, the fix is:

- **Install OpenEB** on the machine: https://docs.prophesee.ai/stable/installation/linux.html  
  This puts the ECF filter at `/usr/lib/x86_64-linux-gnu/hdf5/plugins/`, which `reconstruct.py`
  picks up automatically via `HDF5_PLUGIN_PATH`.
- **No OpenEB available?** Get the pre-converted `events.zip` from a teammate who already ran
  `prepare_sequences.sh`. Place it at `data/processed/sequence_N/events.zip` — the script skips
  h5 conversion when the zip already exists.

### File sizes (team sequences)

| Sequence | events.h5 (local) | events.zip (uploaded to Colab) | Events after 5s skip |
|---|---|---|---|
| sequence_84 | 465 MB | 618 MB | 110.0 M |
| sequence_85 | 256 MB | 357 MB | 62.5 M |
| sequence_201 | 416 MB | 577 MB | 100.1 M |
| sequence_127 | 611 MB | 530 MB | 99.5 M |

**What is needed for reconstruction?** Only the `events.zip` — that is the only file uploaded to
Colab. The `events.h5` stays local; it is the source from which the zip is generated by
`prepare_sequences.sh` and is never needed again after the zip exists.

**"Events after 5s skip"** — the total number of events in the zip, i.e. the events that remain
after discarding the noise burst (everything before `t = 5.0 s`). These are the events E2VID
actually processes. The full sequence has more events but the burst portion is unusable.

The events.zip files are larger than the h5 files because the text format (`timestamp x y p\n`)
is less compact than binary HDF5, even after deflate compression.

---

## 3. Running the Pipeline

The pipeline has two stages, each implemented as a standalone Python script.
Both run on GPU (Colab or Kaggle) or locally (CPU only, much slower).

### Option A — Kaggle (recommended for full runs)

Kaggle GPU notebooks run up to **9 hours** — enough for the full reconstruction + training run in
one session without needing resume. GPU: **T4** (P100 is too old for PyTorch 2.x).

**One-time setup — upload data (~2.1 GB, ~7 min):**

```bash
pip install kaggle          # or: pipx install kaggle
# Place API token at ~/.kaggle/kaggle.json, then:
bash scripts/sync_to_kaggle.sh
```

This creates the `fred-events-ami` Kaggle dataset containing events + coordinates:

| File | Size |
|---|---|
| `data/processed/sequence_84/events.zip` | 618 MB |
| `data/processed/sequence_85/events.zip` | 357 MB |
| `data/processed/sequence_201/events.zip` | 577 MB |
| `data/processed/sequence_127/events.zip` | 530 MB |
| `data/raw/sequence_*/coordinates.txt` | ~1 MB total |

**When only scripts change (~30 KB, seconds):**

```bash
bash scripts/sync_scripts_to_kaggle.sh
```

This updates a separate `fred-scripts-ami` dataset containing only `reconstruct.py` and
`train_yolo.py`. No need to re-upload the 2.1 GB of event data.

**Run the notebook:**

1. Go to kaggle.com → Create → New Notebook
2. **+ Add data** → Your Datasets → add both `fred-events-ami` and `fred-scripts-ami`
3. Import `notebooks/kaggle_launcher.ipynb`
4. Settings → Accelerator → **GPU T4 x2**
5. Run All

Note: Kaggle auto-extracts zip files — `events.zip` becomes `events/events.txt`. `reconstruct.py`
detects this automatically and passes the txt directly to rpg_e2vid (no re-zipping needed).

**Kaggle working directory layout:**

```
/kaggle/
├── input/datasets/gennepy/
│   ├── fred-events-ami/              ← read-only, persistent
│   │   ├── data/processed/sequence_N/events/events.txt   ← auto-extracted by Kaggle
│   │   └── data/raw/sequence_N/coordinates.txt
│   └── fred-scripts-ami/             ← read-only, update via sync_scripts_to_kaggle.sh
│       ├── reconstruct.py
│       └── train_yolo.py
└── working/                          ← writable, saved as output after run
    ├── data/processed/sequence_N/reconstruction_e2vid/   ← frames + timestamps.txt
    ├── scripts/                      ← local copy of scripts
    ├── yolo_e2vid/                   ← YOLO dataset (rebuilt each session)
    ├── yolo_runs/e2vid/weights/      ← last.pt, best.pt
    ├── yolo_e2vid.pt                 ← final weights → download this
    ├── logs/run_YYYYMMDD_HHMMSS.log  ← full run log
    └── work/rpg_e2vid/               ← cloned + patched each session
```

**Download results:**

```bash
# weights + KPIs + logs + training plots (small, fast — do this first)
bash scripts/sync_from_kaggle.sh

# also download reconstructed frames (~2.5 GB JPEG, only if needed locally for the GUI)
bash scripts/sync_from_kaggle.sh --frames
```

> **Note:** `sync_from_kaggle.sh` uses a paginated Python downloader internally. The kernel
> produces ~80 k output files; the standard `kaggle kernels output` CLI only fetches the first
> API page and misses everything past `sequence_127` alphabetically. The script paginates through
> all ~162 pages (~500 files each) automatically.

Then rebuild the Docker service:

```bash
docker compose build e2vid && docker compose up -d
```

---

### Option B — Google Colab

Colab free tier disconnects after ~90 min — requires resume for a full run. Use Kaggle instead
unless you have Colab Pro.

**Upload data to Google Drive first:**

```bash
bash scripts/sync_to_drive.sh
```

This uploads (~2.1 GB total) to `MyDrive/ami/` via rclone (`gdrive` remote required):

| File | Size |
|---|---|
| `scripts/reconstruct.py` | 18 KB |
| `scripts/train_yolo.py` | 14 KB |
| `notebooks/colab_launcher.ipynb` | 11 KB |
| `data/processed/sequence_*/events.zip` | 2.1 GB |
| `data/raw/sequence_*/coordinates.txt` | ~1 MB |

**Then open `notebooks/colab_launcher.ipynb` in Colab:**

1. Runtime → Change runtime type → Hardware accelerator → **T4 GPU**
2. Edit the **Configuration** cell if needed
3. Run all cells top to bottom

**Colab working directory layout:**

```
/content/
├── drive/MyDrive/ami/               ← Google Drive, persistent across sessions
│   ├── data/
│   │   ├── raw/sequence_N/          ← coordinates.txt (read-only input)
│   │   ├── processed/sequence_N/
│   │   │   ├── events.zip           ← input event stream
│   │   │   ├── reconstruction_e2vid/← output frames + timestamps.txt
│   │   │   └── kpis/               ← per-sequence reconstruction KPIs
│   │   ├── yolo_runs/e2vid/         ← training checkpoints (last.pt, best.pt)
│   │   ├── kpis/train_yolo.json
│   │   ├── yolo_e2vid.pt            ← final weights
│   │   └── yolo_e2vid/dataset.yaml  ← redirects ultralytics to local SSD on resume
│   ├── scripts/                     ← reconstruct.py, train_yolo.py
│   └── logs/run_YYYYMMDD_HHMMSS.log
├── scripts/                         ← local SSD copy (avoids Drive cache lag)
├── yolo_e2vid/                      ← YOLO dataset on local SSD (fast random I/O)
└── work/rpg_e2vid/                  ← cloned + patched each session
```

**Key design decisions (Colab):**
- **Reconstruction frames** go to Drive so they survive disconnects and are skipped on resume.
- **YOLO dataset images** go to local SSD — Drive I/O throttling caused `FileNotFoundError` crashes during training when images were read from Drive.
- **Scripts** are copied to local SSD at session start to avoid Drive cache/sync lag.
- **Drive dataset.yaml** is overwritten each session to point to local SSD, so ultralytics resume reads images from SSD even when reloading a Drive checkpoint.

**Colab session limits and resume:**
- Free tier disconnects after ~90 min of inactivity. Keep the browser tab active.
- Set `RESUME = True` in the config cell after a disconnect — skips cleanup and reconstruction, rebuilds the YOLO dataset locally (~2 min), then resumes training from `last.pt` on Drive.

**Download results:**

```bash
bash scripts/sync_from_drive.sh
docker compose build e2vid && docker compose up -d
```

`sync_from_drive.sh` downloads weights, KPIs, training plots, and reconstructed frames from Drive.

---

### Option C — Local (CPU only)

```bash
# Stage 1 — reconstruct
python scripts/reconstruct.py \
  --zip_path   data/raw/sequence_0/events.zip \
  --out_dir    data/processed/sequence_0/reconstruction_e2vid \
  --work_dir   /tmp/e2vid_work \
  --events_per_pixel 0.01

# Stage 2 — train
python scripts/train_yolo.py \
  --sequences  sequence_0 sequence_1 \
  --raw_root   data/raw \
  --recon_root data/processed \
  --out_dir    data/yolo_e2vid \
  --weights    data/yolo_e2vid.pt \
  --epochs     3   # smoke test
```

---

## 4. Reconstruction: reconstruct.py

### What it does

1. Clones `rpg_e2vid` from GitHub into `work_dir/`
2. Downloads the full E2VID model weights (`E2VID.pth.tar`, ~60 MB) from RPG
3. Applies 6 compatibility patches for Python 3.12 / PyTorch 2.x / NumPy 1.24+ / pandas 2.0
4. Reads events from `events.zip`, skips the first `START_S` seconds
5. Runs `run_reconstruction.py` with `--num_events_per_pixel 0.01`
6. Moves output frames to `out_dir/reconstruction_e2vid/`
7. Writes `timestamps.txt` and a KPI JSON

### Event window size

`num_events_per_pixel = 0.01` → one frame per **9,216 events** (0.01 × 1280 × 720).
This suits FRED's sparse event rate (~46–76 k events/s after the noise burst).

Frames do not have a fixed wall-clock duration — they represent equal amounts of event activity.
When the drone moves faster the 9,216 events accumulate faster and more frames are produced per second.

### Key arguments

| Argument | Default | Notes |
|---|---|---|
| `--zip_path` | required | path to `events.zip` |
| `--out_dir` | required | output directory |
| `--work_dir` | required | where rpg_e2vid is cloned |
| `--events_per_pixel` | `0.01` | window size |
| `--start_s` | `5.0` | skip first N seconds (noise burst) |
| `--max_events` | None | cap for smoke tests (e.g. `100000` → ~10 frames) |
| `--compress_jpeg` | off | convert PNGs → JPEG (quality 90) after reconstruction |

---

## 5. YOLO Dataset Preparation

This step is handled inside `train_yolo.py` before training starts. It is documented separately
because the disk constraints on Kaggle drove several non-obvious design decisions.

### What it does

1. Loads bounding-box annotations from `coordinates.txt` for each sequence
2. Matches each reconstructed frame to its nearest annotation (within 0.5 s)
3. Writes a **YOLO label file** (`.txt`) alongside each frame in `reconstruction_e2vid/`
4. Writes `train.txt` and `val.txt` manifests listing absolute image paths
5. Writes `dataset.yaml` pointing to the manifests

### Why labels live alongside frames (not in a separate directory)

The natural YOLO layout (`images/train/`, `labels/train/`) requires **copying** ~40 k frames
into a second directory tree. On Kaggle this failed with `OSError: [Errno 28] No space left on device`
because frames alone occupy ~16 GB on a 20 GB disk.

The fix uses two ultralytics features:
- **Txt manifests** — `dataset.yaml` can point to a `.txt` file listing image paths instead of a directory.
- **Same-directory label discovery** — when an image path contains no `/images/` component,
  ultralytics falls back to looking for `<same_path_stem>.txt` in the same directory.

This means label files (`frame_000001.txt`) are written next to frames (`frame_000001.jpg`) with
no image duplication. Total label storage: ~40 k × ~30 bytes ≈ 1.2 MB.

### Disk space on Kaggle: JPEG compression

Kaggle's `/kaggle/working/` is limited to **20 GB**. The 40 k PNG frames produced by e2vid
measure ~0.41 MB each → ~16.7 GB, leaving only ~3 GB for everything else. Even creating
symlinks or writing tiny label files fills the remaining space.

`reconstruct.py --compress_jpeg` converts each PNG to JPEG (quality 90) and deletes the PNG
immediately after reconstruction of each sequence. Storage drops from ~16.7 GB to **~2.5 GB**.

### Quality tradeoff: JPEG vs PNG

| Format | Storage | Quality |
|---|---|---|
| PNG (lossless) | ~16.7 GB | exact reconstruction output |
| JPEG q=90 | ~2.5 GB | visually lossless; artifacts well below e2vid reconstruction noise |
| PNG at 640×360 | ~4 GB | lossless for YOLO (which rescales to 640 px anyway); loses full resolution |

**JPEG q=90 is acceptable** for two reasons:
1. The e2vid reconstruction is itself an approximation of the real scene from events.
   The reconstruction error dominates over JPEG compression artifacts at quality 90.
2. YOLOv8 is routinely trained on JPEG data (ImageNet, COCO) and is robust to this level of compression.

An alternative that avoids lossy compression: pre-resize frames to **640×360** (PNG) before
training. YOLOv8 downscales all images to 640 px during training anyway, so storing full
1280×720 frames wastes both disk space and data-loading time. This has not yet been implemented.

### Multi-GPU training (T4 x2)

`train_yolo.py` auto-detects multiple GPUs and passes `device='0,1'` to ultralytics, which uses
PyTorch DDP internally. In DDP mode `model.train()` returns `None` in the main process — the
script works around this by deriving `save_dir` from the known `project/name` path and reading
metrics from `results.csv`.

---

## 6. Training: train_yolo.py

### What it does

1. Parses `coordinates.txt` for each sequence → timestamps + bounding boxes
2. Matches each reconstructed frame to its nearest annotation within 0.5 s
3. Builds a YOLO dataset (`images/train`, `images/val`, `labels/`, `dataset.yaml`)
4. Fine-tunes a YOLOv8 model on the dataset (model size configured via `--model`)
5. Copies `best.pt` to `--weights` output path
6. Writes a KPI JSON

### Annotation matching

Frames more than 0.5 s from any annotation are kept as **background frames** (empty label files).
This is intentional — background frames teach the model to suppress false positives.

### Key arguments

| Argument | Default | Notes |
|---|---|---|
| `--sequences` | required | e.g. `sequence_0 sequence_1` |
| `--raw_root` | required | parent of `sequence_N/` dirs |
| `--recon_root` | required | parent of `sequence_N/reconstruction_e2vid/` |
| `--epochs` | `100` | set to `3` for smoke test |
| `--batch` | `8` | `16` on T4 GPU |
| `--match_threshold` | `0.5` | max gap (s) for annotation matching |
| `--split` | `0.8` | train fraction |

---

## 7. Downloading Results: sync_from_drive.sh

Run locally after the Colab notebook finishes:

```bash
bash scripts/sync_from_drive.sh
```

Requires `rclone` at `~/.local/bin/rclone` with a `gdrive` remote configured.

**What it downloads:**

| Source (Drive) | Destination (local) |
|---|---|
| `ami/data/yolo_e2vid.pt` | `services/e2vid/weights/yolo_e2vid.pt` |
| `ami/data/processed/sequence_N/kpis/` | `data/kpis/` |
| `ami/data/kpis/` | `data/kpis/` |
| `ami/data/yolo_runs/` | `data/yolo_runs/` |
| `ami/data/processed/sequence_N/reconstruction_e2vid/` | `data/processed/sequence_N/reconstruction_e2vid/` |

**Important:** if the Colab session died mid-training, `yolo_e2vid.pt` on Drive may be stale.
In that case, copy the correct weights manually:

```bash
cp data/yolo_runs/e2vid/weights/best.pt services/e2vid/weights/yolo_e2vid.pt
# Then update Drive so future syncs don't overwrite:
~/.local/bin/rclone copyto services/e2vid/weights/yolo_e2vid.pt gdrive:ami/data/yolo_e2vid.pt
```

---

## 8. Updating the Docker Service

After new weights are in `services/e2vid/weights/yolo_e2vid.pt`:

```bash
# Build and push to GHCR
docker build -t ghcr.io/gennepy/ami-e2vid:latest services/e2vid/
gh auth token | docker login ghcr.io -u gennepy --password-stdin
docker push ghcr.io/gennepy/ami-e2vid:latest

# Teammates pull the update
docker compose pull e2vid && docker compose up -d e2vid
```

The Dockerfile bakes the weights into the image (`COPY weights/ /app/weights/`),
so the service is fully self-contained with no runtime volume dependency for weights.

---

## 9. Local Folder Structure (complete)

```
ami/
├── data/                       ← all data, never committed to git
│   ├── raw/                    ← original FRED dataset (read-only input)
│   │   ├── sequence_0/         ← full FRED sequence: RGB, Event frames, coordinates
│   │   │   ├── events.zip      ← compressed event stream (Prophesee RAW)
│   │   │   └── coordinates.txt ← bounding-box annotations (timestamp: x1,y1,x2,y2)
│   │   ├── sequence_1/
│   │   │   ├── events.zip
│   │   │   └── coordinates.txt
│   │   └── zips/               ← original download archives (0.zip, 1.zip, ~1.1 GB each)
│   ├── processed/              ← pipeline outputs from reconstruct.py
│   │   ├── sequence_0/
│   │   │   └── reconstruction_e2vid/
│   │   │       ├── frame_000000.png … frame_000870.png   ← 871 grayscale frames
│   │   │       └── timestamps.txt                        ← one timestamp (s) per frame
│   │   └── sequence_1/
│   │       └── reconstruction_e2vid/
│   │           ├── frame_000000.jpg … frame_000554.jpg   ← 555 JPEG frames (--compress_jpeg)
│   │           ├── frame_000000.txt … frame_000554.txt   ← YOLO labels written alongside frames
│   │           └── timestamps.txt
│   ├── kpis/                   ← JSON performance metrics written by each pipeline stage
│   │   ├── reconstruct_sequence_0.json
│   │   ├── reconstruct_sequence_1.json
│   │   └── train_yolo.json
│   ├── yolo_runs/              ← ultralytics training output (logs, plots, weights)
│   │   └── e2vid/
│   │       ├── results.csv     ← per-epoch metrics (loss, mAP50, precision, recall)
│   │       ├── results.png     ← training curves plot
│   │       └── weights/
│   │           ├── best.pt     ← best checkpoint by mAP50 (use this)
│   │           └── last.pt     ← checkpoint from final epoch
│   └── yolo_e2vid/             ← YOLO dataset manifest (tiny — no image copies)
│       ├── train.txt           ← absolute paths to train images
│       ├── val.txt             ← absolute paths to val images
│       └── dataset.yaml        ← points to train.txt / val.txt
├── docs/                       ← project documentation
│   └── e2vid_pipeline.md       ← this file
├── notebooks/                  ← Colab notebooks
│   └── colab_launcher.ipynb    ← the one notebook to run the full pipeline on Colab
├── scripts/                    ← standalone Python/shell scripts
│   ├── reconstruct.py          ← stage 1: event zip → e2vid frames
│   ├── train_yolo.py           ← stage 2: frames + annotations → YOLO weights
│   └── sync_from_drive.sh      ← pull Colab outputs from Google Drive to local
├── services/                   ← Docker microservices
│   ├── e2vid/                  ← FastAPI service: /detect endpoint using YOLOv8
│   │   ├── Dockerfile
│   │   ├── app.py
│   │   ├── requirements.txt
│   │   └── weights/
│   │       └── yolo_e2vid.pt   ← 22 MB, Run 5 YOLOv8s weights (mAP50=79.2%, sequence_127 val)
│   ├── hypere2vid/             ← alternative reconstruction service (HyperE2VID)
│   ├── fusion/                 ← early fusion service
│   └── web/                    ← Streamlit UI
├── outdated/                   ← superseded code kept for reference
├── docker-compose.yml          ← wires all services together
└── setup.sh                    ← creates .env with local data paths
```

---

## 10. KPIs

KPI files are written automatically by `reconstruct.py` and `train_yolo.py`.
They are not yet consumed by the running services but are available for reporting.

### Reconstruction KPI (`data/kpis/reconstruct_sequence_N.json`)

| Field | Description |
|---|---|
| `n_events` | total events processed (after noise-burst skip) |
| `n_frames` | reconstructed frames produced |
| `events_per_frame` | always `width × height × events_per_pixel` |
| `runtime_s` | wall-clock time for reconstruction |
| `frames_per_second` | processing throughput (not real-time fps) |
| `seconds_per_frame` | inverse of above |
| `gpu` | GPU used (`Tesla T4` on Colab, absent on CPU) |
| `peak_ram_mb` | peak RSS memory |

**Actual values (full run, T4 GPU):**

| | seq_84 | seq_85 | seq_201 | seq_127 | Total |
|---|---|---|---|---|---|
| events | 110.0 M | 62.5 M | 100.1 M | 99.5 M | 372.1 M |
| frames | 11,936 | 6,777 | 10,861 | 10,795 | **40,369** |
| runtime | 2,973 s | 1,681 s | 2,733 s | 2,702 s | **2.79 h** |
| throughput | 4.02 fps | 4.03 fps | 3.97 fps | 3.99 fps | ~4.0 fps avg |

Earlier reference sequences (initial development):

| | sequence_0 | sequence_1 |
|---|---|---|
| events | 8,018,782 | 5,113,036 |
| frames | 871 | 555 |
| runtime | 222.9 s | 147.5 s |
| throughput | 3.91 fps | 3.76 fps |

Note: frame counts differ because sequences have different event rates (scene activity), not different durations.

### Training KPI (`data/kpis/train_yolo.json`)

| Field | Description |
|---|---|
| `n_train_images` / `n_val_images` | dataset split sizes |
| `epochs_requested` / `epochs_completed` | requested vs. actual (may differ on disconnect) |
| `best_epoch` | epoch with highest mAP50 |
| `map50` | mAP @ IoU=0.50 at best epoch |
| `map50_95` | mAP @ IoU=0.50–0.95 at best epoch |
| `precision` / `recall` | at best epoch |
| `runtime_per_epoch_s` | wall-clock seconds per epoch |
| `gpu_memory_peak_mb` | peak VRAM usage |

**Full run results — team sequences (Kaggle, 2×T4, sequence-level split):**

| Field | Value |
|---|---|
| train sequences | seq_84, seq_85, seq_201 |
| val sequence | seq_127 |
| train / val images | 29,574 / 10,795 |
| epochs requested | 100 |
| epochs run | **32** (early stopped; best at epoch 12 by YOLO fitness) |
| best mAP50 | **0.291** (epoch 7) |
| best mAP50-95 | 0.072 (epoch 12) |
| best precision | 0.546 (epoch 12) |
| best recall | 0.314 (epoch 12) |
| runtime total | 3.75 h |
| runtime per epoch | ~7 min (2×T4 DDP) |

**Earlier reference run (frame-level split, sequences 0–8):**

| Field | Value |
|---|---|
| train / val images | 1,140 / 286 |
| epochs completed | 83 / 100 |
| mAP50 | 0.874 |
| precision / recall | 0.878 / 0.832 |
| runtime per epoch | 35.6 s |

> **Why the big difference?** The 0.874 used a frame-level split — validation frames came from the
> same sequences as training, just different time windows. The model effectively memorized the
> backgrounds. This was a data-leakage artifact, not a real generalisation score.
>
> The 0.291 uses a **sequence-level split**: the entire sequence_127 is held out. The model is
> evaluated on a scene it has never seen. This is the honest number.

**Why did training stop at epoch 32?**

The validation box loss barely moved across all 32 epochs (2.37 → 2.35 → 2.35), signalling that
YOLOv8n had reached its capacity ceiling on 3 training sequences. Running more epochs would not
have helped — the model was genuinely stuck.

**What could improve performance:**

| Approach | Expected gain | Cost |
|---|---|---|
| YOLOv8s (larger model) | moderate | ~2× time/epoch (~14 min); 50 epochs feasible on Kaggle |
| YOLOv8m | larger | ~4× time/epoch; likely too slow for Kaggle 9h limit |
| More training sequences | likely most impactful | need more FRED data or augmentation |
| Resize frames to 640 px before training | minimal quality change | faster I/O, less disk |
| Disable early stopping (`patience=0`) | more epochs to explore | risk of overfitting |

**Model size vs. training time (estimated, 2×T4 DDP, 40k frames, batch=16):**

| Model | Parameters | Size | ~Time/epoch | 100-epoch ETA | Fits 9h? |
|---|---|---|---|---|---|
| yolov8n | 3.2 M | 6 MB | ~7 min | ~12 h (with early stop: ~3.5 h) | ✓ |
| yolov8s | 11.2 M | 22 MB | ~14 min | ~23 h (with early stop: ~7 h) | borderline |
| yolov8m | 25.9 M | 52 MB | ~28 min | ~47 h | ✗ |

---

## 11. KPI Evolution (Run History)

All runs use sequence-level split: sequences 84, 85, 201, 124 for training; sequence 127 for validation.

| Run | Model | mAP@0.5 | mAP@0.5:95 | Precision | Recall | Epochs (best) | GPU | Notes |
|---|---|---|---|---|---|---|---|---|
| Run 1 | YOLOv8n | 0.291 | 0.072 | 0.546 | 0.314 | 32 (12) | 2×T4 | seq_84/85/201 train only (no seq_124) |
| Run 2 | YOLOv8n | **0.634** | 0.275 | 0.820 | 0.582 | 23 (3) | 2×T4 | added seq_124 to train; lr0=0.01, batch=32 |
| Run 3 | YOLOv8n | 0.634 | — | — | — | 23 (3) | 1×P100 | lr0=0.001 (accidental); identical metrics; weights deleted by cleanup bug |
| Run 4 | YOLOv8n | 0.634 | 0.275 | 0.820 | 0.582 | 23 (3) | 2×T4 | lr0=0.001 (accidental); same result as Run 2 |
| **Run 5** | **YOLOv8s** | **0.792** | **0.395** | **0.954** | **0.707** | **27 (5)** | **2×T4** | upgraded to YOLOv8s; lr0=0.01 (default); mosaic=1.0, mixup=0.0 |

**Key observations:**
- Run 1 → Run 2: the single biggest jump (+0.343 mAP50) came from adding sequence_124 to the training set, not from hyperparameter tuning.
- Run 2 → Run 4: lr0=0.001 vs 0.01 made no measurable difference on YOLOv8n.
- Run 4 → Run 5: upgrading from YOLOv8n (3.2M params) to YOLOv8s (11.2M params) gained +0.158 mAP50 with the same dataset — the model had more capacity to learn.
- Early stopping consistently triggers around epoch 20–30, well before the 100-epoch limit.

**Current production weights:** Run 5 (`ami-e2vid:latest` Docker image, baked in as `yolo_e2vid.pt`).
