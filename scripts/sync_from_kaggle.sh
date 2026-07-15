#!/usr/bin/env bash
# Download results from the AMI Kaggle pipeline.
#
# Usage:
#   bash scripts/sync_from_kaggle.sh                   # session 2: weights + KPIs + logs
#   bash scripts/sync_from_kaggle.sh --frames-zip      # session 1: extract frames_and_detections.zip
#   bash scripts/sync_from_kaggle.sh --frames-zip=/path/to/frames_and_detections.zip
#   bash scripts/sync_from_kaggle.sh --detections      # session 3: download detections/ via API
#   bash scripts/sync_from_kaggle.sh --detections-zip  # session 3: extract JSONs from zip
#
# Session 1 (recon): download frames_and_detections.zip manually from the Kaggle
#   output page, then run with --frames-zip to extract into data/processed/.
#
# Session 2 (training): run without flags to pull weights, KPIs, and logs.
#   Uses file_pattern to skip the 8.9 GB frames zip.
#
# Session 3 (detection cache): run --detections to download the standalone
#   detections/ folder via the Kaggle API (no manual zip download required).
#   Requires notebook v100+ which writes the standalone folder.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KAGGLE="${HOME}/.local/bin/kaggle"
if [ ! -x "$KAGGLE" ]; then
    KAGGLE="$(command -v kaggle)"
fi
KERNEL="gennepy/ami-e2vid-pipeline"
TMPDIR="/tmp/kaggle_output"

FRAMES_ZIP=""
DETECTIONS_ZIP=""
DETECTIONS_API=""
for arg in "$@"; do
    case "$arg" in
        --frames-zip=*)      FRAMES_ZIP="${arg#--frames-zip=}" ;;
        --frames-zip)        FRAMES_ZIP="$HOME/Downloads/frames_and_detections.zip" ;;
        --detections-zip=*)  DETECTIONS_ZIP="${arg#--detections-zip=}" ;;
        --detections-zip)    DETECTIONS_ZIP="$HOME/Downloads/frames_and_detections.zip" ;;
        --detections)        DETECTIONS_API="1" ;;
    esac
done

# ── Session 3: extract only detections_e2vid.json from zip ───────────────────
if [ -n "$DETECTIONS_ZIP" ]; then
    if [ ! -f "$DETECTIONS_ZIP" ]; then
        echo "ERROR: zip not found at $DETECTIONS_ZIP"
        echo "Download frames_and_detections.zip from the Kaggle output page and pass its path:"
        echo "  bash scripts/sync_from_kaggle.sh --detections-zip=/path/to/frames_and_detections.zip"
        exit 1
    fi
    echo "=== Extracting detections_e2vid.json from $DETECTIONS_ZIP → $ROOT ==="
    python3 - <<PYEOF
import zipfile, os
from pathlib import Path

zip_path = "$DETECTIONS_ZIP"
root = Path("$ROOT")
count = 0
with zipfile.ZipFile(zip_path) as zf:
    for name in zf.namelist():
        if name.endswith('detections_e2vid.json'):
            dest = root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(name))
            count += 1
print(f"  Extracted {count} detections_e2vid.json files → data/processed/")
PYEOF
    echo "=== Done ==="
    exit 0
fi

# ── Session 3 (API): download detections/ folder via Kaggle API ──────────────
if [ -n "$DETECTIONS_API" ]; then
    echo "=== Downloading detections/ from $KERNEL via Kaggle API ==="
    rm -rf "$TMPDIR"
    mkdir -p "$TMPDIR"
    python3 - <<'PYEOF'
import sys, os
sys.path.insert(0, os.path.expanduser('~/.local/lib/python3.12/site-packages'))
import kaggle

api = kaggle.api
api.authenticate()

tmpdir = '/tmp/kaggle_output'
kernel = 'gennepy/ami-e2vid-pipeline'
pattern = r'detections_e2vid\.json'
files, _ = api.kernels_output(kernel, path=tmpdir, file_pattern=pattern, force=True, quiet=False)
print(f'Downloaded {len(files)} files to {tmpdir}')
PYEOF

    python3 - <<PYEOF
from pathlib import Path
import shutil

tmpdir = Path('/tmp/kaggle_output')
root = Path("$ROOT")
count = 0
for f in tmpdir.rglob('detections_e2vid.json'):
    # Files land as detections/<seq>/detections_e2vid.json in tmpdir
    # Map to data/processed/<seq>/detections_e2vid.json
    parts = f.relative_to(tmpdir).parts
    if len(parts) >= 2:
        dest = root / 'data' / 'processed' / '/'.join(parts[-2:])
    else:
        dest = root / 'data' / 'processed' / f.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(f, dest)
    count += 1
print(f"  Copied {count} detections_e2vid.json files → data/processed/")
PYEOF
    echo "=== Done ==="
    exit 0
fi

# ── Session 1: extract frames zip ────────────────────────────────────────────
if [ -n "$FRAMES_ZIP" ]; then
    if [ ! -f "$FRAMES_ZIP" ]; then
        echo "ERROR: zip not found at $FRAMES_ZIP"
        echo "Download frames_and_detections.zip from the Kaggle output page and pass its path:"
        echo "  bash scripts/sync_from_kaggle.sh --frames-zip=/path/to/frames_and_detections.zip"
        exit 1
    fi
    echo "=== Extracting $FRAMES_ZIP → $ROOT ==="
    unzip -q -o "$FRAMES_ZIP" -d "$ROOT"
    echo "Counting extracted frames..."
    total=$(find "$ROOT/data/processed" -name 'frame_*.jpg' | wc -l)
    seqs=$(find "$ROOT/data/processed" -name 'reconstruction_e2vid' -type d | wc -l)
    echo "  $total frames across $seqs sequences → $ROOT/data/processed/"
    echo "=== Done ==="
    exit 0
fi

# ── Session 2: download weights + KPIs + logs (skip frames zip) ──────────────
echo "=== Downloading session 2 outputs from $KERNEL (skipping frames zip) ==="
rm -rf "$TMPDIR"
mkdir -p "$TMPDIR"

# file_pattern is a regex matched against filenames.
# Exclude frames_and_detections.zip (8.9 GB) — download everything else.
python3 - <<'PYEOF'
import sys, os, shutil
sys.path.insert(0, os.path.expanduser('~/.local/lib/python3.12/site-packages'))
import kaggle

api = kaggle.api
api.authenticate()

tmpdir = '/tmp/kaggle_output'
kernel = 'gennepy/ami-e2vid-pipeline'

# Download all files except the large frames zip
pattern = r'^(?!frames_and_detections\.zip$)'
files, _ = api.kernels_output(kernel, path=tmpdir, file_pattern=pattern, force=True, quiet=False)
print(f'Downloaded {len(files)} files to {tmpdir}')
PYEOF

# ── Copy to project directories ───────────────────────────────────────────────

# Weights (prefer yolo_e2vid.pt, fall back to best.pt)
PT="$TMPDIR/yolo_e2vid.pt"
BEST="$TMPDIR/best.pt"
if [ -f "$PT" ]; then
    mkdir -p "${ROOT}/services/e2vid/weights"
    cp "$PT" "${ROOT}/services/e2vid/weights/yolo_e2vid.pt"
    echo "  yolo_e2vid.pt → services/e2vid/weights/  ($(du -h "$PT" | cut -f1))"
elif [ -f "$BEST" ]; then
    mkdir -p "${ROOT}/services/e2vid/weights"
    cp "$BEST" "${ROOT}/services/e2vid/weights/yolo_e2vid.pt"
    echo "  best.pt → services/e2vid/weights/yolo_e2vid.pt  ($(du -h "$BEST" | cut -f1))"
else
    echo "  WARNING: no weights file found"
fi

# KPIs + plots (downloaded into kpis/ subdir)
mkdir -p "${ROOT}/data/kpis"
for f in "$TMPDIR"/kpis/train_yolo.json "$TMPDIR"/kpis/reconstruct_summary.json \
          "$TMPDIR"/kpis/*.png "$TMPDIR"/kpis/*.csv "$TMPDIR"/kpis/*.jpg; do
    [ -f "$f" ] && cp "$f" "${ROOT}/data/kpis/" && echo "  $(basename "$f") → data/kpis/"
done

# Logs (downloaded into logs/ subdir)
mkdir -p "${ROOT}/data/logs"
for f in "$TMPDIR"/logs/run_*.log; do
    [ -f "$f" ] && cp "$f" "${ROOT}/data/logs/" && echo "  $(basename "$f") → data/logs/"
done

echo "=== Done ==="
