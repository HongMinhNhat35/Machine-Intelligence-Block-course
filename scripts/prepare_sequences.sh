#!/usr/bin/env bash
# Prepare new FRED sequences for the e2vid pipeline.
# For each sequence: extract events.hdf5 + coordinates.txt from the downloaded zip,
# verify the HDF5 structure, then run reconstruct.py to generate events.zip + frames.
#
# Usage:
#   bash scripts/prepare_sequences.sh 2 3 8

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZIPS_DIR="${ROOT}/data/raw/zips"
SEQUENCES=("${@:-2 3 8}")

echo "=== Preparing sequences: ${SEQUENCES[*]} ==="

for SEQ in "${SEQUENCES[@]}"; do
    ZIP="${ZIPS_DIR}/${SEQ}.zip"
    SEQ_NAME="sequence_${SEQ}"
    RAW_DIR="${ROOT}/data/raw/${SEQ_NAME}"
    PROC_DIR="${ROOT}/data/processed/${SEQ_NAME}"

    echo ""
    echo "──────────────────────────────────────────"
    echo "  ${SEQ_NAME}"
    echo "──────────────────────────────────────────"

    if [ ! -f "$ZIP" ]; then
        echo "  ✗ ${ZIP} not found — skipping"
        continue
    fi

    mkdir -p "$RAW_DIR" "$PROC_DIR"

    # ── Step 1: Extract ───────────────────────────────────────────────────────
    if [ -f "${PROC_DIR}/events.h5" ] && [ -f "${RAW_DIR}/coordinates.txt" ]; then
        echo "  Step 1: Already extracted — skipping"
    else
        echo "  Step 1: Extracting from ${ZIP} ..."
        unzip -p "$ZIP" "${SEQ}/Event/events.hdf5" > "${PROC_DIR}/events.h5"
        unzip -p "$ZIP" "${SEQ}/coordinates.txt"   > "${RAW_DIR}/coordinates.txt"
        echo "    events.h5       : $(du -h "${PROC_DIR}/events.h5" | cut -f1)"
        echo "    coordinates.txt : $(wc -l < "${RAW_DIR}/coordinates.txt") lines"
    fi

    # ── Step 2: Verify HDF5 structure ─────────────────────────────────────────
    echo "  Step 2: Verifying HDF5 structure ..."
    python3 - << PYEOF
import h5py, os, sys
path = "${PROC_DIR}/events.h5"
try:
    with h5py.File(path, 'r') as f:
        if 'CD/events' not in f:
            print(f"  ✗ 'CD/events' path not found. Keys: {list(f.keys())}")
            sys.exit(1)
        ev = f['CD/events']
        fields = list(ev.dtype.names) if hasattr(ev.dtype, 'names') and ev.dtype.names else list(ev.keys()) if hasattr(ev, 'keys') else []
        n = len(ev)
        print(f"  ✓ CD/events found: {n:,} events, fields: {fields}")
except Exception as e:
    print(f"  ✗ Failed to read HDF5: {e}")
    sys.exit(1)
PYEOF
    if [ $? -ne 0 ]; then
        echo "  HDF5 verification failed for ${SEQ_NAME} — stopping"
        exit 1
    fi

    # ── Step 3: Generate events.zip (h5 → text format for rpg_e2vid) ──────────
    # Reconstruction runs on Colab (GPU). Locally we only need the zip to upload.
    if [ -f "${PROC_DIR}/events.zip" ]; then
        echo "  Step 3: events.zip already exists — skipping"
        echo "    (delete ${PROC_DIR}/events.zip to force re-run)"
    else
        echo "  Step 3: Converting HDF5 → events.zip ..."
        export HDF5_PLUGIN_PATH="/tmp"
        python3 - << PYEOF
import sys
import os
os.environ['HDF5_PLUGIN_PATH'] = '/tmp'
sys.path.insert(0, '${ROOT}/scripts')
from reconstruct import convert_h5_to_zip
from pathlib import Path
import hdf5plugin
import h5py
convert_h5_to_zip(
    h5_path    = Path('${PROC_DIR}/events.h5'),
    zip_path   = Path('${PROC_DIR}/events.zip'),
    width      = 1280,
    height     = 720,
    start_s    = 5.0,
    duration_s = None,
)
PYEOF
        if [ $? -ne 0 ]; then
            echo "  ✗ events.zip generation failed for ${SEQ_NAME}"
            exit 1
        fi
        echo "    events.zip: $(du -h "${PROC_DIR}/events.zip" | cut -f1)"
    fi

    echo "  ✓ ${SEQ_NAME} done."
done

echo ""
echo "=== All sequences prepared. ==="
echo "Next step: bash scripts/sync_to_drive.sh"
