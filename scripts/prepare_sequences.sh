#!/usr/bin/env bash
# Prepare FRED sequences for the e2vid pipeline.
# For each sequence: extract events.hdf5 + coordinates.txt from the downloaded zip,
# verify the HDF5 structure, then generate events.zip for Kaggle reconstruction.
#
# Usage:
#   bash scripts/prepare_sequences.sh 44 45 46 47   # specific sequences
#   bash scripts/prepare_sequences.sh --all          # all zips in data/raw/zips/

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZIPS_DIR="${ROOT}/data/raw/zips"

# Per-sequence start_s overrides (default 5.0 for all others)
declare -A START_S_MAP=([47]=7.0)

if [ "${1:-}" = "--all" ]; then
    mapfile -t SEQUENCES < <(ls "$ZIPS_DIR"/*.zip 2>/dev/null | xargs -n1 basename | sed 's/\.zip//' | sort -n)
else
    SEQUENCES=("$@")
fi

if [ ${#SEQUENCES[@]} -eq 0 ]; then
    echo "Usage: $0 <seq_id> [seq_id ...] | --all"
    exit 1
fi

echo "=== Preparing ${#SEQUENCES[@]} sequence(s): ${SEQUENCES[*]} ==="

FAILED=()

for SEQ in "${SEQUENCES[@]}"; do
    ZIP="${ZIPS_DIR}/${SEQ}.zip"
    SEQ_NAME="sequence_${SEQ}"
    RAW_DIR="${ROOT}/data/raw/${SEQ_NAME}"
    PROC_DIR="${ROOT}/data/processed/${SEQ_NAME}"
    START_S="${START_S_MAP[$SEQ]:-5.0}"

    echo ""
    echo "──────────────────────────────────────────"
    echo "  ${SEQ_NAME}  (start_s=${START_S})"
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
        if ! unzip -p "$ZIP" "${SEQ}/Event/events.hdf5" > "${PROC_DIR}/events.h5"; then
            echo "  ✗ Failed to extract events.hdf5 — skipping sequence"
            FAILED+=("$SEQ")
            continue
        fi
        if ! unzip -p "$ZIP" "${SEQ}/coordinates.txt" > "${RAW_DIR}/coordinates.txt"; then
            echo "  ✗ Failed to extract coordinates.txt — skipping sequence"
            FAILED+=("$SEQ")
            continue
        fi
        echo "    events.h5       : $(du -h "${PROC_DIR}/events.h5" | cut -f1)"
        echo "    coordinates.txt : $(wc -l < "${RAW_DIR}/coordinates.txt") lines"
    fi

    # ── Step 2: Verify HDF5 structure ─────────────────────────────────────────
    echo "  Step 2: Verifying HDF5 structure ..."
    if ! python3 - << PYEOF
import h5py, sys
path = "${PROC_DIR}/events.h5"
try:
    with h5py.File(path, 'r') as f:
        if 'CD/events' not in f:
            print(f"  ✗ 'CD/events' not found. Keys: {list(f.keys())}")
            sys.exit(1)
        ev = f['CD/events']
        fields = list(ev.dtype.names) if hasattr(ev.dtype, 'names') and ev.dtype.names else list(ev.keys()) if hasattr(ev, 'keys') else []
        print(f"  ✓ CD/events: {len(ev):,} events, fields: {fields}")
except Exception as e:
    print(f"  ✗ Failed to read HDF5: {e}")
    sys.exit(1)
PYEOF
    then
        echo "  ✗ HDF5 verification failed for ${SEQ_NAME} — skipping"
        FAILED+=("$SEQ")
        continue
    fi

    # ── Step 3: Generate events.zip (h5 → text format for rpg_e2vid) ──────────
    if [ -f "${PROC_DIR}/events.zip" ]; then
        echo "  Step 3: events.zip already exists — skipping"
    else
        echo "  Step 3: Converting HDF5 → events.zip (start_s=${START_S}) ..."
        if ! python3 - << PYEOF
import sys
sys.path.insert(0, '${ROOT}/scripts')
from reconstruct import convert_h5_to_zip
from pathlib import Path
convert_h5_to_zip(
    h5_path    = Path('${PROC_DIR}/events.h5'),
    zip_path   = Path('${PROC_DIR}/events.zip'),
    width      = 1280,
    height     = 720,
    start_s    = ${START_S},
    duration_s = None,
)
PYEOF
        then
            echo "  ✗ events.zip generation failed for ${SEQ_NAME}"
            FAILED+=("$SEQ")
            continue
        fi
        echo "    events.zip: $(du -h "${PROC_DIR}/events.zip" | cut -f1)"
    fi

    # Delete events.h5 — no longer needed once events.zip exists
    if [ -f "${PROC_DIR}/events.h5" ]; then
        rm -f "${PROC_DIR}/events.h5"
        echo "  Deleted events.h5 (temp file)"
    fi

    echo "  ✓ ${SEQ_NAME} done."
done

echo ""
echo "=== All done. ==="
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed sequences (${#FAILED[@]}): ${FAILED[*]}"
    echo "Re-run with: bash scripts/prepare_sequences.sh ${FAILED[*]}"
    exit 1
fi
echo "Next step: bash scripts/sync_to_kaggle.sh"
