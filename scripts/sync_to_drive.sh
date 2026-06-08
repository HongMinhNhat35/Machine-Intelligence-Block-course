#!/usr/bin/env bash
# Push local files to Google Drive so the Colab notebook can run.
# Run this on your laptop before opening colab_launcher.ipynb.
#
# Usage: bash scripts/sync_to_drive.sh

set -uo pipefail

RCLONE="${HOME}/.local/bin/rclone"
REMOTE="gdrive:ami"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # ami/ root

rcopy() {
    "$RCLONE" copy "$1" "$2" --progress
}

echo "=== Syncing ${ROOT} → ${REMOTE} ==="

# ── Scripts ───────────────────────────────────────────────────────────────────
echo ""
echo "--- Scripts ---"
rcopy "${ROOT}/scripts/reconstruct.py"       "${REMOTE}/scripts/"
rcopy "${ROOT}/scripts/train_yolo.py"        "${REMOTE}/scripts/"

# ── Notebook ──────────────────────────────────────────────────────────────────
echo ""
echo "--- Notebook ---"
rcopy "${ROOT}/notebooks/colab_launcher.ipynb" "${REMOTE}/notebooks/"

# ── Event zips (one per sequence, used by reconstruct.py on Colab) ────────────
echo ""
echo "--- Event zips ---"
for SEQ in sequence_0 sequence_1 sequence_2 sequence_3 sequence_8; do
    ZIP="${ROOT}/data/processed/${SEQ}/events.zip"
    if [ -f "$ZIP" ]; then
        echo "  ${SEQ}: $(du -h "$ZIP" | cut -f1)"
        rcopy "$ZIP" "${REMOTE}/data/processed/${SEQ}/"
    else
        echo "  ${SEQ}: events.zip not found — skipping (run prepare_sequences.sh first)"
    fi
done

# ── Annotations ───────────────────────────────────────────────────────────────
echo ""
echo "--- Annotations ---"
for SEQ in sequence_0 sequence_1 sequence_2 sequence_3 sequence_8; do
    COORDS="${ROOT}/data/raw/${SEQ}/coordinates.txt"
    if [ -f "$COORDS" ]; then
        rcopy "$COORDS" "${REMOTE}/data/raw/${SEQ}/"
    else
        echo "  ${SEQ}: coordinates.txt not found — skipping"
    fi
done

echo ""
echo "=== Done. ==="
echo "Next step: open notebooks/colab_launcher.ipynb in Colab and run all cells."
