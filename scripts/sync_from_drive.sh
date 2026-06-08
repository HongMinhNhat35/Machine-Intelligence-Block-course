#!/usr/bin/env bash
# Pull Colab training results from Google Drive into the local ami/ tree.
# Run this on your laptop after the Colab notebook finishes.
#
# Usage: bash scripts/sync_from_drive.sh

set -uo pipefail

RCLONE="${HOME}/.local/bin/rclone"
REMOTE="gdrive:ami"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # ami/ root

# Copy from Drive, silently skip if the source directory doesn't exist yet
rcopy() {
    "$RCLONE" copy "$1" "$2" --progress 2>&1 \
        | grep -v 'directory not found' || true
}

echo "=== Syncing from ${REMOTE} → ${ROOT} ==="

# Trained weights → e2vid service weights directory
echo ""
echo "--- Weights ---"
rcopy "${REMOTE}/data/yolo_e2vid.pt" "${ROOT}/services/e2vid/weights/"

# KPI logs (reconstruction KPIs per sequence + training KPI)
echo ""
echo "--- KPIs ---"
for SEQ in sequence_0 sequence_1 sequence_2 sequence_3 sequence_8; do
    rcopy "${REMOTE}/data/processed/${SEQ}/kpis/" "${ROOT}/data/kpis/"
done
rcopy "${REMOTE}/data/kpis/" "${ROOT}/data/kpis/"

# YOLO training run logs and plots
echo ""
echo "--- Training logs ---"
rcopy "${REMOTE}/data/yolo_runs/" "${ROOT}/data/yolo_runs/"

# Reconstructed frames (large — comment out if not needed locally)
echo ""
echo "--- Reconstructed frames ---"
for SEQ in sequence_0 sequence_1 sequence_2 sequence_3 sequence_8; do
    rcopy "${REMOTE}/data/processed/${SEQ}/reconstruction_e2vid/" \
          "${ROOT}/data/processed/${SEQ}/reconstruction_e2vid/"
done

echo ""
echo "=== Done. ==="
echo "Weights at: ${ROOT}/services/e2vid/weights/yolo_e2vid.pt"
echo "Next step : docker compose build e2vid && docker compose up -d"
