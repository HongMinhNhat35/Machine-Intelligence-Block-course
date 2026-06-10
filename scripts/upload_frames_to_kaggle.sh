#!/usr/bin/env bash
# Upload all reconstructed frames (sequences 84/85/201/127/124) to Kaggle as
# dataset 'fred-frames-all'.  Run after:
#   bash scripts/sync_from_kaggle.sh --frames-zip   # downloads sequence_124 frames
#
# First run:  creates the dataset.
# Re-runs:    adds a new version.
#
# Usage:
#   bash scripts/upload_frames_to_kaggle.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_DIR="/tmp/fred-frames-all"
SLUG="fred-frames-all"
OWNER="gennepy"

SEQUENCES=(84 85 201 127 124)

echo "=== Preparing combined frames dataset ==="
rm -rf "$DATASET_DIR"
mkdir -p "$DATASET_DIR"

for SEQ in "${SEQUENCES[@]}"; do
    SRC="${ROOT}/data/processed/sequence_${SEQ}/reconstruction_e2vid"
    DST="${DATASET_DIR}/data/processed/sequence_${SEQ}/reconstruction_e2vid"
    if [ ! -d "$SRC" ] || [ "$(find "$SRC" -maxdepth 1 -name 'frame_*.jpg' | wc -l)" -eq 0 ]; then
        echo "  ERROR: sequence_${SEQ} frames not found at $SRC"
        echo "  Run: bash scripts/sync_from_kaggle.sh --frames-zip first"
        exit 1
    fi
    echo "  Linking sequence_${SEQ} ..."
    mkdir -p "$(dirname "$DST")"
    # Hard-link instead of copy to avoid doubling disk usage
    cp -rl "$SRC" "$DST"
    n=$(find "$DST" -maxdepth 1 -name 'frame_*.jpg' | wc -l)
    echo "    $n frames"
done

# Write dataset metadata
cat > "$DATASET_DIR/dataset-metadata.json" << EOF
{
  "title": "AMI E2VID Reconstructed Frames (all sequences)",
  "id": "${OWNER}/${SLUG}",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

echo ""
echo "=== Uploading to Kaggle ==="

# Create or update
if kaggle datasets list --user "$OWNER" --mine 2>/dev/null | grep -q "$SLUG"; then
    echo "Dataset exists — creating new version ..."
    kaggle datasets version -p "$DATASET_DIR" -m "All 5 sequences (84/85/201/127/124)" --dir-mode zip
else
    echo "Creating new dataset ..."
    kaggle datasets create -p "$DATASET_DIR" --dir-mode zip
fi

echo ""
echo "=== Done ==="
echo "Dataset: https://www.kaggle.com/datasets/${OWNER}/${SLUG}"
echo "Add it as input in the Kaggle notebook before running Run 2."
