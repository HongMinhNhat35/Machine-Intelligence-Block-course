#!/usr/bin/env bash
# Upload the 5 FRED fusion sequence zips + weights to Kaggle as dataset ami-fusion-g1.
# Source zips: data/raw/zips/{84,85,124,127,201}.zip  (~6.6 GB total)
# Run once; rerun to version-bump the dataset.
#
# Usage:
#   bash scripts/upload_fusion_g1_dataset.sh
#
# Prerequisites: kaggle CLI configured (~/.kaggle/kaggle.json)
set -euo pipefail

KAGGLE="${HOME}/.local/bin/kaggle"
if [ ! -x "$KAGGLE" ]; then KAGGLE="$(command -v kaggle)"; fi

DATASET_ID="ami-fusion-g1"
DATASET_TITLE="AMI Fusion G1 FRED sequences"
STAGING="/tmp/${DATASET_ID}_upload"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZIPS_DIR="${ROOT}/data/raw/zips"
WEIGHTS="${ROOT}/services/fusion/weights"

echo "=== Preparing staging area: $STAGING ==="
rm -rf "$STAGING"
mkdir -p "$STAGING"

# Copy the 5 FRED sequence zips
for n in 84 85 124 127 201; do
    echo "  Copying ${n}.zip ($(du -sh "$ZIPS_DIR/${n}.zip" | cut -f1)) ..."
    cp "$ZIPS_DIR/${n}.zip" "$STAGING/${n}.zip"
done

# Copy fusion weights
echo "  Copying weights ($(du -sh "$WEIGHTS" | cut -f1)) ..."
cp "$WEIGHTS/fusion_rgb.pt"   "$STAGING/fusion_rgb.pt"
cp "$WEIGHTS/fusion_event.pt" "$STAGING/fusion_event.pt"

# Write dataset metadata
cat > "$STAGING/dataset-metadata.json" <<EOF
{
  "id": "gennepy/${DATASET_ID}",
  "title": "${DATASET_TITLE}",
  "licenses": [{"name": "other"}],
  "isPrivate": true
}
EOF

echo ""
echo "=== Staging complete. Total size: $(du -sh "$STAGING" | cut -f1) ==="
echo "=== Uploading to Kaggle (~6.6 GB — may take 20-40 min) ==="
echo ""

# Create or version-update
if "$KAGGLE" datasets list --user gennepy --search "$DATASET_ID" 2>/dev/null | grep -q "$DATASET_ID"; then
    echo "Dataset exists — creating new version ..."
    "$KAGGLE" datasets version -p "$STAGING" -m "Updated fusion_event.pt to run2 (YOLOv8n, A100, mAP50=0.875 test)" --dir-mode zip
else
    echo "Creating new dataset ..."
    "$KAGGLE" datasets create -p "$STAGING" --dir-mode zip
fi

echo ""
echo "=== Done. Dataset: https://www.kaggle.com/datasets/gennepy/${DATASET_ID} ==="
