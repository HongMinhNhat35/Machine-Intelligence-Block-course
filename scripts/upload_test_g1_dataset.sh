#!/usr/bin/env bash
# Upload the 5 FRED test sequence zips + fusion_event.pt to Kaggle as ami-test-g1.
# Test sequences: 8, 9, 12, 20, 21  (~2 GB total)
#
# Usage:
#   bash scripts/upload_test_g1_dataset.sh
#
# Prerequisites: kaggle CLI configured (~/.kaggle/kaggle.json)
set -euo pipefail

KAGGLE="${HOME}/.local/bin/kaggle"
if [ ! -x "$KAGGLE" ]; then KAGGLE="$(command -v kaggle)"; fi

DATASET_ID="ami-test-g1"
DATASET_TITLE="AMI Test G1 FRED test sequences event cache"
STAGING="/tmp/${DATASET_ID}_upload"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZIPS_DIR="${ROOT}/data/raw/zips"
WEIGHTS="${ROOT}/services/fusion/weights"

echo "=== Preparing staging area: $STAGING ==="
rm -rf "$STAGING"
mkdir -p "$STAGING"

for n in 8 9 12 20 21; do
    echo "  Copying ${n}.zip ($(du -sh "$ZIPS_DIR/${n}.zip" | cut -f1)) ..."
    cp "$ZIPS_DIR/${n}.zip" "$STAGING/${n}.zip"
done

echo "  Copying fusion_event.pt ($(du -sh "$WEIGHTS/fusion_event.pt" | cut -f1)) ..."
cp "$WEIGHTS/fusion_event.pt" "$STAGING/fusion_event.pt"

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
echo "=== Uploading to Kaggle (~2 GB) ==="
echo ""

if "$KAGGLE" datasets list --user gennepy --search "$DATASET_ID" 2>/dev/null | grep -q "$DATASET_ID"; then
    echo "Dataset exists — creating new version ..."
    "$KAGGLE" datasets version -p "$STAGING" -m "Initial test sequences (8,9,12,20,21) + fusion_event.pt run2" --dir-mode zip
else
    echo "Creating new dataset ..."
    "$KAGGLE" datasets create -p "$STAGING" --dir-mode zip
fi

echo ""
echo "=== Done. Dataset: https://www.kaggle.com/datasets/gennepy/${DATASET_ID} ==="
