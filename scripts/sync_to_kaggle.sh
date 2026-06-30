#!/usr/bin/env bash
# Upload events.zip and coordinates.txt for Run 8 sequences to Kaggle.
# Run once to create; re-run to update (adds a new dataset version).
# Scripts (reconstruct.py, train_yolo.py) are in a separate dataset —
# update them with sync_scripts_to_kaggle.sh instead.
#
# Prerequisites:
#   pip install kaggle
#   Place kaggle.json in ~/.kaggle/kaggle.json  (chmod 600)
#
# Usage:
#   bash scripts/sync_to_kaggle.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGING="/tmp/fred-events-ami"
KAGGLE="${HOME}/.local/bin/kaggle"

# Fall back to system kaggle if not in ~/.local/bin
if [ ! -x "$KAGGLE" ]; then
    KAGGLE="$(command -v kaggle)"
fi

echo "=== Staging files ==="
rm -rf "$STAGING"

# events.zip per sequence
for SEQ in 84 85 124 201 127 44 45 46 47 146; do
    mkdir -p "${STAGING}/data/processed/sequence_${SEQ}"
    ZIP="${ROOT}/data/processed/sequence_${SEQ}/events.zip"
    if [ -f "$ZIP" ]; then
        cp "$ZIP" "${STAGING}/data/processed/sequence_${SEQ}/events.zip"
        echo "  sequence_${SEQ}/events.zip  $(du -h "$ZIP" | cut -f1)"
    else
        echo "  ✗ ${ZIP} not found — skipping"
    fi
done

# coordinates.txt per sequence
for SEQ in 84 85 124 201 127 44 45 46 47 146; do
    mkdir -p "${STAGING}/data/raw/sequence_${SEQ}"
    COORD="${ROOT}/data/raw/sequence_${SEQ}/coordinates.txt"
    if [ -f "$COORD" ]; then
        cp "$COORD" "${STAGING}/data/raw/sequence_${SEQ}/coordinates.txt"
        echo "  sequence_${SEQ}/coordinates.txt"
    else
        echo "  ✗ ${COORD} not found — skipping"
    fi
done

# dataset metadata
cat > "${STAGING}/dataset-metadata.json" <<'EOF'
{
  "title": "FRED Events AMI",
  "id": "Gennepy/fred-events-ami",
  "licenses": [{"name": "other"}]
}
EOF

echo ""
echo "=== Uploading to Kaggle ==="

# Create on first run, update on subsequent runs
if "$KAGGLE" datasets list --mine 2>/dev/null | grep -q "fred-events-ami"; then
    echo "Dataset exists — creating new version ..."
    "$KAGGLE" datasets version -p "$STAGING" --dir-mode zip -m "Updated $(date +%Y-%m-%d)"
else
    echo "Creating dataset for the first time ..."
    "$KAGGLE" datasets create -p "$STAGING" --dir-mode zip
fi

echo ""
echo "=== Done ==="
echo "Dataset will appear at: https://www.kaggle.com/datasets/Gennepy/fred-events-ami"
echo "Add it to your notebook: Notebook → + Add data → Your Datasets → fred-events-ami"
