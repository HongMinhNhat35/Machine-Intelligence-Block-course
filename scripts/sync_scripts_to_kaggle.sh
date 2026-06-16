#!/usr/bin/env bash
# Upload scripts to Kaggle and push the notebook.
# Run this whenever reconstruct.py, train_yolo.py, or kaggle_launcher.ipynb changes.
# Much faster than sync_to_kaggle.sh (~30 KB vs ~2.1 GB).
#
# Usage:
#   bash scripts/sync_scripts_to_kaggle.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGING_DATASET="/tmp/fred-scripts-ami"
STAGING_NOTEBOOK="/tmp/ami-e2vid-notebook"
KAGGLE="${HOME}/.local/bin/kaggle"
if [ ! -x "$KAGGLE" ]; then
    KAGGLE="$(command -v kaggle)"
fi

# ── 1. Scripts dataset ────────────────────────────────────────────────────────
echo "=== Staging scripts ==="
rm -rf "$STAGING_DATASET"
mkdir -p "$STAGING_DATASET"

cp "${ROOT}/scripts/reconstruct.py" "${STAGING_DATASET}/reconstruct.py"
cp "${ROOT}/scripts/train_yolo.py"  "${STAGING_DATASET}/train_yolo.py"
echo "  reconstruct.py  ($(wc -c < "${ROOT}/scripts/reconstruct.py") bytes)"
echo "  train_yolo.py   ($(wc -c < "${ROOT}/scripts/train_yolo.py") bytes)"

cat > "${STAGING_DATASET}/dataset-metadata.json" <<'EOF'
{
  "title": "FRED Scripts AMI",
  "id": "Gennepy/fred-scripts-ami",
  "licenses": [{"name": "other"}]
}
EOF

echo ""
echo "=== Uploading scripts dataset ==="
if "$KAGGLE" datasets list --mine 2>/dev/null | grep -q "fred-scripts-ami"; then
    "$KAGGLE" datasets version -p "$STAGING_DATASET" -m "Updated $(date +%Y-%m-%d_%H%M%S)"
else
    "$KAGGLE" datasets create -p "$STAGING_DATASET"
fi

# ── 2. Notebook ───────────────────────────────────────────────────────────────
echo ""
echo "=== Pushing notebook ==="
rm -rf "$STAGING_NOTEBOOK"
mkdir -p "$STAGING_NOTEBOOK"

cp "${ROOT}/notebooks/kaggle_launcher.ipynb" "${STAGING_NOTEBOOK}/kaggle_launcher.ipynb"

cat > "${STAGING_NOTEBOOK}/kernel-metadata.json" <<'EOF'
{
  "id": "Gennepy/ami-e2vid-pipeline",
  "title": "AMI E2VID Pipeline",
  "code_file": "kaggle_launcher.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [
    "gennepy/fred-events-ami",
    "gennepy/fred-scripts-ami",
    "gennepy/fred-frames-all"
  ],
  "kernel_sources": []
}
EOF

"$KAGGLE" kernels push -p "$STAGING_NOTEBOOK"

echo ""
echo "=== Done ==="
echo "Scripts : https://www.kaggle.com/datasets/Gennepy/fred-scripts-ami"
echo "Notebook: https://www.kaggle.com/code/Gennepy/ami-e2vid-pipeline"
