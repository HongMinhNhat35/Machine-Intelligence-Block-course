#!/usr/bin/env bash
# Upload all prepared FRED sequences to Kaggle, split across two datasets.
#
#   Dataset 1: gennepy/fred-events-ami    (sequences up to ~13 GB)
#   Dataset 2: gennepy/fred-events-ami-2  (remaining sequences)
#
# Each dataset contains:
#   data/processed/sequence_N/events.zip
#   data/raw/sequence_N/coordinates.txt
#
# Run after prepare_sequences.sh --all is complete.
# Safe to re-run: creates a new dataset version each time.
#
# Usage:
#   bash scripts/sync_to_kaggle_all.sh

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KAGGLE="$(command -v kaggle)"
STAGING_1="/tmp/fred-events-ami-1"
STAGING_2="/tmp/fred-events-ami-2"
SIZE_LIMIT_BYTES=$((13 * 1024 * 1024 * 1024))   # 13 GB per dataset (safe below 20 GB limit)

echo "=== Collecting prepared sequences ==="

# All sequences with events.zip, sorted numerically
mapfile -t SEQS < <(
    find "$ROOT/data/processed" -name "events.zip" \
    | sed 's|.*/sequence_||; s|/events.zip||' \
    | sort -n
)

echo "Found ${#SEQS[@]} sequences with events.zip"
echo ""

# ── Split into two datasets by cumulative size ────────────────────────────────
SPLIT_IDX=0
CUMSIZE=0
for i in "${!SEQS[@]}"; do
    SEQ="${SEQS[$i]}"
    ZIP="$ROOT/data/processed/sequence_${SEQ}/events.zip"
    SZ=$(stat -c%s "$ZIP" 2>/dev/null || echo 0)
    CUMSIZE=$((CUMSIZE + SZ))
    if [ $CUMSIZE -gt $SIZE_LIMIT_BYTES ] && [ $SPLIT_IDX -eq 0 ]; then
        SPLIT_IDX=$i
        echo "Split at sequence_${SEQ} (index $i, cumulative $(( (CUMSIZE - SZ) / 1073741824 )) GB)"
    fi
done

if [ $SPLIT_IDX -eq 0 ]; then
    # Everything fits in one dataset
    SPLIT_IDX=${#SEQS[@]}
    echo "All sequences fit in one dataset"
fi

SEQS_1=("${SEQS[@]:0:$SPLIT_IDX}")
SEQS_2=("${SEQS[@]:$SPLIT_IDX}")

echo "Dataset 1: ${#SEQS_1[@]} sequences (${SEQS_1[0]}–${SEQS_1[-1]})"
[ ${#SEQS_2[@]} -gt 0 ] && echo "Dataset 2: ${#SEQS_2[@]} sequences (${SEQS_2[0]}–${SEQS_2[-1]})"
echo ""

# ── Stage files ───────────────────────────────────────────────────────────────
stage_dataset() {
    local STAGING="$1"; shift
    local SLUG="$1"; shift
    local SEQS=("$@")

    echo "=== Staging $SLUG (${#SEQS[@]} sequences) ==="
    rm -rf "$STAGING"

    local TOTAL=0
    local MISSING=0

    for SEQ in "${SEQS[@]}"; do
        ZIP="$ROOT/data/processed/sequence_${SEQ}/events.zip"
        COORD="$ROOT/data/raw/sequence_${SEQ}/coordinates.txt"

        if [ ! -f "$ZIP" ]; then
            echo "  ✗ sequence_${SEQ}: events.zip not found — skipping"
            MISSING=$((MISSING + 1))
            continue
        fi

        mkdir -p "${STAGING}/data/processed/sequence_${SEQ}"
        mkdir -p "${STAGING}/data/raw/sequence_${SEQ}"

        # Hard-link to avoid copying 13 GB
        ln "$ZIP" "${STAGING}/data/processed/sequence_${SEQ}/events.zip" 2>/dev/null \
            || cp "$ZIP" "${STAGING}/data/processed/sequence_${SEQ}/events.zip"

        if [ -f "$COORD" ]; then
            ln "$COORD" "${STAGING}/data/raw/sequence_${SEQ}/coordinates.txt" 2>/dev/null \
                || cp "$COORD" "${STAGING}/data/raw/sequence_${SEQ}/coordinates.txt"
        fi

        SZ=$(stat -c%s "$ZIP")
        TOTAL=$((TOTAL + SZ))
    done

    echo "  Total staged: $(( TOTAL / 1073741824 )) GB ($(( TOTAL / 1048576 )) MB)"
    [ $MISSING -gt 0 ] && echo "  Skipped (missing): $MISSING sequences"

    # Dataset metadata
    cat > "${STAGING}/dataset-metadata.json" <<EOF
{
  "title": "FRED Events AMI${SLUG##*-}",
  "id": "gennepy/${SLUG}",
  "licenses": [{"name": "other"}]
}
EOF
}

# Fix metadata title for dataset 1 (no suffix)
stage_dataset "$STAGING_1" "fred-events-ami" "${SEQS_1[@]}"
cat > "${STAGING_1}/dataset-metadata.json" <<'EOF'
{
  "title": "FRED Events AMI",
  "id": "gennepy/fred-events-ami",
  "licenses": [{"name": "other"}]
}
EOF

if [ ${#SEQS_2[@]} -gt 0 ]; then
    stage_dataset "$STAGING_2" "fred-events-ami-2" "${SEQS_2[@]}"
fi

echo ""

# ── Upload ────────────────────────────────────────────────────────────────────
upload_dataset() {
    local STAGING="$1"
    local SLUG="$2"

    echo "=== Uploading $SLUG ==="
    if "$KAGGLE" datasets list --mine 2>/dev/null | grep -q "$SLUG"; then
        echo "  Updating existing dataset..."
        "$KAGGLE" datasets version -p "$STAGING" --dir-mode zip \
            -m "All FRED sequences — $(date +%Y-%m-%d)"
    else
        echo "  Creating new dataset..."
        "$KAGGLE" datasets create -p "$STAGING" --dir-mode zip
    fi
    echo "  Done: https://www.kaggle.com/datasets/gennepy/${SLUG}"
}

upload_dataset "$STAGING_1" "fred-events-ami"

if [ ${#SEQS_2[@]} -gt 0 ]; then
    upload_dataset "$STAGING_2" "fred-events-ami-2"
fi

echo ""
echo "=== All uploads complete ==="
echo ""
echo "Next steps:"
echo "  1. Reconstruction notebook: add both datasets as inputs"
echo "  2. Run reconstruction → Kaggle saves frames as kernel output"
echo "  3. Training notebook: add reconstruction output + both event datasets"
echo "  4. Run training (YOLOv8n) → download best.pt + detections"
