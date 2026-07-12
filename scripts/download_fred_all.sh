#!/usr/bin/env bash
# Download all 231 FRED sequences from HuggingFace.
# Skips sequences already present in data/raw/zips/.
# Runs up to PARALLEL concurrent downloads (default 3).
#
# Usage:
#   bash scripts/download_fred_all.sh            # 3 parallel
#   bash scripts/download_fred_all.sh 5          # 5 parallel
#   PARALLEL=1 bash scripts/download_fred_all.sh # sequential

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZIPS_DIR="${ROOT}/data/raw/zips"
PARALLEL="${1:-3}"
BASE_URL="https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/main"
CSV="${ROOT}/data/fred_dataset_analysis/fred_sequence_analysis.csv"
LOG="${ROOT}/data/logs/download_fred_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$ZIPS_DIR" "$(dirname "$LOG")"

# Build list of (id, split) from CSV, sorted numerically
mapfile -t SEQUENCES < <(
    awk -F',' 'NR>1 {gsub(/sequence_/,"",$1); print $1,$12}' "$CSV" | sort -k1 -n
)

echo "FRED download — $(date)" | tee "$LOG"
echo "Total sequences in CSV: ${#SEQUENCES[@]}" | tee -a "$LOG"
echo "Parallel downloads: $PARALLEL" | tee -a "$LOG"
echo "Zip dir: $ZIPS_DIR" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Count what we already have
HAVE=0
NEED=0
for entry in "${SEQUENCES[@]}"; do
    id="${entry%% *}"
    if [ -f "${ZIPS_DIR}/${id}.zip" ]; then
        HAVE=$((HAVE + 1))
    else
        NEED=$((NEED + 1))
    fi
done
echo "Already downloaded: $HAVE  |  Need to download: $NEED" | tee -a "$LOG"
echo "" | tee -a "$LOG"

download_one() {
    local id="$1" split="$2"
    local dest="${ZIPS_DIR}/${id}.zip"
    local url="${BASE_URL}/${split}/${id}.zip"
    local tmp="${dest}.tmp"

    if [ -f "$dest" ]; then
        echo "[SKIP] sequence_${id} already exists" | tee -a "$LOG"
        return 0
    fi

    echo "[START] sequence_${id} (${split})" | tee -a "$LOG"
    if curl -fL --retry 5 --retry-delay 10 --retry-max-time 300 \
             -o "$tmp" "$url" 2>>"$LOG"; then
        mv "$tmp" "$dest"
        SIZE=$(du -h "$dest" | cut -f1)
        echo "[OK]    sequence_${id} → ${SIZE}" | tee -a "$LOG"
    else
        rm -f "$tmp"
        echo "[FAIL]  sequence_${id} (${url})" | tee -a "$LOG"
        return 1
    fi
}

export -f download_one
export ZIPS_DIR BASE_URL LOG

PIDS=()
ACTIVE=0

for entry in "${SEQUENCES[@]}"; do
    id="${entry%% *}"
    split="${entry##* }"

    if [ -f "${ZIPS_DIR}/${id}.zip" ]; then
        continue
    fi

    # Wait if we've hit the parallel limit
    while [ "${#PIDS[@]}" -ge "$PARALLEL" ]; do
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                unset "PIDS[$i]"
            fi
        done
        PIDS=("${PIDS[@]+"${PIDS[@]}"}")
        [ "${#PIDS[@]}" -ge "$PARALLEL" ] && sleep 2 || true
    done

    download_one "$id" "$split" &
    PIDS+=($!)
done

# Wait for all remaining
for pid in "${PIDS[@]+"${PIDS[@]}"}"; do
    wait "$pid" || true
done

echo "" | tee -a "$LOG"
echo "=== Done: $(date) ===" | tee -a "$LOG"
DOWNLOADED=$(ls "${ZIPS_DIR}"/*.zip 2>/dev/null | wc -l)
echo "Zips on disk: $DOWNLOADED / ${#SEQUENCES[@]}" | tee -a "$LOG"

# Report any failures
FAIL_COUNT=$(grep -c "^\[FAIL\]" "$LOG" || true)
if [ "$FAIL_COUNT" -gt 0 ]; then
    echo ""
    echo "=== Failed downloads (re-run script to retry) ==="
    grep "^\[FAIL\]" "$LOG"
fi

echo "Log: $LOG"
