#!/usr/bin/env bash
# Download results from Kaggle after a pipeline run.
#
# The Kaggle kernel produces ~80k output files (40k frames + 40k label .txt).
# `kaggle kernels output` only fetches one API page and misses everything past
# sequence_127 alphabetically. This script uses a Python paginator instead.
#
# Usage:
#   bash scripts/sync_from_kaggle.sh            # weights, KPIs, logs, training plots
#   bash scripts/sync_from_kaggle.sh --frames   # also download reconstructed JPEG frames (~2.5 GB)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMPDIR="/tmp/kaggle_output"
PYBIN="/home/bani/.local/share/pipx/venvs/kaggle/bin/python3"
SCRIPT="$(mktemp /tmp/kaggle_dl_XXXXXX.py)"

trap "rm -f $SCRIPT" EXIT

DOWNLOAD_FRAMES_ARG=""
DOWNLOAD_FRAMES_ZIP=false
if [[ "${1:-}" == "--frames" ]]; then
    DOWNLOAD_FRAMES_ARG="--frames"
elif [[ "${1:-}" == "--frames-zip" ]]; then
    DOWNLOAD_FRAMES_ZIP=true
fi

# ── Inline Python paginator ───────────────────────────────────────────────────
cat > "$SCRIPT" << 'PYEOF'
#!/usr/bin/env python3
"""Paginated download of Kaggle kernel outputs."""
import os, re, sys, time, requests, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

KAGGLE_LIB = "/home/bani/.local/share/pipx/venvs/kaggle/lib/python3.12/site-packages"
sys.path.insert(0, KAGGLE_LIB)
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

KERNEL          = "gennepy/ami-e2vid-pipeline"
OUT_DIR         = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/kaggle_output")
DOWNLOAD_FRAMES = "--frames" in sys.argv
FRAME_PAT       = re.compile(r"frame_\d+\.(jpg|png)$")
LABEL_PAT       = re.compile(r"frame_\d+\.txt$")
PAGE_DELAY      = 0.3
WORKERS         = 16    # parallel download threads
RETRY_DELAYS    = [5, 15, 30, 60]
NET_RETRY_DELAYS= [3, 10, 30, 60]

OUT_DIR.mkdir(parents=True, exist_ok=True)

api = KaggleApi()
api.authenticate()
owner_slug, kernel_slug, _ = api.parse_kernel_string(KERNEL)

print_lock = threading.Lock()
def log(msg):
    with print_lock:
        print(msg, flush=True)

def list_page(kc, owner, slug, token):
    req = ApiListKernelSessionOutputRequest()
    req.user_name, req.kernel_slug = owner, slug
    if token:
        req.page_token = token
    for delay in RETRY_DELAYS:
        try:
            return kc.kernels.kernels_api_client.list_kernel_session_output(req)
        except Exception as e:
            if "429" in str(e):
                log(f"  Rate limited — retrying in {delay}s ...")
                time.sleep(delay)
            else:
                raise
    return kc.kernels.kernels_api_client.list_kernel_session_output(req)

def download_one(fname, url):
    outfile = OUT_DIR / fname
    if outfile.exists() and FRAME_PAT.search(fname):
        return 'skip', fname
    outfile.parent.mkdir(parents=True, exist_ok=True)
    for delay in NET_RETRY_DELAYS:
        try:
            dl = requests.get(url, stream=True, timeout=60)
            dl.raise_for_status()
            outfile.write_bytes(dl.content)
            return 'ok', fname
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout, OSError) as e:
            log(f"  Network error ({e.__class__.__name__}) on {fname} — retrying in {delay}s ...")
            time.sleep(delay)
    return 'fail', fname

# ── Phase 1: collect all URLs by paging the API ───────────────────────────────
page_token, page, total = "", 0, 0
to_download = []   # list of (fname, url)
log(f"Downloading from {KERNEL} → {OUT_DIR}")
log(f"Frames: {'YES' if DOWNLOAD_FRAMES else 'NO'}\n")
log("Phase 1: collecting file list ...")

with api.build_kaggle_client() as kaggle:
    log_written = False
    while True:
        resp = list_page(kaggle, owner_slug, kernel_slug, page_token)
        files = resp.files or []
        total += len(files)
        page  += 1

        if not log_written and resp.log:
            (OUT_DIR / f"{kernel_slug}.log").write_text(resp.log)
            log_written = True

        for item in files:
            fname = item.file_name
            if LABEL_PAT.search(fname):
                continue
            if FRAME_PAT.search(fname) and not DOWNLOAD_FRAMES:
                continue
            if not (OUT_DIR / fname).exists():
                to_download.append((fname, item.url))

        if page % 20 == 0:
            log(f"  ... page {page}, {total} files seen, {len(to_download)} queued")

        next_token = resp.next_page_token
        if not next_token:
            break
        page_token = next_token
        time.sleep(PAGE_DELAY)

log(f"Phase 1 done: {total} files seen, {len(to_download)} to download\n")

# ── Phase 2: parallel download ────────────────────────────────────────────────
if not to_download:
    log("Nothing to download — all files already present.")
else:
    log(f"Phase 2: downloading {len(to_download)} files with {WORKERS} threads ...")
    downloaded, failed = 0, 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(download_one, fname, url): fname
                   for fname, url in to_download}
        for fut in as_completed(futures):
            status, fname = fut.result()
            if status == 'ok':
                downloaded += 1
                if downloaded % 500 == 0 or downloaded <= 10:
                    log(f"  [{downloaded}/{len(to_download)}] {fname}")
            elif status == 'fail':
                failed += 1
                log(f"  FAILED: {fname}")

    log(f"\n=== Done: {downloaded} downloaded, {failed} failed, {total} total files seen ===")
PYEOF

# ── Frames zip shortcut ───────────────────────────────────────────────────────
if [ "$DOWNLOAD_FRAMES_ZIP" = true ]; then
    echo "=== Downloading frames_e2vid.zip ==="
    # Use Python paginator to find and download the zip (kernels output --file-pattern
    # is unreliable for large outputs; we use a direct API download instead).
    ZIPFILE="$TMPDIR/frames_e2vid.zip"
    if [ ! -f "$ZIPFILE" ]; then
        python3 - "$TMPDIR" << 'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, "/home/bani/.local/share/pipx/venvs/kaggle/lib/python3.12/site-packages")
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
import requests

api = KaggleApi(); api.authenticate()
out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True)
owner, slug, _ = api.parse_kernel_string("gennepy/ami-e2vid-pipeline")

with api.build_kaggle_client() as kc:
    token = ""
    while True:
        req = ApiListKernelSessionOutputRequest()
        req.user_name, req.kernel_slug = owner, slug
        if token: req.page_token = token
        resp = kc.kernels.kernels_api_client.list_kernel_session_output(req)
        for item in (resp.files or []):
            if item.file_name == "frames_e2vid.zip":
                print(f"Found: {item.file_name}")
                r = requests.get(item.url, stream=True, timeout=300)
                r.raise_for_status()
                dest = out_dir / "frames_e2vid.zip"
                total = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(65536):
                        f.write(chunk); total += len(chunk)
                        if total % (256*1024*1024) == 0:
                            print(f"  {total//1024//1024} MB ...")
                print(f"Downloaded: {dest}  ({total//1024//1024} MB)")
                sys.exit(0)
        token = resp.next_page_token
        if not token: break
print("frames_e2vid.zip not found in output"); sys.exit(1)
PYEOF
    else
        echo "frames_e2vid.zip already present, skipping download."
    fi

    if [ -f "$ZIPFILE" ]; then
        echo "Unzipping frames ..."
        unzip -q "$ZIPFILE" -d "$TMPDIR/frames_unzipped"
        for SEQ in 84 85 201 127 124; do
            # zip stores: data/processed/sequence_N/reconstruction_e2vid/frame_*.jpg
            SRC="$TMPDIR/frames_unzipped/data/processed/sequence_${SEQ}/reconstruction_e2vid"
            DST="${ROOT}/data/processed/sequence_${SEQ}/reconstruction_e2vid"
            if [ -d "$SRC" ]; then
                mkdir -p "$DST"
                rsync -a "$SRC/" "$DST/"
                n=$(find "$DST" -maxdepth 1 -name "frame_*.jpg" | wc -l)
                echo "  sequence_${SEQ}: ${n} frames → $DST"
            else
                echo "  sequence_${SEQ}: not found in zip"
            fi
        done
        echo "=== Done ==="
    else
        echo "ERROR: frames_e2vid.zip download failed"
        exit 1
    fi
    exit 0
fi

# ── Run paginator ─────────────────────────────────────────────────────────────
mkdir -p "$TMPDIR"
"$PYBIN" "$SCRIPT" "$TMPDIR" $DOWNLOAD_FRAMES_ARG

# ── Copy weights ───────────────────────────────────────────────────────────────
echo ""
echo "--- Weights ---"
PT="$TMPDIR/yolo_e2vid.pt"
if [ -f "$PT" ]; then
    mkdir -p "${ROOT}/services/e2vid/weights"
    cp "$PT" "${ROOT}/services/e2vid/weights/yolo_e2vid.pt"
    echo "  yolo_e2vid.pt → services/e2vid/weights/  ($(du -h "$PT" | cut -f1))"
else
    echo "  yolo_e2vid.pt not found in output"
fi

# ── Copy KPIs ─────────────────────────────────────────────────────────────────
echo ""
echo "--- KPIs ---"
for SEQ in 84 85 201 127; do
    SRC="$TMPDIR/data/processed/sequence_${SEQ}/kpis"
    if [ -d "$SRC" ]; then
        mkdir -p "${ROOT}/data/kpis"
        cp -r "$SRC/." "${ROOT}/data/kpis/"
        echo "  sequence_${SEQ} KPIs → data/kpis/"
    fi
done
if [ -d "$TMPDIR/kpis" ]; then
    mkdir -p "${ROOT}/data/kpis"
    cp -r "$TMPDIR/kpis/." "${ROOT}/data/kpis/"
    echo "  kpis/ → data/kpis/"
fi

# ── Copy training runs ─────────────────────────────────────────────────────────
echo ""
echo "--- Training runs ---"
if [ -d "$TMPDIR/yolo_runs" ]; then
    mkdir -p "${ROOT}/data/yolo_runs"
    cp -r "$TMPDIR/yolo_runs/." "${ROOT}/data/yolo_runs/"
    echo "  yolo_runs/ → data/yolo_runs/"
fi

# ── Copy logs ─────────────────────────────────────────────────────────────────
echo ""
echo "--- Logs ---"
if [ -d "$TMPDIR/logs" ]; then
    mkdir -p "${ROOT}/data/logs"
    cp -r "$TMPDIR/logs/." "${ROOT}/data/logs/"
    echo "  logs/ → data/logs/"
fi
if [ -f "$TMPDIR/ami-e2vid-pipeline.log" ]; then
    mkdir -p "${ROOT}/data/logs"
    cp "$TMPDIR/ami-e2vid-pipeline.log" "${ROOT}/data/logs/"
    echo "  ami-e2vid-pipeline.log → data/logs/"
fi

# ── Copy frames ───────────────────────────────────────────────────────────────
if [[ "${DOWNLOAD_FRAMES_ARG}" == "--frames" ]]; then
    echo ""
    echo "--- Reconstructed frames ---"
    for SEQ in 84 85 201 127; do
        SRC="$TMPDIR/data/processed/sequence_${SEQ}/reconstruction_e2vid"
        DST="${ROOT}/data/processed/sequence_${SEQ}/reconstruction_e2vid"
        if [ -d "$SRC" ]; then
            mkdir -p "$DST"
            rsync -a --include="frame_*.jpg" --include="frame_*.png" \
                  --include="timestamps.txt" --exclude="*" "$SRC/" "$DST/"
            n=$(find "$DST" -name "frame_*.jpg" -o -name "frame_*.png" 2>/dev/null | wc -l)
            echo "  sequence_${SEQ}: ${n} frames  ($(du -sh "$DST" | cut -f1))"
        else
            echo "  sequence_${SEQ}: not found in output"
        fi
    done
else
    echo ""
    echo "Skipping frames (add --frames to download ~2.5 GB of reconstructed JPEGs)"
fi

echo ""
echo "=== Done ==="
echo "Weights at : ${ROOT}/services/e2vid/weights/yolo_e2vid.pt"
echo "Next step  : docker compose build e2vid && docker compose up -d"
