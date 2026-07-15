#!/usr/bin/env bash
set -euo pipefail

RELEASE_V2="https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v2.0"
RELEASE_V5="https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v5.0"
RAW="https://raw.githubusercontent.com/HongMinhNhat35/Machine-Intelligence-Block-course/gui"
DEFAULT_RECON="$HOME/g1-ami-data"

echo "=== AMI Hybrid Vision System — Group 1 Installer ==="
echo ""

read -rp "Where should the reconstruction data be stored? [${DEFAULT_RECON}]: " RECON
RECON="${RECON:-$DEFAULT_RECON}"

echo ""
echo "The FRED raw dataset is required for Late Fusion and the Event/RGB detection modes."
echo "If you have it, enter its path (the folder containing sequence_8/, sequence_84/, …)."
echo "If not, press Enter — the demo works without it for e2vid/hypere2vid detection and comparison."
read -rp "Path to FRED raw dataset [${RECON}]: " FRED
FRED="${FRED:-$RECON}"

if ! mkdir -p "$RECON" 2>/dev/null; then
  echo "Error: cannot create '$RECON' — permission denied."
  echo "Choose a path inside your home directory (e.g. ~/g1-ami-data) or run with sudo."
  exit 1
fi
if [ ! -w "$RECON" ]; then
  echo "Error: '$RECON' exists but is not writable — check ownership/permissions."
  exit 1
fi

echo ""
echo "Downloading pre-computed E2VID reconstruction frames to $RECON ..."
echo "(Run 9 — events_per_pixel=0.1 · test sequences: 8,9,12,20,21 · prototyping: 84,85,124,127,201)"
echo ""

download_seq() {
    local rel=$1; local seq=$2; shift 2
    local parts=("$@")
    local n_parts=${#parts[@]}
    local target="$RECON/$seq/reconstruction_e2vid"
    if [ -d "$target" ] && [ -n "$(ls -A "$target" 2>/dev/null)" ]; then
        echo "  [skip] $seq — already exists"
        echo ""
        return
    fi
    mkdir -p "$RECON/$seq"
    local tmp_files=()
    local i=1
    for part in "${parts[@]}"; do
        if [ "$n_parts" -eq 1 ]; then
            echo "  [$seq] Downloading ..."
        else
            echo "  [$seq] Downloading part $i/$n_parts ..."
        fi
        curl -L --progress-bar -o "/tmp/$part" "$rel/$part"
        tmp_files+=("/tmp/$part")
        (( i++ )) || true
    done
    echo "  [$seq] Extracting ..."
    cat "${tmp_files[@]}" | tar x -C "$RECON/$seq/"
    rm "${tmp_files[@]}"
    echo "  [$seq] Done"
    echo ""
}

# e2vid reconstructions — Run 9 (events_per_pixel=0.1)
# Test sequences
download_seq "$RELEASE_V5" sequence_8   sequence_8_e2vid_run9.tar
download_seq "$RELEASE_V5" sequence_9   sequence_9_e2vid_run9.tar
download_seq "$RELEASE_V5" sequence_12  sequence_12_e2vid_run9.tar
download_seq "$RELEASE_V5" sequence_20  sequence_20_e2vid_run9.tar
download_seq "$RELEASE_V5" sequence_21  sequence_21_e2vid_run9.tar
# Prototyping sequences (HyperE2VID available for these)
download_seq "$RELEASE_V5" sequence_84  sequence_84_e2vid_run9.tar
download_seq "$RELEASE_V5" sequence_85  sequence_85_e2vid_run9.tar
download_seq "$RELEASE_V5" sequence_124 sequence_124_e2vid_run9.tar
download_seq "$RELEASE_V5" sequence_127 sequence_127_e2vid_run9.tar
download_seq "$RELEASE_V5" sequence_201 sequence_201_e2vid_run9.tar

echo ""
echo "The 10 sequences above are the demo set (5 test + 5 prototyping)."
echo "Group 1 ran E2VID reconstruction (Run 9) on the full FRED dataset split:"
echo "  40 training + 10 validation + 5 test sequences (~8 GB total)."
echo "Download the remaining 45 sequences to explore the complete dataset in the GUI."
read -rp "Download all 55 reconstructed sequences? [y/N]: " DL_ALL
DL_ALL="${DL_ALL:-N}"
if [[ "$DL_ALL" =~ ^[Yy] ]]; then
  echo ""
  echo "Downloading E2VID Run 9 — all 55 sequences (skipping already-downloaded ones) ..."
  ALL_SEQS=(0 2 5 8 9 12 14 16 18 20 21 25 28 31 33 38 44 47 49 54 58 62 67 71 75 79 81 84 85 89 95 99 101 104 108 116 120 124 125 127 131 135 140 145 149 150 157 161 168 172 176 182 186 192 199 201 204 211)
  for n in "${ALL_SEQS[@]}"; do
    download_seq "$RELEASE_V5" "sequence_${n}" "sequence_${n}_e2vid_run9.tar"
  done
  echo ""
fi

echo "Downloading HyperE2VID reconstruction frames (~290 MB, 5 sequences, Run 2) ..."
RELEASE_V3="https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v3.0"
for seq in sequence_84 sequence_85 sequence_124 sequence_127 sequence_201; do
  target="$RECON/$seq/reconstruction_hypere2vid"
  if [ -d "$target" ] && [ -n "$(ls -A "$target" 2>/dev/null)" ]; then
    echo "  [skip] $seq hypere2vid — already exists"
    continue
  fi
  echo "  [$seq] Downloading HyperE2VID frames ..."
  curl -L --progress-bar -o "/tmp/${seq}_hypere2vid.tar" "$RELEASE_V3/${seq}_hypere2vid.tar"
  echo "  [$seq] Extracting ..."
  tar xf "/tmp/${seq}_hypere2vid.tar" -C "$RECON/$seq/"
  rm "/tmp/${seq}_hypere2vid.tar"
  echo "  [$seq] Done"
done
echo ""
echo "Downloading pre-cached HyperE2VID detections (Run 2 — YOLOv8n, mAP@0.5=56.7%) ..."
curl -L --progress-bar -o /tmp/detections_hypere2vid_run2.tar.gz "$RELEASE_V3/detections_hypere2vid_run2.tar.gz"
tar xzf /tmp/detections_hypere2vid_run2.tar.gz -C "$RECON/"
rm /tmp/detections_hypere2vid_run2.tar.gz
echo ""

echo "Downloading pre-cached E2VID detections (demo sequences — all 10) ..."
curl -L --progress-bar -o /tmp/detections_e2vid_demo.tar.gz "$RELEASE_V5/detections_e2vid_demo.tar.gz"
tar xzf /tmp/detections_e2vid_demo.tar.gz -C "$RECON/"
rm /tmp/detections_e2vid_demo.tar.gz
echo ""

echo "Downloading pre-cached Late Fusion detections (Run 1 — YOLOv8n RGB+Event) ..."
RELEASE_V4="https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v4.0"
curl -L --progress-bar -o /tmp/detections_fusion_run1.tar.gz "$RELEASE_V4/detections_fusion_run1.tar.gz"
tar xzf /tmp/detections_fusion_run1.tar.gz -C "$RECON/"
rm /tmp/detections_fusion_run1.tar.gz
echo ""

echo "Downloading docker-compose.yml ..."
curl -fsSL -o "$RECON/docker-compose.yml" "$RAW/docker-compose.deploy.yml"

echo "Writing .env ..."
cat > "$RECON/.env" <<EOF
FRED_DATA_PATH=$FRED
RECON_DATA_PATH=$RECON
EOF

echo ""
echo "Pulling Docker images ..."
cd "$RECON"
docker compose pull

echo ""
echo "=== Setup complete ==="
echo ""
echo "  Next:"
echo "    cd $RECON"
echo "    docker compose up -d"
echo "    open http://localhost:8080"
echo ""
echo "  ⚠  Run 'docker compose up -d' only once. If the stack is already"
echo "     running, do not start it again — stop it first with:"
echo "     docker compose down"
echo ""
echo "Note: run 'cd $RECON' in your terminal — the script cannot change"
echo "      your shell's directory for you."
echo ""
cd "$RECON"
