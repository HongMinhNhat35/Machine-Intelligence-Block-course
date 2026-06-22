#!/usr/bin/env bash
set -euo pipefail

RELEASE="https://github.com/HongMinhNhat35/Machine-Intelligence-Block-course/releases/download/v2.0"
RAW="https://raw.githubusercontent.com/HongMinhNhat35/Machine-Intelligence-Block-course/gui"
DEFAULT_RECON="$HOME/g1-ami-data"

echo "=== AMI Hybrid Vision System — Group 1 Installer ==="
echo ""

read -rp "Where should the reconstruction data be stored? [${DEFAULT_RECON}]: " RECON
RECON="${RECON:-$DEFAULT_RECON}"

echo ""
echo "The FRED raw dataset (events.h5, coordinates.txt) is required for late fusion."
echo "If you have it, enter its path (the folder containing sequence_84/, sequence_127/, …)."
echo "If not, press Enter — the demo works without it for e2vid detection and comparison."
read -rp "Path to FRED raw dataset [${RECON}]: " FRED
FRED="${FRED:-$RECON}"

mkdir -p "$RECON"

echo ""
echo "Downloading pre-computed E2VID reconstruction frames to $RECON ..."
echo "(Run 7 — events_per_pixel=0.1 — ~1.5 GB total)"
echo ""

download_seq() {
    local seq=$1; shift
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
        curl -L --progress-bar -o "/tmp/$part" "$RELEASE/$part"
        tmp_files+=("/tmp/$part")
        (( i++ )) || true
    done
    echo "  [$seq] Extracting ..."
    cat "${tmp_files[@]}" | tar x -C "$RECON/$seq/"
    rm "${tmp_files[@]}"
    echo "  [$seq] Done"
    echo ""
}

# e2vid reconstructions — Run 7 (events_per_pixel=0.1)
download_seq sequence_84  sequence_84.tar
download_seq sequence_85  sequence_85.tar
download_seq sequence_124 sequence_124.tar
download_seq sequence_127 sequence_127.tar
download_seq sequence_201 sequence_201.tar

echo "Downloading pre-cached detections (Run 7 — YOLOv8s, mAP@0.5=93.6%) ..."
curl -L --progress-bar -o /tmp/detections_e2vid_run7.tar.gz "$RELEASE/detections_e2vid_run7.tar.gz"
tar xzf /tmp/detections_e2vid_run7.tar.gz -C "$RECON/"
rm /tmp/detections_e2vid_run7.tar.gz
echo ""

echo "Downloading docker-compose.yml ..."
curl -fsSL -o "$RECON/docker-compose.yml" "$RAW/docker-compose.yml"

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
