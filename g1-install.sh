#!/usr/bin/env bash
set -euo pipefail

GL_PROJECT="https://gitlab.lrz.de/ldv/teaching/ami/ami2026/group01"
GL_RAW="$GL_PROJECT/-/raw/gui"
GL_PKG="https://gitlab.lrz.de/api/v4/projects/269843/packages/generic/data/1.0"
GL_REGISTRY="gitlab.lrz.de:5005"
DEFAULT_RECON="$HOME/g1-ami-data"

echo "=== AMI Hybrid Vision System — Group 1 Installer ==="
echo ""
echo "This installer pulls from the LRZ GitLab project."
echo "You need an LRZ account and a Personal Access Token with read_registry"
echo "and read_api scopes. Create one at:"
echo "  https://gitlab.lrz.de/-/user_settings/personal_access_tokens"
echo ""

read -rp  "LRZ username (e.g. go49kon): " GL_USER
read -rsp "Personal Access Token: " GL_TOKEN
echo ""

# Validate token
echo "Checking token ..."
HTTP=$(curl -s -o /dev/null -w "%{http_code}" \
  --header "PRIVATE-TOKEN: $GL_TOKEN" \
  "https://gitlab.lrz.de/api/v4/projects/269843")
if [ "$HTTP" != "200" ]; then
  echo "Error: token validation failed (HTTP $HTTP). Check your username and token."
  exit 1
fi
echo "Token OK."
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

gl_curl() {
  curl -fsSL --header "PRIVATE-TOKEN: $GL_TOKEN" "$@"
}

echo ""
echo "Logging in to GitLab container registry ..."
echo "$GL_TOKEN" | docker login "$GL_REGISTRY" --username "$GL_USER" --password-stdin

echo ""
echo "Downloading docker-compose.yml ..."
gl_curl -o "$RECON/docker-compose.yml" "$GL_RAW/docker-compose.deploy.yml"

echo "Downloading GUI static files ..."
mkdir -p "$RECON/static"
for f in app-admin.js app-compare.js app-core.js app-detect.js app-recon.js app-upload.js app-utils.js index.html style.css gui_user_guide.html fusion_viz.jpeg; do
  gl_curl -o "$RECON/static/$f" "$GL_RAW/services/web/static/$f"
done

echo "Downloading KPI files ..."
mkdir -p "$RECON/kpis"
for f in e2vid_run2.json e2vid_run3.json e2vid_run4.json e2vid_run5.json e2vid_run6.json e2vid_run7.json e2vid_run8.json e2vid_run9.json e2vid_run10.json fusion_event_run1.json fusion_event_run2.json fusion_rgb_run1.json fusion_rgb_run2.json fusion_run1.json fusion_run2.json hypere2vid_run1.json hypere2vid_run2.json; do
  gl_curl -o "$RECON/kpis/$f" "$GL_RAW/services/web/kpis/$f"
done

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
echo "  If you already have the reconstruction data, you can stop here (Ctrl+C)."
echo "  cd $RECON && docker compose up -d"
echo ""
echo "Downloading pre-computed reconstruction frames (~1.6 GB, all sequences) ..."
echo "(E2VID Run 9 + HyperE2VID Run 2)"
echo ""

if [ -d "$RECON/sequence_8/reconstruction_e2vid" ] && [ -n "$(ls -A "$RECON/sequence_8/reconstruction_e2vid" 2>/dev/null)" ]; then
  echo "  [skip] Reconstruction frames already present in $RECON"
else
  curl -L --progress-bar \
    --header "PRIVATE-TOKEN: $GL_TOKEN" \
    -o /tmp/recon_frames.tar.gz \
    "$GL_PKG/recon_frames.tar.gz"
  echo "Extracting ..."
  tar xzf /tmp/recon_frames.tar.gz -C "$RECON/"
  rm /tmp/recon_frames.tar.gz
  echo "Done."
fi

echo ""
echo "Downloading pre-cached detections — all models, all demo sequences (~5 MB) ..."
curl -L --progress-bar \
  --header "PRIVATE-TOKEN: $GL_TOKEN" \
  -o /tmp/detections_all.tar.gz \
  "$GL_PKG/detections_all_sequences.tar.gz"
tar xzf /tmp/detections_all.tar.gz -C "$RECON/"
rm /tmp/detections_all.tar.gz
echo ""

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
