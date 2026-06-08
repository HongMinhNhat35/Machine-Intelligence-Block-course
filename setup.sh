#!/usr/bin/env bash
# Hybrid vision system — first-time setup
# Writes a .env file that docker compose reads automatically

set -e

DEFAULT_FRED="/home/bani/ami/data/raw"
DEFAULT_RECON="/home/bani/ami/data/processed"

read -rp "FRED_DATA_PATH [${DEFAULT_FRED}]: " FRED_INPUT
FRED_DATA_PATH="${FRED_INPUT:-$DEFAULT_FRED}"

read -rp "RECON_DATA_PATH [${DEFAULT_RECON}]: " RECON_INPUT
RECON_DATA_PATH="${RECON_INPUT:-$DEFAULT_RECON}"

cat > .env <<EOF
FRED_DATA_PATH=${FRED_DATA_PATH}
RECON_DATA_PATH=${RECON_DATA_PATH}
EOF

echo ".env written:"
cat .env
