import os
import requests
import streamlit as st
from pathlib import Path

E2VID_URL      = os.getenv("E2VID_URL",      "http://e2vid:8001")
HYPERE2VID_URL = os.getenv("HYPERE2VID_URL", "http://hypere2vid:8002")
FUSION_URL     = os.getenv("FUSION_URL",     "http://fusion:8003")

FRED_ROOT  = Path("/data/fred")
RECON_ROOT = Path("/data/recon")

st.set_page_config(page_title="Hybrid Vision System", layout="wide")
st.title("Hybrid Vision System")

# ── Service health ────────────────────────────────────────────────────────────
st.subheader("Service status")
cols = st.columns(3)
for col, (name, url) in zip(cols, [
    ("e2vid",      E2VID_URL),
    ("hypere2vid", HYPERE2VID_URL),
    ("fusion",     FUSION_URL),
]):
    try:
        r = requests.get(f"{url}/health", timeout=2)
        data = r.json()
        if data.get("weights_loaded"):
            col.success(f"**{name}** — ok · weights loaded")
        else:
            col.warning(f"**{name}** — ok · weights missing")
    except Exception:
        col.error(f"**{name}** — unreachable")

# ── Data volume status ────────────────────────────────────────────────────────
st.subheader("Data volumes")

# FRED raw data
fred_seqs = sorted(FRED_ROOT.glob("sequence_*")) if FRED_ROOT.exists() else []
if fred_seqs:
    st.success(f"FRED data: {len(fred_seqs)} sequence(s) found — {', '.join(s.name for s in fred_seqs)}")
else:
    st.error("FRED data: not found — check FRED_DATA_PATH mount")

# Reconstructions
st.markdown("**Reconstructions**")
recon_cols = st.columns(2)
for col, (label, subdir) in zip(recon_cols, [
    ("e2vid",      "reconstruction_e2vid"),
    ("HyperE2VID", "reconstruction_hypere2vid"),
]):
    ready = []
    for seq in fred_seqs:
        seq_id = seq.name
        recon_path = RECON_ROOT / seq_id / subdir
        if recon_path.exists() and any(recon_path.iterdir()):
            ready.append(seq_id)
    if ready:
        col.success(f"{label}: {len(ready)} sequence(s) ready — {', '.join(ready)}")
    else:
        col.warning(f"{label}: no reconstructed frames found — run Colab notebook first")

st.info("Sequence browser and detection view coming soon")
