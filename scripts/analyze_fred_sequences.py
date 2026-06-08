"""
analyze_fred_sequences.py

Downloads only coordinates.txt (~150 KB) from each FRED sequence zip on
HuggingFace using HTTP Range requests — skips the multi-GB events data.
Then ranks all sequences by training/validation suitability.

Usage:
    pip install huggingface_hub requests numpy pandas
    python scripts/analyze_fred_sequences.py
    python scripts/analyze_fred_sequences.py --plot       # save PNG charts
    python scripts/analyze_fred_sequences.py --cache-dir /tmp/fred_coords
"""

import argparse
import io
import struct
import sys
import time
from pathlib import Path

import numpy as np
import requests

FRAME_W, FRAME_H = 1280, 720
NOISE_SKIP       = 5.0   # seconds — skip noise burst
HF_BASE          = "https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/main"
TIMEOUT          = 30
RETRY_DELAYS     = [5, 15, 30]


# ── ZIP range-request helpers ─────────────────────────────────────────────────

def http_get_range(url: str, start: int, end: int) -> bytes:
    headers = {"Range": f"bytes={start}-{end}"}
    for delay in RETRY_DELAYS:
        try:
            r = requests.get(url, headers=headers, timeout=TIMEOUT)
            if r.status_code in (200, 206):
                return r.content
            time.sleep(delay)
        except requests.RequestException:
            time.sleep(delay)
    raise RuntimeError(f"Failed to fetch {url} bytes {start}-{end}")


def http_head(url: str) -> int:
    """Return Content-Length (file size) via HEAD request."""
    for delay in RETRY_DELAYS:
        try:
            r = requests.head(url, timeout=TIMEOUT, allow_redirects=True)
            if r.status_code == 200:
                return int(r.headers["Content-Length"])
            time.sleep(delay)
        except requests.RequestException:
            time.sleep(delay)
    raise RuntimeError(f"HEAD failed for {url}")


def find_eocd(tail: bytes) -> int:
    """Find End of Central Directory offset within tail bytes."""
    sig = b"PK\x05\x06"
    pos = tail.rfind(sig)
    if pos == -1:
        raise ValueError("EOCD signature not found")
    return pos


def extract_file_from_zip_url(url: str, target_name: str) -> bytes:
    """
    Download only `target_name` from a remote ZIP using Range requests.
    Works for unencrypted, non-split ZIP/ZIP64 archives.
    """
    size = http_head(url)

    # 1. Fetch last 64 KB to find EOCD (handles up to 64 KB ZIP comment)
    tail_size = min(65536, size)
    tail = http_get_range(url, size - tail_size, size - 1)
    eocd_pos = find_eocd(tail)
    eocd = tail[eocd_pos:]

    # 2. Parse EOCD  (signature already consumed, fields start at offset 4)
    # Layout: disk(H) cd_disk(H) disk_entries(H) total_entries(H) cd_size(I) cd_offset(I) comment(H)
    (disk_num, cd_disk, disk_entries, total_entries,
     cd_size, cd_offset, comment_len) = struct.unpack_from("<HHHHIIH", eocd, 4)

    # Handle ZIP64
    if cd_offset == 0xFFFFFFFF:
        # Find ZIP64 EOCD locator just before EOCD
        loc_pos = eocd_pos - 20
        if loc_pos >= 0:
            loc = tail[loc_pos:loc_pos + 20]
            if loc[:4] == b"PK\x06\x07":
                zip64_eocd_offset = struct.unpack_from("<Q", loc, 8)[0]
                zip64_eocd = http_get_range(url, zip64_eocd_offset, zip64_eocd_offset + 56)
                cd_size   = struct.unpack_from("<Q", zip64_eocd, 40)[0]
                cd_offset = struct.unpack_from("<Q", zip64_eocd, 48)[0]

    # 3. Fetch Central Directory
    cd_data = http_get_range(url, cd_offset, cd_offset + cd_size - 1)

    # 4. Walk Central Directory entries to find target_name
    pos = 0
    while pos < len(cd_data):
        if cd_data[pos:pos+4] != b"PK\x01\x02":
            break
        # Correct CD entry format: 16 fields, 42 bytes after signature
        (ver_made, ver_need, flag, method, mod_time, mod_date,
         crc, comp_size, uncomp_size,
         fname_len, extra_len, comment_len2,
         disk_start, int_attr, ext_attr,
         local_hdr_offset) = struct.unpack_from("<HHHHHHIIIHHHHHII", cd_data, pos + 4)

        fname = cd_data[pos + 46: pos + 46 + fname_len].decode("utf-8", errors="replace")

        if fname.endswith(target_name):
            # Handle ZIP64: if sizes are 0xFFFFFFFF, read from extra field
            if comp_size == 0xFFFFFFFF or local_hdr_offset == 0xFFFFFFFF:
                extra = cd_data[pos + 46 + fname_len: pos + 46 + fname_len + extra_len]
                ep = 0
                while ep + 4 <= len(extra):
                    tag, sz = struct.unpack_from("<HH", extra, ep)
                    if tag == 0x0001:  # ZIP64 extended info
                        vals = struct.unpack_from("<" + "Q" * (sz // 8), extra, ep + 4)
                        idx = 0
                        if uncomp_size == 0xFFFFFFFF:
                            uncomp_size = vals[idx]; idx += 1
                        if comp_size == 0xFFFFFFFF:
                            comp_size = vals[idx]; idx += 1
                        if local_hdr_offset == 0xFFFFFFFF:
                            local_hdr_offset = vals[idx]
                        break
                    ep += 4 + sz

            # 5. Fetch local file header to find data offset
            lh = http_get_range(url, local_hdr_offset, local_hdr_offset + 30)
            lh_fname_len, lh_extra_len = struct.unpack_from("<HH", lh, 26)
            data_start = local_hdr_offset + 30 + lh_fname_len + lh_extra_len
            raw = http_get_range(url, data_start, data_start + comp_size - 1)
            if method == 8:
                import zlib
                return zlib.decompress(raw, -15)
            return raw

        pos += 46 + fname_len + extra_len + comment_len2

    raise FileNotFoundError(f"{target_name!r} not found in {url}")


# ── Annotation analysis ───────────────────────────────────────────────────────

def analyze_coordinates(text: str, seq_id: str) -> dict:
    ts, boxes = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            t_str, box_str = line.split(":")
            t = float(t_str.strip())
            b = [float(v) for v in box_str.split(",")[:4]]
            ts.append(t)
            boxes.append(b)
        except ValueError:
            continue

    if not ts:
        return None

    ts    = np.array(ts)
    boxes = np.array(boxes)
    mask  = ts >= NOISE_SKIP
    ts    = ts[mask]
    boxes = boxes[mask]

    if len(ts) == 0:
        return None

    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1) / (FRAME_W * FRAME_H)
    cx    = (x1 + x2) / 2 / FRAME_W
    cy    = (y1 + y2) / 2 / FRAME_H

    duration = ts[-1] - ts[0]

    tiny   = (areas < 0.003).sum()
    small  = ((areas >= 0.003) & (areas < 0.01)).sum()
    medium = (areas >= 0.01).sum()

    return {
        "seq":          seq_id,
        "n_ann":        len(ts),
        "duration_s":   round(float(duration), 1),
        "ann_per_s":    round(float(len(ts) / max(duration, 1)), 1),
        "mean_area_pct":round(float(areas.mean() * 100), 4),
        "tiny_pct":     round(float(tiny / len(areas) * 100), 1),
        "small_pct":    round(float(small / len(areas) * 100), 1),
        "medium_pct":   round(float(medium / len(areas) * 100), 1),
        "cx_std":       round(float(cx.std()), 3),
        "cy_std":       round(float(cy.std()), 3),
        "spatial_spread": round(float(cx.std() + cy.std()), 3),
    }


def training_score(r: dict) -> float:
    """Higher = better for training.  Rewards: many annotations, spatial diversity, mixed sizes."""
    norm_ann    = min(r["n_ann"] / 3000, 1.0)
    norm_spread = min(r["spatial_spread"] / 0.4, 1.0)
    # prefer some small/medium drones mixed in (not 100% tiny)
    size_diversity = min((r["small_pct"] + r["medium_pct"]) / 30, 1.0)
    return 0.4 * norm_ann + 0.4 * norm_spread + 0.2 * size_diversity


def val_score(r: dict, train_mean_area: float) -> float:
    """Higher = better for validation.  Rewards: many annotations, hard (small) drones,
    different mean area from training (domain gap test)."""
    norm_ann   = min(r["n_ann"] / 3000, 1.0)
    area_diff  = min(abs(r["mean_area_pct"] - train_mean_area) / 0.2, 1.0)
    # slightly prefer harder (smaller) drones in val
    hardness   = max(0, 1 - r["mean_area_pct"] / 0.4)
    return 0.4 * norm_ann + 0.3 * area_diff + 0.3 * hardness


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="/tmp/fred_coords",
                        help="Directory to cache downloaded coordinates.txt files")
    parser.add_argument("--plot", action="store_true",
                        help="Save analysis charts to fred_analysis.png")
    parser.add_argument("--local-only", action="store_true",
                        help="Only analyze sequences already in data/raw/ (no download)")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover sequences ────────────────────────────────────────────────────
    print("Discovering FRED sequences on HuggingFace ...")
    try:
        from huggingface_hub import HfApi
        api   = HfApi()
        files = api.list_repo_files("GabrieleMagrini/FRED", repo_type="dataset")
        zips  = [(f, "train" if f.startswith("train/") else "test")
                 for f in files if f.endswith(".zip")]
        print(f"  Found {len(zips)} sequences on HuggingFace")
    except Exception as e:
        print(f"  HuggingFace listing failed ({e}) — using local sequences only")
        zips = []
        args.local_only = True

    # Always include locally available sequences
    local_root = Path(__file__).parent.parent / "data" / "raw"
    local_seqs = {p.name: p / "coordinates.txt"
                  for p in local_root.glob("sequence_*")
                  if (p / "coordinates.txt").exists()}
    print(f"  {len(local_seqs)} sequences available locally")

    # ── Fetch / load coordinates ──────────────────────────────────────────────
    results = []

    if not args.local_only:
        print(f"\nFetching coordinates.txt for {len(zips)} sequences (~200 KB each) ...")
        for i, (zip_path, split) in enumerate(zips):
            # e.g. "train/84.zip" → seq_id="sequence_84", split="train"
            seq_num = zip_path.split("/")[-1].replace(".zip", "")
            seq_id  = f"sequence_{seq_num}"
            cache_f = cache_dir / f"{seq_id}.txt"

            # Use local file if available
            if seq_id in local_seqs:
                text = local_seqs[seq_id].read_text()
            elif cache_f.exists():
                text = cache_f.read_text()
            else:
                url = f"{HF_BASE}/{zip_path}"
                print(f"  [{i+1}/{len(zips)}] {seq_id} ({split}) ...", end=" ", flush=True)
                try:
                    raw  = extract_file_from_zip_url(url, "coordinates.txt")
                    text = raw.decode("utf-8", errors="replace")
                    cache_f.write_text(text)
                    print("ok")
                except Exception as e:
                    print(f"FAILED: {e}")
                    continue

            row = analyze_coordinates(text, seq_id)
            if row:
                row["split"] = split
                results.append(row)
    else:
        print("\nLocal-only mode — analyzing sequences in data/raw/ ...")
        for seq_id, path in sorted(local_seqs.items()):
            row = analyze_coordinates(path.read_text(), seq_id)
            if row:
                row["split"] = "local"
                results.append(row)

    if not results:
        print("No sequences analyzed. Exiting.")
        sys.exit(1)

    # ── Score and rank ────────────────────────────────────────────────────────
    train_mean_area = np.mean([r["mean_area_pct"] for r in results])
    for r in results:
        r["train_score"] = round(training_score(r), 3)
        r["val_score"]   = round(val_score(r, train_mean_area), 3)

    by_train = sorted(results, key=lambda r: r["train_score"], reverse=True)
    by_val   = sorted(results, key=lambda r: r["val_score"],   reverse=True)

    # ── Print results ─────────────────────────────────────────────────────────
    header = f"{'Seq':<14} {'N_ann':>6} {'Dur':>6} {'Area%':>7} {'Tiny%':>6} {'Sm%':>5} {'Spread':>7} {'TrainS':>7} {'ValS':>6}"
    print(f"\n{'='*80}")
    print("TOP 20 FOR TRAINING (ranked by training score):")
    print(header)
    for r in by_train[:20]:
        print(f"{r['seq']:<14} {r['n_ann']:>6} {r['duration_s']:>6} {r['mean_area_pct']:>7.3f} "
              f"{r['tiny_pct']:>6.0f} {r['small_pct']:>5.0f} {r['spatial_spread']:>7.3f} "
              f"{r['train_score']:>7.3f} {r['val_score']:>6.3f}")

    print(f"\n{'='*80}")
    print("TOP 10 FOR VALIDATION (ranked by val score, must not overlap with top-4 train):")
    train_set = {r["seq"] for r in by_train[:4]}
    print(header)
    for r in [r for r in by_val if r["seq"] not in train_set][:10]:
        print(f"{r['seq']:<14} {r['n_ann']:>6} {r['duration_s']:>6} {r['mean_area_pct']:>7.3f} "
              f"{r['tiny_pct']:>6.0f} {r['small_pct']:>5.0f} {r['spatial_spread']:>7.3f} "
              f"{r['train_score']:>7.3f} {r['val_score']:>6.3f}")

    print(f"\n{'='*80}")
    print("RECOMMENDED SPLIT (top-4 train + best val):")
    print(f"  Train : {[r['seq'] for r in by_train[:4]]}")
    val_seq = [r for r in by_val if r["seq"] not in train_set][0]
    print(f"  Val   : {val_seq['seq']}  (train_score={val_seq['train_score']}, val_score={val_seq['val_score']})")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    import csv
    out_csv = Path("data/kpis/fred_sequence_analysis.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(sorted(results, key=lambda r: r["train_score"], reverse=True))
    print(f"\nFull table saved → {out_csv}")

    # ── Optional plot ─────────────────────────────────────────────────────────
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            xs = [r["spatial_spread"] for r in results]
            ys = [r["n_ann"] for r in results]
            cs = [r["mean_area_pct"] for r in results]
            sc = axes[0].scatter(xs, ys, c=cs, cmap="RdYlGn", s=60)
            plt.colorbar(sc, ax=axes[0], label="Mean area %")
            axes[0].set_xlabel("Spatial spread (cx_std + cy_std)")
            axes[0].set_ylabel("Annotation count")
            axes[0].set_title("Training suitability")

            train_scores = [r["train_score"] for r in by_train[:20]]
            axes[1].barh([r["seq"] for r in by_train[:20]], train_scores, color="steelblue")
            axes[1].set_xlabel("Training score")
            axes[1].set_title("Top 20 training candidates")
            axes[1].invert_yaxis()

            val_cands = [r for r in by_val if r["seq"] not in train_set][:10]
            axes[2].barh([r["seq"] for r in val_cands],
                         [r["val_score"] for r in val_cands], color="darkorange")
            axes[2].set_xlabel("Validation score")
            axes[2].set_title("Top 10 validation candidates")
            axes[2].invert_yaxis()

            plt.tight_layout()
            out_png = Path("data/kpis/fred_analysis.png")
            plt.savefig(out_png, dpi=120)
            print(f"Plot saved → {out_png}")
        except ImportError:
            print("matplotlib not installed — skipping plot")


if __name__ == "__main__":
    main()
