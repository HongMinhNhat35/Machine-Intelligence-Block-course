"""
Fusion threshold grid search — runs on cached detections, no GPU needed.

Tests 125 combinations of (event_t, rgb_t, iou_t) and scores each with
mAP@0.5 computed against FRED ground-truth labels.

Usage:
    cd /home/bani/ami/scripts
    python tune_fusion.py

Output:
    fusion_tuning_results.csv   — all 125 rows sorted by mAP@0.5 descending
"""

import bisect
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from fusion_layer_new import fuse_detections, temporal_smooth

# ── Paths ──────────────────────────────────────────────────────────────────────
CACHE_DIR = Path(os.environ.get("RECON_DATA_PATH", Path.home() / "g1-ami-data"))
FRED_DIR  = Path(os.environ.get("FRED_DATA_PATH",  Path.home() / "ami/data/raw"))
TEST_SEQS = [8, 9, 12, 20, 21]

# ── Box format helpers ─────────────────────────────────────────────────────────
# Cache files store [x1, y1, w, h]; fusion_layer_new expects [x1, y1, x2, y2].

def _xywh_to_xyxy(b):
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]

def _xyxy_to_xywh(b):
    return [b[0], b[1], b[2] - b[0], b[3] - b[1]]

# ── Ground-truth loader (mirrors detection_cache.ipynb) ────────────────────────

def _load_gt_list(coord_path):
    gt = []
    for line in Path(coord_path).read_text().splitlines():
        line = line.strip()
        if not line or ':' not in line:
            continue
        ts_str, rest = line.split(':', 1)
        try:
            ts   = float(ts_str.strip())
            vals = [float(x.strip()) for x in rest.split(',')[:4]]
            if len(vals) == 4:
                gt.append((ts, vals))
        except ValueError:
            pass
    return sorted(gt, key=lambda x: x[0])

def _rgb_timestamp(filename):
    stem  = Path(filename).stem          # e.g. Video_8_17_08_44.478342
    parts = stem.rsplit('_', 3)
    if len(parts) < 4:
        return None
    try:
        h, m, s = int(parts[1]), int(parts[2]), float(parts[3])
        return h * 3600 + m * 60 + s
    except ValueError:
        return None

def _match_gt_to_frames(rgb_files, gt_list, tol=0.05):
    """Returns {frame_idx: [x1, y1, x2, y2]}."""
    frame_gt = {}
    if not gt_list or not rgb_files:
        return frame_gt
    t0 = _rgb_timestamp(rgb_files[0].name)
    if t0 is None:
        return frame_gt
    gt_ts = [t for t, _ in gt_list]
    for i, f in enumerate(rgb_files):
        ts_abs = _rgb_timestamp(f.name)
        if ts_abs is None:
            continue
        ts_rel = ts_abs - t0
        idx = bisect.bisect_left(gt_ts, ts_rel)
        candidates = []
        if idx > 0:            candidates.append(gt_list[idx - 1])
        if idx < len(gt_list): candidates.append(gt_list[idx])
        if not candidates:
            continue
        closest_ts, closest_box = min(candidates, key=lambda x: abs(x[0] - ts_rel))
        if abs(closest_ts - ts_rel) <= tol:
            frame_gt[i] = closest_box
    return frame_gt

# ── mAP@0.5 (VOC 11-point interpolation) ──────────────────────────────────────

def _iou(det_xywh, gt_xyxy):
    dx1, dy1, dw, dh = det_xywh
    dx2, dy2 = dx1 + dw, dy1 + dh
    gx1, gy1, gx2, gy2 = gt_xyxy
    ix1, iy1 = max(dx1, gx1), max(dy1, gy1)
    ix2, iy2 = min(dx2, gx2), min(dy2, gy2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    return inter / (dw * dh + (gx2 - gx1) * (gy2 - gy1) - inter)

def compute_map50(detections, frame_gt):
    """
    detections: [{'frame': int, 'bbox': [x1,y1,w,h], 'confidence': float}, ...]
    frame_gt:   {frame_idx: [x1, y1, x2, y2]}
    """
    n_gt = len(frame_gt)
    if n_gt == 0 or not detections:
        return 0.0
    dets = sorted(detections, key=lambda d: d['confidence'], reverse=True)
    matched, tp_list, fp_list = set(), [], []
    for det in dets:
        f = det['frame']
        if f not in frame_gt or f in matched:
            fp_list.append(1); tp_list.append(0)
            continue
        if _iou(det['bbox'], frame_gt[f]) >= 0.5:
            tp_list.append(1); fp_list.append(0)
            matched.add(f)
        else:
            tp_list.append(0); fp_list.append(1)
    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)
    recalls    = tp_cum / n_gt
    precisions = tp_cum / (tp_cum + fp_cum)
    ap = sum(
        (precisions[recalls >= t].max() if (recalls >= t).any() else 0.0)
        for t in np.linspace(0, 1, 11)
    ) / 11
    return float(ap)

# ── Data loading ───────────────────────────────────────────────────────────────

def load_sequence(seq_n):
    """
    Returns (event_by_frame, rgb_by_frame, frame_gt).
      *_by_frame: {frame_idx: [{'box': [x1,y1,x2,y2], 'confidence': float}, ...]}
      frame_gt:   {frame_idx: [x1, y1, x2, y2]}
    """
    seq_dir = CACHE_DIR / f"sequence_{seq_n}"

    def _load_cache(model):
        path = seq_dir / f"detections_{model}.json"
        if not path.exists():
            print(f"  WARNING: {path} not found")
            return {}
        data = json.loads(path.read_text())
        by_frame = {}
        for d in data["detections"]:
            by_frame.setdefault(d["frame"], []).append({
                "box":        _xywh_to_xyxy(d["bbox"]),
                "confidence": d["confidence"],
            })
        return by_frame

    event_by_frame = _load_cache("fusion_event")
    rgb_by_frame   = _load_cache("fusion_rgb")

    fred_seq  = FRED_DIR / f"sequence_{seq_n}"
    rgb_dir   = fred_seq / "RGB"
    coord     = fred_seq / "coordinates_rgb.txt"
    frame_gt  = {}
    if rgb_dir.exists() and coord.exists():
        rgb_files = sorted(rgb_dir.glob("*.jpg"))
        frame_gt  = _match_gt_to_frames(rgb_files, _load_gt_list(coord))
    else:
        print(f"  WARNING: FRED raw data not found for sequence_{seq_n} at {fred_seq}")

    return event_by_frame, rgb_by_frame, frame_gt

# ── Grid search ────────────────────────────────────────────────────────────────

def _run_combo(sequences, et, rt, it, alpha, weighted_box,
               temporal_window=0, temporal_boost=1.0, temporal_decay=1.0):
    all_fused, all_gt = [], {}
    offset = 0
    for seq_n, (ev_by_frame, rgb_by_frame, frame_gt) in sequences.items():
        all_frames = sorted(set(ev_by_frame) | set(rgb_by_frame))
        n = (max(all_frames) + 1) if all_frames else 0

        # per-frame fusion
        fused_by_frame = {}
        for f in all_frames:
            dets = fuse_detections(
                ev_by_frame.get(f, []),
                rgb_by_frame.get(f, []),
                event_t=et, rgb_t=rt, iou_t=it,
                alpha=alpha, weighted_box=weighted_box,
            )
            if dets:
                fused_by_frame[f] = dets

        # idea 3: temporal smoothing
        if temporal_window > 0:
            fused_by_frame = temporal_smooth(
                fused_by_frame,
                window=temporal_window,
                boost=temporal_boost,
                decay=temporal_decay,
            )

        for f, dets in fused_by_frame.items():
            gf = offset + f
            for det in dets:
                all_fused.append({
                    "frame":      gf,
                    "bbox":       _xyxy_to_xywh(det["box"]),
                    "confidence": det["confidence"],
                })
        for f, box in frame_gt.items():
            all_gt[offset + f] = box
        offset += n
    return compute_map50(all_fused, all_gt)


def run_tuning():
    # ── Phase 1: threshold + confidence + box grid (strategies 1 & 2) ──────────
    event_thresholds    = [0.1, 0.2, 0.3, 0.4, 0.5]
    rgb_thresholds      = [0.1, 0.2, 0.3, 0.4, 0.5]
    iou_thresholds      = [0.3, 0.4, 0.5, 0.6, 0.7]
    alpha_values        = [None, 0.1, 0.2, 0.3, 0.5]   # idea 1
    weighted_box_values = [False, True]                  # idea 2

    # ── Phase 2: temporal smoothing grid (strategy 3) ──────────────────────────
    # Run on narrowed threshold range (top region from phase 1) to stay fast.
    temporal_event_t  = [0.1, 0.2, 0.3]
    temporal_rgb_t    = [0.3, 0.4, 0.5]
    temporal_iou_t    = [0.3, 0.4, 0.5]
    temporal_windows  = [1, 2]          # adjacent frames each side
    temporal_boosts   = [1.1, 1.2, 1.3] # multiplier for supported detections
    temporal_decays   = [0.8, 0.9, 1.0] # multiplier for isolated detections

    n_phase1 = (len(event_thresholds) * len(rgb_thresholds) * len(iou_thresholds)
                * len(alpha_values) * len(weighted_box_values))
    n_phase2 = (len(temporal_event_t) * len(temporal_rgb_t) * len(temporal_iou_t)
                * len(temporal_windows) * len(temporal_boosts) * len(temporal_decays))

    print("Loading test sequences …")
    sequences = {}
    for seq_n in TEST_SEQS:
        ev, rgb, gt = load_sequence(seq_n)
        sequences[seq_n] = (ev, rgb, gt)
        print(f"  sequence_{seq_n}: {len(ev)} event-frames, {len(rgb)} rgb-frames, {len(gt)} GT frames")

    # ── Phase 1 ────────────────────────────────────────────────────────────────
    print(f"\n── Phase 1: threshold / confidence / box grid ({n_phase1} combos) ──\n")
    results = []
    for et in event_thresholds:
        for rt in rgb_thresholds:
            for it in iou_thresholds:
                for alpha in alpha_values:
                    for wb in weighted_box_values:
                        score = _run_combo(sequences, et, rt, it, alpha, wb)
                        alpha_str = "max" if alpha is None else str(alpha)
                        results.append({
                            "phase": 1, "event_t": et, "rgb_t": rt, "iou_t": it,
                            "alpha": alpha_str, "weighted_box": wb,
                            "t_window": 0, "t_boost": 1.0, "t_decay": 1.0,
                            "map50": round(score, 4),
                        })
                        print(f"  E={et}  R={rt}  IOU={it}  α={alpha_str:<4}  box={'W' if wb else '-'}  →  mAP@0.5={score:.4f}")

    df1 = pd.DataFrame(results).sort_values("map50", ascending=False).reset_index(drop=True)
    best1 = df1.iloc[0]
    print(f"\n  Phase 1 best → mAP@0.5={best1.map50}  "
          f"(E={best1.event_t} R={best1.rgb_t} IOU={best1.iou_t} "
          f"α={best1.alpha} box={'W' if best1.weighted_box else '-'})")

    # ── Phase 2 ────────────────────────────────────────────────────────────────
    print(f"\n── Phase 2: temporal smoothing grid ({n_phase2} combos) ───────────\n")
    t_results = []
    for et in temporal_event_t:
        for rt in temporal_rgb_t:
            for it in temporal_iou_t:
                for tw in temporal_windows:
                    for tb in temporal_boosts:
                        for td in temporal_decays:
                            score = _run_combo(
                                sequences, et, rt, it,
                                alpha=None, weighted_box=True,  # best from phase 1
                                temporal_window=tw, temporal_boost=tb, temporal_decay=td,
                            )
                            t_results.append({
                                "phase": 2, "event_t": et, "rgb_t": rt, "iou_t": it,
                                "alpha": "max", "weighted_box": True,
                                "t_window": tw, "t_boost": tb, "t_decay": td,
                                "map50": round(score, 4),
                            })
                            print(f"  E={et}  R={rt}  IOU={it}  W={tw}  boost={tb}  decay={td}  →  mAP@0.5={score:.4f}")

    df2 = pd.DataFrame(t_results).sort_values("map50", ascending=False).reset_index(drop=True)
    best2 = df2.iloc[0]

    # ── Summary ────────────────────────────────────────────────────────────────
    df_all = pd.concat([df1, df2], ignore_index=True).sort_values("map50", ascending=False).reset_index(drop=True)

    print("\n── Top 10 overall ─────────────────────────────────────────────")
    print(df_all.head(10).to_string(index=False))

    print("\n── Best per strategy ──────────────────────────────────────────")
    strategies = [
        ("1a  max-conf + original box   (baseline)",
         (df1.alpha == "max") & ~df1.weighted_box & (df1.t_window == 0)),
        ("1b  max-conf + weighted box   (idea 2)",
         (df1.alpha == "max") &  df1.weighted_box & (df1.t_window == 0)),
        ("1c  α-blend  + weighted box   (idea 1+2)",
         (df1.alpha != "max") &  df1.weighted_box & (df1.t_window == 0)),
        ("2   temporal smoothing        (idea 3)",
         pd.Series([True] * len(df2), index=df2.index)),
    ]
    dfs = [df1, df1, df1, df2]
    for (label, mask), df_s in zip(strategies, dfs):
        sub = df_s[mask]
        if sub.empty:
            continue
        b = sub.iloc[0]
        extra = (f"W={int(b.t_window)} boost={b.t_boost} decay={b.t_decay}"
                 if b.t_window > 0 else f"α={b.alpha}")
        print(f"  {label}  →  mAP@0.5={b.map50:.4f}  "
              f"(E={b.event_t} R={b.rgb_t} IOU={b.iou_t} {extra})")

    overall_best = df_all.iloc[0]
    print(f"\n── Overall best ──────────────────────────────────────────────")
    print(f"  mAP@0.5 = {overall_best.map50}  phase={int(overall_best.phase)}  "
          f"E={overall_best.event_t}  R={overall_best.rgb_t}  IOU={overall_best.iou_t}  "
          f"α={overall_best.alpha}  box={'W' if overall_best.weighted_box else '-'}  "
          f"t_window={int(overall_best.t_window)}  t_boost={overall_best.t_boost}  "
          f"t_decay={overall_best.t_decay}")

    out = Path(__file__).parent / "fusion_tuning_results.csv"
    df_all.to_csv(out, index=False)
    print(f"\n{len(df_all)} total results saved to {out}")
    return df_all


if __name__ == "__main__":
    run_tuning()
