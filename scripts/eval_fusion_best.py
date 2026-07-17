"""
Full evaluation of the best temporal-smoothing fusion configuration.

Best params (from fusion_tuning_results.csv, phase 2 top row):
  event_t=0.3, rgb_t=0.5, iou_t=0.5, alpha=max, weighted_box=True
  t_window=2, t_boost=1.3, t_decay=0.8

Computes per-sequence and aggregate:
  mAP@0.5, mAP@0.5:0.95, precision, recall

Usage:
    cd /home/bani/ami/scripts
    python eval_fusion_best.py
"""

import json
from pathlib import Path

import numpy as np

from tune_fusion import (
    load_sequence, _xyxy_to_xywh, _iou, TEST_SEQS
)
from fusion_layer_new import fuse_detections, temporal_smooth

# ── Best params from CSV ───────────────────────────────────────────────────────
EVENT_T  = 0.3
RGB_T    = 0.5
IOU_T    = 0.5
T_WINDOW = 2
T_BOOST  = 1.3
T_DECAY  = 0.8

CONF_T   = 0.1   # operating-point confidence threshold for precision/recall


def _compute_full(detections, frame_gt, iou_thresh=0.5):
    """Returns (ap, precision, recall) at a given IoU threshold."""
    n_gt = len(frame_gt)
    if n_gt == 0 or not detections:
        return 0.0, 0.0, 0.0

    dets = sorted(detections, key=lambda d: d['confidence'], reverse=True)
    matched, tp_list, fp_list = set(), [], []
    for det in dets:
        f = det['frame']
        if f not in frame_gt or f in matched:
            fp_list.append(1); tp_list.append(0)
            continue
        if _iou(det['bbox'], frame_gt[f]) >= iou_thresh:
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

    # Precision/recall at max-F1 operating point
    f1 = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best = int(np.argmax(f1))
    return float(ap), float(precisions[best]), float(recalls[best])


def _compute_map5095(detections, frame_gt):
    """mAP averaged over IoU 0.50:0.05:0.95."""
    aps = [_compute_full(detections, frame_gt, iou_t)[0]
           for iou_t in np.arange(0.50, 1.00, 0.05)]
    return float(np.mean(aps))


def fuse_sequence(seq_n):
    ev_by_frame, rgb_by_frame, frame_gt = load_sequence(seq_n)
    all_frames = sorted(set(ev_by_frame) | set(rgb_by_frame))

    fused_by_frame = {}
    for f in all_frames:
        dets = fuse_detections(
            ev_by_frame.get(f, []),
            rgb_by_frame.get(f, []),
            event_t=EVENT_T, rgb_t=RGB_T, iou_t=IOU_T,
            alpha=None, weighted_box=True,
        )
        if dets:
            fused_by_frame[f] = dets

    fused_by_frame = temporal_smooth(
        fused_by_frame, window=T_WINDOW, boost=T_BOOST, decay=T_DECAY
    )

    all_fused = []
    for f, dets in fused_by_frame.items():
        for det in dets:
            all_fused.append({
                'frame':      f,
                'bbox':       _xyxy_to_xywh(det['box']),
                'confidence': det['confidence'],
            })

    return all_fused, frame_gt


def main():
    print("Loading sequences …")
    agg_fused, agg_gt = [], {}
    offset = 0
    per_seq = {}

    for seq_n in TEST_SEQS:
        fused, gt = fuse_sequence(seq_n)
        n_gt = len(gt)

        # per-sequence metrics (absolute frame indices don't matter here)
        ap50, prec, rec = _compute_full(fused, gt, iou_thresh=0.5)
        map5095 = _compute_map5095(fused, gt)
        n_det = len(fused)
        per_seq[f'sequence_{seq_n}'] = {
            'map50':     round(ap50, 4),
            'map50_95':  round(map5095, 4),
            'precision': round(prec, 4),
            'recall':    round(rec, 4),
            'n_gt':      n_gt,
            'n_det':     n_det,
        }
        print(f"  seq_{seq_n:3d}  mAP50={ap50:.4f}  mAP50:95={map5095:.4f}"
              f"  P={prec:.4f}  R={rec:.4f}  n_gt={n_gt}  n_det={n_det}")

        # accumulate for aggregate (offset frames to avoid collisions)
        all_frames = [d['frame'] for d in fused]
        n = (max(all_frames) + 1) if all_frames else 0
        for d in fused:
            agg_fused.append({**d, 'frame': d['frame'] + offset})
        for f, box in gt.items():
            agg_gt[f + offset] = box
        offset += n

    # aggregate
    agg_ap50, agg_prec, agg_rec = _compute_full(agg_fused, agg_gt, iou_thresh=0.5)
    agg_map5095 = _compute_map5095(agg_fused, agg_gt)

    print(f"\n  AGGREGATE  mAP50={agg_ap50:.4f}  mAP50:95={agg_map5095:.4f}"
          f"  P={agg_prec:.4f}  R={agg_rec:.4f}")

    result = {
        'params': {
            'event_t': EVENT_T, 'rgb_t': RGB_T, 'iou_t': IOU_T,
            't_window': T_WINDOW, 't_boost': T_BOOST, 't_decay': T_DECAY,
        },
        'aggregate': {
            'map50':     round(agg_ap50, 4),
            'map50_95':  round(agg_map5095, 4),
            'precision': round(agg_prec, 4),
            'recall':    round(agg_rec, 4),
        },
        'per_sequence': per_seq,
    }

    out = Path(__file__).parent / 'fusion_best_eval.json'
    out.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved to {out}")


if __name__ == '__main__':
    main()
