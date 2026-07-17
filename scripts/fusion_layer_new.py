import numpy as np


def calculate_iou(box1, box2):
    x_left  = max(box1[0], box2[0])
    y_top   = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom= min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection / (area1 + area2 - intersection)


def find_rgb_match(event_detection, rgb_detections, iou_t=0.5):
    best_match, best_iou = None, 0.0
    for rgb in rgb_detections:
        iou = calculate_iou(event_detection["box"], rgb["box"])
        if iou > best_iou:
            best_iou = iou
            best_match = rgb
    return (best_match, best_iou) if best_iou > iou_t else (None, 0.0)


def _fuse_confidence(ce, cr, alpha):
    """
    Combine event confidence ce and RGB confidence cr.
      alpha=None  →  max(ce, cr)          [original behaviour]
      alpha=0.0   →  ce                   [event only]
      alpha=0.5   →  0.5*cr + 0.5*ce      [equal blend]
      alpha=1.0   →  cr                   [rgb only]
    """
    if alpha is None:
        return max(ce, cr)
    return (1.0 - alpha) * ce + alpha * cr


def _fuse_box(event_box, rgb_box, ce, cr, weighted_box):
    """
    Merge two xyxy boxes.
      weighted_box=False  →  event box kept as-is  [original behaviour]
      weighted_box=True   →  coordinate average weighted by confidence
    """
    if not weighted_box:
        return event_box
    total = ce + cr
    return [(ce * e + cr * r) / total for e, r in zip(event_box, rgb_box)]


def temporal_smooth(fused_by_frame, window=1, match_iou=0.2, boost=1.1, decay=1.0):
    """
    Post-process per-frame fused detections with temporal context.

    For each detection, look at ±window adjacent frames. If a spatially
    similar box (IoU ≥ match_iou) exists in any of those frames, the
    detection is considered temporally supported and its confidence is
    multiplied by boost. Otherwise it is multiplied by decay (set < 1.0
    to suppress isolated false positives).

    Parameters
    ----------
    fused_by_frame : {frame_idx: [{'box', 'confidence', 'source'}, ...]}
    window         : how many frames to look either side
    match_iou      : spatial IoU threshold for cross-frame matching
                     (lower than same-frame iou_t since the drone moves)
    boost          : confidence multiplier for supported detections  (> 1.0)
    decay          : confidence multiplier for isolated detections   (< 1.0)
    """
    smoothed = {}
    for f, dets in fused_by_frame.items():
        new_dets = []
        for det in dets:
            supported = False
            for df in range(f - window, f + window + 1):
                if df == f or df not in fused_by_frame:
                    continue
                for other in fused_by_frame[df]:
                    if calculate_iou(det["box"], other["box"]) >= match_iou:
                        supported = True
                        break
                if supported:
                    break
            factor = boost if supported else decay
            new_dets.append({**det, "confidence": min(1.0, det["confidence"] * factor)})
        smoothed[f] = new_dets
    return smoothed


def fuse_detections(
    event_boxes,
    rgb_boxes,
    event_t=0.5,
    rgb_t=0.8,
    iou_t=0.5,
    alpha=None,
    weighted_box=False,
):
    """
    Late-fusion of event and RGB detections.

    Parameters
    ----------
    event_boxes, rgb_boxes : list of {'box': [x1,y1,x2,y2], 'confidence': float}
    event_t      : confidence threshold for event-only detections
    rgb_t        : confidence threshold for RGB-only detections
    iou_t        : IoU threshold to consider two boxes the same detection
    alpha        : RGB weight in confidence blend (None → use max, 0.0 → event only,
                   1.0 → RGB only)
    weighted_box : if True, merge box coordinates weighted by confidence
    """
    final = []
    matched_rgb = set()

    for event in event_boxes:
        rgb_match, _ = find_rgb_match(event, rgb_boxes, iou_t=iou_t)

        if rgb_match is not None:
            idx = rgb_boxes.index(rgb_match)
            matched_rgb.add(idx)
            ce, cr = event["confidence"], rgb_match["confidence"]
            final.append({
                "box":        _fuse_box(event["box"], rgb_match["box"], ce, cr, weighted_box),
                "confidence": _fuse_confidence(ce, cr, alpha),
                "source":     "fusion",
            })
        else:
            if event["confidence"] >= event_t:
                final.append({
                    "box":        event["box"],
                    "confidence": event["confidence"],
                    "source":     "event",
                })

    # RGB-only detections (unmatched)
    for i, rgb in enumerate(rgb_boxes):
        if i not in matched_rgb and rgb["confidence"] >= rgb_t:
            final.append({
                "box":        rgb["box"],
                "confidence": rgb["confidence"],
                "source":     "rgb",
            })

    return final
