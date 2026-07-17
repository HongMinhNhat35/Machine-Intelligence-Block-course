import numpy as np

# Best params from grid search (tune_fusion.py, phase 2 top result)
EVENT_THRESHOLD = 0.3
RGB_THRESHOLD   = 0.5
IOU_THRESHOLD   = 0.5

# Temporal smoothing params
T_WINDOW = 2
T_BOOST  = 1.3
T_DECAY  = 0.8


def calculate_iou(box1, box2):
    x_left   = max(box1[0], box2[0])
    y_top    = max(box1[1], box2[1])
    x_right  = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection / (area1 + area2 - intersection)


def find_rgb_match(event_detection, rgb_detections):
    best_match, best_iou = None, 0.0
    for rgb in rgb_detections:
        iou = calculate_iou(event_detection["box"], rgb["box"])
        if iou > best_iou:
            best_iou = iou
            best_match = rgb
    return best_match if best_iou > IOU_THRESHOLD else None


def fuse_detections(event_boxes, rgb_boxes):
    final = []
    matched_rgb = set()

    for event in event_boxes:
        rgb_match = find_rgb_match(event, rgb_boxes)
        if rgb_match is not None:
            idx = rgb_boxes.index(rgb_match)
            matched_rgb.add(idx)
            ce, cr = event["confidence"], rgb_match["confidence"]
            total = ce + cr
            merged_box = [(ce * e + cr * r) / total
                          for e, r in zip(event["box"], rgb_match["box"])]
            final.append({
                "box":        merged_box,
                "confidence": max(ce, cr),
                "source":     "fusion",
            })
        elif event["confidence"] >= EVENT_THRESHOLD:
            final.append({
                "box":        event["box"],
                "confidence": event["confidence"],
                "source":     "event",
            })

    for i, rgb in enumerate(rgb_boxes):
        if i not in matched_rgb and rgb["confidence"] >= RGB_THRESHOLD:
            final.append({
                "box":        rgb["box"],
                "confidence": rgb["confidence"],
                "source":     "rgb",
            })

    return final


def temporal_smooth(fused_by_frame, window=T_WINDOW, boost=T_BOOST, decay=T_DECAY):
    smoothed = {}
    for f, dets in fused_by_frame.items():
        new_dets = []
        for det in dets:
            supported = False
            for df in range(f - window, f + window + 1):
                if df == f or df not in fused_by_frame:
                    continue
                for other in fused_by_frame[df]:
                    if calculate_iou(det["box"], other["box"]) >= 0.2:
                        supported = True
                        break
                if supported:
                    break
            factor = boost if supported else decay
            new_dets.append({**det, "confidence": min(1.0, det["confidence"] * factor)})
        smoothed[f] = new_dets
    return smoothed
