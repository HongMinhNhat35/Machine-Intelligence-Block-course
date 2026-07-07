import numpy as np


EVENT_THRESHOLD = 0.5
RGB_THRESHOLD = 0.8
IOU_THRESHOLD = 0.5


def calculate_iou(box1, box2):

    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])

    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])


    if x_right < x_left or y_bottom < y_top:
        return 0.0


    intersection = (
        (x_right - x_left)
        *
        (y_bottom - y_top)
    )


    area1 = (
        box1[2]-box1[0]
    ) * (
        box1[3]-box1[1]
    )


    area2 = (
        box2[2]-box2[0]
    ) * (
        box2[3]-box2[1]
    )


    union = area1 + area2 - intersection


    return intersection / union



def find_rgb_match(event_detection, rgb_detections):

    best_match = None
    best_iou = 0


    for rgb in rgb_detections:

        iou = calculate_iou(
            event_detection["box"],
            rgb["box"]
        )


        if iou > best_iou:
            best_iou = iou
            best_match = rgb


    if best_iou > IOU_THRESHOLD:
        return best_match


    return None



def fuse_detections(event_boxes, rgb_boxes):

    final = []


    # Event is main source

    for event in event_boxes:


        rgb_match = find_rgb_match(
            event,
            rgb_boxes
        )


        # beide sehen gleiche Drohne

        if rgb_match:


            final_confidence = max(
                event["confidence"],
                rgb_match["confidence"]
            )


            final.append(
                {
                    "box": event["box"],
                    "confidence": final_confidence,
                    "source": "fusion"
                }
            )


        # nur Event

        else:


            if event["confidence"] >= EVENT_THRESHOLD:

                final.append(
                    {
                    "box": event["box"],
                    "confidence": event["confidence"],
                    "source": "event"
                    }
                )


    # RGB-only

    for rgb in rgb_boxes:


        if rgb["confidence"] >= RGB_THRESHOLD:


            final.append(
                {
                "box": rgb["box"],
                "confidence": rgb["confidence"],
                "source": "rgb"
                }
            )


    return final
