from ultralytics import YOLO
from fusion.decision_layer import fuse_detections


rgb_model = YOLO(
    "models/rgb.pt"
)

event_model = YOLO(
    "models/event.pt"
)



def convert_yolo_output(result):

    detections = []


    for box in result.boxes:


        xyxy = box.xyxy[0].tolist()

        conf = float(
            box.conf[0]
        )


        detections.append(
            {
            "box": xyxy,
            "confidence": conf
            }
        )


    return detections



def process_frame(rgb_frame, event_frame):


    rgb_result = rgb_model(
        rgb_frame
    )[0]


    event_result = event_model(
        event_frame
    )[0]



    rgb_detections = convert_yolo_output(
        rgb_result
    )


    event_detections = convert_yolo_output(
        event_result
    )



    final = fuse_detections(
        event_detections,
        rgb_detections
    )


    return final