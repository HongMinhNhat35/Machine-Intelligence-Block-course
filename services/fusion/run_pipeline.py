from pathlib import Path
from fusion_layer import fuse_detections

WEIGHTS_ROOT = Path("/app/weights")
FRED_ROOT = Path("/data/fred")

_rgb_model = None
_event_model = None


def _get_models():
    global _rgb_model, _event_model
    if _rgb_model is None:
        from ultralytics import YOLO
        rgb_path = WEIGHTS_ROOT / "fusion_rgb.pt"
        event_path = WEIGHTS_ROOT / "fusion_event.pt"
        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB weights not found: {rgb_path}")
        if not event_path.exists():
            raise FileNotFoundError(f"Event weights not found: {event_path}")
        _rgb_model = YOLO(str(rgb_path))
        _event_model = YOLO(str(event_path))
    return _rgb_model, _event_model


def _get_frame_files(sequence_id: str):
    seq_dir = FRED_ROOT / sequence_id
    rgb_dir = seq_dir / "RGB"
    event_dir = seq_dir / "Event" / "Frames"
    if not event_dir.exists():
        event_dir = seq_dir / "Event" / "images"
    rgb_files = sorted(rgb_dir.glob("*.jpg")) if rgb_dir.exists() else []
    if not rgb_files:
        rgb_files = sorted(rgb_dir.glob("*.png")) if rgb_dir.exists() else []
    event_files = sorted(event_dir.glob("*.jpg")) if event_dir.exists() else []
    if not event_files:
        event_files = sorted(event_dir.glob("*.png")) if event_dir.exists() else []
    return rgb_files, event_files


def convert_yolo_output(result):
    detections = []
    for box in result.boxes:
        xyxy = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        detections.append({"box": xyxy, "confidence": conf})
    return detections


def process_frame(rgb_path: str, event_path: str):
    rgb_model, event_model = _get_models()
    rgb_result = rgb_model(rgb_path)[0]
    event_result = event_model(event_path)[0]
    rgb_detections = convert_yolo_output(rgb_result)
    event_detections = convert_yolo_output(event_result)
    return fuse_detections(event_detections, rgb_detections)


def weights_available() -> bool:
    return (
        (WEIGHTS_ROOT / "fusion_rgb.pt").exists()
        and (WEIGHTS_ROOT / "fusion_event.pt").exists()
    )
