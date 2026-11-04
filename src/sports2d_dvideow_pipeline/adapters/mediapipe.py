"""MediaPipe Pose 0.10.21 video adapter with explicit side and optional ROI."""

from __future__ import annotations

import csv
import json
from pathlib import Path


_LANDMARKS = {
    "left": {"hip": "LEFT_HIP", "knee": "LEFT_KNEE", "ankle": "LEFT_ANKLE", "foot": "LEFT_FOOT_INDEX"},
    "right": {"hip": "RIGHT_HIP", "knee": "RIGHT_KNEE", "ankle": "RIGHT_ANKLE", "foot": "RIGHT_FOOT_INDEX"},
}


def _read_roi(config_path, camera_role, side):
    if config_path is None:
        return None
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return config.get("entries", {}).get(f"{camera_role}:{side}")


def _crop(frame, roi):
    height, width = frame.shape[:2]
    if not roi:
        return frame, (0.0, 0.0, float(width), float(height))
    x, y = int(roi["x"]), int(roi["y"])
    w, h = int(roi["width"]), int(roi["height"])
    pad = float(roi.get("padding", 0.0))
    x0, y0 = max(0, int(x - w * pad)), max(0, int(y - h * pad))
    x1, y1 = min(width, int(x + w * (1 + pad))), min(height, int(y + h * (1 + pad)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI lies outside the video frame")
    crop = frame[y0:y1, x0:x1]
    scale = max(1.0, float(roi.get("scale", 1.0)))
    if scale != 1.0:
        import cv2
        crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
    return crop, (float(x0), float(y0), float(x1 - x0), float(y1 - y0))


def extract(video, output_csv, *, side: str, camera_role: str = "CAM_1",
            roi_config=None, model_complexity: int = 2,
            min_confidence: float = 0.15, static_image_mode: bool = False) -> int:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("install the optional extra with pip install '.[mediapipe]'") from exc
    if side not in _LANDMARKS:
        raise ValueError("side must be left or right")
    if model_complexity not in (0, 1, 2) or not 0 <= min_confidence <= 1:
        raise ValueError("invalid MediaPipe model complexity or confidence threshold")
    video, output_csv = Path(video), Path(output_csv)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("could not open input video")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    roi = _read_roi(roi_config, camera_role, side)
    indices = {name: getattr(mp.solutions.pose.PoseLandmark, value).value for name, value in _LANDMARKS[side].items()}
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame_index = 0
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["frame_index", "camera", "landmark", "x_px", "y_px", "confidence"])
        writer.writeheader()
        try:
            with mp.solutions.pose.Pose(
                static_image_mode=static_image_mode,
                model_complexity=model_complexity,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.10,
                min_tracking_confidence=0.10,
            ) as pose:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    inference, geometry = _crop(frame, roi)
                    result = pose.process(cv2.cvtColor(inference, cv2.COLOR_BGR2RGB))
                    for name, landmark_index in indices.items():
                        x_px = y_px = confidence = ""
                        if result.pose_landmarks is not None:
                            landmark = result.pose_landmarks.landmark[landmark_index]
                            confidence = float(landmark.visibility)
                            if confidence >= min_confidence:
                                x_px = geometry[0] + float(landmark.x) * geometry[2]
                                y_px = geometry[1] + float(landmark.y) * geometry[3]
                        writer.writerow({"frame_index": frame_index, "camera": camera_role,
                                         "landmark": name, "x_px": x_px, "y_px": y_px,
                                         "confidence": confidence})
                    frame_index += 1
        finally:
            cap.release()
    return frame_index
