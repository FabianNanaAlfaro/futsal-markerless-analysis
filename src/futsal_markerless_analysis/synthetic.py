"""Small, explicitly synthetic two-camera trial for the end-to-end example."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


SYNTHETIC_CALIBRATION = {
    "units": "m",
    "coordinate_system": {
        "x": "anteroposterior (+anterior)",
        "y": "mediolateral",
        "z": "vertical (+up)",
        "sagittal_plane": ["x", "z"],
    },
    "projection_matrices": {
        "CAM_1": [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "CAM_2": [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    },
    "data_status": "synthetic example; not participant data or thesis results",
}


def generate_example(output_dir: Path, frames: int = 64) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = output_dir / "synthetic_tracks_2d.csv"
    calibration_path = output_dir / "synthetic_calibration.json"
    with tracks_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["frame_index", "camera", "landmark", "x_px", "y_px", "confidence"])
        writer.writeheader()
        for frame in range(frames):
            phase = frame / max(frames - 1, 1)
            flexion = 0.12 * np.sin(np.pi * phase)
            hip = np.array([0.0, 0.0, 1.0])
            knee = np.array([0.0, 0.0, 0.58])
            ankle = np.array([flexion, 0.0, 0.18])
            foot = ankle + np.array([0.22, 0.0, 0.03 * np.sin(2 * np.pi * phase)])
            for camera, projection in SYNTHETIC_CALIBRATION["projection_matrices"].items():
                matrix = np.asarray(projection, dtype=float)
                for name, xyz in zip(("hip", "knee", "ankle", "foot"), (hip, knee, ankle, foot)):
                    uvw = matrix @ np.r_[xyz, 1.0]
                    uv = uvw[:2] / uvw[2]
                    writer.writerow({"frame_index": frame, "camera": camera, "landmark": name,
                                     "x_px": f"{uv[0]:.8f}", "y_px": f"{uv[1]:.8f}", "confidence": "1.0"})
    calibration_path.write_text(__import__("json").dumps(SYNTHETIC_CALIBRATION, indent=2), encoding="utf-8")
    return tracks_path, calibration_path
