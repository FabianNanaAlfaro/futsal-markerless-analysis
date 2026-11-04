"""Sports2D adapter and native TRC to canonical 2D track conversion."""

from __future__ import annotations

import csv
import math
from copy import deepcopy
from pathlib import Path

from ..schema import homologate_landmark


def _merge_supported(target: dict, source: dict, prefix: str = "") -> list[str]:
    """Merge TOML values accepted by the installed Sports2D configuration."""
    ignored = []
    for key, value in source.items():
        name = f"{prefix}.{key}" if prefix else key
        if key not in target:
            ignored.append(name)
        elif isinstance(target[key], dict) and isinstance(value, dict):
            ignored.extend(_merge_supported(target[key], value, name))
        else:
            target[key] = deepcopy(value)
    return ignored


def read_sports2d_trc(path, *, camera: str, side: str, frame_rate: float) -> list[dict]:
    """Read a Sports2D pixel-space TRC and return functional 2D observations."""
    if side not in ("left", "right") or not camera.strip() or frame_rate <= 0:
        raise ValueError("side, camera and a positive video frame rate are required")
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream, delimiter="\t"))
    header_index = next((i for i, row in enumerate(rows) if row and row[0].strip() == "Frame#"), None)
    if header_index is None or header_index + 2 >= len(rows):
        raise ValueError(f"{path} is not a readable Sports2D TRC file")
    marker_row = rows[header_index]
    marker_names = [marker_row[i].strip() for i in range(2, len(marker_row), 3) if marker_row[i].strip()]
    if not marker_names:
        raise ValueError(f"{path} has no named landmarks")
    records = [record for record in rows[header_index + 2:] if len(record) >= 2]
    try:
        first_time = float(records[0][1])
    except (IndexError, ValueError, TypeError):
        raise ValueError(f"{path} has no valid first timestamp")
    frame_offset = int(round(first_time * frame_rate))
    observations = []
    for record in records:
        try:
            sample_index = int(round(float(record[0])))
        except (ValueError, TypeError):
            continue
        frame_index = frame_offset + sample_index
        for marker_index, name in enumerate(marker_names):
            # Sports2D also exports the generic torso root "Hip". The lower-limb
            # schema needs an explicitly lateralized hip, knee, ankle and foot.
            if not name.lower().startswith(("l", "r")) or homologate_landmark(name, side) is None:
                continue
            x_col = 2 + marker_index * 3
            if len(record) <= x_col + 1:
                continue
            try:
                x_px, y_px = float(record[x_col]), float(record[x_col + 1])
            except (ValueError, TypeError):
                x_px = y_px = math.nan
            # Sports2D writes all-zero coordinates for deleted or unobserved poses.
            if x_px == 0.0 and y_px == 0.0:
                x_px = y_px = math.nan
            observations.append({
                "frame_index": frame_index,
                "camera": camera,
                "landmark": name,
                "x_px": x_px,
                "y_px": y_px,
                "confidence": "",
            })
    if not observations:
        raise RuntimeError(f"{path} contains no landmarks for the selected {side} side")
    if not any(math.isfinite(float(row["x_px"])) and math.isfinite(float(row["y_px"])) for row in observations):
        raise RuntimeError(f"{path} contains no valid observed coordinates for the selected {side} side")
    return observations


def write_canonical_tracks(rows: list[dict], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["frame_index", "camera", "landmark", "x_px", "y_px", "confidence"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def run(video, config, output_dir, *, side: str, camera: str = "CAM_1") -> list[Path]:
    video, config, output_dir = Path(video).resolve(), Path(config).resolve(), Path(output_dir).resolve()
    if not video.is_file() or not config.is_file():
        raise FileNotFoundError("input video and Sports2D TOML configuration must exist")
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    if not camera.strip():
        raise ValueError("camera name cannot be empty")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("choose a new or empty output directory for Sports2D results")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import cv2
        import toml
        from Sports2D import Sports2D as sports2d_api
        from Sports2D.Sports2D import DEFAULT_CONFIG
    except ImportError as exc:
        raise RuntimeError("install the optional [sports2d] extra and a compatible PyTorch build in its own environment") from exc

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"could not open input video: {video.name}")
    frame_rate = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise RuntimeError("could not read a valid frame rate from the input video")

    user_config = toml.loads(config.read_text(encoding="utf-8"))
    runtime_config = deepcopy(DEFAULT_CONFIG)
    ignored = _merge_supported(runtime_config, user_config)
    runtime_config["base"].update({
        "video_input": [str(video)],
        "video_dir": "",
        "result_dir": str(output_dir),
        "show_realtime_results": False,
        "save_vid": False,
        "save_img": False,
        "save_pose": True,
    })
    if ignored:
        import warnings
        warnings.warn("Ignoring Sports2D config keys unsupported by the installed version: " + ", ".join(ignored), RuntimeWarning)
    sports2d_api.process(runtime_config)

    native_trc = sorted(output_dir.rglob("*_person00.trc"))
    if not native_trc:
        raise RuntimeError("Sports2D completed without a person 00 pixel-space TRC; check detection, duration and person-selection settings")
    rows = read_sports2d_trc(native_trc[0], camera=camera, side=side, frame_rate=frame_rate)
    tracks_path = output_dir / f"{video.stem}_tracks_2d.csv"
    write_canonical_tracks(rows, tracks_path)
    return [tracks_path]
