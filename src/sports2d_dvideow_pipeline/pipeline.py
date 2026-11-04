"""File-independent orchestration for 2D-to-3D trial processing."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .geometry import load_projection_matrices, triangulate_point
from .kinematics import (
    ankle_dorsiflexion_angle,
    angular_velocity,
    knee_flexion_angle,
    vector_velocity,
)
from .schema import FUNCTIONAL_POINTS, homologate_landmark
from .temporal import curate_2d_point, filter_common_trace, normalize_to_phase


def read_tracks_csv(path: Path) -> list[dict]:
    required = {"frame_index", "camera", "landmark", "x_px", "y_px"}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("tracks CSV is missing columns: " + ", ".join(sorted(missing)))
        rows = []
        for row in reader:
            rows.append({
                "frame_index": int(row["frame_index"]),
                "camera": row["camera"].strip(),
                "landmark": row["landmark"].strip(),
                "x_px": _number(row.get("x_px")),
                "y_px": _number(row.get("y_px")),
                "confidence": _number(row.get("confidence")),
            })
    if not rows:
        raise ValueError("tracks CSV contains no observations")
    return rows


def _number(value):
    if value is None or not str(value).strip():
        return float("nan")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"coordinate/confidence value {value!r} is not numeric") from exc


def _json_value(value):
    if isinstance(value, np.ndarray):
        return [_json_value(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def process_trial(
    tracks: list[dict],
    calibration: dict,
    *,
    selected_side: str,
    source: str = "markerless",
    sampling_hz: float = 240.0,
    impact_frame: int,
    window_start_frame: int,
    cameras: list[str] | None = None,
    apply_common_filter: bool = True,
) -> dict:
    """Homologate, curate automated 2D tracks, triangulate and derive kinematics.

    ``source='manual'`` skips the markerless 2D Hampel/PCHIP/Savitzky-Golay
    stage. Both sources pass through the declared common 3D filter. This is a
    generic DLT implementation; equivalence to legacy DVIDEOW software is not
    assumed.
    """
    if selected_side not in ("left", "right"):
        raise ValueError("selected_side must be left or right")
    if source not in ("markerless", "manual"):
        raise ValueError("source must be markerless or manual")
    if sampling_hz <= 0 or window_start_frame >= impact_frame:
        raise ValueError("sampling_hz must be positive and window_start_frame before impact_frame")
    projection_matrices = load_projection_matrices(calibration)
    camera_names = list(cameras) if cameras else list(projection_matrices)
    camera_names = [name for name in camera_names if name in projection_matrices]
    if len(camera_names) < 2:
        raise ValueError("select at least two cameras present in the calibration")

    groups = defaultdict(dict)
    all_frames = []
    for row in tracks:
        frame = int(row["frame_index"])
        point = homologate_landmark(row["landmark"], selected_side)
        if point is None or row["camera"] not in camera_names:
            continue
        key = (row["camera"], point)
        if frame in groups[key]:
            raise ValueError(f"duplicate observation for {row['camera']}/{point} at frame {frame}")
        xy = np.array([row.get("x_px", np.nan), row.get("y_px", np.nan)], dtype=float)
        confidence = row.get("confidence")
        if confidence is not None and np.isfinite(confidence) and confidence <= 0:
            xy[:] = np.nan
        groups[key][frame] = xy
        all_frames.append(frame)
    if not all_frames:
        raise ValueError("no track labels matched the selected side and functional-point schema")
    first, last = min(all_frames), max(all_frames)
    frames = np.arange(first, last + 1, dtype=int)

    curated = {}
    for camera in camera_names:
        for point in FUNCTIONAL_POINTS:
            xy = np.full((len(frames), 2), np.nan)
            for index, frame in enumerate(frames):
                if frame in groups.get((camera, point), {}):
                    xy[index] = groups[(camera, point)][int(frame)]
            if source == "markerless":
                xy = curate_2d_point(xy)
            curated[(camera, point)] = xy

    positions = {point: np.full((len(frames), 3), np.nan) for point in FUNCTIONAL_POINTS}
    reprojection = []
    for index, frame in enumerate(frames):
        for point in FUNCTIONAL_POINTS:
            observations = {}
            for camera in camera_names:
                xy = curated[(camera, point)][index]
                if np.isfinite(xy).all():
                    observations[camera] = xy
            try:
                result = triangulate_point(observations, projection_matrices)
            except (ValueError, np.linalg.LinAlgError):
                continue
            positions[point][index] = result["xyz"]
            reprojection.append({
                "frame_index": int(frame),
                "landmark": point,
                "cameras": result["cameras"],
                "errors_px": result["reprojection_error_px"],
                "max_error_px": result["max_reprojection_error_px"],
            })

    filtered = {}
    for point, xyz in positions.items():
        if apply_common_filter:
            filtered[point] = np.column_stack([
                filter_common_trace(xyz[:, axis], sampling_hz=sampling_hz)
                for axis in range(3)
            ])
        else:
            filtered[point] = xyz.copy()

    derived = {}
    for point, xyz in filtered.items():
        velocity = vector_velocity(xyz, sampling_hz)
        speed = np.linalg.norm(velocity, axis=1)
        speed[~np.isfinite(velocity).all(axis=1)] = np.nan
        derived[point] = {"position": xyz, "velocity": velocity, "speed": speed}
    knee = knee_flexion_angle(filtered["hip"], filtered["knee"], filtered["ankle"])
    ankle = ankle_dorsiflexion_angle(filtered["knee"], filtered["ankle"], filtered["foot"])
    knee_velocity = angular_velocity(knee, sampling_hz)
    ankle_velocity = angular_velocity(ankle, sampling_hz)

    kinematics_rows = []
    for i, frame in enumerate(frames):
        row = {"frame_index": int(frame), "time_s": float((frame - first) / sampling_hz)}
        for point in FUNCTIONAL_POINTS:
            for axis, axis_name in enumerate(("x", "y", "z")):
                row[f"{point}_{axis_name}"] = filtered[point][i, axis]
                row[f"{point}_v{axis_name}"] = derived[point]["velocity"][i, axis]
            row[f"{point}_speed"] = derived[point]["speed"][i]
        row["knee_flexion_deg"] = knee[i]
        row["ankle_dorsiflexion_deg"] = ankle[i]
        row["knee_angular_velocity_deg_s"] = knee_velocity[i]
        row["ankle_angular_velocity_deg_s"] = ankle_velocity[i]
        kinematics_rows.append(row)

    normalized = {"phase_percent": np.linspace(0.0, 100.0, 101)}
    for row_name, rows in (("", kinematics_rows),):
        names = [key for key in rows[0] if key not in ("frame_index", "time_s")]
        for name in names:
            values = np.asarray([row[name] for row in rows], dtype=float)
            normalized[name] = normalize_to_phase(frames, values, window_start_frame, impact_frame, 101)

    track_rows = []
    for camera in camera_names:
        for point in FUNCTIONAL_POINTS:
            xy = curated[(camera, point)]
            for i, frame in enumerate(frames):
                track_rows.append({
                    "frame_index": int(frame), "camera": camera, "landmark": point,
                    "x_px": xy[i, 0], "y_px": xy[i, 1],
                    "valid": bool(np.isfinite(xy[i]).all()),
                })
    axes = calibration.get("coordinate_system", {})
    return {
        "tracks_2d": track_rows,
        "kinematics": kinematics_rows,
        "normalized": normalized,
        "reprojection": reprojection,
        "metadata": {
            "schema_version": "1.0",
            "source": source,
            "selected_side": selected_side,
            "sampling_hz": sampling_hz,
            "window_start_frame": int(window_start_frame),
            "impact_frame": int(impact_frame),
            "cameras": camera_names,
            "common_filter": {"type": "Butterworth low-pass + robust LOWESS"} if apply_common_filter else None,
            "coordinate_system": axes,
            "units": calibration.get("units", "as specified by calibration"),
            "dlt_implementation": "linear least-squares triangulation from supplied 3x4 projection matrices",
            "legacy_dvideow_equivalence_validated": False,
        },
    }


def write_result(result: dict, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "tracks_2d_processed.csv", result["tracks_2d"])
    _write_csv(output_dir / "kinematics_3d.csv", result["kinematics"])
    normalized_rows = []
    normalized = result["normalized"]
    for i, phase in enumerate(normalized["phase_percent"]):
        row = {"phase_percent": phase}
        row.update({name: values[i] for name, values in normalized.items() if name != "phase_percent"})
        normalized_rows.append(row)
    _write_csv(output_dir / "kinematics_normalized_101.csv", normalized_rows)
    _write_csv(output_dir / "reprojection_diagnostics.csv", result["reprojection"])
    (output_dir / "run_metadata.json").write_text(
        json.dumps(_json_value(result["metadata"]), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return json.dumps(_json_value(value), ensure_ascii=False)
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return ""
    return value
