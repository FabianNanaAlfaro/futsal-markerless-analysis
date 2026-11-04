"""Multi-view reconstruction from investigator-supplied DLT calibrations."""

from __future__ import annotations

import numpy as np


def projection_matrix_from_dlt(coefficients) -> np.ndarray:
    values = np.asarray(coefficients, dtype=float).reshape(-1)
    if values.size != 11 or not np.isfinite(values).all():
        raise ValueError("classical DLT calibration requires 11 finite coefficients")
    return np.array([
        [values[0], values[1], values[2], values[3]],
        [values[4], values[5], values[6], values[7]],
        [values[8], values[9], values[10], 1.0],
    ])


def load_projection_matrices(calibration: dict) -> dict[str, np.ndarray]:
    """Read 3x4 matrices or 11-coefficient DLT records from JSON data."""
    records = calibration.get("projection_matrices", calibration.get("cameras", {}))
    matrices = {}
    for camera, value in records.items():
        if isinstance(value, dict):
            if "projection_matrix" in value:
                value = value["projection_matrix"]
            elif "dlt_coefficients" in value:
                matrices[camera] = projection_matrix_from_dlt(value["dlt_coefficients"])
                continue
        matrix = np.asarray(value, dtype=float)
        if matrix.size == 11:
            matrix = projection_matrix_from_dlt(matrix)
        if matrix.size != 12 or not np.isfinite(matrix).all():
            raise ValueError(f"camera {camera!r} needs a finite 3x4 matrix or 11 DLT coefficients")
        matrices[camera] = matrix.reshape(3, 4)
    if len(matrices) < 2:
        raise ValueError("at least two calibrated cameras are required")
    return matrices


def project_point(matrix, xyz) -> np.ndarray:
    p = np.asarray(matrix, dtype=float).reshape(3, 4) @ np.r_[np.asarray(xyz, dtype=float).reshape(3), 1.0]
    if not np.isfinite(p).all() or abs(p[2]) < 1e-12:
        raise ValueError("point projects at infinity")
    return p[:2] / p[2]


def triangulate_point(observations: dict, projection_matrices: dict) -> dict:
    cameras = [camera for camera in projection_matrices if camera in observations]
    if len(cameras) < 2:
        raise ValueError("at least two matching camera observations are required")
    rows = []
    clean = {}
    for camera in cameras:
        uv = np.asarray(observations[camera], dtype=float).reshape(2)
        if not np.isfinite(uv).all():
            raise ValueError("camera observations must be finite")
        matrix = np.asarray(projection_matrices[camera], dtype=float).reshape(3, 4)
        rows.extend((uv[0] * matrix[2] - matrix[0], uv[1] * matrix[2] - matrix[1]))
        clean[camera] = uv
    _, singular, vt = np.linalg.svd(np.asarray(rows), full_matrices=True)
    h = vt[-1]
    if abs(h[3]) < 1e-12:
        raise ValueError("DLT solution is at infinity")
    xyz = h[:3] / h[3]
    reprojection = {c: project_point(projection_matrices[c], xyz) for c in cameras}
    errors = {c: float(np.linalg.norm(reprojection[c] - clean[c])) for c in cameras}
    return {
        "xyz": xyz,
        "cameras": cameras,
        "reprojection_error_px": errors,
        "max_reprojection_error_px": max(errors.values()),
        "condition_proxy": float(singular[0] / max(singular[-1], 1e-12)),
    }
