"""Functional-point kinematics and bilateral metrics from the thesis."""

from __future__ import annotations

import numpy as np


def _wrap_degrees(values):
    return (values + 180.0) % 360.0 - 180.0


def vector_velocity(position, sampling_hz: float) -> np.ndarray:
    xyz = np.asarray(position, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or sampling_hz <= 0:
        raise ValueError("position must be (frames,3) and sampling_hz positive")
    out = np.full_like(xyz, np.nan)
    valid = np.isfinite(xyz).all(axis=1)
    edges = np.diff(np.r_[False, valid, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    for start, end in zip(starts, ends):
        if end - start >= 2:
            out[start:end] = np.gradient(xyz[start:end], 1.0 / sampling_hz, axis=0)
    return out


def _oriented_sagittal_angle(first, second):
    """Signed angle in X-Z using +X anterior and +Z superior."""
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    dot = np.sum(a * b, axis=-1)
    det = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    angle = np.degrees(np.arctan2(det, dot))
    good = np.isfinite(a).all(axis=-1) & np.isfinite(b).all(axis=-1)
    good &= (np.linalg.norm(a, axis=-1) > 1e-12) & (np.linalg.norm(b, axis=-1) > 1e-12)
    return np.where(good, angle, np.nan)


def knee_flexion_angle(hip, knee, ankle) -> np.ndarray:
    """Oriented sagittal knee angle: extension 0°, flexion positive."""
    hip, knee, ankle = (np.asarray(p, dtype=float) for p in (hip, knee, ankle))
    thigh = np.stack((hip[..., 0] - knee[..., 0], hip[..., 2] - knee[..., 2]), axis=-1)
    shank = np.stack((ankle[..., 0] - knee[..., 0], ankle[..., 2] - knee[..., 2]), axis=-1)
    return _wrap_degrees(_oriented_sagittal_angle(thigh, shank) + 180.0)


def ankle_dorsiflexion_angle(knee, ankle, foot) -> np.ndarray:
    """Oriented sagittal ankle angle: neutral 0°, dorsiflexion positive.

    The proximal shank vector is ankle-to-knee. This orientation gives the
    neutral reference -90° in X-Z, which is shifted to zero as specified by
    the thesis convention.
    """
    knee, ankle, foot = (np.asarray(p, dtype=float) for p in (knee, ankle, foot))
    shank = np.stack((knee[..., 0] - ankle[..., 0], knee[..., 2] - ankle[..., 2]), axis=-1)
    foot_segment = np.stack((foot[..., 0] - ankle[..., 0], foot[..., 2] - ankle[..., 2]), axis=-1)
    return _wrap_degrees(_oriented_sagittal_angle(shank, foot_segment) + 90.0)


def angular_velocity(angle_deg, sampling_hz: float) -> np.ndarray:
    angle = np.asarray(angle_deg, dtype=float)
    out = np.full_like(angle, np.nan)
    if sampling_hz <= 0:
        raise ValueError("sampling_hz must be positive")
    valid = np.isfinite(angle)
    edges = np.diff(np.r_[False, valid, False].astype(int))
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if end - start >= 2:
            unwrapped = np.unwrap(np.deg2rad(angle[start:end]))
            out[start:end] = np.rad2deg(np.gradient(unwrapped, 1.0 / sampling_hz))
    return out


def signed_symmetry_index(dominant, non_dominant, epsilon: float = 1e-8) -> np.ndarray:
    """Signed speed symmetry index: 200*(D-ND)/(D+ND), NaN at zero denominator."""
    d, nd = np.asarray(dominant, dtype=float), np.asarray(non_dominant, dtype=float)
    denominator = d + nd
    out = np.full(np.broadcast_shapes(d.shape, nd.shape), np.nan)
    np.divide(200.0 * (d - nd), denominator, out=out,
              where=np.isfinite(d) & np.isfinite(nd) & (np.abs(denominator) > epsilon))
    return out


def bilateral_difference(dominant, non_dominant) -> np.ndarray:
    d, nd = np.asarray(dominant, dtype=float), np.asarray(non_dominant, dtype=float)
    return np.where(np.isfinite(d) & np.isfinite(nd), d - nd, np.nan)
