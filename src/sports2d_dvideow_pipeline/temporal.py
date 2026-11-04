"""Thesis-based missing-data, filtering and phase-normalization operations."""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, filtfilt, savgol_filter


def hampel_velocity_mask(values, half_window: int = 5, threshold: float = 3.0) -> np.ndarray:
    """Flag local outliers in a frame-to-frame coordinate derivative."""
    x = np.asarray(values, dtype=float)
    n = x.size
    mask = np.zeros(n, dtype=bool)
    if n < 3:
        return mask
    filled = x.copy()
    valid = np.flatnonzero(np.isfinite(filled))
    if valid.size < 3:
        return mask
    missing = np.flatnonzero(~np.isfinite(filled))
    filled[missing] = np.interp(missing, valid, filled[valid])
    velocity = np.diff(filled)
    for i in range(n - 1):
        lo, hi = max(0, i - half_window), min(len(velocity), i + half_window + 1)
        window = velocity[lo:hi]
        med = float(np.median(window))
        mad = float(np.median(np.abs(window - med)))
        if abs(float(velocity[i]) - med) > threshold * 1.4826 * mad + 1e-12:
            mask[i:i + 2] = True
    return mask & np.isfinite(x)


def interpolate_short_gaps(values, max_gap: int = 5) -> np.ndarray:
    """Fill only interior gaps up to ``max_gap``; retain long gaps and edges."""
    x = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(x)
    i = 0
    while i < len(x):
        if finite[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(x) and not finite[j + 1]:
            j += 1
        if j - i + 1 <= max_gap and i > 0 and j + 1 < len(x) and finite[i - 1] and finite[j + 1]:
            valid = np.flatnonzero(finite)
            local = valid[(valid >= max(0, i - 3)) & (valid <= min(len(x) - 1, j + 3))]
            target = np.arange(i, j + 1)
            if local.size >= 3:
                x[target] = PchipInterpolator(local, x[local])(target)
            else:
                x[target] = np.interp(target, [i - 1, j + 1], [x[i - 1], x[j + 1]])
            finite[target] = True
        i = j + 1
    return x


def curate_2d_point(xy, max_gap: int = 5, hampel_half_window: int = 5,
                    hampel_threshold: float = 3.0, savgol_window: int = 5,
                    savgol_order: int = 2) -> np.ndarray:
    """Apply one shared 2D validity mask, short-gap fill and light SG smoothing."""
    points = np.asarray(xy, dtype=float).copy()
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("xy must have shape (frames, 2)")
    invalid = ~np.isfinite(points).all(axis=1)
    invalid |= hampel_velocity_mask(points[:, 0], hampel_half_window, hampel_threshold)
    invalid |= hampel_velocity_mask(points[:, 1], hampel_half_window, hampel_threshold)
    points[invalid] = np.nan
    for axis in range(2):
        points[:, axis] = interpolate_short_gaps(points[:, axis], max_gap)
        start = 0
        while start < len(points):
            if not np.isfinite(points[start, axis]):
                start += 1
                continue
            end = start
            while end + 1 < len(points) and np.isfinite(points[end + 1, axis]):
                end += 1
            length = end - start + 1
            window = min(savgol_window, length if length % 2 else length - 1)
            if window >= savgol_order + 2 and window % 2:
                points[start:end + 1, axis] = savgol_filter(
                    points[start:end + 1, axis], window, min(savgol_order, window - 1), mode="interp"
                )
            start = end + 1
    return points


def robust_lowess(values, span: float = 0.10, robust_iterations: int = 2) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    n = y.size
    if n < 5:
        return y.copy()
    span_n = min(n, max(3, int(math.ceil(span * n))))
    x = np.arange(n, dtype=float)
    robust = np.ones(n)
    result = y.copy()
    for _ in range(robust_iterations):
        z = np.empty(n)
        half = max(1, span_n // 2)
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            xx = x[lo:hi] - x[i]
            distance = np.max(np.abs(xx))
            if distance == 0:
                z[i] = y[i]
                continue
            u = np.abs(xx) / distance
            weights = ((1 - u**3) ** 3) * robust[lo:hi]
            if np.count_nonzero(weights > 1e-12) < 2:
                z[i] = y[i]
                continue
            design = np.column_stack((np.ones_like(xx), xx))
            try:
                beta = np.linalg.lstsq(design * weights[:, None], y[lo:hi] * weights, rcond=None)[0]
                z[i] = beta[0]
            except np.linalg.LinAlgError:
                z[i] = y[i]
        residual = y - z
        scale = np.median(np.abs(residual)) + 1e-12
        u = residual / (6.0 * scale)
        robust = np.where(np.abs(u) < 1.0, (1 - u**2) ** 2, 0.0)
        result = z
    return result


def filter_common_trace(values, sampling_hz: float = 240.0, cutoff_hz: float = 18.0,
                        order: int = 1, lowess_span: float = 0.10) -> np.ndarray:
    """Butterworth low-pass then robust LOWESS on each finite contiguous run."""
    x = np.asarray(values, dtype=float)
    if sampling_hz <= 0 or cutoff_hz <= 0 or cutoff_hz >= sampling_hz / 2:
        raise ValueError("filter frequencies must satisfy 0 < cutoff_hz < sampling_hz/2")
    if order < 1:
        raise ValueError("order must be positive")
    out = np.full_like(x, np.nan)
    finite = np.isfinite(x)
    edges = np.diff(np.r_[False, finite, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    b, a = butter(order, cutoff_hz / (sampling_hz / 2.0), btype="low")
    for start, end in zip(starts, ends):
        segment = x[start:end]
        if len(segment) < 3:
            out[start:end] = segment
            continue
        try:
            low = filtfilt(b, a, segment, method="gust")
        except (ValueError, FloatingPointError):
            low = segment
        out[start:end] = robust_lowess(low, lowess_span)
    return out


def normalize_to_phase(frame_index, values, start_frame: int, end_frame: int,
                       points: int = 101) -> np.ndarray:
    """Resample the declared start-to-impact interval, preserving internal gaps."""
    frames = np.asarray(frame_index, dtype=float)
    values = np.asarray(values, dtype=float)
    if frames.shape != values.shape or points < 2 or end_frame <= start_frame:
        raise ValueError("invalid frame/value arrays or normalization window")
    target = np.linspace(start_frame, end_frame, points)
    out = np.full(points, np.nan)
    in_window = (frames >= start_frame) & (frames <= end_frame)
    local_frames, local_values = frames[in_window], values[in_window]
    if local_frames.size == 0:
        return out
    finite = np.isfinite(local_frames) & np.isfinite(local_values)
    valid = np.flatnonzero(finite)
    if valid.size < 2:
        return out
    breaks = np.flatnonzero(np.diff(valid) > 1)
    groups = np.split(valid, breaks + 1)
    for group in groups:
        if group.size < 2:
            continue
        lo, hi = local_frames[group[0]], local_frames[group[-1]]
        mask = (target >= lo) & (target <= hi)
        out[mask] = np.interp(target[mask], local_frames[group], local_values[group])
    return out
