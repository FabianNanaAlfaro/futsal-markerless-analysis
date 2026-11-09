#!/usr/bin/env python3
"""Rebuild the per-kick MediaPipe knee-angular-velocity peaks.

The output uses the same full-recording filter, signed sagittal knee angle,
240-Hz differentiation, manual-reference window, and 101-point phase
normalization as the temporal-peak analysis. It is therefore comparable to
the MediaPipe peak values used in the repeated-measures Bland--Altman
analysis.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt


SYSTEM = "MediaPipe Pose"
LANDMARKS = ("Hip", "Knee", "Ankle", "Foot")
FS_HZ = 240.0
CUTOFF_HZ = 18.0
FILTER_ORDER = 1
LOWESS_SPAN = 0.10
N_NORMALIZED = 101
EXPECTED_KICKS = 78


def finite_number(value: str | None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def read_windows(path: Path) -> dict[str, dict[str, str | int]]:
    windows: dict[str, dict[str, str | int]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            trial_id = row["trial_id"]
            if trial_id in windows:
                raise ValueError(f"Duplicate window for {trial_id}")
            windows[trial_id] = {
                "participant_id": row["participant_id"],
                "kicking_side": row["kicking_side"],
                "window_start_frame": int(row["window_start_frame"]),
                "window_end_frame": int(row["window_end_frame"]),
            }
    if len(windows) != EXPECTED_KICKS:
        raise ValueError(f"Expected {EXPECTED_KICKS} trial windows; found {len(windows)}")
    side_counts = defaultdict(int)
    for row in windows.values():
        side_counts[str(row["kicking_side"])] += 1
    if sorted(side_counts.values()) != [39, 39]:
        raise ValueError(f"Expected 39 kicks per side; found {dict(side_counts)}")
    return windows


def read_positions(path: Path) -> dict[str, dict[str, np.ndarray]]:
    records: dict[str, dict[str, dict[int, np.ndarray]]] = defaultdict(
        lambda: {landmark: {} for landmark in LANDMARKS}
    )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("system") != SYSTEM or row.get("landmark") not in LANDMARKS:
                continue
            trial_id = row["trial_id"]
            landmark = row["landmark"]
            frame = int(row["frame"])
            if frame in records[trial_id][landmark]:
                raise ValueError(f"Duplicate {trial_id} {landmark} frame {frame}")
            records[trial_id][landmark][frame] = np.asarray(
                [finite_number(row[key]) for key in ("x_m", "y_m", "z_m")], dtype=float
            )

    output: dict[str, dict[str, np.ndarray]] = {}
    for trial_id, landmarks in records.items():
        observed = sorted({frame for values in landmarks.values() for frame in values})
        if not observed or observed[0] != 1:
            raise ValueError(f"No complete one-based frame series for {trial_id}")
        frames = np.arange(1, observed[-1] + 1, dtype=int)
        trial: dict[str, np.ndarray] = {"frames": frames}
        for landmark, values in landmarks.items():
            array = np.full((frames.size, 3), np.nan, dtype=float)
            for frame, xyz in values.items():
                array[frame - 1] = xyz
            trial[landmark] = array
        output[trial_id] = trial
    return output


def lowess_like(values: np.ndarray, span: float = LOWESS_SPAN) -> np.ndarray:
    """Two-pass robust local-linear smoother used in the article analysis."""
    values = np.asarray(values, dtype=float)
    n = values.size
    if n < 5:
        return values.copy()
    span_n = min(n, max(3, int(math.ceil(span * n))))
    x = np.arange(n, dtype=float)
    robust = np.ones(n, dtype=float)
    output = values.copy()
    for _ in range(2):
        fitted = np.empty(n, dtype=float)
        half = max(1, span_n // 2)
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            local_x = x[lo:hi] - x[i]
            distance = np.max(np.abs(local_x))
            if distance == 0:
                fitted[i] = values[i]
                continue
            u = np.abs(local_x) / distance
            weights = ((1.0 - u**3) ** 3) * robust[lo:hi]
            if np.count_nonzero(weights > 1e-12) < 2:
                fitted[i] = values[i]
                continue
            design = np.column_stack((np.ones_like(local_x), local_x))
            try:
                beta = np.linalg.lstsq(
                    design * weights[:, None], values[lo:hi] * weights, rcond=None
                )[0]
                fitted[i] = beta[0]
            except np.linalg.LinAlgError:
                fitted[i] = values[i]
        residual = values - fitted
        scale = np.median(np.abs(residual)) + 1e-12
        u = residual / (6.0 * scale)
        robust = np.where(np.abs(u) < 1.0, (1.0 - u**2) ** 2, 0.0)
        output = fitted
    return output


def finite_runs(values: np.ndarray):
    valid = np.isfinite(values)
    changes = np.diff(np.r_[False, valid, False].astype(np.int8))
    starts = np.where(changes == 1)[0]
    stops = np.where(changes == -1)[0]
    return zip(starts, stops)


def filter_coordinate(values: np.ndarray) -> np.ndarray:
    """Filter every contiguous finite run; do not discard later valid segments."""
    values = np.asarray(values, dtype=float)
    output = np.full_like(values, np.nan)
    for start, stop in finite_runs(values):
        segment = values[start:stop]
        if segment.size < 3:
            continue
        filtered = segment.copy()
        if segment.size >= 5:
            b, a = butter(
                FILTER_ORDER,
                min(0.999, CUTOFF_HZ / (FS_HZ / 2.0)),
                btype="low",
            )
            try:
                filtered = filtfilt(b, a, segment, method="gust")
            except Exception:
                pad = 3 * max(len(a), len(b))
                if segment.size > pad:
                    filtered = filtfilt(b, a, segment)
        output[start:stop] = lowess_like(filtered)
    return output


def wrap_degrees(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=float) + 180.0) % 360.0 - 180.0


def knee_flexion_angle(positions: dict[str, np.ndarray]) -> np.ndarray:
    """Signed sagittal knee flexion in the study's X-Z coordinate plane."""
    hip = np.asarray(positions["Hip"], dtype=float)[:, [0, 2]]
    knee = np.asarray(positions["Knee"], dtype=float)[:, [0, 2]]
    ankle = np.asarray(positions["Ankle"], dtype=float)[:, [0, 2]]
    thigh, shank = hip - knee, ankle - knee
    cross = thigh[:, 0] * shank[:, 1] - thigh[:, 1] * shank[:, 0]
    dot = np.sum(thigh * shank, axis=1)
    angle_oriented = np.degrees(np.arctan2(cross, dot))
    angle = wrap_degrees(180.0 - angle_oriented)
    valid = np.isfinite(hip).all(axis=1) & np.isfinite(knee).all(axis=1)
    valid &= np.isfinite(ankle).all(axis=1)
    valid &= (np.linalg.norm(thigh, axis=1) > 1e-10)
    valid &= (np.linalg.norm(shank, axis=1) > 1e-10)
    angle[~valid] = np.nan
    return angle


def angular_velocity(angle: np.ndarray) -> np.ndarray:
    output = np.full_like(angle, np.nan, dtype=float)
    for start, stop in finite_runs(angle):
        if stop - start < 3:
            continue
        unwrapped = np.degrees(np.unwrap(np.radians(angle[start:stop])))
        output[start:stop] = np.gradient(unwrapped, 1.0 / FS_HZ)
    return output


def resample_phase(values: np.ndarray, n: int = N_NORMALIZED) -> np.ndarray:
    """Map finite runs onto the original 0-100% window phase without stretching gaps."""
    values = np.asarray(values, dtype=float)
    output = np.full(n, np.nan, dtype=float)
    if values.size == 0:
        return output
    source_phase = np.linspace(0.0, 1.0, values.size)
    target_phase = np.linspace(0.0, 1.0, n)
    for start, stop in finite_runs(values):
        if stop - start < 2:
            continue
        inside = (target_phase >= source_phase[start]) & (target_phase <= source_phase[stop - 1])
        output[inside] = np.interp(
            target_phase[inside], source_phase[start:stop], values[start:stop]
        )
    return output


def summarize_trial(
    trial_id: str,
    trial: dict[str, np.ndarray],
    window: dict[str, str | int],
) -> dict[str, object]:
    filtered = {
        landmark: np.column_stack(
            [filter_coordinate(trial[landmark][:, axis]) for axis in range(3)]
        )
        for landmark in LANDMARKS
    }
    angle = knee_flexion_angle(filtered)
    full_velocity = angular_velocity(angle)
    start = max(0, int(window["window_start_frame"]) - 1)
    stop = min(len(full_velocity), int(window["window_end_frame"]))
    if stop <= start:
        raise ValueError(f"Empty analysis window for {trial_id}")
    normalized_velocity = resample_phase(full_velocity[start:stop])
    finite = np.flatnonzero(np.isfinite(normalized_velocity))
    peak_index = int(finite[np.argmax(np.abs(normalized_velocity[finite]))]) if finite.size else None
    row: dict[str, object] = {
        "participant_id": window["participant_id"],
        "trial_id": trial_id,
        "kicking_side": window["kicking_side"],
        "window_start_frame": int(window["window_start_frame"]),
        "window_end_frame": int(window["window_end_frame"]),
        "window_length_frames": int(window["window_end_frame"]) - int(window["window_start_frame"]) + 1,
        "finite_normalized_velocity_points": int(finite.size),
        "peak_abs_knee_angular_velocity_deg_s": "",
        "peak_phase_pct": "",
    }
    if peak_index is not None:
        row["peak_abs_knee_angular_velocity_deg_s"] = float(abs(normalized_velocity[peak_index]))
        row["peak_phase_pct"] = peak_index
    return row


def compare_reference(rows: list[dict[str, object]], reference_path: Path) -> float:
    reference: dict[str, float] = {}
    with reference_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("system") == SYSTEM and row.get("feature") == "peak_abs_knee_angvel":
                reference[row["trial_id"]] = float(row["method_peak"])
    calculated = {
        str(row["trial_id"]): float(row["peak_abs_knee_angular_velocity_deg_s"])
        for row in rows
        if row["peak_abs_knee_angular_velocity_deg_s"] != ""
    }
    if set(calculated) != set(reference):
        raise ValueError(
            f"Reference trial IDs differ: calculated={len(calculated)}, reference={len(reference)}"
        )
    differences = {key: abs(calculated[key] - reference[key]) for key in calculated}
    maximum = max(differences.values(), default=0.0)
    if maximum > 1e-6:
        worst = max(differences, key=differences.get)
        raise ValueError(
            f"Peak values disagree: max |difference|={maximum:.9g} at {worst}"
        )
    return maximum


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking", type=Path, required=True, help="Path to tracking_3d_long.csv")
    parser.add_argument(
        "--coverage", type=Path,
        default=root / "supplementary_data" / "S1_Data_temporal_coverage.csv",
        help="Per-kick common analysis windows",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "supplementary_data" / "S2_Data_MP_knee_velocity_by_trial.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--reference-metrics", type=Path,
        help="Optional metricas_temporales_picos.csv for exact independent comparison",
    )
    args = parser.parse_args()

    windows = read_windows(args.coverage)
    positions = read_positions(args.tracking)
    if set(windows) != set(positions):
        raise ValueError(
            f"Trial IDs differ: windows={len(windows)}, trajectories={len(positions)}"
        )
    rows = [summarize_trial(trial_id, positions[trial_id], windows[trial_id]) for trial_id in sorted(windows)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {
                key: (f"{value:.8f}" if isinstance(value, float) and math.isfinite(value) else value)
                for key, value in row.items()
            }
            for row in rows
        )
    finite_count = sum(row["peak_abs_knee_angular_velocity_deg_s"] != "" for row in rows)
    print(f"Wrote {len(rows)} trial rows; finite normalized peaks available in {finite_count}.")
    if args.reference_metrics:
        maximum = compare_reference(rows, args.reference_metrics)
        print(f"Reference comparison passed for all {len(rows)} kicks; max absolute difference={maximum:.3g} deg/s.")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

