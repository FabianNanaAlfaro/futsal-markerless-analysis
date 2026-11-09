"""Build the per-kick temporal coverage file from released 3D trajectories.

Counts describe finite reconstructed coordinates; they do not distinguish direct
2D detections from interpolated values.
"""
from __future__ import annotations
import argparse, csv, math
from collections import defaultdict
from pathlib import Path

LANDMARKS = ("Hip", "Knee", "Ankle", "Foot")
SYSTEMS = {
    "manual_reference": "Manual reference",
    "sports2d": "Sports2D",
    "mediapipe_pose": "MediaPipe Pose",
    "deeplabcut_resnet101": "DeepLabCut (ResNet-101)",
}

def finite_xyz(row):
    if row is None:
        return False
    try:
        return all(math.isfinite(float(row[key])) for key in ("x_m", "y_m", "z_m"))
    except (KeyError, TypeError, ValueError):
        return False

def build(tracking_csv: Path):
    metadata = {}
    points = defaultdict(lambda: defaultdict(dict))
    with tracking_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            trial_id = row["trial_id"]
            meta = {key: row[key] for key in ("participant_id", "bilateral_pair_id", "kicking_side")}
            if trial_id in metadata and metadata[trial_id] != meta:
                raise ValueError(f"Inconsistent trial metadata for {trial_id}")
            metadata[trial_id] = meta
            points[(trial_id, row["system"], int(row["frame"]))][row["landmark"]] = row
    if len(metadata) != 78:
        raise ValueError(f"Expected 78 kicks; found {len(metadata)}")
    if len({m["participant_id"] for m in metadata.values()}) != 13:
        raise ValueError("Expected 13 participant codes")
    side_counts = {side: sum(m["kicking_side"] == side for m in metadata.values()) for side in ("left", "right")}
    if side_counts != {"left": 39, "right": 39}:
        raise ValueError(f"Expected 39 kicks per side; found {side_counts}")
    output = []
    for trial_id in sorted(metadata):
        series = {
            system: {frame: landmarks for (tid, sys, frame), landmarks in points.items()
                     if tid == trial_id and sys == system}
            for system in SYSTEMS.values()
        }
        if any(not series[system] for system in SYSTEMS.values()):
            raise ValueError(f"Missing a tracking route for {trial_id}")
        complete_manual = [f for f, pts in series["Manual reference"].items()
                           if all(finite_xyz(pts.get(p)) for p in LANDMARKS)]
        if not complete_manual:
            raise ValueError(f"No complete manual-reference frame for {trial_id}")
        start, end = min(complete_manual), max(complete_manual)
        row = {
            "participant_id": metadata[trial_id]["participant_id"],
            "trial_id": trial_id,
            "bilateral_pair_id": metadata[trial_id]["bilateral_pair_id"],
            "kicking_side": metadata[trial_id]["kicking_side"],
            "window_start_frame": start,
            "window_end_frame": end,
            "window_length_frames": end - start + 1,
        }
        for prefix, system in SYSTEMS.items():
            frames = series[system]
            for landmark in LANDMARKS:
                row[f"{prefix}_{landmark.lower()}_finite_frames"] = sum(
                    finite_xyz(frames.get(f, {}).get(landmark)) for f in range(start, end + 1))
            row[f"{prefix}_complete_4_landmark_frames"] = sum(
                all(finite_xyz(frames.get(f, {}).get(p)) for p in LANDMARKS)
                for f in range(start, end + 1))
        output.append(row)
    return output

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking", type=Path, required=True,
                        help="Path to data/tracking_3d_long.csv from the dataset release")
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV for S1 Data")
    args = parser.parse_args()
    rows = build(args.tracking)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} trial records: 13 participants, 78 kicks, 39 per side.")
    print(f"Output: {args.output}")

if __name__ == "__main__":
    main()

