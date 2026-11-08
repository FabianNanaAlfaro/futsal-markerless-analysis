"""Validate the published PLOS ONE 78-kick dataset structure and checksums."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


EXPECTED_SYSTEMS = {
    "Manual reference",
    "Sports2D",
    "MediaPipe Pose",
    "DeepLabCut (ResNet-101)",
}
EXPECTED_LANDMARKS = {"Hip", "Knee", "Ankle", "Foot"}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        return reader.fieldnames, list(reader)


def verify_checksums(root: Path) -> int:
    sums_file = root / "SHA256SUMS.txt"
    expected: dict[str, str] = {}
    for line in sums_file.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        expected[relative.strip()] = digest

    csv_files = {p.relative_to(root).as_posix(): p for p in root.rglob("*.csv")}
    if set(expected) != set(csv_files):
        raise ValueError("SHA256SUMS.txt does not list exactly the distributed CSV files")
    for relative, path in csv_files.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected[relative]:
            raise ValueError(f"Checksum mismatch: {relative}")
    return len(csv_files)


def validate(root: Path) -> None:
    manifest_path = root / "data" / "trial_manifest.csv"
    manifest_fields, manifest = read_csv(manifest_path)
    required_manifest = {"participant_id", "trial_id", "bilateral_pair_id", "kicking_side"}
    if not required_manifest.issubset(manifest_fields):
        raise ValueError(f"Manifest is missing fields: {sorted(required_manifest - set(manifest_fields))}")
    trial_ids = {row["trial_id"] for row in manifest}
    participants = {row["participant_id"] for row in manifest}
    pairs = {row["bilateral_pair_id"] for row in manifest}
    side_counts: dict[str, int] = {}
    for row in manifest:
        side_counts[row["kicking_side"]] = side_counts.get(row["kicking_side"], 0) + 1
    if len(manifest) != 78 or len(trial_ids) != 78:
        raise ValueError("The trial manifest must contain 78 unique kicks")
    if len(participants) != 13 or len(pairs) != 39 or sorted(side_counts.values()) != [39, 39]:
        raise ValueError("Expected 13 participants, 39 kicks per side, and 39 bilateral pairs")

    tracking_fields, tracking = read_csv(root / "data" / "tracking_3d_long.csv")
    required_tracking = {"trial_id", "system", "landmark", "frame", "x_m", "y_m", "z_m"}
    if not required_tracking.issubset(tracking_fields):
        raise ValueError(f"Tracking data are missing fields: {sorted(required_tracking - set(tracking_fields))}")
    if {row["trial_id"] for row in tracking} != trial_ids:
        raise ValueError("Tracking records do not map exactly to the 78 manifest kicks")
    if {row["system"] for row in tracking} != EXPECTED_SYSTEMS:
        raise ValueError("Tracking data do not contain the four documented routes")
    if {row["landmark"] for row in tracking} != EXPECTED_LANDMARKS:
        raise ValueError("Tracking data do not contain the four documented landmarks")
    route_landmark_trials = {
        (system, landmark): set()
        for system in EXPECTED_SYSTEMS
        for landmark in EXPECTED_LANDMARKS
    }
    for row in tracking:
        route_landmark_trials[(row["system"], row["landmark"])].add(row["trial_id"])
    if any(trials != trial_ids for trials in route_landmark_trials.values()):
        raise ValueError("Each route and landmark must map to all 78 kicks")

    for path in (root / "derived").glob("*.csv"):
        fields, rows = read_csv(path)
        if "trial_id" in fields:
            unknown = {row["trial_id"] for row in rows} - trial_ids
            if unknown:
                raise ValueError(f"Unknown trial IDs in {path.name}: {sorted(unknown)[:5]}")

    checksum_count = verify_checksums(root)
    print(
        "Dataset validated: 13 participants; 78 kicks (39 per side); "
        f"39 bilateral pairs; {len(tracking):,} tracking rows; "
        f"{len(EXPECTED_SYSTEMS)} routes; {checksum_count} checksummed CSV files."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/plos-one-78"))
    args = parser.parse_args()
    validate(args.data_root.resolve())


if __name__ == "__main__":
    main()
