"""Command-line interface for pose extraction, reconstruction and kinematics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .adapters.dlc import analyze as run_dlc, model_registry
from .adapters.mediapipe import extract as run_mediapipe
from .adapters.sports2d import run as run_sports2d
from .kinematics import bilateral_difference, signed_symmetry_index
from .metrics import coordinate_agreement, trajectory_association
from .pipeline import process_trial, read_tracks_csv, write_result
from .synthetic import generate_example


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="futsal-markerless", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="process canonical multi-camera 2D tracks through DLT and kinematics")
    run.add_argument("--tracks", type=Path, required=True, help="CSV: frame_index,camera,landmark,x_px,y_px[,confidence]")
    run.add_argument("--calibration", type=Path, required=True, help="JSON with 3x4 camera matrices or 11 DLT coefficients")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--side", choices=("left", "right"), required=True)
    run.add_argument("--source", choices=("markerless", "manual"), default="markerless")
    run.add_argument("--sampling-hz", type=float, default=240.0)
    run.add_argument("--impact-frame", type=int, required=True)
    run.add_argument("--window-start-frame", type=int, required=True)
    run.add_argument("--cameras", nargs="+", help="two or more calibrated camera names")
    run.add_argument("--skip-common-filter", action="store_true")

    demo = commands.add_parser("demo", help="generate and process an explicitly synthetic trial")
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--frames", type=int, default=64)

    models = commands.add_parser("models", help="list the five DeepLabCut models and thesis metadata")
    models.add_argument("--json", action="store_true")

    mp = commands.add_parser("pose-mediapipe", help="extract four functional points from one video")
    mp.add_argument("--video", type=Path, required=True)
    mp.add_argument("--output", type=Path, required=True)
    mp.add_argument("--side", choices=("left", "right"), required=True)
    mp.add_argument("--camera", default="CAM_1")
    mp.add_argument("--roi-config", type=Path)
    mp.add_argument("--model-complexity", type=int, choices=(0, 1, 2), default=2)
    mp.add_argument("--min-confidence", type=float, default=0.15)
    mp.add_argument("--static-image-mode", action="store_true")

    s2d = commands.add_parser("pose-sports2d", help="run Sports2D and export canonical 2D tracks")
    s2d.add_argument("--video", type=Path, required=True)
    s2d.add_argument("--config", type=Path, required=True)
    s2d.add_argument("--output", type=Path, required=True)
    s2d.add_argument("--side", choices=("left", "right"), required=True)
    s2d.add_argument("--camera", default="CAM_1")

    dlc = commands.add_parser("pose-dlc", help="run a registered DeepLabCut model on new videos")
    dlc.add_argument("--model", choices=tuple(model_registry()["models"]), required=True)
    dlc.add_argument("--model-bundle", type=Path, required=True, help="local compatible DeepLabCut model bundle")
    dlc.add_argument("--video", type=Path, action="append", required=True)
    dlc.add_argument("--camera", default="CAM_1", help="camera label written to canonical 2D tracks")
    dlc.add_argument("--device", help="PyTorch device such as cuda:0 or cpu")
    dlc.add_argument("--batch-size", type=int, help="DLC inference batch size; defaults to the model setting")
    dlc.add_argument("--output", type=Path, required=True)

    bilateral = commands.add_parser("bilateral", help="compute signed dominant/non-dominant metrics on normalized curves")
    bilateral.add_argument("--dominant", type=Path, required=True)
    bilateral.add_argument("--non-dominant", type=Path, required=True)
    bilateral.add_argument("--output", type=Path, required=True)

    agreement = commands.add_parser("agreement", help="compare paired coordinate columns from two kinematics CSV files")
    agreement.add_argument("--reference", type=Path, required=True)
    agreement.add_argument("--prediction", type=Path, required=True)
    agreement.add_argument("--columns", nargs="+", required=True)
    agreement.add_argument("--kind", choices=("coordinate", "trajectory"), default="coordinate")
    agreement.add_argument("--normalization-scale", type=float)
    agreement.add_argument("--output", type=Path, required=True)
    return parser


def _read_calibration(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_number(row, key):
    value = row.get(key)
    return float(value) if value not in (None, "") else float("nan")


def _load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _run_bilateral(args) -> dict:
    dominant, non_dominant = _load_csv(args.dominant), _load_csv(args.non_dominant)
    if len(dominant) != len(non_dominant):
        raise ValueError("normalized dominant and non-dominant CSV files must have the same number of phase rows")
    phases_d = np.array([_csv_number(row, "phase_percent") for row in dominant])
    phases_n = np.array([_csv_number(row, "phase_percent") for row in non_dominant])
    if not np.allclose(phases_d, phases_n, equal_nan=True):
        raise ValueError("normalized curves must use the same phase_percent grid")
    rows = []
    for i, phase in enumerate(phases_d):
        row = {"phase_percent": phase}
        for point in ("hip", "knee", "ankle", "foot"):
            d = _csv_number(dominant[i], f"{point}_speed")
            nd = _csv_number(non_dominant[i], f"{point}_speed")
            row[f"{point}_speed_is_percent"] = signed_symmetry_index([d], [nd])[0]
        for name in ("knee_flexion_deg", "ankle_dorsiflexion_deg",
                     "knee_angular_velocity_deg_s", "ankle_angular_velocity_deg_s"):
            row[f"{name}_dominant_minus_non_dominant"] = bilateral_difference(
                [_csv_number(dominant[i], name)], [_csv_number(non_dominant[i], name)]
            )[0]
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"rows": len(rows), "output": str(args.output)}


def _run_agreement(args) -> dict:
    reference_rows, prediction_rows = _load_csv(args.reference), _load_csv(args.prediction)
    ref_by_frame = {row["frame_index"]: row for row in reference_rows if row.get("frame_index")}
    pred_by_frame = {row["frame_index"]: row for row in prediction_rows if row.get("frame_index")}
    common = sorted(set(ref_by_frame) & set(pred_by_frame), key=lambda value: int(value))
    if not common:
        raise ValueError("the input CSV files have no paired frame_index values")
    missing = [c for c in args.columns if c not in reference_rows[0] or c not in prediction_rows[0]]
    if missing:
        raise ValueError("requested columns are missing: " + ", ".join(missing))
    ref = np.array([[_csv_number(ref_by_frame[f], c) for c in args.columns] for f in common])
    pred = np.array([[_csv_number(pred_by_frame[f], c) for c in args.columns] for f in common])
    if args.kind == "trajectory":
        result = {column: trajectory_association(ref[:, i], pred[:, i]) for i, column in enumerate(args.columns)}
    else:
        result = coordinate_agreement(ref, pred, normalization_scale=args.normalization_scale)
    result["paired_frames"] = len(common)
    result["columns"] = args.columns
    return result


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            result = process_trial(
                read_tracks_csv(args.tracks), _read_calibration(args.calibration),
                selected_side=args.side, source=args.source, sampling_hz=args.sampling_hz,
                impact_frame=args.impact_frame, window_start_frame=args.window_start_frame,
                cameras=args.cameras, apply_common_filter=not args.skip_common_filter,
            )
            write_result(result, args.output)
            print(json.dumps({"status": "ok", "output": str(args.output), "frames": len(result["kinematics"])}))
        elif args.command == "demo":
            tracks, calibration = generate_example(args.output / "inputs", args.frames)
            result = process_trial(
                read_tracks_csv(tracks), _read_calibration(calibration), selected_side="right",
                source="markerless", sampling_hz=240.0, impact_frame=args.frames - 1,
                window_start_frame=0, cameras=["CAM_1", "CAM_2"],
            )
            write_result(result, args.output / "results")
            print(json.dumps({"status": "ok", "data": "synthetic", "output": str(args.output / 'results')}))
        elif args.command == "models":
            registry = model_registry()
            print(json.dumps(registry if args.json else registry["models"], indent=2))
        elif args.command == "pose-mediapipe":
            count = run_mediapipe(args.video, args.output, side=args.side,
                                  camera_role=args.camera, roi_config=args.roi_config,
                                  model_complexity=args.model_complexity,
                                  min_confidence=args.min_confidence,
                                  static_image_mode=args.static_image_mode)
            print(json.dumps({"status": "ok", "frames": count, "output": str(args.output)}))
        elif args.command == "pose-sports2d":
            outputs = run_sports2d(args.video, args.config, args.output, side=args.side, camera=args.camera)
            print(json.dumps({"status": "ok", "tracks_2d": [str(path) for path in outputs]}))
        elif args.command == "pose-dlc":
            outputs = run_dlc(
                args.model, args.model_bundle, args.video, args.output,
                camera=args.camera, device=args.device, batch_size=args.batch_size,
            )
            print(json.dumps({"status": "ok", "model": args.model, "tracks_2d": [str(path) for path in outputs]}))
        elif args.command == "bilateral":
            print(json.dumps(_run_bilateral(args)))
        elif args.command == "agreement":
            result = _run_agreement(args)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps({"status": "ok", "output": str(args.output), "paired_frames": result["paired_frames"]}))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
