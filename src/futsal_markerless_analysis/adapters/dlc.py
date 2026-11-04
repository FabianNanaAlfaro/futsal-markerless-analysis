"""Run one of the five thesis-trained DeepLabCut models on new videos."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from importlib.resources import files
from pathlib import Path


def model_registry() -> dict:
    path = files("futsal_markerless_analysis").joinpath("models/registry.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _bundle_paths(model_name: str, bundle: Path, model: dict, metadata: dict) -> tuple[Path, Path, Path]:
    if metadata.get("model") != model_name:
        raise ValueError(f"bundle contains {metadata.get('model')!r}, not {model_name!r}")
    iteration = int(metadata.get("iteration", 0))
    checkpoint = bundle / "dlc-models-pytorch" / f"iteration-{iteration}" / metadata["source_model_folder"] / "train" / model["checkpoint"]
    detector = bundle / "dlc-models-pytorch" / f"iteration-{iteration}" / metadata["source_model_folder"] / "train" / metadata["detector_checkpoint"]
    config_yaml = bundle / "config.yaml"
    for path in (checkpoint, detector, config_yaml):
        if not path.is_file():
            raise FileNotFoundError(f"incomplete model bundle; missing {path.name}")
    if _sha256(checkpoint) != model["sha256"] or _sha256(checkpoint) != metadata["selected_checkpoint_sha256"]:
        raise ValueError("pose checkpoint SHA-256 does not match the thesis model registry or supplied bundle metadata")
    if _sha256(detector) != metadata["detector_checkpoint_sha256"]:
        raise ValueError("person-detector checkpoint SHA-256 does not match the model metadata")
    return config_yaml, checkpoint, detector


def _snapshot_index_for_checkpoint(snapshots, checkpoint: Path) -> int:
    """Resolve a DLC snapshot index from the exact checkpoint path in the registry."""
    checkpoint = checkpoint.resolve()
    matches = [
        index
        for index, snapshot in enumerate(snapshots)
        if Path(snapshot.path).resolve() == checkpoint
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one DeepLabCut snapshot for {checkpoint.name}, found {len(matches)}"
        )
    return matches[0]


def _standardize_dlc_csv(csv_path: Path, output_csv: Path, *, camera: str) -> int:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("DeepLabCut's pandas dependency is required to export canonical 2D tracks") from exc

    frame = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    if not isinstance(frame.columns, pd.MultiIndex) or frame.columns.nlevels != 3:
        raise ValueError(f"unexpected DeepLabCut CSV structure in {csv_path.name}")
    if not len(frame.index):
        raise ValueError(f"DeepLabCut output contains no frames: {csv_path.name}")
    try:
        frame_indices = [int(value) for value in frame.index]
    except (TypeError, ValueError):
        frame_indices = list(range(len(frame.index)))

    # DeepLabCut's single-animal export uses scorer/bodypart/coordinate columns.
    bodyparts = list(dict.fromkeys(str(column[1]) for column in frame.columns))
    coords_by_bodypart = {
        bodypart: {str(column[2]).lower(): column for column in frame.columns if str(column[1]) == bodypart}
        for bodypart in bodyparts
    }
    required_coords = {"x", "y", "likelihood"}
    missing = [name for name, coords in coords_by_bodypart.items() if not required_coords.issubset(coords)]
    if missing:
        raise ValueError("DeepLabCut CSV lacks x/y/likelihood coordinates for: " + ", ".join(missing))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["frame_index", "camera", "landmark", "x_px", "y_px", "confidence"],
        )
        writer.writeheader()
        for row_number, frame_index in enumerate(frame_indices):
            for bodypart, coords in coords_by_bodypart.items():
                writer.writerow({
                    "frame_index": frame_index,
                    "camera": camera,
                    "landmark": bodypart,
                    "x_px": frame.iloc[row_number][coords["x"]],
                    "y_px": frame.iloc[row_number][coords["y"]],
                    "confidence": frame.iloc[row_number][coords["likelihood"]],
                })
                count += 1
    return count


def analyze(
    model_name: str,
    model_bundle,
    videos,
    output_dir,
    *,
    camera: str = "CAM_1",
    device: str | None = None,
    batch_size: int | None = None,
) -> list[Path]:
    """Run the exact selected checkpoint and export its native and common 2D CSV files."""
    registry = model_registry()
    models = registry["models"]
    if model_name not in models:
        raise ValueError(f"unknown model {model_name!r}; choose from {', '.join(models)}")
    bundle = Path(model_bundle).resolve()
    video_paths = [Path(video).resolve() for video in videos]
    if not bundle.is_dir() or not video_paths or any(not path.is_file() for path in video_paths):
        raise FileNotFoundError("model bundle and every input video must exist")
    if not camera.strip():
        raise ValueError("camera name cannot be empty")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    try:
        metadata = json.loads((bundle / "model_metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FileNotFoundError("model bundle is missing a readable model_metadata.json") from exc
    config_yaml, checkpoint, detector = _bundle_paths(model_name, bundle, models[model_name], metadata)

    try:
        import deeplabcut
        import yaml
        from deeplabcut.pose_estimation_pytorch.apis.utils import get_model_snapshots
        from deeplabcut.pose_estimation_pytorch.task import Task
    except ImportError as exc:
        raise RuntimeError("install DeepLabCut in a dedicated Python 3.10–3.12 environment") from exc

    pose_snapshots = get_model_snapshots("all", checkpoint.parent, Task.TOP_DOWN)
    detector_snapshots = get_model_snapshots("all", detector.parent, Task.DETECT)
    pose_snapshot_index = _snapshot_index_for_checkpoint(pose_snapshots, checkpoint)
    detector_snapshot_index = _snapshot_index_for_checkpoint(detector_snapshots, detector)

    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("choose a new or empty output directory so DeepLabCut cannot reuse previous predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    project_config = yaml.safe_load(config_yaml.read_text(encoding="utf-8"))
    project_config["project_path"] = str(bundle)
    project_config["video_sets"] = {}
    runtime_config = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".yaml", prefix=".dlc-runtime-",
            dir=bundle, delete=False,
        ) as stream:
            yaml.safe_dump(project_config, stream, sort_keys=False, allow_unicode=True)
            runtime_config = Path(stream.name)
        deeplabcut.analyze_videos(
            str(runtime_config),
            [str(path) for path in video_paths],
            shuffle=int(models[model_name]["shuffle"]),
            trainingsetindex=0,
            engine=deeplabcut.Engine.PYTORCH,
            snapshot_index=pose_snapshot_index,
            detector_snapshot_index=detector_snapshot_index,
            save_as_csv=True,
            destfolder=str(output_dir),
            dynamic=(False, 0.5, 10),
            device=device,
            batch_size=batch_size,
        )
    finally:
        if runtime_config is not None:
            runtime_config.unlink(missing_ok=True)

    standardized = []
    for video in video_paths:
        candidates = [
            path for path in output_dir.glob(f"{video.stem}*.csv")
            if not path.name.endswith("_tracks_2d.csv")
        ]
        if not candidates:
            raise RuntimeError(f"DeepLabCut completed without a native CSV for {video.name}")
        native_csv = max(candidates, key=lambda path: path.stat().st_mtime_ns)
        target = output_dir / f"{video.stem}_tracks_2d.csv"
        _standardize_dlc_csv(native_csv, target, camera=camera)
        standardized.append(target)
    return standardized
