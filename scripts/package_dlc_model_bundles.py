from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
import zipfile
from pathlib import Path

import yaml


REGISTRY_NAME = Path(__file__).resolve().parents[1] / "src/sports2d_dvideow_pipeline/models/registry.json"
DETECTOR_CHECKPOINT = "snapshot-detector-best-020.pt"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest().upper()


def remove_local_paths(value):
    if isinstance(value, dict):
        result = {}
        for child_key, child_value in value.items():
            lowered = str(child_key).lower()
            if lowered in {"video_sets", "video_sets_for_training"}:
                result[child_key] = {}
            elif lowered in {"project_path", "dataset"}:
                result[child_key] = "."
            else:
                result[child_key] = remove_local_paths(child_value)
        return result
    if isinstance(value, list):
        return [remove_local_paths(item) for item in value]
    return value


def validate_no_local_paths(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    slash = chr(47)
    normalized = text.replace(chr(92), slash).lower()
    windows_user_path = re.compile(r"(?<![a-z0-9_])[a-z]:" + slash + "u" + "sers" + slash)
    unix_user_path = slash + "home" + slash
    if windows_user_path.search(normalized) or unix_user_path in normalized:
        raise ValueError(f"personal or machine-specific path remains in {path}")

def write_yaml(source: Path, destination: Path) -> None:
    content = yaml.safe_load(source.read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(remove_local_paths(content), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    validate_no_local_paths(destination)


def distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create portable, inference-ready bundles for the five thesis DLC checkpoints."
    )
    parser.add_argument("--source-project", type=Path, required=True, help="DeepLabCut project root")
    parser.add_argument("--output-dir", type=Path, required=True, help="new or empty directory for model bundles and ZIP assets")
    parser.add_argument(
        "--registry",
        type=Path,
        default=REGISTRY_NAME,
        help="package model registry JSON (defaults to the checked-in registry)",
    )
    args = parser.parse_args()
    source = args.source_project.resolve()
    out = args.output_dir.resolve()
    config_path = source / "config.yaml"
    if not config_path.is_file():
        parser.error(f"missing DLC project config: {config_path}")
    if out.exists() and any(out.iterdir()):
        parser.error("output directory must be empty; existing files are left untouched")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    fractions = config.get("TrainingFraction")
    if not fractions or len(fractions) != 1:
        parser.error("expected the thesis benchmark project to contain its single training fraction")
    fraction = float(fractions[0])
    iteration = int(config.get("iteration", 0))
    task_date = f"{config['Task']}{config['date']}"
    model_base = source / "dlc-models-pytorch" / f"iteration-{iteration}"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "training_project": config["Task"],
        "training_date": str(config["date"]),
        "training_fraction": fraction,
        "training_labels": registry["benchmark"]["labels_train"],
        "validation_labels": registry["benchmark"]["labels_validation"],
        "total_labels": registry["benchmark"]["labels_total"],
        "training_frame_stride": registry["benchmark"]["training_frame_stride"],
        "training_frame_rate_hz": registry["benchmark"]["training_frame_rate_hz"],
        "landmarks": config.get("bodyparts", registry["benchmark"]["landmarks"]),
        "detector_checkpoint": DETECTOR_CHECKPOINT,
        "models": {},
    }

    for model_name, model in registry["models"].items():
        folder_name = f"{task_date}-trainset{int(100 * fraction)}shuffle{model['shuffle']}"
        source_folder = model_base / folder_name
        source_train = source_folder / "train"
        pose_source = source_folder / "test" / "pose_cfg.yaml"
        snapshot_source = source_train / model["checkpoint"]
        detector_source = source_train / DETECTOR_CHECKPOINT
        train_config_source = source_train / "pytorch_config.yaml"
        required = (snapshot_source, detector_source, train_config_source, pose_source)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing DLC model files:\n" + "\n".join(missing))
        if digest(snapshot_source) != model["sha256"]:
            raise ValueError(f"checkpoint SHA-256 mismatch for {model_name}")
        expected_detector_sha = registry["benchmark"].get("detector_checkpoint_sha256")
        if expected_detector_sha and digest(detector_source) != expected_detector_sha:
            raise ValueError(f"detector SHA-256 mismatch for {model_name}")

        bundle_dir = out / model_name
        if bundle_dir.exists():
            parser.error(f"bundle already exists and was left untouched: {bundle_dir}")
        bundle_model_dir = bundle_dir / "dlc-models-pytorch" / f"iteration-{iteration}" / folder_name
        train_dest = bundle_model_dir / "train"
        test_dest = bundle_model_dir / "test"
        train_dest.mkdir(parents=True)
        test_dest.mkdir(parents=True)
        shutil.copy2(snapshot_source, train_dest / model["checkpoint"])
        shutil.copy2(detector_source, train_dest / DETECTOR_CHECKPOINT)
        write_yaml(train_config_source, train_dest / "pytorch_config.yaml")
        write_yaml(pose_source, test_dest / "pose_cfg.yaml")

        portable_config = remove_local_paths(config)
        portable_config["project_path"] = "."
        portable_config["video_sets"] = {}
        portable_config["snapshotindex"] = -1
        (bundle_dir / "config.yaml").write_text(
            yaml.safe_dump(portable_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        validate_no_local_paths(bundle_dir / "config.yaml")

        model_metadata = {
            "model": model_name,
            "architecture": model["architecture"],
            "iteration": iteration,
            "shuffle": int(model["shuffle"]),
            "selected_epoch": int(model["selected_epoch"]),
            "selected_checkpoint": model["checkpoint"],
            "selected_checkpoint_sha256": digest(snapshot_source),
            "detector_checkpoint": DETECTOR_CHECKPOINT,
            "detector_checkpoint_sha256": digest(detector_source),
            "checkpoint_size_bytes": snapshot_source.stat().st_size,
            "detector_size_bytes": detector_source.stat().st_size,
            "training_fraction": fraction,
            "inference_snapshot_index": -1,
            "detector_snapshot_index": -1,
            "inference_pcutoff_for_reported_metrics": registry["benchmark"]["inference_pcutoff"],
            "bodyparts": config.get("bodyparts", []),
            "source_model_folder": folder_name,
            "training_software_versions_recorded": False,
            "bundle_creation_environment": {
                "deeplabcut": distribution_version("deeplabcut"),
                "torch": distribution_version("torch"),
                "torchvision": distribution_version("torchvision"),
            },
            "selection_validation": {
                "rmse_px": model["selection_validation_rmse_px"],
                "rmse_at_pcutoff_px": model["selection_validation_rmse_pcutoff_px"],
                "mAP": model.get("selection_validation_map"),
                "mAR": model.get("selection_validation_mar"),
            },
            "benchmark_scope": registry["benchmark"]["interpretation"],
            "training_data_and_labels_included": False,
        }
        (bundle_dir / "model_metadata.json").write_text(
            json.dumps(model_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        license_path = Path(__file__).resolve().parents[1] / "LICENSE"
        shutil.copy2(license_path, bundle_dir / "LICENSE.txt")
        bundle_readme = f"""# {model_name} DeepLabCut model bundle

This bundle contains the thesis-trained {model_name} lower-limb pose checkpoint and the shared SSDLite person detector used by the five-model benchmark.

## Files

- config.yaml: portable DLC project settings with study video paths removed.
- dlc-models-pytorch/: selected pose checkpoint, shared detector checkpoint, architecture/training configuration and inference configuration.
- model_metadata.json: selected epoch, checkpoint hashes, benchmark metadata and inference snapshot indexes.

The training videos, participant annotations and calibration files are not included. The training-set label interval was one frame in four at 240 Hz. It describes annotation sampling only.

## Inference

From the package repository, install the dlc extra in a Python 3.10–3.12 environment with a compatible PyTorch CPU or CUDA installation. Then run:

    sports2d-dvideow pose-dlc --model "{model_name}" --model-bundle ./{model_name} --video input.mp4 --camera CAM_1 --output predictions

The command fixes both snapshot indexes to -1; each bundle contains only the selected pose checkpoint and detector checkpoint. Use a new or empty output directory so DLC does not reuse previous predictions. DLC outputs preserve its native files and the command also writes a canonical <video>_tracks_2d.csv. Optional device and batch-size settings are available in the CLI.

## License and attribution

The pose and detector checkpoint artifacts in this bundle are released under the MIT License included as LICENSE.txt. The repository NOTICE.md describes attribution and source-data boundaries. DeepLabCut and its dependencies retain their own licenses. Do not redistribute the training videos or annotations through this bundle.
"""
        (bundle_dir / "README.md").write_text(bundle_readme, encoding="utf-8")
        (bundle_dir / "NOTICE.txt").write_text(
            "Trained model parameters by Fabian A. Nana for the futsal-kicking thesis project. "
            "The original study videos, annotations and calibration records are not included. "
            "DeepLabCut and dependency licenses remain applicable to their respective software. "
            "See the repository NOTICE.md for project attribution and source-data boundaries.\n",
            encoding="utf-8",
        )

        for text_path in bundle_dir.rglob("*"):
            if text_path.is_file() and text_path.suffix.lower() in {".yaml", ".yml", ".json", ".md", ".txt"}:
                validate_no_local_paths(text_path)

        archive = out / model["release_asset"]
        if archive.exists():
            parser.error(f"archive already exists and was left untouched: {archive}")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle_zip:
            for item in sorted(bundle_dir.rglob("*")):
                if item.is_file():
                    bundle_zip.write(item, item.relative_to(bundle_dir).as_posix())
        manifest["models"][model_name] = {
            "archive": archive.name,
            "archive_size_bytes": archive.stat().st_size,
            "selected_checkpoint_sha256": model_metadata["selected_checkpoint_sha256"],
            "detector_checkpoint_sha256": model_metadata["detector_checkpoint_sha256"],
            "model_folder": folder_name,
        }

    manifest_path = out / "bundles_manifest.json"
    if manifest_path.exists():
        parser.error(f"manifest already exists and was left untouched: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
