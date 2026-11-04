# Sports2D–DVIDEOW thesis pipeline

[![CI](https://github.com/FabianNanaAlfaro/sports2d-dvideow-thesis-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/FabianNanaAlfaro/sports2d-dvideow-thesis-pipeline/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/code-MIT-2ea44f)](LICENSE)

A Python library and command-line tool for lower-limb pose processing and kicking kinematics. It implements the reusable stages described in *Implementación y validación de algoritmos de inteligencia artificial para analizar la asimetría en el movimiento del pateo en futsal*: homologation of four functional landmarks, multi-camera DLT, temporal processing, 3D velocities, sagittal knee and ankle angles, bilateral metrics, and analysis exports.

The project supports four distinct 2D sources: manual DVIDEOW tracking as the reference, Sports2D, MediaPipe Pose, and five DeepLabCut PyTorch configurations. The reference and markerless routes remain separately identified in the output metadata.

## Capabilities

- Installable Python package with reusable functions and the `sports2d-dvideow` CLI.
- MediaPipe Pose extraction for a user-selected side, optional camera-specific ROI, and explicit mapping of cropped coordinates back to the original pixels.
- Sports2D Python API adapter using a user-supplied TOML configuration.
- DeepLabCut inference for the five thesis checkpoints, once a model bundle and DeepLabCut environment are available.
- Anatomical homologation to hip, knee, ankle, and foot.
- User-supplied projection matrices or 11-coefficient DLT calibrations for 3D reconstruction.
- Hampel-based 2D outlier removal, PCHIP interpolation of gaps up to five frames, linear fallback, and Savitzky–Golay smoothing for markerless tracks.
- Common first-order 18 Hz Butterworth filtering followed by robust LOWESS (span 0.10), with missing intervals preserved.
- 3D position and speed, oriented sagittal knee and ankle angles, angular velocity, phase-normalized curves, bilateral speed symmetry, and signed dominant-minus-nondominant angle differences.
- Coordinate agreement, temporal correlation, peak error, participant-cluster Bland–Altman bootstrap, multiplicity corrections, and optional SPM1D repeated-measures analysis.
- CSV and JSON outputs; an end-to-end example generated entirely from synthetic data.

## Installation

The numerical core supports Python 3.10–3.12. Clone the repository, create a virtual environment, and install the package:

```powershell
git clone https://github.com/FabianNanaAlfaro/sports2d-dvideow-thesis-pipeline.git
cd sports2d-dvideow-thesis-pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
```

On macOS or Linux, activate with `source .venv/bin/activate` instead.

Pose-estimation systems are optional extras. Use separate environments because their computer-vision and deep-learning dependencies have different version and GPU requirements.

For MediaPipe Pose:

```powershell
python -m pip install ".[mediapipe]"
```

For the tested CPU Sports2D setup:

```powershell
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install ".[sports2d]"
```

For DeepLabCut, install the compatible PyTorch and torchvision builds first, then install the package extra:

```powershell
python -m pip install ".[dlc]"
```

MediaPipe is pinned to 0.10.21 to match the thesis route. DeepLabCut 3.0.1 supports Python 3.10–3.12. The tested Sports2D runtime uses PyTorch 2.5.1 and torchvision 0.20.1 on CPU; for CUDA, select matching wheels with the official [PyTorch selector](https://pytorch.org/get-started/locally/). The DLC bundles were verified with PyTorch 2.5.1+cu121 and torchvision 0.20.1+cu121. See the official [DeepLabCut installation guide](https://deeplabcut.github.io/DeepLabCut/docs/installation.html). Sports2D is pinned to the tested 0.8.20 interface. Install the optional statistics dependency with `python -m pip install ".[spm]"` when SPM1D analysis is needed.

## Run the synthetic example

```powershell
sports2d-dvideow demo --output demo-output --frames 64
```

This creates synthetic two-camera 2D coordinates and calibration, processes the trial, and writes:

- `tracks_2d_processed.csv`
- `kinematics_3d.csv`
- `kinematics_normalized_101.csv`
- `reprojection_diagnostics.csv`
- `run_metadata.json`

The data are generated for demonstration and contain no thesis observations.

## Process a study video

Run MediaPipe separately on each synchronized camera view, with the participant side specified explicitly:

```powershell
sports2d-dvideow pose-mediapipe `
  --video input_cam1.mp4 --camera CAM_1 --side right `
  --roi-config roi.json --output cam1_tracks.csv
```

Repeat for the second camera, then combine both files into a CSV with columns `frame_index,camera,landmark,x_px,y_px,confidence`. Landmark names can be canonical (`hip`, `knee`, `ankle`, `foot`) or common system labels. Calibration JSON uses a `projection_matrices` object containing one 3×4 matrix or 11 DLT coefficients per camera; see [`examples`](examples) for the schema.

Process the 2D coordinates and derive kinematics:

```powershell
sports2d-dvideow run `
  --tracks paired_tracks.csv --calibration calibration.json `
  --side right --source markerless --sampling-hz 240 `
  --window-start-frame 180 --impact-frame 240 --output trial-output
```

For manual DVIDEOW 2D tracks, use `--source manual`; this skips markerless 2D curation while retaining the common 3D processing. Supply the investigator’s camera calibrations and the correct frame range for the pre-impact-to-impact window.

Sports2D can be run with its own thesis-derived TOML or a study-specific TOML:

```powershell
sports2d-dvideow pose-sports2d `
  --video input_cam1.mp4 --config sports2d-config.toml `
  --side right --camera CAM_1 --output sports2d-output
```

The adapter calls the installed Sports2D Python API, retains its native products and writes `input_cam1_tracks_2d.csv` in the common 2D schema. It selects functional landmarks for the declared side and maps the native sample index and initial TRC timestamp to the original video frame index.

## Pose systems and study settings

### Sports2D

The thesis route used YOLOX with the Halpe-26 whole-body pose model, selected the person with the greatest displacement, and tracked with DeepSORT. A manual target selection rerun was used when an initial tracking result was unusable. See [`docs/sports2d.md`](docs/sports2d.md) for the runnable configuration and native plus canonical outputs. Sports2D’s own 2D filtering and this package’s shared markerless 2D processing are distinct stages.

### MediaPipe Pose

MediaPipe Pose 0.10.21 supplies 33 landmarks. The pipeline selects the kicking-side hip, knee, ankle, and Foot Index from the declared side and maps ROI coordinates back to source-image pixels.

### DeepLabCut

The thesis benchmark used one participant-level training/validation split, one exploratory run per configuration, 3,181 labels, and eight bilateral points. Training labels were sampled every four frames from 240 Hz recordings; this is the training-set annotation interval, not the inference or analysis frame rate. The ten frames immediately before impact belong to the strict 3D evaluation subset and are a different sampling decision.

| Model | Selected epoch | Selection-validation RMSE (px) | RMSE at p-cutoff (px) |
| --- | ---: | ---: | ---: |
| ResNet-101 | 25 | 12.55 | 12.25 |
| HRNet-W48 | 35 | 12.49 | 12.26 |
| RTMPose-S | 15 | 13.82 | 13.64 |
| RTMPose-M | 28 | 15.54 | 14.58 |
| RTMPose-X | 4 | 15.62 | 15.83 |

These are single-run results on the study’s selection partition. ResNet-101 was selected for the integrated analysis using the thesis criterion; the benchmark does not establish a general architecture ranking. Downloadable model bundles, checkpoint hashes and setup instructions are listed in [`docs/deeplabcut-benchmark.md`](docs/deeplabcut-benchmark.md).

Example inference after installing DeepLabCut and extracting a model release bundle:

```powershell
sports2d-dvideow pose-dlc `
  --model ResNet-101 --model-bundle models\ResNet-101 `
  --video new_kick.mp4 --camera CAM_1 --device cuda:0 `
  --batch-size 1 --output predictions
```

The DLC adapter verifies both checkpoint hashes, selects the thesis pose and detector snapshots, retains native DeepLabCut outputs and writes `<video>_tracks_2d.csv` in the common schema. Give each run a new or empty output directory so DeepLabCut cannot reuse older predictions. Run each synchronized camera view separately and provide both standardized CSV files to the reconstruction command.

## Biomechanical processing

The 3D coordinate frame follows the thesis: X anterior–posterior, Y mediolateral, and Z vertical. The sagittal joint angles use the X–Z projection, +X anterior, and +Z upward. Knee extension is 0°; flexion is positive and hyperextension negative. Ankle neutral is 0°; dorsiflexion is positive and plantarflexion negative. Speeds are 3D vector magnitudes, and angular velocity is derived using the physical sampling interval before phase normalization.

The signed speed symmetry index is `200 × (D − ND) / (D + ND)`. Knee and ankle angle and angular-velocity differences are signed `D − ND`. The thesis contains conflicting wording about whether the speed index is non-negative; the implementation follows its displayed signed equation and the result interpretation, and records that discrepancy in [`docs/reproducibility-notes.md`](docs/reproducibility-notes.md).

The Python DLT routine accepts calibration matrices supplied by the researcher. Its output is not represented as bitwise or numerical reproduction of DVIDEOW’s legacy implementation; the reconstruction path must be checked against a researcher’s calibration and reference outputs for each study.

## Inputs and exported results

The core accepts synchronized 2D observations in a simple CSV and calibration in JSON. It does not require the original study’s folder names, participant identifiers, videos, or calibration files. Outputs include processed 2D tracks, per-frame 3D positions, vector velocities, speed magnitudes, knee/ankle angles, angular velocities, a 101-point phase export, and per-frame reprojection diagnostics.

Use `sports2d-dvideow bilateral --dominant dominant.csv --non-dominant non_dominant.csv --output bilateral.csv` to compare paired normalized curves. Use `sports2d-dvideow agreement --reference reference.csv --prediction estimate.csv --columns hip_x hip_y hip_z --output agreement.json` for coordinate agreement. See `sports2d-dvideow --help` for the full command list.

## Reuse and fine-tuning

Use the numerical modules directly from `sports2d_dvideow_pipeline` or adapt the CSV schema to another markerless estimator. The five DeepLabCut bundles preserve each selected PyTorch checkpoint, inference configuration, landmark names and available training settings. Fine-tuning requires new annotations and the compatible DeepLabCut/PyTorch environment; the trained checkpoints are starting weights, not a supplied training dataset. The training software versions were not recorded in the source project files; the verified inference environment is listed in the model documentation.

## Documentation and references

- [Processing workflow](docs/pipeline.md)
- [Scientific methods and conventions](docs/methods.md)
- [DeepLabCut model metadata and bundles](docs/deeplabcut-benchmark.md)
- [Sports2D route](docs/sports2d.md)
- [MediaPipe route and ROI schema](configs/mediapipe.roi.example.json)
- [Reproducibility notes](docs/reproducibility-notes.md)
- [Sources and citations](docs/sources.md)

The Python package, documentation, examples and five trained model bundles are MIT-licensed. External pose-estimation software and its dependencies retain their own terms. See [`NOTICE.md`](NOTICE.md) and cite the methods and systems listed in [`CITATION.cff`](CITATION.cff).
