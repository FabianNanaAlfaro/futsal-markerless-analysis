# Futsal Markerless Analysis

[![CI](https://github.com/FabianNanaAlfaro/futsal-markerless-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/FabianNanaAlfaro/futsal-markerless-analysis/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/code-MIT-2ea44f)](LICENSE)

<p align="center">
  <img src="https://raw.githubusercontent.com/FabianNanaAlfaro/futsal-markerless-analysis/main/site/assets/brand/tehealp-logo.png#gh-light-mode-only" width="280" alt="TeHealP logo, light theme">
  <img src="https://raw.githubusercontent.com/FabianNanaAlfaro/futsal-markerless-analysis/main/site/assets/brand/tehealp-logo-white.svg#gh-dark-mode-only" width="280" alt="TeHealP logo, dark theme">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/FabianNanaAlfaro/futsal-markerless-analysis/main/site/assets/brand/pucp-logo-color.png#gh-light-mode-only" width="150" alt="Pontificia Universidad Católica del Perú logo, light theme">
  <img src="https://raw.githubusercontent.com/FabianNanaAlfaro/futsal-markerless-analysis/main/site/assets/brand/pucp-logo-white.svg#gh-dark-mode-only" width="150" alt="Pontificia Universidad Católica del Perú logo, dark theme">
</p>

A Python package and command-line tool for lower-limb pose processing and futsal-kicking kinematics. It brings manual DVIDEOW tracking and three markerless routes—Sports2D, MediaPipe Pose, and DeepLabCut—into a shared analysis workflow.

## Study scope and acquisition

The study compares the outcome-specific agreement and limitations of Sports2D, MediaPipe Pose, and DeepLabCut relative to manual reference tracking during three-dimensional analysis of the rapid futsal instep kick. The source corpus includes 13 professional players, 78 kicks (39 per side), and 156 analyzed camera views from a four-camera setup. Recordings were acquired at 240 Hz and 848 × 480 pixels. This low spatial resolution makes landmark localization demanding during a rapid movement, particularly for distal points such as the foot.

## Capabilities

- Installable Python package with reusable functions and the `futsal-markerless` CLI.
- MediaPipe Pose extraction for a user-selected side, optional camera-specific ROI, and explicit mapping of cropped coordinates back to the original pixels.
- Sports2D Python API adapter using a user-supplied TOML configuration.
- DeepLabCut inference with a compatible local model bundle for any of the five study configurations.
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
git clone https://github.com/FabianNanaAlfaro/futsal-markerless-analysis.git
cd futsal-markerless-analysis
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

MediaPipe is pinned to 0.10.21 to match the thesis route. DeepLabCut 3.0.1 supports Python 3.10–3.12. The tested Sports2D runtime uses PyTorch 2.5.1 and torchvision 0.20.1 on CPU; for CUDA, select matching wheels with the official [PyTorch selector](https://pytorch.org/get-started/locally/). The verified DeepLabCut inference environment used PyTorch 2.5.1+cu121 and torchvision 0.20.1+cu121. See the official [DeepLabCut installation guide](https://deeplabcut.github.io/DeepLabCut/docs/installation.html). Sports2D is pinned to the tested 0.8.20 interface. Install the optional statistics dependency with `python -m pip install ".[spm]"` when SPM1D analysis is needed.

## Run the synthetic example

```powershell
futsal-markerless demo --output demo-output --frames 64
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
futsal-markerless pose-mediapipe `
  --video input_cam1.mp4 --camera CAM_1 --side right `
  --roi-config roi.json --output cam1_tracks.csv
```

Repeat for the second camera, then combine both files into a CSV with columns `frame_index,camera,landmark,x_px,y_px,confidence`. Landmark names can be canonical (`hip`, `knee`, `ankle`, `foot`) or common system labels. Calibration JSON uses a `projection_matrices` object containing one 3×4 matrix or 11 DLT coefficients per camera; see [`examples`](examples) for the schema.

Process the 2D coordinates and derive kinematics:

```powershell
futsal-markerless run `
  --tracks paired_tracks.csv --calibration calibration.json `
  --side right --source markerless --sampling-hz 240 `
  --window-start-frame 180 --impact-frame 240 --output trial-output
```

For manual DVIDEOW 2D tracks, use `--source manual`; this skips markerless 2D curation while retaining the common 3D processing. Supply the investigator’s camera calibrations and the correct frame range for the pre-impact-to-impact window.

Sports2D can be run with its own thesis-derived TOML or a study-specific TOML:

```powershell
futsal-markerless pose-sports2d `
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

An exploratory model-selection step evaluated ResNet-101, HRNet-W48, RTMPose-S, RTMPose-M, and RTMPose-X using one participant-level training/validation split and one run per configuration. The study used 3,181 annotations across eight bilateral landmarks, sampled every fourth frame from the 240-Hz recordings. ResNet-101 was selected for the integrated analysis using validation RMSE at the 0.6 confidence cutoff. Full model-selection metrics and metadata are available in [`docs/deeplabcut-benchmark.md`](docs/deeplabcut-benchmark.md). The training-frame interval describes annotation sampling, not inference cadence. Trained checkpoints are not publicly distributed.

Example inference with a compatible model bundle already available on the local computer:

```powershell
futsal-markerless pose-dlc `
  --model ResNet-101 --model-bundle .\study-models\ResNet-101 `
  --video new_kick.mp4 --camera CAM_1 --device cuda:0 `
  --batch-size 1 --output predictions
```

The DLC adapter verifies the pose checkpoint hash against the thesis registry and supplied bundle metadata, and the detector hash against the supplied bundle metadata. It selects the pose and detector snapshots, retains native DeepLabCut outputs and writes `<video>_tracks_2d.csv` in the common schema. Give each run a new or empty output directory so DeepLabCut cannot reuse older predictions. Run each synchronized camera view separately and provide both standardized CSV files to the reconstruction command.

## Biomechanical processing

The 3D coordinate frame follows the thesis: X anterior–posterior, Y mediolateral, and Z vertical. The sagittal joint angles use the X–Z projection, +X anterior, and +Z upward. Knee extension is 0°; flexion is positive and hyperextension negative. Ankle neutral is 0°; dorsiflexion is positive and plantarflexion negative. Speeds are 3D vector magnitudes, and angular velocity is derived using the physical sampling interval before phase normalization.

The signed speed symmetry index is `200 × (D − ND) / (D + ND)`, where D and ND are non-negative speed magnitudes. Positive values indicate greater speed in the dominant leg, negative values indicate greater speed in the non-dominant leg, and zero indicates equal magnitudes. Knee and ankle angle and angular-velocity differences are signed `D − ND`.

The Python DLT routine accepts calibration matrices supplied by the researcher. Its output is not represented as bitwise or numerical reproduction of DVIDEOW’s legacy implementation; the reconstruction path must be checked against a researcher’s calibration and reference outputs for each study.

## Inputs and exported results

The core accepts synchronized 2D observations in a simple CSV and calibration in JSON. It does not require the original study’s folder names, participant identifiers, videos, or calibration files. Outputs include processed 2D tracks, per-frame 3D positions, vector velocities, speed magnitudes, knee/ankle angles, angular velocities, a 101-point phase export, and per-frame reprojection diagnostics.

Use `futsal-markerless bilateral --dominant dominant.csv --non-dominant non_dominant.csv --output bilateral.csv` to compare paired normalized curves. Use `futsal-markerless agreement --reference reference.csv --prediction estimate.csv --columns hip_x hip_y hip_z --output agreement.json` for coordinate agreement. See `futsal-markerless --help` for the full command list.

## Reuse and fine-tuning

Use the numerical modules directly from `futsal_markerless_analysis` or adapt the CSV schema to another markerless estimator. The DeepLabCut registry preserves the five benchmark configurations, checkpoint identifiers and hashes, inference settings, landmark names and reported training metadata. Checkpoint weights and the annotated training dataset are not distributed. Inference and fine-tuning require compatible local model files, new annotations as needed, and the corresponding DeepLabCut/PyTorch environment. The original training software versions were not recorded in the source project files; the verified inference environment is listed in the model documentation.

## Authors, laboratory, and citation

**Developed by TeHealP@PUCP.** Fabian Alcides Ñaña Alfaro and Luiz Henrique Palucci Vieira are TeHealP@PUCP members linked to this project. The six software authors below are listed in citation order.

| Author | Affiliation(s) | ORCID |
| --- | --- | --- |
| Fabian Alcides Ñaña Alfaro | TeHealP@PUCP; Departamento Académico de Ingeniería, Pontificia Universidad Católica del Perú, Lima, Perú | [0009-0000-7881-1223](https://orcid.org/0009-0000-7881-1223) |
| Paulo Roberto Pereira Santiago | School of Physical Education and Sport of Ribeirão Preto, University of São Paulo, Ribeirão Preto, Brazil | [0000-0002-9460-8847](https://orcid.org/0000-0002-9460-8847) |
| Bruno Luiz de Souza Bedo | School of Physical Education and Sport, University of São Paulo, São Paulo, Brazil | [0000-0003-3821-2327](https://orcid.org/0000-0003-3821-2327) |
| Rodrigo Leal de Queiroz Thomaz de Aquino | Federal University of Espírito Santo, Vitória, Brazil; Brazilian National Football Confederation, Rio de Janeiro, Brazil | [0000-0002-4885-7316](https://orcid.org/0000-0002-4885-7316) |
| Luiz Henrique Palucci Vieira | TeHealP@PUCP; Departamento Académico de Ingeniería, Pontificia Universidad Católica del Perú, Lima, Perú | [0000-0001-6981-756X](https://orcid.org/0000-0001-6981-756X) |
| César Armando Beltrán Castañón | Grupo de Inteligencia Artificial PUCP (IA-PUCP); Departamento Académico de Ingeniería, Pontificia Universidad Católica del Perú, Lima, Perú | [0000-0002-0173-4140](https://orcid.org/0000-0002-0173-4140) |

Use the software citation in [`CITATION.md`](CITATION.md) or [`CITATION.cff`](CITATION.cff). The related publication, *Preliminary analysis of markerless three-dimensional trajectory reconstruction during the futsal instep kicking using Sports2D*, retains its original five authors: Fabian A. Nana, Paulo R. P. Santiago, Bruno L. S. Bedo, Rodrigo Aquino, and Luiz H. Palucci Vieira ([DOI](https://doi.org/10.1080/14763141.2026.2680520)). It is a separate citation from the six-author software record.

The package code is MIT licensed; its copyright notice is in [`LICENSE`](LICENSE). TeHealP@PUCP is identified as the developing laboratory, separately from the software authors and copyright statement. See [`NOTICE.md`](NOTICE.md) for third-party software and institutional-mark terms.

## Reproducing the PLOS ONE analysis

The processed three-dimensional trajectories and analysis tables for the 78-kick study are available in [`data/plos-one-78`](data/plos-one-78/README.md). The release includes manual reference tracking and the Sports2D, MediaPipe Pose, and DeepLabCut (ResNet-101) outputs. It contains no source videos, extracted frames, participant-code key, camera-calibration files, or trained pose-model checkpoints. Use the scripts in [`paper-reproduction/plos-one-78`](paper-reproduction/plos-one-78/README.md) to rebuild Figures 2–5, the repeated-measures Bland–Altman table, and the S1/S2 supplementary tables for temporal coverage and MediaPipe knee angular velocity.
## Documentation and references

- [Processing workflow](docs/pipeline.md)
- [Scientific methods and conventions](docs/methods.md)
- [DeepLabCut model metadata and bundles](docs/deeplabcut-benchmark.md)
- [Sports2D route](docs/sports2d.md)
- [MediaPipe route and ROI schema](configs/mediapipe.roi.example.json)
- [Reproducibility notes](docs/reproducibility-notes.md)
- [Sources and citations](docs/sources.md)

The Python package, documentation and examples are MIT-licensed. DeepLabCut weights are not distributed by this repository. External pose-estimation software, model artifacts and institutional marks retain their own terms. See [`NOTICE.md`](NOTICE.md) and cite the software and related work listed in [`CITATION.cff`](CITATION.cff).
