# Data dictionary

## Study coverage

The release contains coded tracking records from 13 professional futsal players: 78 kicks, 39 from each kicking side, 39 dominant–non-dominant pairs, and 156 camera views. Each participant contributed three kicks with each side. The four three-dimensional tracking routes are manual reference tracking in DVIDEOW, Sports2D, MediaPipe Pose, and DeepLabCut (ResNet-101).

## Core data files

### `data/tracking_3d_long.csv`

Long-form frame-level three-dimensional coordinates in metres for all 78 kicks and all four tracking routes.

| Field | Description |
| --- | --- |
| `participant_id` | Coded participant identifier. |
| `trial_id` | Coded participant, kicking side, and kick identifier. |
| `bilateral_pair_id` | Identifier linking a participant's dominant- and non-dominant-side kicks. |
| `kick_number` | Kick index within kicking side (1–3). |
| `kicking_side` | Right- or left-kicking limb. |
| `limb_role` | Dominant or non-dominant limb. |
| `system` | Manual reference, Sports2D, MediaPipe Pose, or DeepLabCut (ResNet-101). |
| `landmark` | Hip, knee, ankle, or foot. |
| `frame` | Source-video frame index; acquisition rate was 240 Hz. |
| `x_m`, `y_m`, `z_m` | Reconstructed three-dimensional coordinates in metres. |

### `data/trial_manifest.csv`

One coded record per kick, linking participant, kicking side, bilateral pair, and the number of finite reconstructed landmark-frame records available for each route.

## Derived tables

Files in `derived/` provide the article analysis inputs and reported results:

- Spatial reconstruction errors by kick, landmark, and system, plus summary tables.
- Normalized sagittal position, angle, linear velocity, and angular-velocity curves.
- Temporal shape and peak metrics.
- Bilateral outcomes for 39 dominant–non-dominant pairs.
- Statistical parametric mapping results and multiplicity-adjustment records.
- Trial-level inputs and summaries for repeated-measures Bland–Altman analyses.
- The five-configuration DeepLabCut benchmark table reported in the Supporting Information.

Column names, units, and analysis conventions are retained in each CSV. Missing coordinates remain missing; they have not been imputed.

## Scope

The files are processed tracking and reconstruction outputs used for the reported analyses. The release excludes source videos, extracted frames, the participant-code key, camera-calibration files, training images, and trained pose-model checkpoints. The supplied scripts rebuild data-derived figures and tables from these CSVs; they do not rerun pose inference or reconstruct three-dimensional coordinates from source recordings.
