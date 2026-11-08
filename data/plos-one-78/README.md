# PLOS ONE futsal markerless kinematics dataset

Version 1.0.0 · 2026

This release provides the processed tracking trajectories and analysis tables used for the PLOS ONE study of markerless three-dimensional kinematics during futsal instep kicking.

## Study coverage

The dataset covers 13 professional futsal players, 78 kicks (39 from each kicking side), 39 dominant–non-dominant pairs, and 156 camera views. It includes frame-level three-dimensional coordinates for manual reference tracking in DVIDEOW, Sports2D, MediaPipe Pose, and DeepLabCut (ResNet-101). Participant and trial identifiers are coded.

## Contents

- `data/tracking_3d_long.csv` — frame-level three-dimensional coordinates in metres for the four tracking routes.
- `data/trial_manifest.csv` — one coded record per kick, linking kicking side and bilateral pair and summarizing available reconstructed landmark-frame records.
- `derived/` — trial-level spatial errors, kinematic curves, temporal metrics, bilateral outcomes, SPM results, and repeated-measures Bland–Altman inputs and summaries used in the article and Supporting Information.
- `DATA_DICTIONARY.md` — variable definitions and file-level descriptions.
- `LICENSE.md` — dataset reuse terms.
- `SHA256SUMS.txt` — checksums for the distributed CSV files.

Missing coordinates remain missing and have not been imputed. The release does not include source videos, extracted frames, the participant-code key, camera-calibration files, training images, or trained pose-model checkpoints.

## Reproduce figures and tables

From the repository root, install the pinned dependencies and run the article-specific rebuild script:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r paper-reproduction\plos-one-78\requirements-figures.txt
python paper-reproduction\plos-one-78\rebuild_all78_article_figures.py `
  --spatial-errors data\plos-one-78\derived\spatial_error_per_trial_78.csv `
  --angles data\plos-one-78\derived\curvas_angulos_corpus78.csv `
  --angular-velocities data\plos-one-78\derived\curvas_velocidad_angular_corpus78.csv `
  --bilateral-summary data\plos-one-78\derived\bilateral_outcomes_summary_78.csv `
  --bland-altman data\plos-one-78\derived\bland_altman_variables_especificas.csv `
  --out outputs\figures `
  --tables-out outputs\tables `
  --bootstrap 5000
```

The script regenerates Figures 2–5 and the spatial-error and repeated-measures Bland–Altman tables. Figure 1 is a fixed workflow schematic in `paper-reproduction/plos-one-78/assets/`. The input tables are processed outputs; this release does not rerun pose-estimation inference, video synchronization, camera calibration, or DLT reconstruction from recordings.

Validate the trial coverage and CSV checksums from the repository root with:

```powershell
python paper-reproduction\plos-one-78\validate_public_dataset.py
```

## Citation

Please cite the dataset using the citation in the manuscript and include the version and immutable GitHub commit URL associated with the article's Data Availability statement. No DOI is assigned to this release.
