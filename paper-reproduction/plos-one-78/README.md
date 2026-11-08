# Reproducing the PLOS ONE 78-kick analysis

This directory contains the article-specific scripts for rebuilding Figures 2–5 and the repeated-measures Bland–Altman summary from the processed analysis tables published with the study.

## Study corpus

The analysis covers 13 players, 78 kicks (39 per kicking side), 39 bilateral pairs, and 156 camera views. The input tables retain the pseudonymous participant and trial identifiers needed to preserve repeated-measures structure.

## Data

The input CSV files are available in [`data/plos-one-78`](../../data/plos-one-78/README.md). The release covers 13 players, 78 kicks (39 per kicking side), 39 bilateral pairs, and 156 camera views. It includes coded three-dimensional trajectories for manual reference tracking, Sports2D, MediaPipe Pose, and DeepLabCut (ResNet-101), together with the derived analysis tables used in the article and Supporting Information. Source videos, extracted frames, the participant-code key, camera-calibration files, and trained pose-model checkpoints are not distributed.

## Environment

The scripts were tested with Python 3.11.9. From the repository root, create an isolated environment and install the pinned dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r paper-reproduction\plos-one-78\requirements-figures.txt
```

## Rebuild Figures 2–5 and the Bland–Altman table

Run the command below from the repository root:

```powershell
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

The script checks the corpus counts, system labels, bilateral pairing, 101-point curve structure, and complete 78-kick coverage required by the reported summaries. Participant-cluster bootstrap resampling uses a fixed seed. Figure 1 is supplied as a fixed workflow schematic.

Validate the trial coverage and CSV checksums from the repository root with:

```powershell
python paper-reproduction\plos-one-78\validate_public_dataset.py
```

## Scope

These scripts regenerate the spatial-error, sagittal-kinematics, bilateral-error, and repeated-measures Bland–Altman figures (Figures 2–5) and the corresponding spatial-error and Bland–Altman tables. They start from processed analysis tables; they do not rerun pose-estimation inference, video synchronization, camera calibration, or DLT reconstruction. Those operations require source recordings and calibration resources that are not distributed.
