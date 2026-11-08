# Reproducing the PLOS ONE 78-kick analysis

This directory contains the article-specific scripts for rebuilding Figures 2–5 and the repeated-measures Bland–Altman summary from approved processed analysis tables.

## Study corpus

The analysis covers 13 players, 78 kicks (39 per kicking side), 39 bilateral pairs, and 156 camera views. The input tables retain the pseudonymous participant and trial identifiers needed to preserve repeated-measures structure.

## Data access

Participant-level trajectories and derived analysis tables are not hosted publicly. Movement profiles from this small professional cohort may permit indirect identification. Requests should be made through the PUCP Research Ethics Committee for Life Sciences and Technologies (CEI-CVyT) and the corresponding author, using the institutional route in the article's Data Availability Statement. Access requires institutional ethics and data-custody review. Source videos, calibration files, and trained pose-model checkpoints are not distributed.

## Environment

The scripts were tested with Python 3.11.9. From the repository root, create an isolated environment and install the pinned dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r paper-reproduction\plos-one-78\requirements-figures.txt
```

## Rebuild Figures 2–5 and the Bland–Altman table

After access is approved, place the processed inputs in a local `data_release_candidate\derived` directory and run from the repository root:

```powershell
python paper-reproduction\plos-one-78\rebuild_all78_article_figures.py `
  --spatial-errors data_release_candidate\derived\spatial_error_per_trial_78.csv `
  --angles data_release_candidate\derived\curvas_angulos_corpus78.csv `
  --angular-velocities data_release_candidate\derived\curvas_velocidad_angular_corpus78.csv `
  --bilateral-summary data_release_candidate\derived\bilateral_outcomes_summary_78.csv `
  --bland-altman data_release_candidate\derived\bland_altman_variables_especificas.csv `
  --out outputs\figures `
  --tables-out outputs\tables `
  --bootstrap 5000
```

The script checks the corpus counts, system labels, bilateral pairing, 101-point curve structure, and complete 78-kick coverage required by the reported summaries. Participant-cluster bootstrap resampling uses a fixed seed. Figure 1 is supplied as a fixed workflow schematic.

## Scope

These scripts regenerate the spatial-error, sagittal-kinematics, bilateral-error, and repeated-measures Bland–Altman figures (Figures 2–5) and the corresponding spatial-error and Bland–Altman tables. They start from processed analysis tables; they do not rerun pose-estimation inference, video synchronization, camera calibration, or DLT reconstruction. Those operations require source recordings and calibration resources that are not distributed.
