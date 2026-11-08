"""Rebuild data-derived manuscript figures from the validated 78-kick tables.

The input CSVs are supplied separately from the Overleaf project. This script
recreates Figures 2--5 without local thesis paths. The workflow schematic
(Figure 1) is supplied as a fixed manuscript asset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import article_figure_helpers as helpers
from recompute_bland_altman import (
    build_figure as build_bland_altman_figure,
    prepare_data as prepare_bland_altman_data,
    summarize as summarize_bland_altman,
    write_tex_table as write_bland_altman_table,
)


PROJECT = Path(__file__).resolve().parent
SYSTEMS = ("Manual reference", "Sports2D", "MediaPipe Pose", "DeepLabCut (ResNet-101)")
SYSTEM_LABELS = {
    "Manual reference": "Manual reference tracking",
    "Sports2D": "Sports2D",
    "MediaPipe Pose": "MediaPipe Pose",
    "DeepLabCut (ResNet-101)": "DeepLabCut (ResNet-101)",
}
SYSTEM_COLORS = {
    "Manual reference": "#000080",
    "Sports2D": "#14823B",
    "MediaPipe Pose": "#D81B9C",
    "DeepLabCut (ResNet-101)": "#F07800",
}
SYSTEM_STYLES = {"Manual reference": "-", "Sports2D": "--", "MediaPipe Pose": "-.", "DeepLabCut (ResNet-101)": ":"}
MARKERS = ("Knee", "Ankle")
SIDE_LABELS = {"right": "right-kicking limb", "left": "left-kicking limb"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_curve_table(path: Path, expected_markers: set[str]) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"side", "system", "marker", "phase_pct", "median", "ci_lo", "ci_hi", "n"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
    data["system"] = data["system"].replace({"Oro": "Manual reference", "Manual reference tracking": "Manual reference", "MediaPipe": "MediaPipe Pose", "DeepLabCut-ResNet-101": "DeepLabCut (ResNet-101)", "DeepLabCut": "DeepLabCut (ResNet-101)"})
    data = data[data["marker"].isin(expected_markers)].copy()
    if set(data["side"].dropna().unique()) != set(SIDE_LABELS):
        raise ValueError(f"{path.name} must contain the right- and left-kicking limbs")
    if set(data["system"].dropna().unique()) != set(SYSTEMS):
        raise ValueError(f"{path.name} must contain the manual reference and all three markerless systems")
    for (side, system, marker), group in data.groupby(["side", "system", "marker"]):
        if len(group) != 101 or group["phase_pct"].nunique() != 101:
            raise ValueError(f"Expected 101 normalized phases for {side}/{system}/{marker}")
        available_n = pd.to_numeric(group["n"], errors="coerce").to_numpy()
        if not np.all(np.isfinite(available_n)) or available_n.max() != 39 or available_n.min() < 1:
            raise ValueError(f"Expected each curve to draw from the 39-kick side corpus for {side}/{system}/{marker}")
    return data


def figure_sagittal_kinematics(angles_csv: Path, angular_velocity_csv: Path, out_dir: Path) -> list[Path]:
    angles = validate_curve_table(angles_csv, {"Knee", "Ankle"})
    angular_velocity = validate_curve_table(angular_velocity_csv, {"Knee", "Ankle"})
    rows = [
        (angles, "right", "Sagittal angle"),
        (angles, "left", "Sagittal angle"),
        (angular_velocity, "right", "Angular velocity"),
        (angular_velocity, "left", "Angular velocity"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(7.25, 8.4), sharex=True)
    fig.subplots_adjust(left=0.15, right=0.99, top=0.94, bottom=0.07, hspace=0.34, wspace=0.24)
    for row_index, (frame, side, family) in enumerate(rows):
        for col_index, marker in enumerate(MARKERS):
            ax = axes[row_index, col_index]
            for system in SYSTEMS:
                curve = frame[(frame["side"] == side) & (frame["system"] == system) & (frame["marker"] == marker)].sort_values("phase_pct")
                x = curve["phase_pct"].to_numpy(dtype=float)
                median = curve["median"].to_numpy(dtype=float)
                low = curve["ci_lo"].to_numpy(dtype=float)
                high = curve["ci_hi"].to_numpy(dtype=float)
                ax.plot(x, median, color=SYSTEM_COLORS[system], linestyle=SYSTEM_STYLES[system], linewidth=1.15, label=SYSTEM_LABELS[system])
                ax.fill_between(x, low, high, color=SYSTEM_COLORS[system], alpha=0.12, linewidth=0)
            ax.text(0.015, 0.96, f"({chr(97 + row_index * 2 + col_index)})", transform=ax.transAxes,
                    ha="left", va="top", fontsize=7.3, fontweight="bold")
            if family == "Sagittal angle":
                ylabel = "Knee flexion (°)" if marker == "Knee" else "Ankle angle (°)"
            else:
                ylabel = r"Knee angular velocity (° s$^{-1}$)" if marker == "Knee" else r"Ankle angular velocity (° s$^{-1}$)"
            ax.set_ylabel(ylabel, fontsize=7.1)
            ax.set_xlim(0, 100)
            ax.set_xticks(np.arange(0, 101, 20))
            if row_index in (1, 3):
                ax.set_xlabel("Normalized pre-impact-to-impact phase (%)", fontsize=7.1)
            helpers.set_clean_axes(ax)
            ax.tick_params(labelsize=6.6, pad=1.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.56, 0.995), ncol=4,
               frameon=False, handletextpad=0.35, columnspacing=1.0, fontsize=7.0)
    return helpers.save_figure(fig, out_dir, "figure3_sagittal_kinematics")


def figure_bilateral_errors(summary_csv: Path, out_dir: Path) -> list[Path]:
    summary = pd.read_csv(summary_csv)
    required = {"outcome", "system", "n_kicks", "n_pairs", "n_participants", "MAE_median_participant", "RMSE_median_participant"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"{summary_csv.name} is missing columns: {', '.join(sorted(missing))}")
    if len(summary) != 24 or not (summary["n_kicks"] == 78).all() or not (summary["n_pairs"] == 39).all() or not (summary["n_participants"] == 13).all():
        raise ValueError("Bilateral summary must contain 24 outcomes for 78 kicks, 39 pairs, and 13 participants")
    names = {
        "SI Hip speed": "Speed symmetry index: hip",
        "SI Knee speed": "Speed symmetry index: knee",
        "SI Ankle speed": "Speed symmetry index: ankle",
        "SI Foot speed": "Speed symmetry index: foot",
        "Angle difference Knee": "Knee angle difference",
        "Angle difference Ankle": "Ankle angle difference",
        "Angular velocity difference Knee": "Knee angular-velocity difference",
        "Angular velocity difference Ankle": "Ankle angular-velocity difference",
    }
    rows = [["Outcome", "System", "MAE", "RMSE", "CCC"]]
    for outcome in names:
        for system in ("Sports2D", "MediaPipe Pose", "DeepLabCut (ResNet-101)"):
            row = summary[(summary["outcome"] == outcome) & (summary["system"] == system)]
            if len(row) != 1:
                raise ValueError(f"Missing unique bilateral row: {outcome}/{system}")
            record = row.iloc[0]
            label = {"Sports2D": "Sports2D", "MediaPipe Pose": "MediaPipe Pose", "DeepLabCut (ResNet-101)": "DeepLabCut (ResNet-101)"}[system]
            rows.append([names[outcome], label, f"{record.MAE_median_participant:.2f}", f"{record.RMSE_median_participant:.2f}", f"{record.CCC_group_median_curves:.3f}"])
    return helpers.figure_bilateral_errors([rows], out_dir)


def figure_bland_altman(source_csv: Path, out_dir: Path, tables_out: Path, replicates: int) -> list[Path]:
    data = prepare_bland_altman_data(source_csv, angles_already_signed=True)
    if len(data) != 936 or data["trial_id"].nunique() != 78 or data["participant"].nunique() != 13:
        raise ValueError("Bland–Altman source must contain 78 kicks, 13 participants, and 12 outcomes per kick")
    counts = data.groupby(["system", "feature"])["trial_id"].nunique()
    if not (counts == 78).all():
        raise ValueError("Every Bland–Altman system/outcome must use the same 78 kicks")
    summary = summarize_bland_altman(data, replicates=replicates)
    if not (summary["n_observations"] == 78).all() or not (summary["n_participants"] == 13).all():
        raise ValueError("Repeated-measures Bland–Altman summaries do not cover the full corpus")
    tables_out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables_out / "bland_altman_medidas_repetidas.csv", index=False, float_format="%.8f")
    write_bland_altman_table(summary, tables_out / "bland_altman_table_s4.tex")
    stem = out_dir / "figure5_bland_altman"
    build_bland_altman_figure(data, summary, stem)
    return [stem.with_suffix(".pdf"), stem.with_suffix(".png"), stem.with_suffix(".tif")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial-errors", type=Path, required=True, help="78-kick trial-level spatial-error CSV")
    parser.add_argument("--angles", type=Path, required=True, help="101-phase bilateral sagittal-angle curves for 78 kicks")
    parser.add_argument("--angular-velocities", type=Path, required=True, help="101-phase bilateral angular-velocity curves for 78 kicks")
    parser.add_argument("--bilateral-summary", type=Path, required=True, help="Bilateral-outcome summary for 39 pairs")
    parser.add_argument("--bland-altman", type=Path, required=True, help="Trial-level Bland–Altman data for 78 kicks")
    parser.add_argument("--out", type=Path, default=PROJECT / "figures" / "main")
    parser.add_argument("--tables-out", type=Path, default=PROJECT / "tables")
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    sources = [args.spatial_errors, args.angles, args.angular_velocities, args.bilateral_summary, args.bland_altman]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = []
    generated, _ = helpers.figure_spatial_error(args.spatial_errors, args.out, args.tables_out)
    outputs.extend(generated)
    outputs.extend(figure_sagittal_kinematics(args.angles, args.angular_velocities, args.out))
    outputs.extend(figure_bilateral_errors(args.bilateral_summary, args.out))
    outputs.extend(figure_bland_altman(args.bland_altman, args.out, args.tables_out, args.bootstrap))
    helpers.captions_file(PROJECT / "figure_captions.md")
    fixed_fig1 = PROJECT / "assets" / "Fig1.tif"
    manifest = {
        "analysis_corpus": {"participants": 13, "kicks": 78, "kicks_per_side": 39, "bilateral_pairs": 39},
        "figure_generation": "Figures 2–5 are generated from the 78-kick spatial, sagittal-curve, bilateral, and trial-level agreement tables. Knee and ankle angles in the Bland–Altman source use the signed study convention.",
        "fixed_workflow_schematic": ({"file": "paper-reproduction/plos-one-78/assets/Fig1.tif", "sha256": sha256(fixed_fig1)} if fixed_fig1.is_file() else None),
        "inputs": [{"file": p.name, "sha256": sha256(p)} for p in sources],
        "main_figures": [{"file": p.name, "sha256": sha256(p)} for p in sorted(args.out.glob("figure*.pdf"))],
        "supplementary_figures": [p.name for p in sorted((PROJECT / "figures" / "supplementary").glob("supplementary_figure*.pdf"))],
    }
    (PROJECT / "figure_provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Generated {len(outputs)} files for Figures 2--5 from validated 78-kick analysis tables.")
    print(f"Kicks=78; participants=13; kicks per side=39; bilateral pairs=39; bootstrap={args.bootstrap}.")
    for item in outputs:
        print(f"{item} ({item.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
