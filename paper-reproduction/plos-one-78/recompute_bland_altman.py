"""Recompute repeated-measures Bland–Altman summaries and Figure 5.

The supplied trial-level CSV has one row per participant/kick/system/outcome
and columns: trial_id, participant, system, feature, gold, method. No videos or
pose-estimation checkpoints are required to reproduce this analysis.

Knee and ankle inputs are included segment angles; this script applies the
study's signed conventions (180° − included knee angle; 90° − included ankle
angle) before calculating method-minus-reference differences.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


SYSTEMS = ("Sports2D", "MediaPipe Pose", "DeepLabCut (ResNet-101)")
FEATURES = (
    "peak_foot_speed",
    "knee_angle_impact",
    "ankle_angle_impact",
    "peak_abs_knee_angvel",
)
LABELS = {
    "peak_foot_speed": ("Foot speed (m s$^{-1}$)", "Difference (m s$^{-1}$)"),
    "knee_angle_impact": ("Knee flexion angle (°)", "Difference (°)"),
    "ankle_angle_impact": ("Ankle angle (°)", "Difference (°)"),
    "peak_abs_knee_angvel": ("Peak absolute knee angular velocity (° s$^{-1}$)", "Difference (° s$^{-1}$)"),
}
COLORS = {"Sports2D": "#245A8D", "MediaPipe Pose": "#A94A76", "DeepLabCut (ResNet-101)": "#477A62"}
MARKERS = {"Sports2D": "o", "MediaPipe Pose": "s", "DeepLabCut (ResNet-101)": "^"}
SEED = 20261105
BOOTSTRAP_REPLICATES = 5000


def prepare_data(path: Path, angles_already_signed: bool = False) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"participant", "system", "feature", "gold", "method"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {', '.join(sorted(missing))}")
    if "trial_id" not in data.columns:
        if "record_id" in data.columns:
            data["trial_id"] = data["record_id"].astype(str)
        elif {"side", "trial"}.issubset(data.columns):
            data["trial_id"] = data["participant"].astype(str) + "_" + data["side"].astype(str).str[0].str.upper() + "_" + data["trial"].astype(str)
        else:
            raise ValueError("Input requires trial_id or record_id (or side and trial columns)")
    aliases = {"MediaPipe": "MediaPipe Pose", "DeepLabCut": "DeepLabCut (ResNet-101)", "DeepLabCut-ResNet-101": "DeepLabCut (ResNet-101)"}
    data["system"] = data["system"].replace(aliases)
    data = data.loc[data["feature"].isin(FEATURES) & data["system"].isin(SYSTEMS)].copy()
    data["reference"] = pd.to_numeric(data["gold"], errors="coerce")
    data["estimate"] = pd.to_numeric(data["method"], errors="coerce")

    if not angles_already_signed:
        knee = data["feature"].eq("knee_angle_impact")
        ankle = data["feature"].eq("ankle_angle_impact")
        data.loc[knee, ["reference", "estimate"]] = 180.0 - data.loc[knee, ["reference", "estimate"]]
        data.loc[ankle, ["reference", "estimate"]] = 90.0 - data.loc[ankle, ["reference", "estimate"]]
    data["mean"] = (data["reference"] + data["estimate"]) / 2.0
    data["difference"] = data["estimate"] - data["reference"]
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["reference", "estimate", "mean", "difference"])

    duplicates = data.duplicated(["trial_id", "system", "feature"], keep=False)
    if duplicates.any():
        raise ValueError("Input contains duplicate participant/kick/system/outcome rows")
    return data


def repeated_measures_limits(frame: pd.DataFrame, group_column: str = "participant") -> dict[str, float]:
    """One-way ANOVA variance components for repeated differences per subject.

    The target is the difference for a single future kick. Its variance is the
    sum of within-participant and between-participant difference variances.
    The unequal-replication divisor follows Bland and Altman (2007).
    """
    groups = [g["difference"].to_numpy(dtype=float) for _, g in frame.groupby(group_column, sort=False)]
    j = len(groups)
    n = sum(len(g) for g in groups)
    if j < 2 or n <= j:
        raise ValueError("At least two participants and within-participant replication are required")
    all_values = np.concatenate(groups)
    bias = float(np.mean(all_values))
    residual_ss = sum(float(np.sum((g - np.mean(g)) ** 2)) for g in groups)
    within_ms = residual_ss / (n - j)
    between_ss = sum(len(g) * (float(np.mean(g)) - bias) ** 2 for g in groups)
    between_ms = between_ss / (j - 1)
    n0 = (n - sum(len(g) ** 2 for g in groups) / n) / (j - 1)
    if n0 <= 0:
        raise ValueError("Unequal-replication variance divisor is not positive")
    between_variance = max(0.0, (between_ms - within_ms) / n0)
    total_sd = math.sqrt(within_ms + between_variance)
    return {
        "n_observations": int(n),
        "n_participants": int(j),
        "bias": bias,
        "within_sd": math.sqrt(max(0.0, within_ms)),
        "between_sd": math.sqrt(between_variance),
        "total_sd": total_sd,
        "loa_low": bias - 1.96 * total_sd,
        "loa_high": bias + 1.96 * total_sd,
    }


def cluster_bootstrap(frame: pd.DataFrame, seed: int, replicates: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    grouped = [g["difference"].to_numpy(dtype=float) for _, g in frame.groupby("participant", sort=True)]
    j = len(grouped)
    if j < 2:
        raise ValueError("Participant-cluster bootstrap requires at least two participants")
    counts = np.asarray([len(g) for g in grouped], dtype=float)
    sums = np.asarray([np.sum(g) for g in grouped], dtype=float)
    sum_squares = np.asarray([np.sum(g * g) for g in grouped], dtype=float)
    picks = rng.integers(0, j, size=(replicates, j))
    draw_n = counts[picks]
    draw_sum = sums[picks]
    draw_squares = sum_squares[picks]
    total_n = draw_n.sum(axis=1)
    grand = draw_sum.sum(axis=1) / total_n
    residual_ss = (draw_squares - draw_sum**2 / draw_n).sum(axis=1)
    within_ms = residual_ss / (total_n - j)
    means = draw_sum / draw_n
    between_ss = (draw_n * (means - grand[:, None])**2).sum(axis=1)
    between_ms = between_ss / (j - 1)
    n0 = (total_n - (draw_n**2).sum(axis=1) / total_n) / (j - 1)
    between_variance = np.maximum(0.0, (between_ms - within_ms) / n0)
    total_sd = np.sqrt(within_ms + between_variance)
    biases = grand
    low = biases - 1.96 * total_sd
    high = biases + 1.96 * total_sd
    result: dict[str, float] = {}
    for key, values in (("bias", biases), ("loa_low", low), ("loa_high", high)):
        result[key + "_ci_low"] = float(np.quantile(values, 0.025))
        result[key + "_ci_high"] = float(np.quantile(values, 0.975))
    result["bootstrap_replicates"] = int(replicates)
    return result


def summarize(data: pd.DataFrame, replicates: int = BOOTSTRAP_REPLICATES) -> pd.DataFrame:
    rows = []
    for system_index, system in enumerate(SYSTEMS):
        for feature_index, feature in enumerate(FEATURES):
            frame = data[(data["system"] == system) & (data["feature"] == feature)].copy()
            if frame.empty:
                continue
            point = repeated_measures_limits(frame)
            ci = cluster_bootstrap(frame, SEED + system_index * len(FEATURES) + feature_index, replicates)
            rows.append({"system": system, "feature": feature, **point, **ci})
    return pd.DataFrame(rows)


def build_figure(data: pd.DataFrame, summary: pd.DataFrame, out: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 4.9), constrained_layout=True)
    axes = axes.ravel()
    for idx, feature in enumerate(FEATURES):
        ax = axes[idx]
        panel = data[data["feature"] == feature]
        limits = summary[summary["feature"] == feature]
        for system in SYSTEMS:
            rows = panel[panel["system"] == system]
            row = limits[limits["system"] == system].iloc[0]
            color = COLORS[system]
            ax.scatter(rows["mean"], rows["difference"], s=17, alpha=0.62,
                       color=color, edgecolors="white", linewidths=0.25,
                       marker=MARKERS[system], label=system, rasterized=True)
            ax.axhline(row["bias"], color=color, linestyle="-", linewidth=1.0, alpha=0.95)
            ax.axhline(row["loa_low"], color=color, linestyle="--", linewidth=0.9, alpha=0.8)
            ax.axhline(row["loa_high"], color=color, linestyle="--", linewidth=0.9, alpha=0.8)
        ax.axhline(0, color="#333333", linewidth=0.7, alpha=0.55)
        x_label, y_label = LABELS[feature]
        ax.set_xlabel("Mean of method and reference\n" + x_label)
        ax.set_ylabel("Method − reference\n" + y_label)
        ax.text(0.015, 0.98, f"({chr(97 + idx)})", transform=ax.transAxes,
                ha="left", va="top", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#d9dde1", linewidth=0.5, alpha=0.7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:3], labels[:3], loc="upper center", bbox_to_anchor=(0.5, 1.025),
               ncol=3, frameon=False, handletextpad=0.45, columnspacing=1.8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    tiff = out.with_suffix(".tif")
    # Keep the conversion in memory. On Windows, deleting a temporary TIFF
    # immediately after Pillow closes it can fail while another library still
    # holds a file handle; BytesIO avoids that platform-specific lock.
    buffer = BytesIO()
    fig.savefig(buffer, format="tiff", dpi=300, bbox_inches="tight", facecolor="white")
    buffer.seek(0)
    with Image.open(buffer) as source:
        rgb = source.convert("RGB")
        rgb.save(tiff, format="TIFF", compression="tiff_lzw", dpi=(300, 300))
        rgb.close()
    buffer.close()
    plt.close(fig)


def write_tex_table(summary: pd.DataFrame, out: Path) -> None:
    outcome_names = {
        "peak_foot_speed": "Peak foot speed (m s$^{-1}$)",
        "knee_angle_impact": "Knee angle at impact ($^\circ$)",
        "ankle_angle_impact": "Ankle angle at impact ($^\circ$)",
        "peak_abs_knee_angvel": "Peak absolute knee\\\\angular velocity ($^\circ$ s$^{-1}$)",
    }
    lines = [
        "% Generated by recompute_bland_altman.py from the supplied trial-level measurements.",
        r"\begin{table}[p]",
        r"\caption{Repeated-measures Bland--Altman estimates for each markerless system and outcome. Bias and limits of agreement (LoA) are method minus manual reference. Confidence intervals are percentile intervals from 5,000 participant-cluster bootstrap samples. Each estimate uses all 78 kicks from 13 participants (39 per kicking side).}",
        r"\label{tab:s4}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}p{3.2cm}>{\centering\arraybackslash}p{2.9cm}>{\centering\arraybackslash}p{2.9cm}>{\centering\arraybackslash}p{2.9cm}}",
        r"\toprule",
        r"System / outcome & Bias (95\% CI) & Lower LoA (95\% CI) & Upper LoA (95\% CI) " + r"\\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        feature = row["feature"]
        lines.append(
            f"\\parbox[t]{{3.2cm}}{{{row['system']}\\\\{outcome_names[feature]}}} & "
            rf"\shortstack{{{row['bias']:.2f}\\{{[{row['bias_ci_low']:.2f}, {row['bias_ci_high']:.2f}]}}}} & "
            rf"\shortstack{{{row['loa_low']:.2f}\\{{[{row['loa_low_ci_low']:.2f}, {row['loa_low_ci_high']:.2f}]}}}} & "
            rf"\shortstack{{{row['loa_high']:.2f}\\{{[{row['loa_high_ci_low']:.2f}, {row['loa_high_ci_high']:.2f}]}}}} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Trial-level measurements CSV in the supplied data package")
    parser.add_argument("--output-dir", type=Path, default=Path("figures/main"))
    parser.add_argument("--summary-csv", type=Path, default=Path("tables/bland_altman_repeated_measures.csv"))
    parser.add_argument("--table-tex", type=Path, default=Path("tables/bland_altman_table_s4.tex"))
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--angles-already-signed", action="store_true",
                        help="Input knee and ankle angles already use the study's signed convention")
    args = parser.parse_args()
    data = prepare_data(args.source, angles_already_signed=args.angles_already_signed)
    summary = summarize(data, replicates=args.bootstrap)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False, float_format="%.8f")
    write_tex_table(summary, args.table_tex)
    build_figure(data, summary, args.output_dir / "figure5_bland_altman")
    print(f"Rows: {len(data)}; participants: {data['participant'].nunique()}; complete-case trial IDs: {data['trial_id'].nunique()}")
    print(summary[["system", "feature", "n_observations", "bias", "loa_low", "loa_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
