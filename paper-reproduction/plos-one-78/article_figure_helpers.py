"""Plotting helpers for the PLOS ONE article reproduction scripts."""
from __future__ import annotations
import re
import unicodedata
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.patches import Patch
PROJECT = Path(__file__).resolve().parent


METHODS = ["DVIDEOW (manual)", "Sports2D", "MediaPipe Pose", "DeepLabCut"]

MARKERLESS = METHODS[1:]

COLORS = {
    "DVIDEOW (manual)": "#000080",
    "Sports2D": "#14823B",
    "MediaPipe Pose": "#D81B9C",
    "DeepLabCut": "#F07800",
}

JOINTS = ["Hip", "Knee", "Ankle", "Foot"]

JOINT_LABELS = {"Hip": "Hip", "Knee": "Knee", "Ankle": "Ankle", "Foot": "Foot"}

SYSTEM_COLUMN_MAP = {
    "oro": "DVIDEOW (manual)",
    "dvideow": "DVIDEOW (manual)",
    "s2d": "Sports2D",
    "sports2d": "Sports2D",
    "mediapipe": "MediaPipe Pose",
    "mediapipe pose": "MediaPipe Pose",
    "deeplabcut": "DeepLabCut",
    "deeplabcut-resnet-101": "DeepLabCut",
    "deeplabcut (resnet-101)": "DeepLabCut",
    "deeplabcut (resnet 101)": "DeepLabCut",
    "deeplabcut - resnet - 101": "DeepLabCut",
    "deeplabcut resnet 101": "DeepLabCut",
}

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8.2,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.3,
        "axes.linewidth": 0.65,
        "lines.linewidth": 1.25,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)

def ascii_key(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip().lower()

def find_table(tables: list[list[list[str]]], required: set[str]) -> list[list[str]]:
    for table in tables:
        if not table:
            continue
        header = {ascii_key(x) for x in table[0]}
        if required.issubset(header):
            return table
    raise ValueError(f"Required input table not found: {sorted(required)}")

def to_number(value: str) -> float:
    text = str(value).strip().replace("−", "-").replace("–", "-")
    text = re.sub(r"\s+", "", text)
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    return float(text)

def save_figure(
    fig: plt.Figure, out_dir: Path, stem: str, *, preserve_facecolor: bool = False
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(ax.get_title() for ax in fig.axes):
        raise ValueError(f"Embedded plot title detected in {stem}")
    outputs = []
    for suffix, dpi in ((".pdf", None), (".png", 300), (".tif", 300)):
        path = out_dir / f"{stem}{suffix}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
        if preserve_facecolor:
            kwargs["facecolor"] = fig.get_facecolor()
        if dpi:
            kwargs["dpi"] = dpi
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"} if suffix == ".tif" else {}
        fig.savefig(path, **kwargs)
        if suffix == ".tif":
            with Image.open(path) as source:
                rgba = source.convert("RGBA")
                background = Image.new("RGB", rgba.size, "white")
                background.paste(rgba, mask=rgba.getchannel("A"))
                temporary = path.with_name(path.stem + ".rgb.tif")
                background.save(temporary, format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi))
            temporary.replace(path)
        outputs.append(path)
    plt.close(fig)
    return outputs

def set_clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.5, width=0.6, pad=2)
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.45, alpha=0.8)
    ax.set_axisbelow(True)

def panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.12,
        1.06,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

def figure_spatial_error(
    spatial_errors_path: Path, out_dir: Path, tables_out: Path | None = None
) -> tuple[list[Path], Path]:
    source = spatial_errors_path.expanduser().resolve()
    frame = pd.read_csv(source)
    aliases = {"MediaPipe": "MediaPipe Pose", "DeepLabCut-ResNet-101": "DeepLabCut", "DeepLabCut (ResNet-101)": "DeepLabCut"}
    frame["system"] = frame["system"].replace(aliases)
    participant_col = "participant_code" if "participant_code" in frame.columns else "participant"
    kick_col = "kick" if "kick" in frame.columns else "trial"
    frame_count_col = "n_paired_frames" if "n_paired_frames" in frame.columns else "n_frames"
    required = {participant_col, kick_col, "side", "system", "marker", "median_error_m", frame_count_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Spatial-error CSV is missing columns: {', '.join(sorted(missing))}")
    frame = frame[frame["marker"].isin(JOINTS)].copy()
    frame = frame[frame["system"].isin(MARKERLESS)]
    frame = frame.dropna(subset=["median_error_m"])
    if frame.empty:
        raise ValueError(f"No trial-level joint errors found in {source}")
    counts = (
        frame.drop_duplicates([participant_col, "side", kick_col, "system", "marker"])
        .groupby(["system", "marker"]).size()
    )
    expected = pd.MultiIndex.from_product([MARKERLESS, JOINTS], names=["system", "marker"])
    counts = counts.reindex(expected, fill_value=0)
    if not counts.eq(78).all():
        raise ValueError(
            "Expected 78 kicks for every markerless system and landmark; "
            f"observed counts: {counts.to_dict()}"
        )
    table_dir = (tables_out or (PROJECT / "tables")).expanduser().resolve()
    table_dir.mkdir(parents=True, exist_ok=True)
    table_rows = []
    system_order = ["Sports2D", "MediaPipe Pose", "DeepLabCut"]
    for system in system_order:
        median_row = [system]
        iqr_row = [""]
        for joint in JOINTS:
            sample = frame.loc[(frame.system == system) & (frame.marker == joint), "median_error_m"]
            q1, median, q3 = sample.quantile([0.25, 0.5, 0.75]).to_numpy()
            median_row.append(f"{median:.3f}")
            iqr_row.append(f"({q1:.3f}--{q3:.3f})")
        table_rows.append(" & ".join(median_row) + r" \\")
        table_rows.append(" & ".join(iqr_row) + r" \\")
    table = [
        r"\begin{table}[t]",
        r"\caption{Trial-level median three-dimensional positional error by markerless system and anatomical landmark. Values are median (interquartile range) in metres across 78 kicks per cell. Each per-kick median uses available manually paired frames over the pre-impact-to-contact interval (2--11 frames per kick and landmark). Kicks are repeated observations nested within 13 participants.}",
        r"\label{tab:spatial}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrr}",
        r"\toprule",
        r"System & Hip & Knee & Ankle & Foot \\",
        r"\midrule",
        *table_rows,
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table}",
    ]
    (table_dir / "spatial_error_summary.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    frame_counts = frame.groupby(["system", "marker"])[frame_count_col].sum().unstack("marker")
    count_rows = [
        " & ".join([system] + [str(int(frame_counts.loc[system, joint])) for joint in JOINTS]) + r" \\"
        for system in system_order
    ]
    frame_count_table = [
        r"\begin{table}[t]",
        r"\caption{Number of finite, frame-matched three-dimensional observations contributing to the 78-kick available-case spatial summaries. Values are frame--landmark pairs by system and joint; each kick contributed 2--11 pairs in each cell.}",
        r"\label{tab:spatial-frame-counts}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"System & Hip & Knee & Ankle & Foot \\",
        r"\midrule",
        *count_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (table_dir / "spatial_error_frame_counts.tex").write_text("\n".join(frame_count_table) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7.05, 3.55))
    positions = {"Sports2D": -0.24, "MediaPipe Pose": 0.0, "DeepLabCut": 0.24}
    width = 0.19
    rng = np.random.default_rng(2026)
    legend_handles = []
    for sidx, system in enumerate(MARKERLESS):
        vals, pos = [], []
        for jidx, joint in enumerate(JOINTS):
            sample = frame.loc[(frame.system == system) & (frame.marker == joint), "median_error_m"].to_numpy(float)
            vals.append(sample)
            pos.append(jidx + positions[system])
            jitter = rng.normal(0, 0.035, size=len(sample))
            ax.scatter(
                np.full(len(sample), jidx + positions[system]) + jitter,
                sample,
                s=7,
                color=COLORS[system],
                alpha=0.24,
                linewidths=0,
                zorder=2,
            )
        bp = ax.boxplot(
            vals,
            positions=pos,
            widths=width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.0},
            whiskerprops={"color": COLORS[system], "linewidth": 0.8},
            capprops={"color": COLORS[system], "linewidth": 0.8},
            boxprops={"edgecolor": COLORS[system], "linewidth": 0.8},
            zorder=3,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(COLORS[system])
            patch.set_alpha(0.22)
        legend_handles.append(Patch(facecolor=COLORS[system], edgecolor=COLORS[system], alpha=0.45, label=system))
    ax.set_xticks(range(len(JOINTS)), [JOINT_LABELS[j] for j in JOINTS])
    ax.set_ylabel("Trial-level median 3D position error (m)")
    ax.set_xlabel("Anatomical landmark")
    ax.legend(handles=legend_handles, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    set_clean_axes(ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    outputs = save_figure(fig, out_dir, "figure2_spatial_error")
    return outputs, source

def variable_label(value: str) -> str:
    key = ascii_key(value).replace(" ", "")
    compact = key.replace(":", "").replace("-", "")
    if "speedsymmetryindex" in compact and "hip" in compact:
        return "Speed symmetry index — Hip (%)"
    if "speedsymmetryindex" in compact and "knee" in compact:
        return "Speed symmetry index — Knee (%)"
    if "speedsymmetryindex" in compact and "ankle" in compact:
        return "Speed symmetry index — Ankle (%)"
    if "speedsymmetryindex" in compact and "foot" in compact:
        return "Speed symmetry index — Foot (%)"
    if "kneeangledifference" in compact:
        return "Knee angle difference (°)"
    if "ankleangledifference" in compact:
        return "Ankle angle difference (°)"
    if "kneeangularvelocitydifference" in compact:
        return "Knee angular-velocity difference (°/s)"
    if "ankleangularvelocitydifference" in compact:
        return "Ankle angular-velocity difference (°/s)"
    if "isvelocidad" in key and "cadera" in key:
        return "Speed symmetry index — Hip (%)"
    if "isvelocidad" in key and "rodilla" in key:
        return "Speed symmetry index — Knee (%)"
    if "isvelocidad" in key and "tobillo" in key:
        return "Speed symmetry index — Ankle (%)"
    if "isvelocidad" in key and "pie" in key:
        return "Speed symmetry index — Foot (%)"
    if "θ" in value or "theta" in key:
        if "rodilla" in key:
            return "Knee angle difference (°)"
        if "tobillo" in key:
            return "Ankle angle difference (°)"
    if "ω" in value or "omega" in key:
        if "rodilla" in key:
            return "Knee angular-velocity difference (°/s)"
        if "tobillo" in key:
            return "Ankle angular-velocity difference (°/s)"
    raise ValueError(f"Unmapped bilateral outcome in input table: {value}")

def figure_bilateral_errors(tables: list[list[list[str]]], out_dir: Path) -> list[Path]:
    try:
        source_table = find_table(tables, {"variable", "sistema", "mae", "rmse", "ccc"})
    except ValueError:
        source_table = find_table(tables, {"outcome", "system", "mae", "rmse", "ccc"})
    rows = []
    for row in source_table[1:]:
        if len(row) < 5 or not row[0].strip():
            continue
        system = SYSTEM_COLUMN_MAP.get(ascii_key(row[1]))
        if system not in MARKERLESS:
            continue
        rows.append(
            {
                "outcome_raw": row[0],
                "outcome": variable_label(row[0]),
                "system": system,
                "MAE": to_number(row[2]),
                "RMSE": to_number(row[3]),
            }
        )
    frame = pd.DataFrame(rows)
    outcomes = list(dict.fromkeys(frame["outcome"].tolist()))
    if len(outcomes) != 8 or len(frame) != 24:
        raise ValueError(f"Unexpected bilateral input table structure: {len(outcomes)} outcomes, {len(frame)} rows")
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 4.55), sharey=True)
    cmap = plt.get_cmap("YlGnBu")
    all_mappables = []
    for pidx, metric in enumerate(("MAE", "RMSE")):
        ax = axes[pidx]
        matrix = np.zeros((len(outcomes), len(MARKERLESS)))
        for ridx, outcome in enumerate(outcomes):
            for cidx, system in enumerate(MARKERLESS):
                val = frame.loc[(frame.outcome == outcome) & (frame.system == system), metric]
                if len(val) != 1:
                    raise ValueError(f"Missing unique {metric} for {outcome} / {system}")
                matrix[ridx, cidx] = float(val.iloc[0])
        normalized = matrix.copy()
        for ridx in range(len(outcomes)):
            lo, hi = matrix[ridx].min(), matrix[ridx].max()
            normalized[ridx] = 0.5 if np.isclose(lo, hi) else (matrix[ridx] - lo) / (hi - lo)
        image = ax.imshow(normalized, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        all_mappables.append(image)
        ax.set_xticks(range(len(MARKERLESS)), ["Sports2D", "MediaPipe\nPose", "DeepLabCut\n(ResNet-101)"])
        ax.set_yticks(range(len(outcomes)), outcomes)
        for ridx in range(len(outcomes)):
            for cidx in range(len(MARKERLESS)):
                text_color = "white" if normalized[ridx, cidx] >= 0.56 else "#17252B"
                ax.text(cidx, ridx, f"{matrix[ridx, cidx]:.2f}", ha="center", va="center", fontsize=6.7, color=text_color)
        if pidx == 0:
            ax.set_ylabel("Bilateral outcome")
        else:
            ax.tick_params(axis="y", labelleft=False)
            ax.set_ylabel("")
        ax.set_xlabel("Pose-estimation system")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(MARKERLESS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(outcomes), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.4)
        ax.tick_params(which="minor", bottom=False, left=False)
        panel_letter(ax, "a" if pidx == 0 else "b")
    fig.text(0.5, 0.045, "Cell shading shows relative error within each outcome (lighter = lower; darker = higher).", ha="center", va="bottom", fontsize=7.0)
    fig.subplots_adjust(left=0.31, right=0.99, top=0.94, bottom=0.22, wspace=0.12)
    return save_figure(fig, out_dir, "figure4_bilateral_errors")

def captions_file(path: Path) -> None:
    text = """# Figure captions\n\n"""
    captions = [
        "**Figure 1. Study workflow.** The same synchronized recordings were analyzed by Sports2D, MediaPipe Pose, and DeepLabCut, with manual reference tracking performed in DVIDEOW. Automated detections were anatomically harmonized and matched by camera and frame. The manual reference retained radial-distortion correction and the original quadratic LOESS smoothing (span = 0.7). All trajectories were reconstructed by DLT with camera-specific calibrations and processed before calculation of kinematic and bilateral outcomes.",
        "**Figure 2. Trial-level three-dimensional positional error.** Distribution of per-kick median position errors for hip, knee, ankle, and foot landmarks for Sports2D, MediaPipe Pose, and DeepLabCut relative to manual reference tracking. Boxes show medians and interquartile ranges; whiskers extend to 1.5 times the interquartile range; points represent 78 kicks per system and landmark. Each kick median uses 2--11 available frame pairs.",
        "**Figure 3. Sagittal kinematics across the pre-impact-to-impact window.** The upper four panels show knee and ankle angles; the lower four show their angular velocities. Within each 2 × 2 group, columns show knee and ankle, and rows show right- and left-kicking limbs. Curves show manual reference tracking and the three markerless systems.",
        "**Figure 4. Bilateral outcome errors.** Mean absolute error (MAE) and root-mean-square error (RMSE) for the speed symmetry index and dominant-minus-nondominant sagittal angle and angular-velocity differences. Cell labels show the corresponding error values. Color is scaled within each outcome to show the relative error among systems.",
        "**Figure 5. Repeated-measures Bland–Altman analyses.** Method-minus-reference differences are plotted against the means of each method and manual reference for peak foot speed, knee and ankle angles, and peak absolute knee angular velocity. Each point represents one kick from the 78-trial corpus. Solid lines indicate bias and dotted lines indicate the 95% limits of agreement.",
    ]
    text += "\n\n".join(captions) + "\n"
    path.write_text(text, encoding="utf-8")
