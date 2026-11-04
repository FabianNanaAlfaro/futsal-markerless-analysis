"""Agreement, temporal association and participant-cluster Bland-Altman tools."""

from __future__ import annotations

import math

import numpy as np


def _pearson_r(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    dx, dy = x - np.mean(x), y - np.mean(y)
    denominator = math.sqrt(float(np.sum(dx * dx) * np.sum(dy * dy)))
    return float(np.sum(dx * dy) / denominator) if denominator else float("nan")


def _ranks(values):
    values = np.asarray(values, float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _beta_continued_fraction(a, b, x):
    max_iterations, epsilon, floor = 200, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (floor if abs(d) < floor else d)
    h = d
    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = floor if abs(d) < floor else d
        c = 1.0 + aa / c
        c = floor if abs(c) < floor else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = floor if abs(d) < floor else d
        c = 1.0 + aa / c
        c = floor if abs(c) < floor else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return h


def _regularized_beta(x, a, b):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _beta_continued_fraction(a, b, x) / a
    return 1.0 - bt * _beta_continued_fraction(b, a, 1.0 - x) / b


def _correlation_pvalue(r, n):
    if n < 3 or not np.isfinite(r):
        return float("nan")
    if abs(r) >= 1.0:
        return 0.0
    degrees = n - 2
    x = degrees / (degrees + r * r / max(1e-300, 1.0 - r * r))
    return float(_regularized_beta(x, degrees / 2.0, 0.5))


def concordance_correlation(reference, prediction) -> float:
    x, y = np.asarray(reference, float), np.asarray(prediction, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2:
        return float("nan")
    vx, vy = np.var(x), np.var(y)
    denominator = vx + vy + (np.mean(x) - np.mean(y)) ** 2
    return float(2 * np.mean((x - np.mean(x)) * (y - np.mean(y))) / denominator) if denominator else float("nan")


def coordinate_agreement(reference, prediction, normalization_scale=None) -> dict:
    """Return Euclidean error, axis bias, absolute/centered RMSE and P95."""
    ref, pred = np.asarray(reference, float), np.asarray(prediction, float)
    if ref.shape != pred.shape or ref.ndim != 2:
        raise ValueError("reference and prediction must be paired 2D arrays of equal shape")
    valid = np.isfinite(ref).all(axis=1) & np.isfinite(pred).all(axis=1)
    ref, pred = ref[valid], pred[valid]
    if not len(ref):
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "centered_rmse": float("nan"), "p95": float("nan"), "axis_bias": []}
    error = pred - ref
    distance = np.linalg.norm(error, axis=1)
    mean_error = np.mean(error, axis=0)
    centered = error - mean_error
    result = {
        "n": int(len(ref)),
        "mae": float(np.mean(distance)),
        "rmse": float(np.sqrt(np.mean(distance ** 2))),
        "centered_rmse": float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))),
        "p95": float(np.percentile(distance, 95)),
        "axis_bias": mean_error.tolist(),
        "ccc_by_axis": [concordance_correlation(ref[:, i], pred[:, i]) for i in range(ref.shape[1])],
    }
    if normalization_scale is not None:
        scale = float(normalization_scale)
        if scale <= 0:
            raise ValueError("normalization_scale must be positive")
        result["nme"] = float(np.mean(distance) / scale)
        normalized_error = distance / scale
        result["pck"] = {str(threshold): float(np.mean(normalized_error <= threshold))
                         for threshold in (0.05, 0.10, 0.20)}
    return result


def trajectory_association(reference, prediction) -> dict:
    x, y = np.asarray(reference, float), np.asarray(prediction, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 3:
        return {"n": int(x.size), "pearson_r": float("nan"), "pearson_p": float("nan"),
                "spearman_rho": float("nan"), "spearman_p": float("nan"), "ccc": float("nan")}
    pearson = _pearson_r(x, y)
    spearman = _pearson_r(_ranks(x), _ranks(y))
    return {"n": int(x.size), "pearson_r": pearson, "pearson_p": _correlation_pvalue(pearson, x.size),
            "spearman_rho": spearman, "spearman_p": _correlation_pvalue(spearman, x.size),
            "ccc": concordance_correlation(x, y)}


def peak_error(reference, prediction, sampling_hz: float, criterion: str = "max_abs") -> dict:
    """Compare peak magnitude and phase; criterion is max, min or max_abs."""
    x, y = np.asarray(reference, float), np.asarray(prediction, float)
    if x.shape != y.shape or x.ndim != 1 or sampling_hz <= 0:
        raise ValueError("traces must be equal one-dimensional arrays and sampling_hz positive")
    if criterion not in ("max", "min", "max_abs"):
        raise ValueError("criterion must be max, min or max_abs")
    if not np.isfinite(x).any() or not np.isfinite(y).any():
        return {"reference_peak": float("nan"), "prediction_peak": float("nan"),
                "magnitude_error": float("nan"), "phase_shift_s": float("nan")}
    if criterion == "max":
        ix, iy = int(np.nanargmax(x)), int(np.nanargmax(y))
    elif criterion == "min":
        ix, iy = int(np.nanargmin(x)), int(np.nanargmin(y))
    else:
        ix, iy = int(np.nanargmax(np.abs(x))), int(np.nanargmax(np.abs(y)))
    return {"reference_peak": float(x[ix]), "prediction_peak": float(y[iy]),
            "magnitude_error": float(abs(y[iy]) - abs(x[ix])),
            "phase_shift_s": float((iy - ix) / sampling_hz)}


def adjust_pvalues(p_values, method: str = "holm") -> np.ndarray:
    """Holm family-wise or Benjamini-Hochberg false-discovery-rate correction."""
    p = np.asarray(p_values, float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("p_values must be a finite one-dimensional array in [0, 1]")
    if method not in ("holm", "bh"):
        raise ValueError("method must be holm or bh")
    n = len(p)
    if n == 0:
        return p.copy()
    order = np.argsort(p)
    sorted_p = p[order]
    if method == "holm":
        adjusted = np.maximum.accumulate((n - np.arange(n)) * sorted_p)
    else:
        adjusted = sorted_p * n / np.arange(1, n + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty(n)
    restored[order] = np.clip(adjusted, 0, 1)
    return restored


def repeated_measures_spm(curves, method_names=None, alpha: float = 0.05) -> dict:
    """Run a one-way repeated-measures SPM ANOVA on subject×method×phase curves.

    This optional function requires ``pip install '.[spm]'``. Data must be
    complete and balanced. Pairwise automated-versus-reference SPM t-tests
    are returned alongside the four-method omnibus test.
    """
    try:
        import spm1d
    except ImportError as exc:
        raise RuntimeError("install the optional dependency with pip install '.[spm]'") from exc
    data = np.asarray(curves, float)
    if data.ndim != 3 or data.shape[0] < 2 or data.shape[1] < 2 or data.shape[2] < 3:
        raise ValueError("curves must have shape (subjects, methods, phase_points)")
    if not np.isfinite(data).all():
        raise ValueError("SPM requires complete curves; resolve missing data before inference")
    n_subjects, n_methods, n_points = data.shape
    if method_names is None:
        method_names = [f"method_{i + 1}" for i in range(n_methods)]
    if len(method_names) != n_methods:
        raise ValueError("method_names must match the method dimension")
    y = data.transpose(1, 0, 2).reshape(n_methods * n_subjects, n_points)
    conditions = np.repeat(np.arange(n_methods), n_subjects)
    subjects = np.tile(np.arange(n_subjects), n_methods)
    omnibus = spm1d.stats.anova1rm(y, conditions, subjects, equal_var=True).inference(alpha=alpha)
    pairwise = {}
    reference_index = 0
    for index in range(1, n_methods):
        test = spm1d.stats.ttest_paired(data[:, reference_index, :], data[:, index, :]).inference(
            alpha=alpha, two_tailed=True
        )
        pairwise[f"{method_names[index]}_vs_{method_names[reference_index]}"] = _spm_summary(test)
    return {"n_subjects": n_subjects, "n_methods": n_methods, "phase_points": n_points,
            "alpha": alpha, "omnibus": _spm_summary(omnibus), "pairwise": pairwise,
            "non_sphericity_correction": "not provided by spm1d.anova1rm; use the model's stated assumption"}


def _spm_summary(inference) -> dict:
    clusters = []
    for cluster in getattr(inference, "clusters", []):
        endpoints = getattr(cluster, "endpoints", None)
        clusters.append({
            "endpoints": [float(value) for value in endpoints] if endpoints is not None else None,
            "p_value": float(getattr(cluster, "P", float("nan"))),
        })
    return {"statistic": str(getattr(inference, "STAT", "")),
            "threshold": float(getattr(inference, "zstar", float("nan"))),
            "clusters": clusters}


def cluster_bootstrap_bland_altman(reference, prediction, cluster_ids,
                                   replicates: int = 2000, seed: int = 2026) -> dict:
    """Bland-Altman limits and percentile CIs, resampling participants as clusters."""
    x, y, clusters = np.asarray(reference, float), np.asarray(prediction, float), np.asarray(cluster_ids)
    if x.shape != y.shape or x.shape != clusters.shape or x.ndim != 1:
        raise ValueError("reference, prediction and cluster_ids must be equal one-dimensional arrays")
    valid = np.isfinite(x) & np.isfinite(y)
    x, y, clusters = x[valid], y[valid], clusters[valid]
    unique = np.unique(clusters)
    if len(unique) < 2:
        raise ValueError("at least two participant clusters are required")
    if replicates < 1:
        raise ValueError("replicates must be positive")

    def summarize(indices):
        difference = y[indices] - x[indices]
        mean = float(np.mean(difference))
        sd = float(np.std(difference, ddof=1)) if len(difference) > 1 else float("nan")
        return np.array([mean, mean - 1.96 * sd, mean + 1.96 * sd])

    point = summarize(np.arange(len(x)))
    rng = np.random.default_rng(seed)
    per_cluster = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    draws = np.empty((replicates, 3))
    for r in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([per_cluster[item] for item in sampled])
        draws[r] = summarize(indices)
    ci = np.percentile(draws, [2.5, 97.5], axis=0)
    return {"n": int(len(x)), "n_clusters": int(len(unique)),
            "mean_bias": float(point[0]), "lower_limit": float(point[1]), "upper_limit": float(point[2]),
            "ci95": {"mean_bias": ci[:, 0].tolist(), "lower_limit": ci[:, 1].tolist(),
                     "upper_limit": ci[:, 2].tolist()},
            "bootstrap_replicates": int(replicates), "seed": int(seed)}
