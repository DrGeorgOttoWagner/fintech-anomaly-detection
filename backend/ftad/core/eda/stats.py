"""Small, dependency-light statistical helpers used across EDA and reporting."""

from __future__ import annotations

import numpy as np
from scipy import stats as sstats


def describe(values: np.ndarray) -> dict:
    v = values[np.isfinite(values)]
    n_missing = int(len(values) - len(v))
    if len(v) == 0:
        return {"count": 0, "missing": n_missing}
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    out = {
        "count": int(len(v)),
        "missing": n_missing,
        "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=1)) if len(v) > 1 else None,
        "min": float(np.min(v)),
        "q1": float(q1),
        "median": float(med),
        "q3": float(q3),
        "max": float(np.max(v)),
        "iqr": float(q3 - q1),
        "n_zero": int(np.sum(v == 0)),
        "n_unique": int(len(np.unique(v))) if len(v) <= 2_000_000 else None,
    }
    if len(v) > 2 and np.std(v) > 0:
        out["skewness"] = float(sstats.skew(v, bias=False))
        out["kurtosis"] = float(sstats.kurtosis(v, bias=False))
    else:
        out["skewness"] = None
        out["kurtosis"] = None
    p = np.percentile(v, [1, 5, 95, 99])
    out.update({"p01": float(p[0]), "p05": float(p[1]), "p95": float(p[2]), "p99": float(p[3])})
    return out


def box_stats(values: np.ndarray) -> dict:
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return {"count": 0}
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = v[(v >= lo_fence) & (v <= hi_fence)]
    return {
        "count": int(len(v)),
        "min": float(v.min()),
        "q1": float(q1),
        "median": float(med),
        "q3": float(q3),
        "max": float(v.max()),
        "mean": float(v.mean()),
        "whisker_low": float(inside.min()) if len(inside) else float(q1),
        "whisker_high": float(inside.max()) if len(inside) else float(q3),
        "n_outliers": int(len(v) - len(inside)),
        "outlier_pct": float((len(v) - len(inside)) / len(v)),
    }


def cohens_d(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else None


def two_sample_tests(a: np.ndarray, b: np.ndarray) -> dict:
    """Non-parametric comparison of two groups (a = positive, b = negative)."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return {"available": False, "reason": "Fewer than two observations in a group."}
    mw = sstats.mannwhitneyu(a, b, alternative="two-sided")
    ks = sstats.ks_2samp(a, b)
    # Rank-biserial correlation as effect size for Mann-Whitney U.
    rbc = 1 - 2 * mw.statistic / (len(a) * len(b))
    return {
        "available": True,
        "mann_whitney_u": float(mw.statistic),
        "mann_whitney_p": float(mw.pvalue),
        "rank_biserial": float(-rbc),
        "ks_statistic": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "cohens_d": cohens_d(a, b),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
    }


def shared_histogram(
    groups: dict[str, np.ndarray], bins: int = 50, log: bool = False, clip: tuple[float, float] | None = None
) -> dict:
    """Histogram with common bin edges for several groups (counts and per-group densities)."""
    all_vals = np.concatenate([g[np.isfinite(g)] for g in groups.values()])
    if len(all_vals) == 0:
        return {"edges": [], "series": {}}
    transform = (lambda x: np.log1p(x)) if log else (lambda x: x)
    if log and all_vals.min() < 0:
        raise ValueError("Log scale requires non-negative values.")
    t_all = transform(all_vals)
    lo, hi = (np.percentile(t_all, clip) if clip else (t_all.min(), t_all.max()))
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, bins + 1)
    series = {}
    for name, vals in groups.items():
        tv = transform(vals[np.isfinite(vals)])
        n_total = len(tv)
        n_clipped = int(np.sum((tv < lo) | (tv > hi)))
        counts, _ = np.histogram(np.clip(tv, lo, hi), bins=edges)
        widths = np.diff(edges)
        density = counts / (max(1, n_total) * widths)
        series[name] = {
            "counts": counts.astype(int).tolist(),
            "share": (counts / max(1, n_total)).tolist(),
            "density": density.tolist(),
            "n": int(n_total),
            "n_clipped_into_edge_bins": n_clipped,
        }
    return {
        "edges": edges.tolist(),
        "display_edges": (np.expm1(edges) if log else edges).tolist(),
        "log1p": log,
        "clipped_percentiles": list(clip) if clip else None,
        "series": series,
    }
