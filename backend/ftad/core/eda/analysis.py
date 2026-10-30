"""Descriptive statistics and exploratory data analysis.

Every number returned here is computed from the uploaded data. Statistical summaries,
histograms, box statistics, correlations and tests always use the *full* dataset.
Only scatter-style visual projections use a seeded sample, and they are labelled so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from ftad.core.eda.context import AnalysisContext
from ftad.core.eda.stats import box_stats, describe, shared_histogram, two_sample_tests

MI_SAMPLE = 60_000
PROJECTION_POS = 2_000
PROJECTION_NEG = 4_000
MAX_CORR_COLUMNS = 60


# ---------------------------------------------------------------------------
# Overview / descriptive statistics
# ---------------------------------------------------------------------------
def overview(ctx: AnalysisContext) -> dict:
    df, cfg = ctx.df, ctx.cfg
    n = len(df)
    missing = df.isna().sum()
    dup_mask = df.duplicated(keep="first")
    n_dup = int(dup_mask.sum())
    analysis_cols = [c for c in df.columns if cfg.roles.get(c) not in ("transaction_id", "ignore")]
    n_dup_content = int(df.duplicated(subset=analysis_cols, keep="first").sum()) if analysis_cols else 0
    conflicting = 0
    if ctx.has_target and cfg.features:
        tmp = df[cfg.features].copy()
        tmp["__y"] = ctx.y
        g = tmp.groupby(cfg.features, dropna=False, sort=False)["__y"].nunique()
        conflicting = int((g > 1).sum())

    role_counts: dict[str, int] = {}
    for r in cfg.roles.values():
        role_counts[r] = role_counts.get(r, 0) + 1
    kind_counts = {
        "numeric": int(sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)),
        "non_numeric": int(sum(not pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)),
    }

    columns = []
    for c in df.columns:
        s = df[c]
        entry = {
            "name": c,
            "role": cfg.roles.get(c),
            "dtype": str(s.dtype),
            "missing": int(missing[c]),
            "missing_pct": float(missing[c] / n) if n else 0.0,
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            entry.update(describe(s.to_numpy(dtype=float)))
        columns.append(entry)

    out = {
        "n_records": n,
        "n_columns": int(df.shape[1]),
        "n_features": len(cfg.features),
        "n_numeric_features": len(cfg.columns_with("numeric")),
        "n_categorical_features": len(cfg.categorical_features),
        "n_pca_features": len(cfg.pca_features),
        "role_counts": role_counts,
        "kind_counts": kind_counts,
        "total_missing_cells": int(missing.sum()),
        "columns_with_missing": int((missing > 0).sum()),
        "duplicate_rows": n_dup,
        "duplicate_rows_pct": n_dup / n if n else 0.0,
        "duplicate_rows_excluding_ids": n_dup_content,
        "duplicate_feature_vectors_with_conflicting_labels": conflicting,
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1e6),
        "columns": columns,
        "class_balance": class_balance(ctx) if ctx.has_target else None,
    }
    out["warnings"] = quality_warnings(ctx, out)
    return out


def quality_warnings(ctx: AnalysisContext, ov: dict) -> list[dict]:
    w: list[dict] = []
    cfg = ctx.cfg
    if ov["total_missing_cells"]:
        w.append({"level": "warning", "code": "missing", "message": f"{ov['total_missing_cells']:,} missing cells in {ov['columns_with_missing']} column(s)."})
    if ov["duplicate_rows"]:
        w.append({
            "level": "warning",
            "code": "duplicates",
            "message": f"{ov['duplicate_rows']:,} exact duplicate rows ({ov['duplicate_rows_pct']:.3%}). "
            "Duplicates can leak between train and test partitions if not handled.",
        })
    if ov["duplicate_feature_vectors_with_conflicting_labels"]:
        w.append({"level": "warning", "code": "label_conflict", "message": f"{ov['duplicate_feature_vectors_with_conflicting_labels']} identical feature vectors carry different labels."})
    for col in ov["columns"]:
        if col["n_unique"] <= 1:
            w.append({"level": "info", "code": "constant", "message": f"Column '{col['name']}' is constant."})
        if col.get("skewness") is not None and abs(col["skewness"]) > 5 and col["role"] in ("numeric",):
            w.append({"level": "info", "code": "skew", "message": f"'{col['name']}' is highly skewed (skewness {col['skewness']:.1f}); consider robust scaling or a log transform."})
    cb = ov.get("class_balance")
    if cb:
        if cb["prevalence"] < 0.01:
            w.append({"level": "warning", "code": "imbalance", "message": f"Severe class imbalance: {cb['prevalence']:.3%} {ctx.pos_name.lower()} records (1 in {cb['one_in_n']:,.0f})."})
        if cb["n_positive"] < 50:
            w.append({"level": "warning", "code": "few_positives", "message": f"Only {cb['n_positive']} positive records — metric estimates will have wide uncertainty."})
    if ctx.has_target:
        for col in cfg.numeric_features:
            v = ctx.numeric(col)
            ok = np.isfinite(v)
            if ok.sum() < 10 or np.std(v[ok]) == 0:
                continue
            auc = roc_auc_score(ctx.y[ok], v[ok])
            if max(auc, 1 - auc) > 0.995:
                w.append({"level": "warning", "code": "leakage", "message": f"'{col}' separates the classes almost perfectly (univariate AUC {max(auc, 1 - auc):.3f}) — check for target leakage."})
    return w


def class_balance(ctx: AnalysisContext) -> dict:
    y = ctx.y
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    raw_counts = ctx.df[ctx.cfg.target].astype(str).value_counts().to_dict()
    minority, majority = min(n_pos, n_neg), max(n_pos, n_neg)
    return {
        "n": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "positive_name": ctx.pos_name,
        "negative_name": ctx.neg_name,
        "positive_label": ctx.cfg.positive_label,
        "prevalence": n_pos / n,
        "negative_share": n_neg / n,
        "minority_to_majority": minority / majority if majority else None,
        "majority_to_minority": majority / minority if minority else None,
        "one_in_n": n / n_pos if n_pos else None,
        "majority_baseline_accuracy": majority / n,
        "majority_class": ctx.neg_name if n_neg >= n_pos else ctx.pos_name,
        "raw_value_counts": {str(k): int(v) for k, v in raw_counts.items()},
        # A majority-class classifier never flags a positive: recall 0, precision undefined.
        "majority_baseline": {"accuracy": majority / n, "recall": 0.0, "precision": None, "f1": 0.0},
    }


# ---------------------------------------------------------------------------
# Univariate analysis of a value column (e.g. Amount), split by class
# ---------------------------------------------------------------------------
def value_analysis(ctx: AnalysisContext, column: str, bins: int = 60, log: bool = False) -> dict:
    v = ctx.numeric(column)
    groups: dict[str, np.ndarray] = {"All": v}
    result: dict = {"column": column, "overall": describe(v), "box": {"All": box_stats(v)}}
    if ctx.has_target:
        pos, neg = v[ctx.y == 1], v[ctx.y == 0]
        groups = {ctx.neg_name: neg, ctx.pos_name: pos}
        result["by_class"] = {ctx.neg_name: describe(neg), ctx.pos_name: describe(pos)}
        result["box"].update({ctx.neg_name: box_stats(neg), ctx.pos_name: box_stats(pos)})
        result["tests"] = two_sample_tests(pos, neg)
        finite = np.isfinite(v)
        if finite.sum() and np.std(v[finite]) > 0:
            auc = roc_auc_score(ctx.y[finite], v[finite])
            result["univariate_auc"] = float(auc)
    can_log = bool(np.nanmin(v) >= 0) if np.isfinite(v).any() else False
    result["log_available"] = can_log
    result["histogram"] = shared_histogram(groups, bins=bins, log=log and can_log)
    if not log:
        result["histogram_clipped"] = shared_histogram(groups, bins=bins, clip=(0.0, 99.0))
    # Value bands: how the positive rate changes across quantile bands of the variable.
    if ctx.has_target:
        result["bands"] = _rate_by_quantile_band(v, ctx.y)
    return result


def _rate_by_quantile_band(v: np.ndarray, y: np.ndarray, n_bands: int = 10) -> list[dict]:
    ok = np.isfinite(v)
    v, y = v[ok], y[ok]
    edges = np.unique(np.percentile(v, np.linspace(0, 100, n_bands + 1)))
    if len(edges) < 2:
        return []
    idx = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        rows.append({
            "low": float(edges[b]),
            "high": float(edges[b + 1]),
            "n": int(m.sum()),
            "n_positive": int(y[m].sum()),
            "positive_rate": float(y[m].mean()),
        })
    return rows


# ---------------------------------------------------------------------------
# Temporal analysis
# ---------------------------------------------------------------------------
def time_analysis(ctx: AnalysisContext, bin_size: float | None = None) -> dict:
    cfg = ctx.cfg
    if not cfg.time_column:
        return {"available": False, "reason": "No time column configured in the schema."}
    col = cfg.time_column
    kind = cfg.time_kind or "ordinal"
    raw = ctx.df[col]
    if kind == "timestamp":
        ts = pd.to_datetime(raw, errors="coerce", format="mixed")
        ok = ts.notna().to_numpy()
        seconds = (ts - ts.min()).dt.total_seconds().to_numpy(dtype=float)
        origin = ts.min().isoformat() if ok.any() else None
        unit_label = "seconds since first timestamp"
    else:
        seconds = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(seconds)
        origin = None
        unit_label = "elapsed seconds since first record" if kind == "elapsed_seconds" else "ordinal units (unit unknown)"
    span = float(np.nanmax(seconds[ok]) - np.nanmin(seconds[ok])) if ok.any() else 0.0
    start = float(np.nanmin(seconds[ok])) if ok.any() else 0.0

    if bin_size is None or bin_size <= 0:
        if kind in ("elapsed_seconds", "timestamp"):
            bin_size = 3600.0 if span <= 14 * 86400 else 86400.0
        else:
            bin_size = max(span / 48.0, 1.0)
    n_bins = int(np.ceil((span + 1e-9) / bin_size)) or 1
    if n_bins > 2000:
        bin_size = span / 2000
        n_bins = 2000
    idx = np.clip(((seconds - start) // bin_size).astype(np.int64, copy=False), 0, n_bins - 1)
    idx = np.where(ok, idx, -1)

    y = ctx.y
    series = []
    for b in range(n_bins):
        m = idx == b
        n_b = int(m.sum())
        row = {"bin": b, "start": start + b * bin_size, "end": start + (b + 1) * bin_size, "n": n_b}
        if y is not None:
            n_pos = int(y[m].sum())
            row.update({"n_positive": n_pos, "positive_rate": n_pos / n_b if n_b else None})
        series.append(row)

    result: dict = {
        "available": True,
        "column": col,
        "kind": kind,
        "unit": unit_label,
        "origin": origin,
        "span_seconds": span,
        "start": start,
        "bin_size": bin_size,
        "series": series,
        "interpretation_note": (
            "Time is the number of seconds elapsed since the first transaction in the file; it is not a calendar "
            "timestamp. Hour-of-cycle values are offsets from the first record, not clock hours."
            if kind == "elapsed_seconds"
            else None
        ),
    }

    if kind in ("elapsed_seconds", "timestamp") and span >= 86400 * 0.9:
        hour = (((seconds - start) // 3600) % 24).astype(int)
        cyc = []
        for h in range(24):
            m = ok & (hour == h)
            row = {"hour": h, "n": int(m.sum())}
            if y is not None:
                n_pos = int(y[m].sum())
                row.update({"n_positive": n_pos, "positive_rate": n_pos / row["n"] if row["n"] else None})
            cyc.append(row)
        result["hour_of_cycle"] = cyc
        result["n_cycles_covered"] = span / 86400

    if y is not None:
        pos, neg = seconds[ok & (y == 1)], seconds[ok & (y == 0)]
        result["by_class"] = {ctx.neg_name: describe(neg), ctx.pos_name: describe(pos)}
        result["tests"] = two_sample_tests(pos, neg)
        result["histogram"] = shared_histogram({ctx.neg_name: neg, ctx.pos_name: pos}, bins=48)
        rates = [r["positive_rate"] for r in series if r["positive_rate"] is not None and r["n"] >= 100]
        if rates:
            result["rate_range"] = {"min": float(min(rates)), "max": float(max(rates)), "min_bin_n": 100}
    return result


# ---------------------------------------------------------------------------
# Feature analysis
# ---------------------------------------------------------------------------
def correlation(ctx: AnalysisContext, method: str = "pearson", columns: list[str] | None = None) -> dict:
    cols = columns or ctx.numeric_columns
    note = None
    if len(cols) > MAX_CORR_COLUMNS:
        note = f"Showing the first {MAX_CORR_COLUMNS} of {len(cols)} numeric columns."
        cols = cols[:MAX_CORR_COLUMNS]
    data = ctx.df[cols].apply(pd.to_numeric, errors="coerce")
    if ctx.has_target:
        data = data.assign(**{f"{ctx.cfg.target} (target)": ctx.y.astype(float)})
    if method == "spearman":
        data = data.rank()
    elif method != "pearson":
        raise ValueError("method must be 'pearson' or 'spearman'")
    mat = data.corr(method="pearson").to_numpy()
    labels = list(data.columns)
    pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if np.isfinite(mat[i, j]):
                pairs.append((abs(mat[i, j]), labels[i], labels[j], mat[i, j]))
    pairs.sort(reverse=True)
    return {
        "method": method,
        "labels": labels,
        "matrix": np.round(mat, 4).tolist(),
        "top_pairs": [{"a": a, "b": b, "r": float(r)} for _, a, b, r in pairs[:15]],
        "note": note,
        "computed_on": f"all {len(data):,} records",
    }


def feature_relevance(ctx: AnalysisContext, seed: int = 42) -> dict:
    if not ctx.has_target:
        return {"available": False, "reason": "A target is required for relevance indicators."}
    y = ctx.y
    rows = []
    cols = ctx.numeric_columns
    for col in cols:
        v = ctx.numeric(col)
        ok = np.isfinite(v)
        if ok.sum() < 10 or np.std(v[ok]) == 0:
            continue
        auc = roc_auc_score(y[ok], v[ok])
        tests = two_sample_tests(v[ok & (y == 1)], v[ok & (y == 0)])
        rows.append({
            "feature": col,
            "role": ctx.cfg.roles.get(col),
            "point_biserial_r": float(np.corrcoef(v[ok], y[ok])[0, 1]),
            "univariate_auc": float(auc),
            "separability": float(max(auc, 1 - auc)),
            "ks_statistic": tests.get("ks_statistic"),
            "cohens_d": tests.get("cohens_d"),
            "mean_positive": float(np.mean(v[ok & (y == 1)])) if (ok & (y == 1)).any() else None,
            "mean_negative": float(np.mean(v[ok & (y == 0)])) if (ok & (y == 0)).any() else None,
        })
    # Mutual information on a stratified sample (bounded cost), clearly labelled.
    mi_note = None
    if rows:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(y))
        if len(y) > MI_SAMPLE:
            pos_idx, neg_idx = idx[y == 1], idx[y == 0]
            n_pos = min(len(pos_idx), max(1, int(round(MI_SAMPLE * y.mean()))))
            idx = np.concatenate([rng.choice(pos_idx, n_pos, replace=False), rng.choice(neg_idx, MI_SAMPLE - n_pos, replace=False)])
            mi_note = f"Mutual information estimated on a stratified random sample of {len(idx):,} records (seed {seed})."
        feats = [r["feature"] for r in rows]
        X = ctx.df[feats].iloc[idx].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median())
        mi = mutual_info_classif(X.to_numpy(), y[idx], random_state=seed)
        for r, m in zip(rows, mi):
            r["mutual_information"] = float(m)
    rows.sort(key=lambda r: r["ks_statistic"] or 0, reverse=True)
    return {
        "available": True,
        "features": rows,
        "mi_note": mi_note,
        "note": "Indicators are univariate and associative; they do not imply causation and ignore interactions.",
    }


def class_conditional(ctx: AnalysisContext, column: str, bins: int = 50) -> dict:
    v = ctx.numeric(column)
    if ctx.has_target:
        groups = {ctx.neg_name: v[ctx.y == 0], ctx.pos_name: v[ctx.y == 1]}
    else:
        groups = {"All": v}
    return {
        "column": column,
        "role": ctx.cfg.roles.get(column),
        "histogram": shared_histogram(groups, bins=bins, clip=(0.5, 99.5)),
        "box": {k: box_stats(g) for k, g in groups.items()},
        "stats": {k: describe(g) for k, g in groups.items()},
        "note": "Display range clipped to the 0.5–99.5th percentiles; values outside are counted in the edge bins.",
    }


def pca_exploration(ctx: AnalysisContext) -> dict:
    cols = ctx.cfg.pca_features
    if not cols:
        return {"available": False, "reason": "No columns have the Existing PCA role."}
    X = ctx.df[cols].apply(pd.to_numeric, errors="coerce")
    var = X.var()
    total = float(var.sum())
    comps = []
    for c in cols:
        entry = {"component": c, "variance": float(var[c]), "variance_share": float(var[c] / total) if total else None, "mean": float(X[c].mean())}
        if ctx.has_target:
            entry["mean_positive"] = float(X[c][ctx.y == 1].mean())
            entry["mean_negative"] = float(X[c][ctx.y == 0].mean())
            entry["std_positive"] = float(X[c][ctx.y == 1].std())
            entry["std_negative"] = float(X[c][ctx.y == 0].std())
        comps.append(entry)
    corr = np.corrcoef(X.fillna(0).to_numpy(), rowvar=False)
    off = corr[~np.eye(len(cols), dtype=bool)] if len(cols) > 1 else np.array([0.0])
    return {
        "available": True,
        "components": comps,
        "n_components": len(cols),
        "max_abs_inter_component_corr": float(np.nanmax(np.abs(off))),
        "variance_monotone_decreasing": bool(np.all(np.diff(var.to_numpy()) <= 1e-9)),
        "note": (
            "These columns were provided already transformed (precomputed PCA). Their original variables and "
            "business meaning are unknown; FTAD reports only their statistical properties."
        ),
    }


def projection(ctx: AnalysisContext, seed: int = 42) -> dict:
    cols = ctx.cfg.numeric_features
    if len(cols) < 2:
        return {"available": False, "reason": "At least two numeric features are required."}
    X = ctx.df[cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()).to_numpy()
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=seed).fit(Xs)
    rng = np.random.default_rng(seed)
    if ctx.has_target:
        pos_idx, neg_idx = np.flatnonzero(ctx.y == 1), np.flatnonzero(ctx.y == 0)
        pos_s = rng.choice(pos_idx, min(len(pos_idx), PROJECTION_POS), replace=False)
        neg_s = rng.choice(neg_idx, min(len(neg_idx), PROJECTION_NEG), replace=False)
        idx = np.concatenate([neg_s, pos_s])
        labels = np.concatenate([np.zeros(len(neg_s), int), np.ones(len(pos_s), int)])
    else:
        idx = rng.choice(len(X), min(len(X), PROJECTION_NEG), replace=False)
        labels = np.zeros(len(idx), int)
    P = pca.transform(Xs[idx])
    return {
        "available": True,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "points": [{"x": float(p[0]), "y": float(p[1]), "label": int(l), "row": int(i)} for p, l, i in zip(P, labels, idx)],
        "sample_note": (
            f"Visual sample only: {int((labels == 1).sum()):,} {ctx.pos_name.lower()} and {int((labels == 0).sum()):,} "
            f"{ctx.neg_name.lower()} records (seed {seed}); positives are over-represented for visibility. "
            "The 2-D projection is fitted on all standardised numeric features for exploration only and is never used for modelling."
        ),
    }


# ---------------------------------------------------------------------------
# Generated interpretation (factual statements derived from computed values)
# ---------------------------------------------------------------------------
def findings(ctx: AnalysisContext, ov: dict, value: dict | None, time: dict | None, relevance: dict | None) -> list[str]:
    out: list[str] = []
    cb = ov.get("class_balance")
    pos, neg = ctx.pos_name.lower(), ctx.neg_name.lower()
    if cb:
        out.append(
            f"The dataset contains {cb['n']:,} records, of which {cb['n_positive']:,} ({cb['prevalence']:.3%}) are "
            f"{pos} and {cb['n_negative']:,} are {neg} — roughly 1 {pos} record per {cb['one_in_n']:,.0f}."
        )
        out.append(
            f"A trivial model that always predicts '{cb['majority_class']}' reaches {cb['majority_baseline_accuracy']:.3%} "
            f"accuracy while detecting none of the {pos} cases; accuracy is therefore not an informative metric here."
        )
    if ov["duplicate_rows"]:
        out.append(f"{ov['duplicate_rows']:,} rows are exact duplicates of earlier rows ({ov['duplicate_rows_pct']:.3%}).")
    if ov["total_missing_cells"] == 0:
        out.append("No missing values were found in any column.")
    if value and value.get("by_class"):
        a = value["by_class"]
        col = value["column"]
        out.append(
            f"'{col}': {ctx.pos_name} median {a[ctx.pos_name]['median']:,.2f} vs. {ctx.neg_name} median "
            f"{a[ctx.neg_name]['median']:,.2f}; mean {a[ctx.pos_name]['mean']:,.2f} vs. {a[ctx.neg_name]['mean']:,.2f}."
        )
        ov_stats = value["overall"]
        if ov_stats.get("skewness") is not None:
            out.append(f"'{col}' is right-skewed (skewness {ov_stats['skewness']:.2f}); the mean ({ov_stats['mean']:,.2f}) exceeds the median ({ov_stats['median']:,.2f}).")
        t = value.get("tests") or {}
        if t.get("available"):
            out.append(
                f"The '{col}' distributions of the two classes differ (Mann–Whitney p = {t['mann_whitney_p']:.2g}, "
                f"KS D = {t['ks_statistic']:.3f}); a large value alone does not indicate a {pos} record."
            )
    if time and time.get("available") and time.get("by_class"):
        rr = time.get("rate_range")
        if rr:
            out.append(
                f"The {pos} rate per time bin varies between {rr['min']:.3%} and {rr['max']:.3%} (bins with ≥{rr['min_bin_n']} records), "
                "so the share of positives is not constant over time."
            )
        if time.get("kind") == "elapsed_seconds":
            out.append(f"'{time['column']}' covers {time['span_seconds'] / 3600:,.1f} hours of elapsed time; it is not a calendar timestamp.")
    if relevance and relevance.get("available") and relevance["features"]:
        top = relevance["features"][:5]
        out.append(
            "Features with the strongest univariate class separation (KS statistic): "
            + ", ".join(f"{r['feature']} ({r['ks_statistic']:.2f})" for r in top)
            + ". These are associations, not causal effects."
        )
    return out
