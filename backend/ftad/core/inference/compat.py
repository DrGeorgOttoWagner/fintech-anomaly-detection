"""Scoring new datasets with a previously trained, persisted pipeline.

The stored pipeline (preprocessing fitted on the original training partition + model)
is applied unchanged; nothing is refitted. Compatibility is checked first and
incompatible inputs are rejected with explicit reasons.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ftad.core.models import get_spec, scores_from

PCA_MEAN_SHIFT = 0.25  # in training standard deviations
PCA_STD_RATIO = (0.67, 1.5)


def load_model(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError("The model artifact for this run does not exist.")
    return joblib.load(path)


def check_compatibility(meta: dict, df: pd.DataFrame, dataset_sha: str) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    info: list[str] = []
    features = meta["features"]
    cols = set(map(str, df.columns))
    missing = [f for f in features if f not in cols]
    if missing:
        errors.append(f"Missing required feature columns ({len(missing)}): " + ", ".join(missing[:15]))
    unexpected = sorted(cols - set(features))
    if unexpected:
        info.append(f"{len(unexpected)} column(s) are not used by the model and will be ignored: " + ", ".join(unexpected[:15]))
    for f in meta["numeric_features"]:
        if f in cols and not pd.api.types.is_numeric_dtype(df[f]):
            errors.append(f"Feature '{f}' must be numeric (found {df[f].dtype}).")
    for f in meta["categorical_features"]:
        if f in cols:
            seen = set(meta["categorical_levels"].get(f, []))
            new = set(map(str, df[f].dropna().unique())) - seen
            if new:
                warnings.append(f"Categorical feature '{f}' has {len(new)} level(s) unseen in training; they are encoded as unknown.")
    present = [f for f in features if f in cols]
    if present:
        n_missing = int(df[present].isna().sum().sum())
        if n_missing:
            warnings.append(f"{n_missing:,} missing feature values will be imputed with training-fitted statistics.")

    drift = []
    pca = [f for f in meta.get("pca_features", []) if f in cols and pd.api.types.is_numeric_dtype(df[f])]
    if pca:
        for f in pca:
            st = meta["training_stats"].get(f)
            if not st or not st["std"]:
                continue
            v = df[f].astype(float)
            shift = abs(float(v.mean()) - st["mean"]) / st["std"]
            ratio = float(v.std()) / st["std"] if len(v) > 1 else 1.0
            flagged = shift > PCA_MEAN_SHIFT or not (PCA_STD_RATIO[0] <= ratio <= PCA_STD_RATIO[1])
            drift.append({"feature": f, "mean_shift_sd": shift, "std_ratio": ratio, "flagged": flagged})
        n_flag = sum(d["flagged"] for d in drift)
        warnings.append(
            f"The model uses {len(pca)} precomputed PCA components. Matching column names do NOT prove that the new data "
            "was projected with the same PCA transformation; confirm the provenance before trusting the scores."
        )
        if n_flag:
            warnings.append(f"{n_flag} PCA component(s) show a distribution shift versus training (mean shift > {PCA_MEAN_SHIFT} SD or std ratio outside {PCA_STD_RATIO}).")
    if dataset_sha == meta.get("dataset_sha256"):
        warnings.append("This is the dataset the model was trained on; scores for training rows are in-sample and optimistic.")
    status = "incompatible" if errors else ("warnings" if warnings else "compatible")
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "info": info,
        "pca_drift": drift,
        "model": {k: meta.get(k) for k in ("algorithm_name", "score_type", "threshold", "threshold_basis", "schema_hash", "dataset_sha256")},
        "n_features": len(features),
    }


def score_frame(bundle: dict, df: pd.DataFrame) -> dict:
    meta = bundle["meta"]
    spec = get_spec(meta["algorithm"])
    X = df[meta["features"]].copy()
    for c in meta["categorical_features"]:
        X[c] = X[c].astype("string").astype(object)
    scores = scores_from(bundle["pipeline"], X, spec)
    thr = meta["threshold"]
    out = {"scores": scores, "predicted": (scores >= thr).astype(np.int8), "threshold": thr}
    if meta["score_type"] == "probability":
        out["model_probability"] = scores
    if bundle.get("calibrator") is not None:
        out["calibrated_probability"] = bundle["calibrator"].predict_proba(scores.reshape(-1, 1))[:, 1]
    return out
