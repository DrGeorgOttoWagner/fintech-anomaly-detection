"""Leakage-safe preprocessing built from scikit-learn transformers.

The returned transformer is *unfitted*; it becomes part of the model pipeline and is
fitted on the training partition only (and on training folds only during CV).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler


class PreprocessConfig(BaseModel):
    numeric_imputation: Literal["median", "mean", "zero"] = "median"
    categorical_imputation: Literal["most_frequent", "constant"] = "most_frequent"
    scaling: Literal["auto", "standard", "robust", "minmax", "none"] = "auto"
    log_amount: bool = False
    encoding: Literal["onehot", "ordinal"] = "onehot"
    max_categories: int = Field(50, ge=2, le=500)
    pca: bool = False
    pca_variance: float = Field(0.95, gt=0.0, le=1.0)
    feature_selection: Literal["none", "variance", "kbest"] = "none"
    k_best: int = Field(20, ge=1)


def log1p_nonnegative(x):
    """Module-level (picklable) log1p transform; negative inputs are clipped to 0."""
    return np.log1p(np.clip(x, 0, None))


def _scaler(name: str):
    return {"standard": StandardScaler(), "robust": RobustScaler(), "minmax": MinMaxScaler()}.get(name)


def resolved_scaling(cfg: PreprocessConfig, needs_scaling: bool) -> str:
    if cfg.scaling != "auto":
        return cfg.scaling
    if cfg.pca:
        return "standard"
    return "standard" if needs_scaling else "none"


def build_preprocessor(
    cfg: PreprocessConfig,
    numeric: list[str],
    categorical: list[str],
    amount_column: str | None,
    *,
    needs_scaling: bool,
    supervised: bool,
) -> tuple[Pipeline, dict]:
    scaling = resolved_scaling(cfg, needs_scaling)

    def imputer():
        if cfg.numeric_imputation == "zero":
            return SimpleImputer(strategy="constant", fill_value=0.0)
        return SimpleImputer(strategy=cfg.numeric_imputation)

    transformers = []
    log_cols = [amount_column] if (cfg.log_amount and amount_column in numeric) else []
    plain_numeric = [c for c in numeric if c not in log_cols]

    def numeric_steps(extra=None):
        steps = [("impute", imputer())]
        if extra is not None:
            steps.append(("log1p", extra))
        sc = _scaler(scaling)
        if sc is not None:
            steps.append(("scale", sc))
        return Pipeline(steps)

    if plain_numeric:
        transformers.append(("num", numeric_steps(), plain_numeric))
    if log_cols:
        log1p = FunctionTransformer(log1p_nonnegative, feature_names_out="one-to-one")
        transformers.append(("num_log", numeric_steps(log1p), log_cols))
    if categorical:
        cat_imp = (
            SimpleImputer(strategy="most_frequent")
            if cfg.categorical_imputation == "most_frequent"
            else SimpleImputer(strategy="constant", fill_value="__missing__")
        )
        enc = (
            OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=cfg.max_categories, sparse_output=False)
            if cfg.encoding == "onehot"
            else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        )
        transformers.append(("cat", Pipeline([("impute", cat_imp), ("encode", enc)]), categorical))

    steps: list = [("columns", ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False))]
    notes: list[str] = []
    if cfg.feature_selection == "variance":
        steps.append(("select", VarianceThreshold(1e-8)))
    elif cfg.feature_selection == "kbest":
        if supervised:
            steps.append(("select", SelectKBest(f_classif, k=min(cfg.k_best, len(numeric) + 10 * len(categorical)))))
        else:
            notes.append("SelectKBest needs labels and was skipped for this unsupervised detector.")
    if cfg.pca:
        steps.append(("pca", PCA(n_components=cfg.pca_variance, svd_solver="full", random_state=0)))
        notes.append(
            f"Application-level PCA retaining {cfg.pca_variance:.0%} variance, fitted on training data only "
            "(distinct from any precomputed PCA columns in the dataset)."
        )
    description = {
        "numeric_columns": len(numeric),
        "categorical_columns": len(categorical),
        "numeric_imputation": cfg.numeric_imputation,
        "scaling": scaling,
        "log1p_columns": log_cols,
        "encoding": cfg.encoding if categorical else None,
        "feature_selection": cfg.feature_selection,
        "pca": cfg.pca,
        "notes": notes,
    }
    return Pipeline(steps), description


def transformed_feature_names(prep: Pipeline) -> list[str]:
    try:
        return [str(n) for n in prep.get_feature_names_out()]
    except Exception:  # some transformer chains cannot report names
        return []
