"""Universal schema configuration: maps dataset columns to analytical roles."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

Role = Literal["target", "numeric", "categorical", "pca", "time", "transaction_id", "group_id", "ignore"]
FEATURE_ROLES = ("numeric", "categorical", "pca")
TimeKind = Literal["elapsed_seconds", "timestamp", "ordinal"]


class SchemaConfig(BaseModel):
    roles: dict[str, Role]
    target: str | None = None
    positive_label: str | None = None
    positive_name: str = "Positive"
    negative_name: str = "Negative"
    time_column: str | None = None
    time_kind: TimeKind | None = None
    amount_column: str | None = None
    id_column: str | None = None
    group_column: str | None = None
    domain: Literal["fraud", "generic"] = "generic"
    notes: list[str] = Field(default_factory=list)

    # -- derived views ------------------------------------------------------
    def columns_with(self, *roles: str) -> list[str]:
        return [c for c, r in self.roles.items() if r in roles]

    @property
    def features(self) -> list[str]:
        return self.columns_with(*FEATURE_ROLES)

    @property
    def numeric_features(self) -> list[str]:
        return self.columns_with("numeric", "pca")

    @property
    def categorical_features(self) -> list[str]:
        return self.columns_with("categorical")

    @property
    def pca_features(self) -> list[str]:
        return self.columns_with("pca")

    def schema_hash(self) -> str:
        payload = self.model_dump(exclude={"notes"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


class SchemaValidation(BaseModel):
    valid: bool
    errors: list[str]
    warnings: list[str]


def encode_target(series: pd.Series, positive_label: str) -> np.ndarray:
    """Binary target: 1 where the value equals the configured positive label, else 0.

    Multi-class targets are binarised as *positive label vs. rest* — this is reported
    explicitly in the schema validation warnings.
    """
    if pd.api.types.is_bool_dtype(series):
        pos = str(positive_label).strip().lower() in ("true", "1", "yes")
        return (series.astype(bool) == pos).to_numpy(dtype=np.int8)
    if pd.api.types.is_numeric_dtype(series):
        try:
            value = float(positive_label)
        except (TypeError, ValueError):
            return np.zeros(len(series), dtype=np.int8)
        return (series.astype(float) == value).to_numpy(dtype=np.int8)
    return (series.astype(str).str.strip() == str(positive_label).strip()).to_numpy(dtype=np.int8)


def label_values(series: pd.Series, limit: int = 50) -> list[str]:
    values = series.dropna().unique()[:limit]
    out = []
    for v in values:
        if isinstance(v, (float, np.floating)) and float(v).is_integer():
            out.append(str(int(v)))
        else:
            out.append(str(v))
    return sorted(set(out), key=lambda s: (len(s), s))


def validate_schema(cfg: SchemaConfig, df: pd.DataFrame) -> SchemaValidation:
    errors: list[str] = []
    warnings: list[str] = []
    cols = set(map(str, df.columns))
    unknown = [c for c in cfg.roles if c not in cols]
    if unknown:
        errors.append("Unknown columns in schema: " + ", ".join(unknown[:10]))
    missing = [c for c in cols if c not in cfg.roles]
    if missing:
        errors.append("Columns without a role: " + ", ".join(sorted(missing)[:10]))

    targets = cfg.columns_with("target")
    if len(targets) > 1:
        errors.append("Only one column can have the Target role.")
    if cfg.target and cfg.roles.get(cfg.target) != "target":
        errors.append(f"Target '{cfg.target}' must have the Target role.")
    if targets and not cfg.target:
        errors.append("A Target role is assigned but no target column is selected.")

    if cfg.target and cfg.target in cols:
        series = df[cfg.target]
        if series.isna().any():
            errors.append(f"Target '{cfg.target}' contains {int(series.isna().sum())} missing values.")
        values = label_values(series, limit=1000)
        if len(values) < 2:
            errors.append(f"Target '{cfg.target}' has fewer than two distinct values.")
        if cfg.positive_label is None:
            errors.append("Select the positive (anomaly) label of the target.")
        elif cfg.positive_label not in values:
            errors.append(f"Positive label '{cfg.positive_label}' does not occur in '{cfg.target}'.")
        else:
            y = encode_target(series.dropna(), cfg.positive_label)
            n_pos = int(y.sum())
            if n_pos == 0 or n_pos == len(y):
                errors.append("The positive label must leave both classes non-empty.")
            elif n_pos < 10:
                warnings.append(f"Only {n_pos} positive records — any evaluation will be very unstable.")
        if len(values) > 2:
            warnings.append(
                f"Target has {len(values)} distinct values; it is binarised as '{cfg.positive_label}' vs. rest."
            )
    elif not cfg.target:
        warnings.append("No target selected — EDA is limited to unsupervised summaries and no models can be trained.")

    if not cfg.features:
        errors.append("Select at least one feature column (Numeric, Categorical or Existing PCA).")
    for col in cfg.numeric_features:
        if col in cols and not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            errors.append(f"Column '{col}' is not numeric and cannot have the Numeric/PCA role.")
    for key, role in (("time_column", "time"), ("id_column", "transaction_id"), ("group_column", "group_id")):
        col = getattr(cfg, key)
        if col and cfg.roles.get(col) != role:
            errors.append(f"'{col}' is selected as {key.replace('_', ' ')} but does not have the matching role.")
    if cfg.time_column and not cfg.time_kind:
        errors.append("Select how the time column should be interpreted.")
    if cfg.time_column and cfg.time_kind == "elapsed_seconds" and cfg.time_column in cols:
        if not pd.api.types.is_numeric_dtype(df[cfg.time_column]):
            errors.append("Elapsed-seconds time columns must be numeric.")
    if cfg.amount_column:
        if cfg.amount_column not in cols:
            errors.append(f"Amount column '{cfg.amount_column}' does not exist.")
        elif not pd.api.types.is_numeric_dtype(df[cfg.amount_column]):
            errors.append("The amount/value column must be numeric.")
    for col in cfg.columns_with("transaction_id", "group_id"):
        if col in cfg.features:
            errors.append(f"Identifier '{col}' cannot also be a feature.")
    if cfg.target in cfg.features:
        errors.append("The target cannot be used as a feature.")
    return SchemaValidation(valid=not errors, errors=errors, warnings=warnings)
