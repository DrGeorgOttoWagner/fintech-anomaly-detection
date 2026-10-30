"""Fraud-domain schema hints.

Recognises card-transaction layouts such as the public ULB/Kaggle "Credit Card Fraud
Detection" dataset (Time, V1..Vn, Amount, Class) *by structure*, and only refines the
generic suggestion — the user can change every role afterwards.
"""

from __future__ import annotations

import re

import pandas as pd

from ftad.core.schema import SchemaConfig, register_hint_provider

FRAUD_TARGET = re.compile(r"^(class|is_?fraud|fraud(_?flag)?|fraud_?label)$", re.I)


@register_hint_provider
def fraud_hints(df: pd.DataFrame, profiles: list[dict], cfg: SchemaConfig) -> SchemaConfig:
    if not cfg.target or not FRAUD_TARGET.match(cfg.target):
        return cfg
    has_amount = cfg.amount_column is not None
    n_pca = len(cfg.pca_features)
    if not (has_amount or "fraud" in cfg.target.lower()):
        return cfg

    updates: dict = {
        "domain": "fraud",
        "positive_name": "Fraud",
        "negative_name": "Genuine",
        "notes": list(cfg.notes),
    }
    if cfg.time_column and pd.api.types.is_numeric_dtype(df[cfg.time_column]):
        t = df[cfg.time_column]
        mostly_ordered = len(t) < 3 or float((t.diff().dropna() >= 0).mean()) > 0.99
        if t.min() >= 0 and mostly_ordered:
            updates["time_kind"] = "elapsed_seconds"
            updates["notes"].append(
                f"'{cfg.time_column}' is (almost entirely) non-decreasing and starts at {t.min():g}; it is interpreted as elapsed "
                "seconds since the first transaction (not a calendar timestamp)."
            )
    if n_pca:
        updates["notes"].append(
            f"{n_pca} columns match an anonymised component pattern and were marked as Existing PCA features. "
            "Their business meaning is unknown and must not be interpreted."
        )
    updates["notes"].append("Fraud-transaction layout detected: positive label is treated as 'Fraud'.")
    return cfg.model_copy(update=updates)
