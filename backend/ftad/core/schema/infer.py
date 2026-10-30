"""Automatic column profiling and role suggestions.

Suggestions are heuristics only; the user confirms or edits every role in the UI.
Domain modules (e.g. ``ftad.modules.fraud``) can register hint providers that refine
the generic suggestion without the core knowing about any specific dataset.
"""

from __future__ import annotations

import re
from typing import Callable

import numpy as np
import pandas as pd

from ftad.core.schema.config import SchemaConfig, label_values

TARGET_NAME = re.compile(r"^(class|label|target|y|is_?fraud|fraud(_?flag)?|anomaly|is_?anomaly|outcome|default|churn|flag)$", re.I)
PCA_NAME = re.compile(r"^(v|pc|pca|comp|component)[_\-]?\d{1,3}$", re.I)
TIME_NAME = re.compile(r"(^|_)(time|timestamp|date|datetime|ts|elapsed|step|event_?time)($|_)", re.I)
ID_NAME = re.compile(r"(^id$|_id$|^id_|uuid|guid|^transaction(_?id)?$|^txn(_?id)?$|^unnamed: ?0$|^index$)", re.I)
GROUP_NAME = re.compile(r"(customer|user|account|card|client|merchant|entity|device|member|cust)", re.I)
AMOUNT_NAME = re.compile(r"(^|_)(amount|amt|value|sum|price|transaction_?amount)($|_)", re.I)
POSITIVE_WORDS = ("1", "true", "yes", "fraud", "anomaly", "positive", "bad", "default")

HintProvider = Callable[[pd.DataFrame, list[dict], SchemaConfig], SchemaConfig]
_HINT_PROVIDERS: list[HintProvider] = []


def register_hint_provider(fn: HintProvider) -> HintProvider:
    _HINT_PROVIDERS.append(fn)
    return fn


def _kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    non_null = series.dropna()
    if non_null.empty:
        return "categorical"
    sample = non_null.astype(str).head(500)
    if sample.str.contains(r"[-/:]", regex=True).mean() > 0.9:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed.notna().mean() > 0.9:
            return "datetime"
    return "categorical"


def profile_columns(df: pd.DataFrame) -> list[dict]:
    n = len(df)
    profiles = []
    for col in df.columns:
        s = df[col]
        kind = _kind(s)
        n_missing = int(s.isna().sum())
        n_unique = int(s.nunique(dropna=True))
        p: dict = {
            "name": str(col),
            "dtype": str(s.dtype),
            "kind": kind,
            "n_missing": n_missing,
            "missing_pct": n_missing / n if n else 0.0,
            "n_unique": n_unique,
            "unique_ratio": n_unique / max(1, n - n_missing),
            "is_constant": n_unique <= 1,
            "sample_values": [str(v) for v in s.dropna().unique()[:5]],
        }
        if kind == "numeric":
            vals = s.dropna().astype(float)
            p["min"] = float(vals.min()) if len(vals) else None
            p["max"] = float(vals.max()) if len(vals) else None
            p["mean"] = float(vals.mean()) if len(vals) else None
            p["std"] = float(vals.std()) if len(vals) > 1 else None
            p["is_integer"] = bool(len(vals) and np.all(np.mod(vals.to_numpy(), 1) == 0))
            p["monotonic_increasing"] = bool(s.is_monotonic_increasing)
        name = str(col)
        p["potential_target"] = bool(
            (2 <= n_unique <= 10 and TARGET_NAME.match(name)) or (n_unique == 2 and kind in ("numeric", "boolean", "categorical"))
        )
        p["target_name_match"] = bool(TARGET_NAME.match(name))
        p["potential_pca"] = bool(kind == "numeric" and PCA_NAME.match(name))
        p["potential_time"] = bool(kind == "datetime" or (TIME_NAME.search(name) and kind == "numeric"))
        p["potential_amount"] = bool(kind == "numeric" and AMOUNT_NAME.search(name))
        id_like_values = kind in ("numeric", "categorical") and p["unique_ratio"] > 0.98 and (
            kind == "categorical" or p.get("is_integer", False)
        )
        # An entity key (customer_id, card_id, ...) repeats across rows; a record identifier is (near) unique.
        p["potential_group"] = bool(GROUP_NAME.search(name) and 1 < n_unique and p["unique_ratio"] < 0.9)
        p["potential_identifier"] = bool(
            (ID_NAME.search(name) and not p["potential_group"]) or (id_like_values and not p["potential_time"] and n > 50)
        )
        profiles.append(p)
    return profiles


def _suggest_positive(series: pd.Series) -> str | None:
    values = label_values(series)
    if len(values) < 2:
        return None
    lowered = {v.lower(): v for v in values}
    for word in POSITIVE_WORDS:
        if word in lowered:
            return lowered[word]
    counts = series.astype(str).value_counts()
    return str(counts.index[-1])  # minority class


def suggest_schema(df: pd.DataFrame, profiles: list[dict] | None = None) -> SchemaConfig:
    profiles = profiles or profile_columns(df)
    roles: dict[str, str] = {}
    notes: list[str] = []

    target_candidates = [p for p in profiles if p["potential_target"]]
    target_candidates.sort(key=lambda p: (not p["target_name_match"], p["n_unique"]))
    target = target_candidates[0]["name"] if target_candidates else None
    if target and not target_candidates[0]["target_name_match"]:
        notes.append(f"'{target}' was suggested as target only because it is binary — please verify.")

    time_col = next((p["name"] for p in profiles if p["potential_time"] and p["name"] != target), None)
    id_col = next(
        (p["name"] for p in profiles if p["potential_identifier"] and p["name"] not in (target, time_col)), None
    )
    group_col = next(
        (p["name"] for p in profiles if p["potential_group"] and p["name"] not in (target, time_col, id_col)), None
    )
    amount_col = next((p["name"] for p in profiles if p["potential_amount"] and p["name"] != target), None)

    for p in profiles:
        name = p["name"]
        if name == target:
            roles[name] = "target"
        elif name == time_col:
            roles[name] = "time"
        elif name == id_col:
            roles[name] = "transaction_id"
        elif name == group_col:
            roles[name] = "group_id"
        elif p["is_constant"]:
            roles[name] = "ignore"
            notes.append(f"'{name}' is constant and was set to Ignore.")
        elif p["potential_identifier"]:
            roles[name] = "ignore"
            notes.append(f"'{name}' looks like an identifier and was set to Ignore.")
        elif p["potential_pca"]:
            roles[name] = "pca"
        elif p["kind"] in ("numeric", "boolean"):
            roles[name] = "numeric"
        elif p["kind"] == "categorical" and p["n_unique"] <= 100:
            roles[name] = "categorical"
        else:
            roles[name] = "ignore"
            if p["kind"] == "categorical":
                notes.append(f"'{name}' has {p['n_unique']} distinct text values (high cardinality) and was set to Ignore.")
            else:
                notes.append(f"'{name}' is an additional date/time column and was set to Ignore.")

    time_kind = None
    if time_col:
        prof = next(p for p in profiles if p["name"] == time_col)
        time_kind = "timestamp" if prof["kind"] == "datetime" else "ordinal"
    cfg = SchemaConfig(
        roles=roles,
        target=target,
        positive_label=_suggest_positive(df[target]) if target else None,
        time_column=time_col,
        time_kind=time_kind,
        amount_column=amount_col,
        id_column=id_col,
        group_column=group_col,
        notes=notes,
    )
    for provider in _HINT_PROVIDERS:
        cfg = provider(df, profiles, cfg)
    return cfg
