"""Train / validation / test split strategies.

Splits operate on row positions only. They are computed once per experiment, persisted,
and every model in the experiment is trained and evaluated on exactly the same partitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator
from sklearn.model_selection import GroupShuffleSplit, train_test_split

SplitMethod = Literal["stratified", "random", "chronological", "group"]
MIN_TRAIN_POSITIVES = 5
MIN_EVAL_POSITIVES = 1
WARN_EVAL_POSITIVES = 20


class SplitConfig(BaseModel):
    method: str = Field("stratified", description="One of the registered split strategies (see STRATEGIES)")
    train: float = Field(0.70, gt=0, lt=1)
    validation: float = Field(0.15, gt=0, lt=1)
    test: float = Field(0.15, gt=0, lt=1)
    seed: int = 42

    @model_validator(mode="after")
    def _sum_to_one(self) -> "SplitConfig":
        if self.method not in STRATEGIES:
            raise ValueError(f"Unknown split method '{self.method}'. Available: {', '.join(STRATEGIES)}")
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Partition fractions must sum to 100% (got {total * 100:.1f}%).")
        return self


@dataclass
class Split:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    method: str
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


class SplitStrategy(Protocol):
    """Extension point: future walk-forward / purged temporal splits implement this."""

    def split(self, n: int, y: np.ndarray | None, cfg: SplitConfig, *, time: np.ndarray | None, groups: np.ndarray | None) -> Split: ...


class SplitError(ValueError):
    pass


def _stratified(n, y, cfg, *, time=None, groups=None, stratify=True) -> Split:
    idx = np.arange(n)
    strat = y if (stratify and y is not None) else None
    rest_frac = cfg.validation + cfg.test
    try:
        train, rest = train_test_split(idx, test_size=rest_frac, random_state=cfg.seed, stratify=strat)
        strat_rest = y[rest] if strat is not None else None
        val, test = train_test_split(rest, test_size=cfg.test / rest_frac, random_state=cfg.seed, stratify=strat_rest)
    except ValueError as exc:
        raise SplitError(f"Stratified split impossible: {exc}") from exc
    return Split(np.sort(train), np.sort(val), np.sort(test), "stratified" if strat is not None else "random")


def _chronological(n, y, cfg, *, time=None, groups=None) -> Split:
    if time is None:
        raise SplitError("Chronological splitting requires a configured time column.")
    order = np.lexsort((np.arange(n), time))  # stable: ties keep file order
    n_train = int(round(n * cfg.train))
    n_val = int(round(n * cfg.validation))
    train, val, test = order[:n_train], order[n_train : n_train + n_val], order[n_train + n_val :]
    notes = [
        "Time order is preserved: validation follows training and test follows validation.",
        "Class proportions are not stratified and may differ between partitions (temporal drift).",
    ]
    t_train_max = time[train].max()
    t_test_min = time[test].min()
    if t_test_min < t_train_max:
        notes.append("Some timestamps are tied across the boundary; ties are broken by original row order.")
    return Split(train, val, test, "chronological", notes)


def _group(n, y, cfg, *, time=None, groups=None) -> Split:
    if groups is None:
        raise SplitError("Group-aware splitting requires a Group / Entity ID column.")
    idx = np.arange(n)
    rest_frac = cfg.validation + cfg.test
    gss = GroupShuffleSplit(n_splits=1, test_size=rest_frac, random_state=cfg.seed)
    train, rest = next(gss.split(idx, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=cfg.test / rest_frac, random_state=cfg.seed)
    v_rel, t_rel = next(gss2.split(rest, groups=groups[rest]))
    val, test = rest[v_rel], rest[t_rel]
    notes = [
        f"{len(np.unique(groups)):,} entities assigned wholly to one partition; sizes are approximate.",
    ]
    return Split(np.sort(train), np.sort(val), np.sort(test), "group", notes)


STRATEGIES = {
    "stratified": _stratified,
    "random": lambda n, y, cfg, **kw: _stratified(n, y, cfg, stratify=False, **kw),
    "chronological": _chronological,
    "group": _group,
}


def make_split(df: pd.DataFrame, y: np.ndarray | None, cfg: SplitConfig, time_col: str | None, group_col: str | None) -> Split:
    time = None
    if time_col:
        t = df[time_col]
        time = (
            pd.to_datetime(t, errors="coerce", format="mixed").astype("int64").to_numpy()
            if not pd.api.types.is_numeric_dtype(t)
            else t.to_numpy(dtype=float)
        )
    groups = df[group_col].astype(str).to_numpy() if group_col else None
    if cfg.method == "stratified" and y is None:
        raise SplitError("Stratified splitting requires a target.")
    split = STRATEGIES[cfg.method](len(df), y, cfg, time=time, groups=groups)
    _assert_disjoint(split, groups)
    return split


def _assert_disjoint(split: Split, groups: np.ndarray | None) -> None:
    parts = split.as_dict()
    names = list(parts)
    for i in range(3):
        for j in range(i + 1, 3):
            if np.intersect1d(parts[names[i]], parts[names[j]]).size:
                raise SplitError(f"Internal error: {names[i]} and {names[j]} partitions overlap.")
    if groups is not None and split.method == "group":
        for i in range(3):
            for j in range(i + 1, 3):
                if np.intersect1d(groups[parts[names[i]]], groups[parts[names[j]]]).size:
                    raise SplitError(f"Entities appear in both {names[i]} and {names[j]}.")


def summarize(split: Split, y: np.ndarray | None, pos_name: str = "Positive") -> dict:
    """Partition sizes, class distribution and validity checks for the UI preview."""
    total = len(split.train) + len(split.validation) + len(split.test)
    parts = {}
    errors: list[str] = []
    warnings: list[str] = []
    for name, idx in split.as_dict().items():
        entry = {"n": int(len(idx)), "share": len(idx) / total if total else 0}
        if y is not None:
            n_pos = int(y[idx].sum())
            entry.update({"n_positive": n_pos, "n_negative": int(len(idx) - n_pos), "prevalence": n_pos / len(idx) if len(idx) else None})
            if name == "train" and n_pos < MIN_TRAIN_POSITIVES:
                errors.append(f"Training partition has only {n_pos} {pos_name.lower()} records (minimum {MIN_TRAIN_POSITIVES}).")
            if name != "train":
                if n_pos < MIN_EVAL_POSITIVES:
                    errors.append(f"The {name} partition contains no {pos_name.lower()} records — evaluation would be impossible.")
                elif n_pos < WARN_EVAL_POSITIVES:
                    warnings.append(f"The {name} partition contains only {n_pos} {pos_name.lower()} records; metrics will be highly uncertain.")
                if n_pos == len(idx):
                    errors.append(f"The {name} partition contains only {pos_name.lower()} records.")
        if len(idx) == 0:
            errors.append(f"The {name} partition is empty.")
        parts[name] = entry
    return {"method": split.method, "partitions": parts, "notes": split.notes, "errors": errors, "warnings": warnings, "valid": not errors}
