"""Extension points of the generic research engine.

FTAD's core is domain-agnostic. Future work (e.g. a custom PyTorch financial network,
walk-forward evaluation on market data) plugs in through the registries below instead of
modifying the experiment runner. See docs/EXTENDING.md for worked examples.

Registry / hook                         | Where                                   | Contract
----------------------------------------|-----------------------------------------|-----------------------------------------
``register_model(ModelSpec)``           | ftad.core.models.registry               | sklearn-compatible estimator factory +
                                        |                                         | score method + resource caps
``register_split(name, fn)``            | ftad.core.splitting.strategies          | ``SplitStrategy`` protocol → ``Split``
``register_hint_provider(fn)``          | ftad.core.schema.infer                  | refine schema suggestions per domain
``register_evidence_provider(fn)``      | ftad.core.reporting.report              | append evidence sections to reports
``ThresholdPolicy`` / ``evaluate``      | ftad.core.evaluation                    | metrics computed from scores + labels
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from ftad.core.models.registry import REGISTRY, SPECS, ModelSpec
from ftad.core.splitting.strategies import STRATEGIES, Split, SplitConfig, SplitStrategy


@runtime_checkable
class DatasetAdapter(Protocol):
    """Turns an external source (CSV, Parquet, market-data API, ...) into a DataFrame."""

    def load(self) -> pd.DataFrame: ...

    def fingerprint(self) -> str: ...


@runtime_checkable
class ScoringModel(Protocol):
    """Minimal estimator contract the runner relies on (sklearn-compatible).

    A custom neural network only needs ``fit`` and one of ``predict_proba`` /
    ``decision_function`` / ``score_samples`` — typically via a thin sklearn wrapper
    (e.g. skorch, or a hand-written ``BaseEstimator`` subclass around a PyTorch module).
    """

    def fit(self, X: Any, y: Any = None) -> Any: ...


class MetricFn(Protocol):
    def __call__(self, y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict: ...


def register_model(spec: ModelSpec) -> ModelSpec:
    """Add a model adapter to the registry (available in the API/UI on next start)."""
    if spec.key in REGISTRY:
        raise ValueError(f"Model key '{spec.key}' already registered.")
    SPECS.append(spec)
    REGISTRY[spec.key] = spec
    return spec


def register_split(name: str, fn: SplitStrategy) -> None:
    """Add a split strategy, e.g. walk-forward or purged/embargoed temporal splits."""
    if name in STRATEGIES:
        raise ValueError(f"Split strategy '{name}' already registered.")
    STRATEGIES[name] = fn  # type: ignore[assignment]


__all__ = [
    "DatasetAdapter",
    "ScoringModel",
    "MetricFn",
    "ModelSpec",
    "Split",
    "SplitConfig",
    "SplitStrategy",
    "register_model",
    "register_split",
]
