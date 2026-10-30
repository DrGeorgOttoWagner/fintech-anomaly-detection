"""Experiment configuration (fully serialisable → reproducible manifest)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ftad.core.evaluation import ThresholdPolicy
from ftad.core.imbalance import IMBALANCE_STRATEGIES
from ftad.core.models import QUICK_SET, REGISTRY
from ftad.core.preprocessing import PreprocessConfig
from ftad.core.splitting import SplitConfig

Mode = Literal["quick", "full", "hpo"]


class HPOConfig(BaseModel):
    n_iter: int = Field(10, ge=1, le=60)
    cv_folds: int = Field(3, ge=2, le=10)
    scoring: Literal["average_precision", "roc_auc", "f1", "balanced_accuracy"] = "average_precision"
    max_rows: int = Field(80_000, ge=1_000, description="Training rows used for the search (stratified subsample).")


class ExperimentConfig(BaseModel):
    name: str = Field("Experiment", max_length=120)
    mode: Mode = "quick"
    split: SplitConfig = Field(default_factory=SplitConfig)
    features: list[str] | None = None
    preprocessing: PreprocessConfig = Field(default_factory=PreprocessConfig)
    algorithms: list[str] = Field(default_factory=lambda: list(QUICK_SET))
    imbalance: list[str] = Field(default_factory=lambda: ["none"])
    resample_ratio: float = Field(1.0, gt=0.0, le=1.0, description="Minority/majority ratio after resampling")
    hyperparameters: dict[str, dict] = Field(default_factory=dict)
    threshold: ThresholdPolicy = Field(default_factory=ThresholdPolicy)
    hpo: HPOConfig = Field(default_factory=HPOConfig)
    resource_profile: Literal["safe", "extended"] = "safe"
    permutation_importance: bool = True
    calibrate: bool = False
    anomaly_training: Literal["unsupervised", "normal_only"] = "unsupervised"

    @field_validator("algorithms")
    @classmethod
    def _known_algorithms(cls, v: list[str]) -> list[str]:
        unknown = [a for a in v if a not in REGISTRY]
        if unknown:
            raise ValueError("Unknown algorithms: " + ", ".join(unknown))
        if not v:
            raise ValueError("Select at least one algorithm.")
        return list(dict.fromkeys(v))

    @field_validator("imbalance")
    @classmethod
    def _known_imbalance(cls, v: list[str]) -> list[str]:
        unknown = [s for s in v if s not in IMBALANCE_STRATEGIES]
        if unknown:
            raise ValueError("Unknown imbalance strategies: " + ", ".join(unknown))
        if not v:
            raise ValueError("Select at least one imbalance strategy.")
        return list(dict.fromkeys(v))

    @field_validator("hyperparameters")
    @classmethod
    def _plain_values(cls, v: dict[str, dict]) -> dict[str, dict]:
        for algo, params in v.items():
            if algo not in REGISTRY:
                raise ValueError(f"Hyperparameters given for unknown algorithm '{algo}'.")
            for k, val in params.items():
                if not isinstance(val, (int, float, str, bool, type(None), list)):
                    raise ValueError(f"Unsupported value for {algo}.{k}")
                if isinstance(val, list):
                    params[k] = tuple(val)  # e.g. MLP hidden_layer_sizes
        return v
