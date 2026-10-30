"""Shared analysis context: a dataset frame bound to a confirmed schema."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from ftad.core.schema import SchemaConfig, encode_target


@dataclass
class AnalysisContext:
    df: pd.DataFrame
    cfg: SchemaConfig

    @cached_property
    def y(self) -> np.ndarray | None:
        if not self.cfg.target or self.cfg.positive_label is None:
            return None
        return encode_target(self.df[self.cfg.target], self.cfg.positive_label)

    @property
    def has_target(self) -> bool:
        return self.y is not None

    @property
    def pos_name(self) -> str:
        return self.cfg.positive_name

    @property
    def neg_name(self) -> str:
        return self.cfg.negative_name

    @cached_property
    def numeric_columns(self) -> list[str]:
        """Numeric columns that are part of the analysis (features + amount + numeric time)."""
        cols = list(self.cfg.numeric_features)
        for extra in (self.cfg.amount_column, self.cfg.time_column):
            if extra and extra not in cols and pd.api.types.is_numeric_dtype(self.df[extra]):
                cols.append(extra)
        return cols

    def numeric(self, col: str) -> np.ndarray:
        return pd.to_numeric(self.df[col], errors="coerce").to_numpy(dtype=float)
