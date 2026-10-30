"""Hypothetical business-impact simulator.

All monetary values are *user-supplied assumptions*. Nothing here is observed in the
dataset except, optionally, the transaction amounts of missed positive records.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from ftad.core.evaluation import threshold_sweep


class CostAssumptions(BaseModel):
    missed_positive_cost: float = Field(100.0, ge=0, description="Cost of one missed fraud (false negative)")
    false_positive_cost: float = Field(5.0, ge=0, description="Cost of a false alarm (customer friction, blocked payment)")
    review_cost: float = Field(2.0, ge=0, description="Manual review cost per alert (true or false)")
    use_amount_for_missed: bool = Field(False, description="Use the actual transaction amount of each missed positive instead of a flat cost")


def _counts(y: np.ndarray, s: np.ndarray, t: float, amounts: np.ndarray | None) -> dict:
    pred = s >= t
    out = {"threshold": float(t), "tp": int((pred & (y == 1)).sum()), "fp": int((pred & (y == 0)).sum()), "fn": int((~pred & (y == 1)).sum())}
    out["alerts"] = out["tp"] + out["fp"]
    out["fn_amount"] = float(amounts[~pred & (y == 1)].sum()) if amounts is not None else None
    return out


def _cost(p: dict, a: CostAssumptions, use_amount: bool) -> float:
    fn_cost = p["fn_amount"] if use_amount else p["fn"] * a.missed_positive_cost
    return float(fn_cost + p["fp"] * a.false_positive_cost + (p["tp"] + p["fp"]) * a.review_cost)


def cost_analysis(
    a: CostAssumptions,
    frozen_threshold: float,
    val: tuple[np.ndarray, np.ndarray, np.ndarray | None],
    test: tuple[np.ndarray, np.ndarray, np.ndarray | None],
) -> dict:
    (y_v, s_v, amt_v), (y_t, s_t, amt_t) = val, test
    use_amount = bool(a.use_amount_for_missed and amt_v is not None)
    sweep = threshold_sweep(y_v, s_v, amt_v)["points"]
    curve = []
    for p in sweep:
        p = {**p, "alerts": p["tp"] + p["fp"], "fn_amount": p.get("fn_amount")}
        curve.append({"t": p["t"], "cost": _cost(p, a, use_amount), "alerts": p["alerts"], "tp": p["tp"], "fp": p["fp"], "fn": p["fn"]})
    best = min(curve, key=lambda c: c["cost"]) if curve else None

    def summary(y, s, amt, t):
        c = _counts(y, s, t, amt)
        return c | {"cost": _cost(c, a, use_amount)}

    def no_detection(y, amt):
        return float(amt[y == 1].sum()) if use_amount else float(y.sum() * a.missed_positive_cost)

    return {
        "available": True,
        "assumptions": a.model_dump() | {"use_amount_for_missed": use_amount},
        "disclaimer": "Hypothetical scenario: all costs are user assumptions, not values observed in the dataset.",
        "validation_curve": curve,
        "cost_optimal_threshold_validation": best["t"] if best else None,
        "validation": {
            "frozen_threshold": summary(y_v, s_v, amt_v, frozen_threshold),
            "cost_optimal": summary(y_v, s_v, amt_v, best["t"]) if best else None,
            "no_detection_cost": no_detection(y_v, amt_v),
        },
        "test": {
            "frozen_threshold": summary(y_t, s_t, amt_t, frozen_threshold),
            "cost_optimal_from_validation": summary(y_t, s_t, amt_t, best["t"]) if best else None,
            "no_detection_cost": no_detection(y_t, amt_t),
        },
        "note": "The cost-optimal threshold is selected on validation and then applied unchanged to the test partition.",
    }
