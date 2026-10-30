"""Evaluation metrics computed from actual predictions.

Metrics that are undefined for a partition (e.g. ROC-AUC with a single class, precision
with zero alerts) are returned as ``None`` together with a reason, never as 0 or 1.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve

MAX_CURVE_POINTS = 300
MAX_SWEEP_POINTS = 400


class ThresholdPolicy(BaseModel):
    policy: Literal["max_f1", "default", "target_recall", "target_precision", "fixed"] = "max_f1"
    value: float | None = Field(None, description="Recall/precision target or fixed threshold")


def confusion(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _div(a: float, b: float) -> float | None:
    return a / b if b else None


def threshold_metrics(cm: dict) -> dict:
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
    n = tp + fp + tn + fn
    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    specificity = _div(tn, tn + fp)
    f1 = None
    if precision is not None and recall is not None:
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    elif recall is not None and tp + fp == 0:
        f1 = 0.0  # no alerts raised: F1 is 0 when positives exist
    bal = (recall + specificity) / 2 if recall is not None and specificity is not None else None
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = float((tp * tn - fp * fn) / denom) if denom else None
    undefined = []
    if precision is None:
        undefined.append("precision: no positive predictions (zero alerts)")
    if recall is None:
        undefined.append("recall: no positive records in this partition")
    if specificity is None:
        undefined.append("specificity: no negative records in this partition")
    return {
        "accuracy": _div(tp + tn, n),
        "balanced_accuracy": bal,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": mcc,
        "false_positive_rate": _div(fp, fp + tn),
        "alerts": tp + fp,
        "alert_rate": _div(tp + fp, n),
        "undefined": undefined,
    }


def ranking_metrics(y: np.ndarray, scores: np.ndarray) -> dict:
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        reason = "only one class present in this partition"
        return {"roc_auc": None, "pr_auc": None, "ranking_undefined": reason}
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "pr_auc_baseline": n_pos / len(y),  # PR-AUC of a random ranking = prevalence
    }


def evaluate(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(np.int8)
    cm = confusion(y, pred)
    out = {"n": int(len(y)), "n_positive": int(y.sum()), "threshold": float(threshold), "confusion": cm}
    out.update(threshold_metrics(cm))
    out.update(ranking_metrics(y, scores))
    return out


def _downsample(*arrays: np.ndarray, max_points: int = MAX_CURVE_POINTS) -> list[np.ndarray]:
    n = len(arrays[0])
    if n <= max_points:
        return list(arrays)
    idx = np.unique(np.linspace(0, n - 1, max_points).round().astype(int))
    return [a[idx] for a in arrays]


def curves(y: np.ndarray, scores: np.ndarray, score_type: str) -> dict:
    out: dict = {}
    if 0 < y.sum() < len(y):
        fpr, tpr, _ = roc_curve(y, scores)
        fpr, tpr = _downsample(fpr, tpr)
        prec, rec, _ = precision_recall_curve(y, scores)
        rec, prec = _downsample(rec[::-1], prec[::-1])
        out["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        out["pr"] = {"recall": rec.tolist(), "precision": prec.tolist()}
        if score_type == "probability" and np.all((scores >= 0) & (scores <= 1)):
            frac_pos, mean_pred = calibration_curve(y, scores, n_bins=10, strategy="quantile")
            out["calibration"] = {
                "mean_predicted": mean_pred.tolist(),
                "fraction_positive": frac_pos.tolist(),
                "brier": float(brier_score_loss(y, scores)),
                "note": "Quantile bins. Model probabilities are not guaranteed to be calibrated, especially after resampling or class weighting.",
            }
    return out


def threshold_sweep(y: np.ndarray, scores: np.ndarray, amounts: np.ndarray | None = None, max_points: int = MAX_SWEEP_POINTS) -> dict:
    """Confusion counts for a grid of thresholds (unique score quantiles) — powers the threshold studio."""
    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = y[order].astype(np.int64)
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)
    amt_cum = np.cumsum((amounts[order] * y_sorted)) if amounts is not None else None
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    uniq = np.unique(s_sorted)
    if len(uniq) > max_points:
        qs = np.unique(np.quantile(s_sorted, np.linspace(0, 1, max_points)))
        # always include the top tail densely: rare positives live there
        tail = np.unique(s_sorted[: min(len(s_sorted), max_points // 2)])
        cand = np.unique(np.concatenate([qs, tail]))
    else:
        cand = uniq
    rows = []
    total_amt = float(amounts[y == 1].sum()) if amounts is not None else None
    for t in cand[::-1]:
        k = int(np.searchsorted(-s_sorted, -t, side="right"))  # number of scores >= t
        tp = int(tp_cum[k - 1]) if k else 0
        fp = int(fp_cum[k - 1]) if k else 0
        row = {"t": float(t), "tp": tp, "fp": fp, "fn": n_pos - tp, "tn": n_neg - fp}
        if amt_cum is not None:
            caught = float(amt_cum[k - 1]) if k else 0.0
            row["fn_amount"] = total_amt - caught
        rows.append(row)
    return {"n_positive": n_pos, "n_negative": n_neg, "points": rows, "positive_amount_total": total_amt}


def select_threshold(y: np.ndarray, scores: np.ndarray, policy: ThresholdPolicy, default: float) -> tuple[float, str]:
    """Choose the operating threshold on *validation* data according to the policy."""
    if policy.policy == "default":
        return default, "model default threshold"
    if policy.policy == "fixed":
        if policy.value is None:
            raise ValueError("A fixed threshold requires a value.")
        return float(policy.value), "fixed threshold set by the user"
    if y.sum() == 0:
        return default, "validation has no positives — fell back to the model default"
    if np.ptp(scores) == 0:
        return default, "constant scores carry no ranking information — model default threshold used"
    sweep = threshold_sweep(y, scores, max_points=2000)["points"]
    best_t, best_val = default, -1.0
    for p in sweep:
        prec = p["tp"] / (p["tp"] + p["fp"]) if (p["tp"] + p["fp"]) else 0.0
        rec = p["tp"] / (p["tp"] + p["fn"]) if (p["tp"] + p["fn"]) else 0.0
        if policy.policy == "max_f1":
            val = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        elif policy.policy == "target_recall":
            val = prec if rec >= (policy.value or 0.8) else -1.0
        else:  # target_precision
            val = rec if prec >= (policy.value or 0.8) else -1.0
        if val > best_val:
            best_t, best_val = p["t"], val
    if best_val < 0:
        return default, f"no threshold reached the {policy.policy.replace('_', ' ')} of {policy.value} on validation — fell back to the model default"
    label = {"max_f1": "maximises F1 on validation", "target_recall": f"highest precision with recall ≥ {policy.value} on validation",
             "target_precision": f"highest recall with precision ≥ {policy.value} on validation"}[policy.policy]
    return float(best_t), label
