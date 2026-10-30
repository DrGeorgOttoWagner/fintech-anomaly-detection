"""Model registry: standardised adapters around established library implementations.

Each ``ModelSpec`` declares how the estimator is built, what kind of score it emits,
whether it needs feature scaling, how (if at all) it supports class weighting, and the
resource caps that keep training bounded on large datasets. Nothing is substituted
silently: an unavailable library or an inapplicable strategy is reported as such.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
from scipy.stats import loguniform, randint, uniform

ScoreMethod = Literal["predict_proba", "decision_function", "anomaly"]
WeightSupport = Literal["class_weight", "scale_pos_weight", "auto_class_weights", None]


@dataclass
class BuildContext:
    seed: int
    n_jobs: int
    n_pos: int
    n_neg: int


@dataclass
class ModelSpec:
    key: str
    name: str
    family: str
    kind: Literal["supervised", "anomaly"]
    library: str
    factory: Callable[[dict, BuildContext], Any]
    default_params: dict = field(default_factory=dict)
    search_space: dict = field(default_factory=dict)
    score_method: ScoreMethod = "predict_proba"
    needs_scaling: bool = False
    weight_support: WeightSupport = None
    max_train_rows: int | None = None
    cost: Literal["low", "medium", "high"] = "low"
    quick: bool = False
    requires: tuple[str, ...] = ()
    notes: str = ""

    @property
    def score_type(self) -> str:
        return {"predict_proba": "probability", "decision_function": "decision_score", "anomaly": "anomaly_score"}[self.score_method]

    def availability(self) -> tuple[bool, str | None]:
        for mod in self.requires:
            try:
                importlib.import_module(mod)
            except Exception as exc:  # ImportError or a broken native library
                return False, f"{mod} could not be imported: {exc.__class__.__name__}: {str(exc).splitlines()[0][:160]}"
        return True, None

    def supports_imbalance(self, strategy: str) -> tuple[bool, str | None]:
        if strategy == "none":
            return True, None
        if self.kind == "anomaly":
            return False, "Unsupervised detectors do not use labels during training; imbalance treatment does not apply."
        if strategy == "class_weight" and self.weight_support is None:
            return False, f"{self.name} has no class-weighting option."
        return True, None

    def build(self, params: dict, ctx: BuildContext, imbalance: str) -> Any:
        merged = {**self.default_params, **(params or {})}
        if imbalance == "class_weight":
            if self.weight_support == "class_weight":
                merged["class_weight"] = "balanced"
            elif self.weight_support == "scale_pos_weight":
                merged["scale_pos_weight"] = ctx.n_neg / max(1, ctx.n_pos)
            elif self.weight_support == "auto_class_weights":
                merged["auto_class_weights"] = "Balanced"
        return self.factory(merged, ctx)

    def public(self) -> dict:
        ok, reason = self.availability()
        return {
            "key": self.key,
            "name": self.name,
            "family": self.family,
            "kind": self.kind,
            "library": self.library,
            "score_type": self.score_type,
            "needs_scaling": self.needs_scaling,
            "class_weight": self.weight_support is not None,
            "max_train_rows": self.max_train_rows,
            "cost": self.cost,
            "quick": self.quick,
            "hpo": bool(self.search_space),
            "default_params": self.default_params,
            "search_space": {k: _describe_dist(v) for k, v in self.search_space.items()},
            "available": ok,
            "unavailable_reason": reason,
            "notes": self.notes,
        }


def _describe_dist(d) -> str:
    if isinstance(d, list):
        return "one of " + ", ".join(map(str, d))
    name = getattr(getattr(d, "dist", None), "name", type(d).__name__)
    args = getattr(d, "args", ())
    return f"{name}{tuple(round(a, 6) if isinstance(a, float) else a for a in args)}"


# ---------------------------------------------------------------------------
# Factories (imports are local so a missing optional library cannot break the API)
# ---------------------------------------------------------------------------
def _dummy(p, c):
    from sklearn.dummy import DummyClassifier

    return DummyClassifier(**p)


def _logreg(p, c):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(random_state=c.seed, **p)


def _ridge(p, c):
    from sklearn.linear_model import RidgeClassifier

    return RidgeClassifier(**p)


def _sgd(p, c):
    from sklearn.linear_model import SGDClassifier

    return SGDClassifier(random_state=c.seed, **p)


def _lda(p, c):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    return LinearDiscriminantAnalysis(**p)


def _tree(p, c):
    from sklearn.tree import DecisionTreeClassifier

    return DecisionTreeClassifier(random_state=c.seed, **p)


def _rf(p, c):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(random_state=c.seed, n_jobs=c.n_jobs, **p)


def _et(p, c):
    from sklearn.ensemble import ExtraTreesClassifier

    return ExtraTreesClassifier(random_state=c.seed, n_jobs=c.n_jobs, **p)


def _gb(p, c):
    from sklearn.ensemble import GradientBoostingClassifier

    return GradientBoostingClassifier(random_state=c.seed, **p)


def _hgb(p, c):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(random_state=c.seed, **p)


def _xgb(p, c):
    from xgboost import XGBClassifier

    return XGBClassifier(random_state=c.seed, n_jobs=c.n_jobs, **p)


def _lgbm(p, c):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(random_state=c.seed, n_jobs=c.n_jobs, **p)


def _cat(p, c):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(random_seed=c.seed, thread_count=c.n_jobs, **p)


def _knn(p, c):
    from sklearn.neighbors import KNeighborsClassifier

    return KNeighborsClassifier(n_jobs=c.n_jobs, **p)


def _gnb(p, c):
    from sklearn.naive_bayes import GaussianNB

    return GaussianNB(**p)


def _svc(p, c):
    from sklearn.svm import SVC

    return SVC(random_state=c.seed, **p)


def _mlp(p, c):
    from sklearn.neural_network import MLPClassifier

    return MLPClassifier(random_state=c.seed, **p)


def _iforest(p, c):
    from sklearn.ensemble import IsolationForest

    return IsolationForest(random_state=c.seed, n_jobs=c.n_jobs, **p)


def _lof(p, c):
    from sklearn.neighbors import LocalOutlierFactor

    return LocalOutlierFactor(novelty=True, n_jobs=c.n_jobs, **p)


def _ocsvm(p, c):
    from sklearn.svm import OneClassSVM

    return OneClassSVM(**p)


SPECS: list[ModelSpec] = [
    ModelSpec("dummy", "Dummy (majority class)", "baseline", "supervised", "scikit-learn", _dummy,
              {"strategy": "most_frequent"}, quick=True,
              notes="Always predicts the majority class; the reference point every model must beat."),
    ModelSpec("logreg", "Logistic Regression", "linear", "supervised", "scikit-learn", _logreg,
              {"max_iter": 2000, "C": 1.0}, {"C": loguniform(1e-3, 1e2)}, needs_scaling=True,
              weight_support="class_weight", quick=True),
    ModelSpec("ridge", "Ridge Classifier", "linear", "supervised", "scikit-learn", _ridge,
              {"alpha": 1.0}, {"alpha": loguniform(1e-3, 1e2)}, score_method="decision_function",
              needs_scaling=True, weight_support="class_weight",
              notes="Emits an uncalibrated decision score, not a probability."),
    ModelSpec("sgd", "SGD Classifier (log-loss)", "linear", "supervised", "scikit-learn", _sgd,
              {"loss": "log_loss", "alpha": 1e-4, "max_iter": 50, "tol": 1e-4},
              {"alpha": loguniform(1e-6, 1e-2), "penalty": ["l2", "l1", "elasticnet"]},
              needs_scaling=True, weight_support="class_weight"),
    ModelSpec("lda", "Linear Discriminant Analysis", "linear", "supervised", "scikit-learn", _lda,
              {"solver": "svd"}, needs_scaling=False),
    ModelSpec("tree", "Decision Tree", "tree", "supervised", "scikit-learn", _tree,
              {"max_depth": 8, "min_samples_leaf": 5},
              {"max_depth": randint(3, 20), "min_samples_leaf": randint(1, 50)}, weight_support="class_weight"),
    ModelSpec("rf", "Random Forest", "ensemble", "supervised", "scikit-learn", _rf,
              {"n_estimators": 200, "min_samples_leaf": 1, "max_features": "sqrt"},
              {"n_estimators": randint(100, 400), "max_depth": [None, 8, 12, 20], "min_samples_leaf": randint(1, 10)},
              weight_support="class_weight", cost="medium", quick=True),
    ModelSpec("et", "Extra Trees", "ensemble", "supervised", "scikit-learn", _et,
              {"n_estimators": 200, "max_features": "sqrt"},
              {"n_estimators": randint(100, 400), "max_depth": [None, 8, 12, 20], "min_samples_leaf": randint(1, 10)},
              weight_support="class_weight", cost="medium"),
    ModelSpec("gb", "Gradient Boosting", "boosting", "supervised", "scikit-learn", _gb,
              {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 3, "subsample": 0.8},
              {"n_estimators": randint(80, 250), "learning_rate": loguniform(0.02, 0.3), "max_depth": randint(2, 5)},
              max_train_rows=120_000, cost="high",
              notes="Classic (non-histogram) boosting is single-threaded; training rows are capped."),
    ModelSpec("hgb", "HistGradientBoosting", "boosting", "supervised", "scikit-learn", _hgb,
              {"max_iter": 300, "learning_rate": 0.1, "early_stopping": False},
              {"max_iter": randint(100, 500), "learning_rate": loguniform(0.02, 0.3), "max_leaf_nodes": randint(15, 63)},
              weight_support="class_weight"),
    ModelSpec("xgb", "XGBoost", "boosting", "supervised", "xgboost", _xgb,
              {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.9, "colsample_bytree": 0.9,
               "tree_method": "hist", "eval_metric": "aucpr"},
              {"n_estimators": randint(100, 500), "max_depth": randint(3, 9), "learning_rate": loguniform(0.02, 0.3),
               "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.6, 0.4)},
              weight_support="scale_pos_weight", quick=True, requires=("xgboost",)),
    ModelSpec("lgbm", "LightGBM", "boosting", "supervised", "lightgbm", _lgbm,
              {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "verbose": -1, "deterministic": True,
               "force_row_wise": True},
              {"n_estimators": randint(100, 500), "learning_rate": loguniform(0.02, 0.3), "num_leaves": randint(15, 127)},
              weight_support="class_weight", requires=("lightgbm",)),
    ModelSpec("catboost", "CatBoost", "boosting", "supervised", "catboost", _cat,
              {"iterations": 300, "depth": 6, "learning_rate": 0.1, "verbose": 0, "allow_writing_files": False},
              {"iterations": randint(150, 500), "depth": randint(4, 9), "learning_rate": loguniform(0.02, 0.3)},
              weight_support="auto_class_weights", cost="medium", requires=("catboost",)),
    ModelSpec("knn", "K-Nearest Neighbors", "neighbors", "supervised", "scikit-learn", _knn,
              {"n_neighbors": 15, "weights": "distance"}, {"n_neighbors": randint(3, 50)},
              needs_scaling=True, max_train_rows=50_000, cost="high",
              notes="Prediction cost grows with training size; training rows are capped."),
    ModelSpec("gnb", "Gaussian Naive Bayes", "bayes", "supervised", "scikit-learn", _gnb,
              {"var_smoothing": 1e-9}, {"var_smoothing": loguniform(1e-12, 1e-3)}),
    ModelSpec("svm", "Support Vector Machine (RBF)", "svm", "supervised", "scikit-learn", _svc,
              {"kernel": "rbf", "C": 1.0, "gamma": "scale"}, {"C": loguniform(1e-2, 1e2)},
              score_method="decision_function", needs_scaling=True, weight_support="class_weight",
              max_train_rows=20_000, cost="high",
              notes="Kernel SVMs scale quadratically; training rows are capped. Emits decision scores, not probabilities."),
    ModelSpec("mlp", "Multi-Layer Perceptron", "neural", "supervised", "scikit-learn", _mlp,
              {"hidden_layer_sizes": (64, 32), "alpha": 1e-4, "max_iter": 200, "early_stopping": True},
              {"alpha": loguniform(1e-6, 1e-2), "hidden_layer_sizes": [(32,), (64, 32), (128, 64)]},
              needs_scaling=True, cost="medium"),
    ModelSpec("iforest", "Isolation Forest", "anomaly", "anomaly", "scikit-learn", _iforest,
              {"n_estimators": 200, "contamination": "auto"}, score_method="anomaly", quick=True,
              notes="Unsupervised: labels are used only for threshold selection on validation and for evaluation."),
    ModelSpec("lof", "Local Outlier Factor", "anomaly", "anomaly", "scikit-learn", _lof,
              {"n_neighbors": 20}, score_method="anomaly", needs_scaling=True, max_train_rows=30_000, cost="high",
              notes="Novelty mode (fit on training data, score unseen data). Training rows are capped."),
    ModelSpec("ocsvm", "One-Class SVM", "anomaly", "anomaly", "scikit-learn", _ocsvm,
              {"kernel": "rbf", "nu": 0.01, "gamma": "scale"}, score_method="anomaly", needs_scaling=True,
              max_train_rows=10_000, cost="high", notes="Training rows are capped (quadratic complexity)."),
]

REGISTRY: dict[str, ModelSpec] = {s.key: s for s in SPECS}
QUICK_SET = [s.key for s in SPECS if s.quick]


def get_spec(key: str) -> ModelSpec:
    if key not in REGISTRY:
        raise KeyError(f"Unknown algorithm '{key}'.")
    return REGISTRY[key]


def scores_from(model, X, spec: ModelSpec) -> np.ndarray:
    """Continuous score where *higher means more likely positive / more anomalous*."""
    if spec.score_method == "predict_proba":
        proba = model.predict_proba(X)
        return proba[:, 1] if proba.shape[1] > 1 else np.zeros(len(proba))
    if spec.score_method == "decision_function":
        return np.asarray(model.decision_function(X), dtype=float)
    return -np.asarray(model.score_samples(X), dtype=float)


def default_threshold(model, spec: ModelSpec) -> float:
    if spec.score_method == "predict_proba":
        return 0.5
    if spec.score_method == "decision_function":
        return 0.0
    est = model[-1] if hasattr(model, "__getitem__") else model
    # Detectors flag a record when decision_function = score_samples - offset_ < 0, i.e. -score_samples > -offset_.
    return float(-np.ravel(getattr(est, "offset_", 0.0))[0])
