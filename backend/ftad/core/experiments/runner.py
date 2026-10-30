"""Experiment execution.

Protocol (identical for every model in an experiment):
  1. One split (train / validation / test) is computed from the configured seed and persisted.
  2. Preprocessing + optional resampling + model form one pipeline fitted on training rows only.
  3. Hyperparameter search (HPO mode) uses cross-validation inside the training partition only.
  4. The operating threshold is chosen on validation and frozen.
  5. The test partition is scored once with the frozen threshold.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
import traceback
import warnings
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, TimeSeriesSplit

from ftad.config import Settings
from ftad.core.evaluation import curves, evaluate, select_threshold, threshold_sweep
from ftad.core.experiments.config import ExperimentConfig
from ftad.core.imbalance import make_sampler
from ftad.core.models import BuildContext, ModelSpec, default_threshold, get_spec, scores_from
from ftad.core.preprocessing import build_preprocessor, transformed_feature_names
from ftad.core.schema import SchemaConfig, encode_target
from ftad.core.splitting import Split, make_split, summarize
from ftad.db import Database, dumps, loads, utcnow

TRACKED_PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "xgboost", "lightgbm", "catboost", "joblib", "pyarrow"]
PERMUTATION_MAX_ROWS = 30_000


def package_versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for p in TRACKED_PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def experiment_dir(settings: Settings, exp_id: str) -> Path:
    d = settings.experiments_dir / exp_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def stratified_subsample(idx: np.ndarray, y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Subsample positions ``idx`` to ``n`` rows, preserving class proportions."""
    if len(idx) <= n:
        return idx
    rng = np.random.default_rng(seed)
    yi = y[idx]
    pos, neg = idx[yi == 1], idx[yi == 0]
    n_pos = min(len(pos), max(1, int(round(n * len(pos) / len(idx)))))
    take = np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n - n_pos, replace=False)])
    return np.sort(take)


def compute_split(df: pd.DataFrame, y: np.ndarray, cfg: ExperimentConfig, schema: SchemaConfig) -> tuple[Split, dict]:
    split = make_split(df, y, cfg.split, schema.time_column, schema.group_column)
    summary = summarize(split, y, schema.positive_name)
    h = hashlib.sha256()
    for part in (split.train, split.validation, split.test):
        h.update(part.astype(np.int64).tobytes())
    summary["indices_sha256"] = h.hexdigest()
    return split, summary


class ExperimentRunner:
    def __init__(self, db: Database, settings: Settings, exp_id: str) -> None:
        self.db = db
        self.settings = settings
        self.exp_id = exp_id
        exp = db.one("SELECT * FROM experiments WHERE id=?", (exp_id,))
        if not exp:
            raise KeyError(exp_id)
        self.exp = exp
        self.cfg = ExperimentConfig.model_validate_json(exp["config_json"])
        schema_row = db.one("SELECT * FROM schemas WHERE id=?", (exp["schema_id"],))
        self.schema = SchemaConfig.model_validate_json(schema_row["config_json"])
        self.dataset = db.one("SELECT * FROM datasets WHERE id=?", (exp["dataset_id"],))
        self.dir = experiment_dir(settings, exp_id)
        self.t0 = time.time()

    # -- progress helpers ------------------------------------------------
    def _progress(self, **kw) -> None:
        current = loads(self.db.one("SELECT progress_json FROM experiments WHERE id=?", (self.exp_id,))["progress_json"]) or {}
        current.update(kw)
        current["elapsed_seconds"] = round(time.time() - self.t0, 1)
        current["updated_at"] = utcnow()
        self.db.execute("UPDATE experiments SET progress_json=? WHERE id=?", (dumps(current), self.exp_id))

    def _run_update(self, run_id: str, **fields) -> None:
        sets = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE runs SET {sets} WHERE id=?", (*fields.values(), run_id))

    # -- main -----------------------------------------------------------------
    def run(self) -> None:
        self.db.execute("UPDATE experiments SET status='running', started_at=? WHERE id=?", (utcnow(), self.exp_id))
        try:
            self._progress(stage="loading data")
            df = pd.read_parquet(self.settings.data_dir / self.dataset["frame_path"])
            y = encode_target(df[self.schema.target], self.schema.positive_label).astype(np.int8)
            features = self.cfg.features or self.schema.features
            missing = [f for f in features if f not in df.columns]
            if missing:
                raise ValueError("Features not in dataset: " + ", ".join(missing))
            self.features = features
            self.numeric = [f for f in features if f in self.schema.numeric_features]
            self.categorical = [f for f in features if f in self.schema.categorical_features]
            for c in self.categorical:
                df[c] = df[c].astype("string").astype(object)
            self._progress(stage="splitting")
            split, summary = compute_split(df, y, self.cfg, self.schema)
            if not summary["valid"]:
                raise ValueError("Invalid split: " + " ".join(summary["errors"]))
            np.savez_compressed(self.dir / "split.npz", train=split.train, validation=split.validation, test=split.test)
            self.db.execute("UPDATE experiments SET split_json=? WHERE id=?", (dumps(summary), self.exp_id))
            X = df[features]
            amounts = df[self.schema.amount_column].to_numpy(dtype=float) if self.schema.amount_column else None

            runs = self.db.query("SELECT * FROM runs WHERE experiment_id=? ORDER BY seq", (self.exp_id,))
            pending = [r for r in runs if r["status"] == "pending"]
            for i, run in enumerate(pending):
                self._progress(stage="training", run_index=i + 1, n_runs=len(pending), current_algorithm=get_spec(run["algorithm"]).name, current_imbalance=run["imbalance"], current_run=run["id"])
                self._run_update(run["id"], status="running", started_at=utcnow())
                try:
                    result = self._run_one(run, X, y, split, amounts)
                    self._run_update(run["id"], status="completed", result_json=dumps(result), finished_at=utcnow())
                except Exception as exc:  # one failing model must not kill the experiment
                    self._run_update(run["id"], status="failed", error=f"{type(exc).__name__}: {exc}", finished_at=utcnow())
                    (self.dir / f"{run['id']}.error.txt").write_text(traceback.format_exc())
            failed = self.db.one("SELECT COUNT(*) AS n FROM runs WHERE experiment_id=? AND status='failed'", (self.exp_id,))["n"]
            status = "completed" if not failed else "completed_with_errors"
            self._progress(stage="finished", current_algorithm=None)
            self.db.execute("UPDATE experiments SET status=?, finished_at=? WHERE id=?", (status, utcnow(), self.exp_id))
            self._write_manifest()
        except Exception as exc:
            self.db.execute(
                "UPDATE experiments SET status='failed', error=?, finished_at=? WHERE id=?",
                (f"{type(exc).__name__}: {exc}", utcnow(), self.exp_id),
            )
            (self.dir / "error.txt").write_text(traceback.format_exc())
            raise

    # -- single model run -----------------------------------------------------
    def _run_one(self, run: dict, X: pd.DataFrame, y: np.ndarray, split: Split, amounts: np.ndarray | None) -> dict:
        spec = get_spec(run["algorithm"])
        cfg = self.cfg
        seed = cfg.split.seed
        notes: list[str] = []
        imbalance = run["imbalance"]

        train_idx = split.train
        cap = spec.max_train_rows
        if cap and cfg.resource_profile == "extended":
            cap *= 4
        if cap and len(train_idx) > cap:
            train_idx = stratified_subsample(train_idx, y, cap, seed)
            notes.append(f"Trained on a stratified subsample of {cap:,} of {len(split.train):,} training rows (resource cap for {spec.name}).")
        if spec.kind == "anomaly" and cfg.anomaly_training == "normal_only":
            train_idx = train_idx[y[train_idx] == 0]
            notes.append("Detector fitted on negative-class training records only (semi-supervised novelty detection).")
        y_tr = y[train_idx]
        n_pos, n_neg = int(y_tr.sum()), int(len(y_tr) - y_tr.sum())

        prep, prep_desc = build_preprocessor(
            cfg.preprocessing, self.numeric, self.categorical, self.schema.amount_column,
            needs_scaling=spec.needs_scaling, supervised=spec.kind == "supervised",
        )
        notes += prep_desc["notes"]
        sampler = None
        if imbalance in ("undersample", "oversample", "smote"):
            current = n_pos / max(1, n_neg)
            if cfg.resample_ratio <= current:
                notes.append(f"Resampling skipped: training ratio {current:.4f} already ≥ requested {cfg.resample_ratio}.")
            else:
                sampler = make_sampler(imbalance, cfg.resample_ratio, seed, n_pos)
        params = dict(cfg.hyperparameters.get(spec.key, {}))
        ctx = BuildContext(seed=seed, n_jobs=self.settings.max_workers_per_model, n_pos=n_pos, n_neg=n_neg)
        # imblearn pipelines must be flat: preprocessing steps, then the sampler, then the model.
        self.n_prep = len(prep.steps)
        steps = list(prep.steps) + ([("sampler", sampler)] if sampler is not None else []) + [("model", spec.build(params, ctx, imbalance))]
        pipe = ImbPipeline(steps)

        hpo_result = None
        if cfg.mode == "hpo":
            if spec.kind == "supervised" and spec.search_space:
                hpo_result, best = self._search(pipe, spec, X, y, train_idx)
                pipe.set_params(**{f"model__{k}": v for k, v in best.items()})
                params.update(best)
            else:
                notes.append("No hyperparameter search for this model (unsupervised detector or no search space); defaults used.")

        X_tr = X.iloc[train_idx]
        t = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipe.fit(X_tr, y_tr)
        train_seconds = time.perf_counter() - t
        seen: set[str] = set()
        for w in caught:  # surface e.g. convergence warnings instead of hiding them
            msg = f"{w.category.__name__}: {str(w.message).splitlines()[0][:180]}"
            if w.category.__name__ in ("DeprecationWarning", "FutureWarning") or msg in seen:
                continue
            seen.add(msg)
            if len(seen) <= 3:
                notes.append("Training warning — " + msg)
        self._progress(stage="evaluating")

        def score(idx):
            t0 = time.perf_counter()
            s = scores_from(pipe, X.iloc[idx], spec)
            return s, time.perf_counter() - t0

        s_tr, _ = score(train_idx)
        s_val, val_seconds = score(split.validation)
        s_te, test_seconds = score(split.test)
        y_val, y_te = y[split.validation], y[split.test]
        d_thr = default_threshold(pipe, spec)
        thr, basis = select_threshold(y_val, s_val, cfg.threshold, d_thr)

        results = {
            "train": evaluate(y_tr, s_tr, thr),
            "validation": evaluate(y_val, s_val, thr),
            "validation_default_threshold": evaluate(y_val, s_val, d_thr),
            "test": evaluate(y_te, s_te, thr),
        }
        calibration = None
        calibrator = None
        if cfg.calibrate and 0 < y_val.sum() < len(y_val):
            calibrator = LogisticRegression().fit(s_val.reshape(-1, 1), y_val)
            p_te = calibrator.predict_proba(s_te.reshape(-1, 1))[:, 1]
            calibration = {
                "method": "Platt scaling fitted on validation scores",
                "test_brier": float(brier_score_loss(y_te, p_te)),
                "curve": curves(y_te, p_te, "probability").get("calibration"),
            }
        importance = self._importance(pipe, spec, X, y, split.validation, seed)

        detail = {
            "curves": {"validation": curves(y_val, s_val, spec.score_type), "test": curves(y_te, s_te, spec.score_type)},
            "sweep": {
                "validation": threshold_sweep(y_val, s_val, amounts[split.validation] if amounts is not None else None),
                "test": threshold_sweep(y_te, s_te, amounts[split.test] if amounts is not None else None),
            },
            "importance": importance,
            "hpo": hpo_result,
            "calibration": calibration,
        }
        run_dir = self.dir / run["id"]
        run_dir.mkdir(exist_ok=True)
        (run_dir / "detail.json").write_text(dumps(detail))
        np.savez_compressed(
            run_dir / "scores.npz",
            validation_idx=split.validation, validation_scores=s_val,
            test_idx=split.test, test_scores=s_te,
        )
        train_frame = X.iloc[train_idx]
        model_meta = {
            "algorithm": spec.key,
            "algorithm_name": spec.name,
            "features": self.features,
            "numeric_features": self.numeric,
            "categorical_features": self.categorical,
            "pca_features": [f for f in self.features if f in self.schema.pca_features],
            "feature_dtypes": {c: str(X[c].dtype) for c in self.features},
            "training_stats": {c: {"mean": float(train_frame[c].mean()), "std": float(train_frame[c].std())} for c in self.numeric},
            "categorical_levels": {c: sorted(map(str, train_frame[c].dropna().unique()))[:200] for c in self.categorical},
            "schema_hash": self.exp["schema_hash"],
            "dataset_sha256": self.exp["dataset_sha256"],
            "threshold": thr,
            "threshold_basis": basis,
            "score_type": spec.score_type,
            "positive_name": self.schema.positive_name,
            "negative_name": self.schema.negative_name,
            "versions": package_versions(),
        }
        joblib.dump({"pipeline": pipe, "calibrator": calibrator, "meta": model_meta}, run_dir / "model.joblib", compress=3)

        return {
            "algorithm_name": spec.name,
            "family": spec.family,
            "kind": spec.kind,
            "score_type": spec.score_type,
            "params": _plain(params) | {k: _plain(v) for k, v in spec.default_params.items() if k not in params},
            "imbalance": imbalance,
            "threshold": thr,
            "default_threshold": d_thr,
            "threshold_basis": basis,
            "train_rows_used": int(len(train_idx)),
            "train_rows_available": int(len(split.train)),
            "train_seconds": train_seconds,
            "inference_seconds": {"validation": val_seconds, "test": test_seconds},
            "inference_ms_per_1k": 1000 * test_seconds / max(1, len(split.test)) * 1000,
            "preprocessing": prep_desc,
            "transformed_features": len(transformed_feature_names(pipe[: self.n_prep])) or None,
            "metrics": results,
            "cv": hpo_result["best"] if hpo_result else None,
            "calibrated_test_brier": calibration["test_brier"] if calibration else None,
            "notes": notes,
        }

    def _search(self, pipe, spec: ModelSpec, X, y, train_idx) -> tuple[dict, dict]:
        cfg = self.cfg
        seed = cfg.split.seed
        idx = stratified_subsample(train_idx, y, cfg.hpo.max_rows, seed)
        if cfg.split.method == "chronological":
            cv = TimeSeriesSplit(n_splits=cfg.hpo.cv_folds)
            cv_desc = f"TimeSeriesSplit({cfg.hpo.cv_folds}) on the training partition"
            idx = np.sort(train_idx)[-cfg.hpo.max_rows:] if len(train_idx) > cfg.hpo.max_rows else train_idx
        else:
            cv = StratifiedKFold(n_splits=cfg.hpo.cv_folds, shuffle=True, random_state=seed)
            cv_desc = f"StratifiedKFold({cfg.hpo.cv_folds}, shuffle, seed {seed}) on the training partition"
        dists = {f"model__{k}": v for k, v in spec.search_space.items()}
        search = RandomizedSearchCV(
            pipe, dists, n_iter=cfg.hpo.n_iter, scoring=cfg.hpo.scoring, cv=cv,
            random_state=seed, n_jobs=1, refit=False, error_score=np.nan,
        )
        t = time.perf_counter()
        search.fit(X.iloc[idx], y[idx])
        cvr = search.cv_results_
        order = np.argsort(-np.nan_to_num(cvr["mean_test_score"], nan=-np.inf))
        best_i = int(order[0])
        best = {k.removeprefix("model__"): _plain(v) for k, v in cvr["params"][best_i].items()}
        result = {
            "method": "RandomizedSearchCV",
            "cv": cv_desc,
            "scoring": cfg.hpo.scoring,
            "n_iter": cfg.hpo.n_iter,
            "rows": int(len(idx)),
            "seconds": time.perf_counter() - t,
            "search_space": {k: str(v) for k, v in dists.items()},
            "best": {"params": best, "mean_score": float(cvr["mean_test_score"][best_i]), "std_score": float(cvr["std_test_score"][best_i]), "scoring": cfg.hpo.scoring},
            "candidates": [
                {"params": {k.removeprefix("model__"): _plain(v) for k, v in cvr["params"][i].items()},
                 "mean_score": _plain(cvr["mean_test_score"][i]), "std_score": _plain(cvr["std_test_score"][i])}
                for i in order[:15]
            ],
            "note": "Search used training-partition cross-validation only; validation and test data were not used.",
        }
        return result, best

    def _importance(self, pipe, spec: ModelSpec, X, y, val_idx, seed) -> dict:
        out: dict = {}
        model = pipe.named_steps["model"]
        names = transformed_feature_names(pipe[: self.n_prep])
        native = None
        if hasattr(model, "feature_importances_"):
            native = np.asarray(model.feature_importances_, dtype=float)
            kind = "impurity/gain importance (model-native)"
        elif hasattr(model, "coef_"):
            native = np.ravel(model.coef_).astype(float)
            kind = "coefficient on transformed (scaled) features"
        if native is not None and len(names) == len(native):
            order = np.argsort(-np.abs(native))[:30]
            out["native"] = {"kind": kind, "features": [{"feature": names[i], "value": float(native[i])} for i in order]}
        if self.cfg.permutation_importance and spec.cost != "high" and 0 < y[val_idx].sum():
            idx = stratified_subsample(val_idx, y, PERMUTATION_MAX_ROWS, seed)
            yv = y[idx]
            if yv.sum() > 0:
                def scorer(est, Xs, ys):
                    return average_precision_score(ys, scores_from(est, Xs, spec))

                r = permutation_importance(pipe, X.iloc[idx], yv, scoring=scorer, n_repeats=3, random_state=seed, n_jobs=1)
                order = np.argsort(-r.importances_mean)[:30]
                out["permutation"] = {
                    "kind": f"drop in PR-AUC when a feature is shuffled (validation, {len(idx):,} rows, 3 repeats)",
                    "features": [{"feature": self.features[i], "mean": float(r.importances_mean[i]), "std": float(r.importances_std[i])} for i in order],
                }
        elif self.cfg.permutation_importance and spec.cost == "high":
            out["permutation_skipped"] = "Skipped for high-cost models to bound runtime."
        return out

    def _write_manifest(self) -> None:
        runs = self.db.query("SELECT id, seq, algorithm, imbalance, status, params_json, result_json, error FROM runs WHERE experiment_id=? ORDER BY seq", (self.exp_id,))
        exp = self.db.one("SELECT * FROM experiments WHERE id=?", (self.exp_id,))
        manifest = build_manifest(exp, self.dataset, self.schema, runs)
        (self.dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def build_manifest(exp: dict, dataset: dict, schema: SchemaConfig, runs: list[dict]) -> dict:
    def metrics_of(r):
        res = loads(r["result_json"])
        if not res:
            return None
        return {p: {k: res["metrics"][p].get(k) for k in ("precision", "recall", "f1", "roc_auc", "pr_auc", "confusion")} for p in ("validation", "test")}

    return {
        "ftad_manifest_version": 1,
        "experiment": {k: exp[k] for k in ("id", "name", "mode", "status", "created_at", "started_at", "finished_at")},
        "dataset": {"id": dataset["id"], "file": dataset["original_filename"], "sha256": dataset["sha256"], "rows": dataset["n_rows"], "columns": dataset["n_cols"]},
        "schema": {"hash": exp["schema_hash"], "config": schema.model_dump()},
        "config": loads(exp["config_json"]),
        "split": loads(exp["split_json"]),
        "package_versions": package_versions(),
        "runs": [
            {"id": r["id"], "algorithm": r["algorithm"], "imbalance": r["imbalance"], "status": r["status"], "error": r["error"],
             "threshold": (loads(r["result_json"]) or {}).get("threshold"), "metrics": metrics_of(r)}
            for r in runs
        ],
        "reproduce": "Re-run this configuration on a dataset with the same SHA-256 fingerprint and schema hash; "
        "split indices are deterministic for the recorded seed and are stored in split.npz.",
    }


def _plain(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, tuple):
        return list(v)
    return v
