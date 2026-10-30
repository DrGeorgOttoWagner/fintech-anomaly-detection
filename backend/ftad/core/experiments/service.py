"""Experiment registry operations (create, list, inspect, reproduce)."""

from __future__ import annotations

import json
import uuid

import numpy as np

from ftad.config import Settings
from ftad.core.experiments.config import ExperimentConfig
from ftad.core.models import get_spec
from ftad.db import Database, dumps, loads, utcnow


class ExperimentError(ValueError):
    pass


def plan_runs(cfg: ExperimentConfig) -> list[dict]:
    """Expand algorithms × imbalance strategies; inapplicable combinations are recorded as skipped."""
    planned = []
    for algo in cfg.algorithms:
        spec = get_spec(algo)
        available, reason = spec.availability()
        strategies = cfg.imbalance
        if spec.kind == "anomaly":
            strategies = ["none"]  # anomaly detectors: one run, labels not used for fitting
        for strat in strategies:
            status, why = "pending", None
            if not available:
                status, why = "skipped", f"Unavailable: {reason}"
            else:
                ok, msg = spec.supports_imbalance(strat)
                if not ok:
                    status, why = "skipped", msg
            planned.append({"algorithm": algo, "imbalance": strat, "status": status, "error": why})
    return planned


def create_experiment(db: Database, dataset: dict, schema_row: dict, cfg: ExperimentConfig) -> dict:
    schema = json.loads(schema_row["config_json"])
    if not schema.get("target"):
        raise ExperimentError("The confirmed schema has no target column — supervised evaluation is impossible.")
    allowed = {c for c, r in schema["roles"].items() if r in ("numeric", "categorical", "pca")}
    if cfg.features is not None:
        bad = [f for f in cfg.features if f not in allowed]
        if bad:
            raise ExperimentError("Not schema features: " + ", ".join(bad[:10]))
        if not cfg.features:
            raise ExperimentError("Select at least one feature.")
    if cfg.split.method == "chronological" and not schema.get("time_column"):
        raise ExperimentError("Chronological splitting requires a time column in the schema.")
    if cfg.split.method == "group" and not schema.get("group_column"):
        raise ExperimentError("Group-aware splitting requires a Group / Entity ID column in the schema.")

    exp_id = "exp_" + uuid.uuid4().hex[:10]
    db.execute(
        "INSERT INTO experiments(id,dataset_id,dataset_sha256,schema_id,schema_hash,name,mode,config_json,status,progress_json,created_at)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (exp_id, dataset["id"], dataset["sha256"], schema_row["id"], schema_row["schema_hash"], cfg.name, cfg.mode,
         cfg.model_dump_json(), "created", dumps({"stage": "queued"}), utcnow()),
    )
    for seq, r in enumerate(plan_runs(cfg)):
        db.execute(
            "INSERT INTO runs(id,experiment_id,seq,algorithm,imbalance,status,params_json,error) VALUES(?,?,?,?,?,?,?,?)",
            ("run_" + uuid.uuid4().hex[:10], exp_id, seq, r["algorithm"], r["imbalance"], r["status"],
             dumps(cfg.hyperparameters.get(r["algorithm"], {})), r["error"]),
        )
    return get_experiment(db, exp_id)


def run_summary(r: dict) -> dict:
    res = loads(r.get("result_json"))
    out = {k: r[k] for k in ("id", "seq", "algorithm", "imbalance", "status", "error", "started_at", "finished_at")}
    spec = get_spec(r["algorithm"])
    out.update({"algorithm_name": spec.name, "family": spec.family, "kind": spec.kind, "score_type": spec.score_type})
    if res:
        out.update({k: res.get(k) for k in ("threshold", "default_threshold", "threshold_basis", "train_seconds", "inference_ms_per_1k",
                                            "train_rows_used", "train_rows_available", "notes", "params", "cv", "calibrated_test_brier")})
        out["metrics"] = res["metrics"]
    return out


def get_experiment(db: Database, exp_id: str, with_runs: bool = True) -> dict:
    exp = db.one("SELECT * FROM experiments WHERE id=?", (exp_id,))
    if not exp:
        raise KeyError(exp_id)
    out = {k: v for k, v in exp.items() if k not in ("config_json", "progress_json", "split_json")}
    out["config"] = loads(exp["config_json"])
    out["progress"] = loads(exp["progress_json"])
    out["split"] = loads(exp["split_json"])
    if with_runs:
        out["runs"] = [run_summary(r) for r in db.query("SELECT * FROM runs WHERE experiment_id=? ORDER BY seq", (exp_id,))]
    return out


def list_experiments(db: Database, dataset_id: str | None = None) -> list[dict]:
    sql = "SELECT e.*, (SELECT COUNT(*) FROM runs r WHERE r.experiment_id=e.id) AS n_runs, " \
          "(SELECT COUNT(*) FROM runs r WHERE r.experiment_id=e.id AND r.status='completed') AS n_completed FROM experiments e"
    params: tuple = ()
    if dataset_id:
        sql += " WHERE e.dataset_id=?"
        params = (dataset_id,)
    rows = db.query(sql + " ORDER BY e.created_at DESC", params)
    return [{k: v for k, v in r.items() if k not in ("config_json", "progress_json", "split_json")} | {"progress": loads(r["progress_json"])} for r in rows]


def get_run(db: Database, run_id: str) -> tuple[dict, dict]:
    r = db.one("SELECT * FROM runs WHERE id=?", (run_id,))
    if not r:
        raise KeyError(run_id)
    exp = db.one("SELECT * FROM experiments WHERE id=?", (r["experiment_id"],))
    return r, exp


def run_detail(settings: Settings, db: Database, run_id: str) -> dict:
    r, exp = get_run(db, run_id)
    path = settings.experiments_dir / exp["id"] / run_id / "detail.json"
    detail = json.loads(path.read_text()) if path.exists() else {}
    return run_summary(r) | {"experiment_id": exp["id"], "dataset_id": exp["dataset_id"], "detail": detail}


def run_scores(settings: Settings, db: Database, run_id: str) -> dict:
    r, exp = get_run(db, run_id)
    path = settings.experiments_dir / exp["id"] / run_id / "scores.npz"
    if not path.exists():
        raise KeyError(f"No scores for run {run_id}")
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
