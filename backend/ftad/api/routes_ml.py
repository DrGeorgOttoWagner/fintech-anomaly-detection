"""Model registry, experiments, runs, leaderboard, record explorer and cost simulator."""

from __future__ import annotations

import csv
import io
import json

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ftad.api.state import AppState
from ftad.core.experiments.config import ExperimentConfig
from ftad.core.experiments.service import (
    ExperimentError,
    create_experiment,
    get_experiment,
    get_run,
    list_experiments,
    run_detail,
    run_scores,
)
from ftad.core.imbalance import IMBALANCE_STRATEGIES
from ftad.core.jsonsafe import clean
from ftad.core.models import QUICK_SET, SPECS
from ftad.core.schema import SchemaConfig, encode_target
from ftad.db import loads
from ftad.modules.fraud.costs import CostAssumptions, cost_analysis

OBJECTIVES = {"pr_auc": "PR-AUC (average precision)", "roc_auc": "ROC-AUC", "f1": "F1 at frozen threshold",
              "recall": "Recall at frozen threshold", "precision": "Precision at frozen threshold",
              "balanced_accuracy": "Balanced accuracy", "mcc": "Matthews correlation"}


class CreateBody(BaseModel):
    dataset_id: str
    config: ExperimentConfig


def build_router(state: AppState) -> APIRouter:
    r = APIRouter(prefix="/api")

    @r.get("/models")
    def models():
        return {"models": [s.public() for s in SPECS], "quick_set": QUICK_SET}

    @r.get("/imbalance-strategies")
    def imbalance():
        return [{"key": k, **v} for k, v in IMBALANCE_STRATEGIES.items()]

    # -- experiments -----------------------------------------------------------
    @r.post("/experiments")
    def create(body: CreateBody):
        dataset = state.dataset(body.dataset_id)
        schema_row, _ = state.require_schema(body.dataset_id)
        try:
            exp = create_experiment(state.db, dataset, schema_row, body.config)
        except ExperimentError as exc:
            raise HTTPException(422, str(exc))
        if not any(run["status"] == "pending" for run in exp["runs"]):
            state.db.execute("UPDATE experiments SET status='failed', error='No runnable model/strategy combination.' WHERE id=?", (exp["id"],))
            return get_experiment(state.db, exp["id"])
        state.jobs.submit(exp["id"])
        return get_experiment(state.db, exp["id"])

    @r.get("/experiments")
    def experiments(dataset_id: str | None = None):
        return list_experiments(state.db, dataset_id)

    @r.get("/experiments/{exp_id}")
    def experiment(exp_id: str):
        try:
            return clean(get_experiment(state.db, exp_id))
        except KeyError:
            raise HTTPException(404, "Experiment not found.")

    @r.post("/experiments/{exp_id}/cancel")
    def cancel(exp_id: str):
        if not state.jobs.cancel(exp_id):
            raise HTTPException(409, "Experiment is not queued or running.")
        return experiment(exp_id)

    @r.post("/experiments/{exp_id}/reproduce")
    def reproduce(exp_id: str):
        exp = experiment(exp_id)
        dataset = state.dataset(exp["dataset_id"])
        if dataset["sha256"] != exp["dataset_sha256"]:
            raise HTTPException(409, "Dataset fingerprint differs from the original experiment.")
        schema_row = state.db.one("SELECT * FROM schemas WHERE id=?", (exp["schema_id"],))
        cfg = ExperimentConfig.model_validate(exp["config"])
        cfg.name = f"{cfg.name} (reproduction of {exp_id})"
        new = create_experiment(state.db, dataset, schema_row, cfg)
        state.jobs.submit(new["id"])
        return get_experiment(state.db, new["id"])

    @r.delete("/experiments/{exp_id}")
    def delete(exp_id: str):
        exp = experiment(exp_id)
        if exp["status"] in ("queued", "running"):
            raise HTTPException(409, "Cancel the experiment before deleting it.")
        state.db.execute("DELETE FROM experiments WHERE id=?", (exp_id,))
        import shutil

        shutil.rmtree(state.settings.experiments_dir / exp_id, ignore_errors=True)
        return {"deleted": exp_id}

    @r.get("/experiments/{exp_id}/manifest")
    def manifest(exp_id: str):
        path = state.settings.experiments_dir / exp_id / "manifest.json"
        if not path.exists():
            raise HTTPException(404, "Manifest is written when the experiment finishes.")
        return json.loads(path.read_text())

    # -- runs ---------------------------------------------------------------------
    @r.get("/runs/{run_id}")
    def run(run_id: str):
        try:
            return clean(run_detail(state.settings, state.db, run_id))
        except KeyError:
            raise HTTPException(404, "Run not found.")

    def _records(run_id: str, partition: str):
        if partition not in ("validation", "test"):
            raise HTTPException(422, "partition must be validation or test")
        try:
            row, exp = get_run(state.db, run_id)
            scores = run_scores(state.settings, state.db, run_id)
        except KeyError:
            raise HTTPException(404, "Run results not found.")
        res = loads(row["result_json"])
        schema = SchemaConfig.model_validate_json(state.db.one("SELECT config_json FROM schemas WHERE id=?", (exp["schema_id"],))["config_json"])
        df = state.store.frame(exp["dataset_id"])
        idx = scores[f"{partition}_idx"]
        s = scores[f"{partition}_scores"]
        y = encode_target(df[schema.target], schema.positive_label)[idx]
        thr = res["threshold"]
        pred = (s >= thr).astype(int)
        outcome = np.where(pred == 1, np.where(y == 1, "TP", "FP"), np.where(y == 1, "FN", "TN"))
        ids = df[schema.id_column].iloc[idx].astype(str).to_numpy() if schema.id_column else None
        return dict(idx=idx, s=s, y=y, pred=pred, outcome=outcome, ids=ids, thr=thr, res=res, schema=schema, exp=exp)

    @r.get("/runs/{run_id}/records")
    def records(run_id: str, partition: str = "test", outcome: str | None = None, sort: str = "score_desc", offset: int = 0, limit: int = 50):
        d = _records(run_id, partition)
        mask = np.ones(len(d["idx"]), bool)
        if outcome:
            wanted = set(outcome.upper().split(","))
            mask = np.isin(d["outcome"], list(wanted))
        sel = np.flatnonzero(mask)
        key = d["s"][sel]
        order = sel[np.argsort(-key if sort == "score_desc" else key, kind="mergesort")]
        page = order[offset : offset + max(1, min(limit, 500))]
        counts = {k: int((d["outcome"] == k).sum()) for k in ("TP", "FP", "TN", "FN")}
        return clean({
            "run_id": run_id,
            "dataset_id": d["exp"]["dataset_id"],
            "partition": partition,
            "threshold": d["thr"],
            "threshold_basis": d["res"]["threshold_basis"],
            "score_type": d["res"]["score_type"],
            "positive_name": d["schema"].positive_name,
            "negative_name": d["schema"].negative_name,
            "id_column": d["schema"].id_column,
            "counts": counts,
            "total": int(len(sel)),
            "offset": offset,
            "records": [
                {"row": int(d["idx"][i]), "id": d["ids"][i] if d["ids"] is not None else None, "actual": int(d["y"][i]),
                 "predicted": int(d["pred"][i]), "score": float(d["s"][i]), "outcome": str(d["outcome"][i]),
                 "correct": bool(d["pred"][i] == d["y"][i])}
                for i in page
            ],
        })

    @r.get("/runs/{run_id}/records.csv")
    def records_csv(run_id: str, partition: str = "test", outcome: str | None = None):
        d = _records(run_id, partition)
        mask = np.isin(d["outcome"], outcome.upper().split(",")) if outcome else np.ones(len(d["idx"]), bool)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["row_index", "record_id", "actual_label", "predicted_label", "score", "score_type", "threshold", "outcome", "correct"])
        for i in np.flatnonzero(mask):
            w.writerow([int(d["idx"][i]), d["ids"][i] if d["ids"] is not None else "", int(d["y"][i]), int(d["pred"][i]),
                        f"{d['s'][i]:.8g}", d["res"]["score_type"], f"{d['thr']:.8g}", d["outcome"][i], bool(d["pred"][i] == d["y"][i])])
        buf.seek(0)
        return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                                 headers={"Content-Disposition": f'attachment; filename="{run_id}_{partition}_records.csv"'})

    # -- leaderboard -------------------------------------------------------------
    @r.get("/datasets/{dataset_id}/leaderboard")
    def leaderboard(dataset_id: str, objective: str = "pr_auc", experiment_id: str | None = None):
        if objective not in OBJECTIVES:
            raise HTTPException(422, f"objective must be one of {', '.join(OBJECTIVES)}")
        state.dataset(dataset_id)
        schema_row = state.schema_row(dataset_id)
        if not schema_row:
            return {"dataset_id": dataset_id, "groups": [], "objective": objective}
        sql = ("SELECT r.*, e.name AS exp_name, e.split_json, e.mode, e.created_at AS exp_created, e.id AS exp_id FROM runs r "
               "JOIN experiments e ON e.id = r.experiment_id WHERE e.dataset_id=? AND e.schema_hash=? AND r.status='completed'")
        params: list = [dataset_id, schema_row["schema_hash"]]
        if experiment_id:
            sql += " AND e.id=?"
            params.append(experiment_id)
        rows = state.db.query(sql + " ORDER BY e.created_at DESC, r.seq", tuple(params))
        groups: dict[str, dict] = {}
        for row in rows:
            split = loads(row["split_json"]) or {}
            key = split.get("indices_sha256", "unknown")
            res = loads(row["result_json"])
            g = groups.setdefault(key, {"protocol_id": key[:12], "split": split, "experiments": {}, "runs": []})
            g["experiments"][row["exp_id"]] = row["exp_name"]
            g["runs"].append({
                "run_id": row["id"], "experiment_id": row["exp_id"], "experiment_name": row["exp_name"], "mode": row["mode"],
                "algorithm": row["algorithm"], "algorithm_name": res["algorithm_name"], "family": res["family"], "kind": res["kind"],
                "imbalance": row["imbalance"], "score_type": res["score_type"], "params": res["params"], "threshold": res["threshold"],
                "threshold_basis": res["threshold_basis"], "train_seconds": res["train_seconds"], "inference_ms_per_1k": res["inference_ms_per_1k"],
                "train_rows_used": res["train_rows_used"], "cv": res.get("cv"),
                "validation": _pick(res["metrics"]["validation"]), "test": _pick(res["metrics"]["test"]),
            })
        out = []
        for g in groups.values():
            ranked = sorted(g["runs"], key=lambda x: -(x["validation"].get(objective) if x["validation"].get(objective) is not None else -1))
            best = ranked[0] if ranked else None
            out.append({**g, "runs": ranked, "experiments": [{"id": k, "name": v} for k, v in g["experiments"].items()],
                        "best_run_id": best["run_id"] if best else None})
        return clean({
            "dataset_id": dataset_id,
            "objective": objective,
            "objective_label": OBJECTIVES[objective],
            "selection_basis": f"Models are ranked by validation {OBJECTIVES[objective]}. The test column is reported for the "
                               "selected model only as a final, single-use estimate; choosing among models by their test score would bias it.",
            "groups": out,
        })

    # -- business impact simulator ---------------------------------------------------
    @r.post("/runs/{run_id}/costs")
    def costs(run_id: str, body: CostAssumptions):
        v = _records(run_id, "validation")
        t = _records(run_id, "test")
        df = state.store.frame(v["exp"]["dataset_id"])
        amt_col = v["schema"].amount_column
        amounts = df[amt_col].to_numpy(dtype=float) if amt_col else None
        pick = lambda d: (d["y"], d["s"], amounts[d["idx"]] if amounts is not None else None)  # noqa: E731
        return clean({"run_id": run_id, "dataset_id": v["exp"]["dataset_id"], "positive_name": v["schema"].positive_name,
                      "amount_column": amt_col, **cost_analysis(body, v["thr"], pick(v), pick(t))})

    return r


def _pick(m: dict) -> dict:
    keys = ("precision", "recall", "f1", "specificity", "accuracy", "balanced_accuracy", "mcc", "roc_auc", "pr_auc", "alerts", "n", "n_positive", "confusion", "pr_auc_baseline")
    return {k: m.get(k) for k in keys}
