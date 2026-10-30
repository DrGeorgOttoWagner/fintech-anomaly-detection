"""Model registry, metrics, thresholds, leakage prevention, experiments, inference and reports."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from conftest import confirm_suggested_schema, synthetic_mixed, synthetic_transactions, upload, wait_for
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from ftad.core.evaluation import ThresholdPolicy, evaluate, select_threshold, threshold_sweep
from ftad.core.models import SPECS


def test_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.1).astype(int)
    s = rng.random(2000) + y * 0.5
    m = evaluate(y, s, 0.8)
    pred = (s >= 0.8).astype(int)
    assert m["precision"] == pytest.approx(precision_score(y, pred))
    assert m["recall"] == pytest.approx(recall_score(y, pred))
    assert m["f1"] == pytest.approx(f1_score(y, pred))
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, s))
    assert m["pr_auc"] == pytest.approx(average_precision_score(y, s))
    cm = m["confusion"]
    assert cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"] == 2000


def test_undefined_metrics_are_none_not_zero():
    y = np.zeros(50, int)
    m = evaluate(y, np.random.default_rng(1).random(50), 2.0)
    assert m["roc_auc"] is None and m["recall"] is None and m["precision"] is None
    assert m["undefined"]


def test_threshold_policy_on_validation():
    rng = np.random.default_rng(2)
    y = (rng.random(3000) < 0.05).astype(int)
    s = rng.normal(y * 2.0, 1)
    t, basis = select_threshold(y, s, ThresholdPolicy(policy="max_f1"), 0.0)
    sweep = threshold_sweep(y, s, max_points=5000)["points"]
    best = max(sweep, key=lambda p: 2 * p["tp"] / (2 * p["tp"] + p["fp"] + p["fn"]) if p["tp"] else 0)
    assert t == pytest.approx(best["t"]) and "validation" in basis
    t2, _ = select_threshold(y, s, ThresholdPolicy(policy="target_recall", value=0.9), 0.0)
    assert evaluate(y, s, t2)["recall"] >= 0.9
    t3, b3 = select_threshold(y, np.zeros(3000), ThresholdPolicy(), 0.5)
    assert t3 == 0.5 and "constant" in b3


def test_registry_declares_score_types():
    keys = {s.key for s in SPECS}
    assert {"dummy", "logreg", "rf", "xgb", "iforest", "lof", "ocsvm", "lgbm", "catboost", "svm", "mlp", "knn"} <= keys
    for s in SPECS:
        if s.kind == "anomaly":
            assert s.score_type == "anomaly_score"
    models = {m["key"]: m for m in __import__("ftad.core.models", fromlist=["x"]).SPECS and [s.public() for s in SPECS]}
    assert models["svm"]["score_type"] == "decision_score"


def _setup(client, df, name="fixture.csv"):
    ds = upload(client, df, name).json()["dataset"]
    confirm_suggested_schema(client, ds["id"])
    return ds


def _run(client, ds, **cfg):
    body = {"dataset_id": ds["id"], "config": {"name": "test", **cfg}}
    r = client.post("/api/experiments", json=body)
    assert r.status_code == 200, r.text
    exp = wait_for(client, r.json()["id"])
    return exp


def test_quick_benchmark_end_to_end(client, settings):
    df = synthetic_transactions(4000, prevalence=0.04, seed=11)
    ds = _setup(client, df)
    exp = _run(client, ds, mode="quick", permutation_importance=True)
    assert exp["status"] == "completed", exp
    runs = {r["algorithm"]: r for r in exp["runs"]}
    assert set(runs) == {"dummy", "logreg", "rf", "xgb", "iforest"}
    split = exp["split"]["partitions"]
    assert split["train"]["n"] + split["validation"]["n"] + split["test"]["n"] == 4000
    for r in runs.values():
        assert r["status"] == "completed", r
        test_m = r["metrics"]["test"]
        assert test_m["n"] == split["test"]["n"]  # every model evaluated on the same test partition
        assert test_m["n_positive"] == split["test"]["n_positive"]  # original prevalence kept
    assert runs["dummy"]["metrics"]["test"]["roc_auc"] == pytest.approx(0.5)
    assert runs["xgb"]["metrics"]["test"]["pr_auc"] > runs["dummy"]["metrics"]["test"]["pr_auc"]
    assert runs["iforest"]["score_type"] == "anomaly_score"

    # metrics must be reproducible from the stored scores
    rid = runs["logreg"]["id"]
    rec = client.get(f"/api/runs/{rid}/records?partition=test&limit=500").json()
    assert sum(rec["counts"].values()) == split["test"]["n"]
    cm = runs["logreg"]["metrics"]["test"]["confusion"]
    assert rec["counts"] == {"TP": cm["tp"], "FP": cm["fp"], "TN": cm["tn"], "FN": cm["fn"]}
    csv_text = client.get(f"/api/runs/{rid}/records.csv?partition=test&outcome=FN").text
    assert csv_text.count("\n") == cm["fn"] + 1

    detail = client.get(f"/api/runs/{rid}").json()["detail"]
    assert "pr" in detail["curves"]["test"] and detail["sweep"]["validation"]["points"]
    assert "permutation" in detail["importance"]

    # reproducibility: an identical re-run yields identical metrics
    rep = wait_for(client, client.post(f"/api/experiments/{exp['id']}/reproduce").json()["id"])
    rep_runs = {r["algorithm"]: r for r in rep["runs"]}
    for k in ("logreg", "rf", "xgb", "iforest"):
        assert rep_runs[k]["metrics"]["test"]["pr_auc"] == pytest.approx(runs[k]["metrics"]["test"]["pr_auc"])
    assert rep["split"]["indices_sha256"] == exp["split"]["indices_sha256"]
    manifest = client.get(f"/api/experiments/{exp['id']}/manifest").json()
    assert manifest["dataset"]["sha256"] == ds["sha256"] and manifest["package_versions"]["scikit-learn"]

    lb = client.get(f"/api/datasets/{ds['id']}/leaderboard?objective=pr_auc").json()
    assert len(lb["groups"]) == 1 and len(lb["groups"][0]["runs"]) == 10  # same protocol → one group
    costs = client.post(f"/api/runs/{rid}/costs", json={"missed_positive_cost": 100, "false_positive_cost": 5, "review_cost": 1, "use_amount_for_missed": True}).json()
    assert costs["available"] and costs["test"]["frozen_threshold"]["fn"] == cm["fn"]


def test_resampling_only_touches_training_data(client):
    df = synthetic_transactions(3000, prevalence=0.04, seed=12)
    ds = _setup(client, df)
    exp = _run(client, ds, mode="full", algorithms=["logreg", "tree", "gnb"], imbalance=["none", "class_weight", "undersample", "oversample", "smote"],
               permutation_importance=False)
    by = {(r["algorithm"], r["imbalance"]): r for r in exp["runs"]}
    assert by[("gnb", "class_weight")]["status"] == "skipped"  # not silently substituted
    test_n = exp["split"]["partitions"]["test"]["n_positive"]
    for (algo, strat), r in by.items():
        if r["status"] == "completed":
            assert r["metrics"]["test"]["n_positive"] == test_n
            assert r["metrics"]["validation"]["n_positive"] == exp["split"]["partitions"]["validation"]["n_positive"]
    smote = by[("logreg", "smote")]
    assert smote["status"] == "completed"
    assert smote["train_rows_used"] == exp["split"]["partitions"]["train"]["n"]  # resampling happens inside the pipeline


def test_hpo_mode_and_mixed_schema(client):
    df = synthetic_mixed(1500)
    ds = _setup(client, df)
    exp = _run(client, ds, mode="hpo", algorithms=["logreg", "hgb", "iforest"], hpo={"n_iter": 3, "cv_folds": 3},
               split={"method": "group", "train": 0.6, "validation": 0.2, "test": 0.2, "seed": 3}, permutation_importance=False)
    assert exp["status"] == "completed", exp
    lr = next(r for r in exp["runs"] if r["algorithm"] == "logreg")
    assert lr["cv"]["scoring"] == "average_precision" and "C" in lr["cv"]["params"]
    detail = client.get(f"/api/runs/{lr['id']}").json()["detail"]
    assert "training partition" in detail["hpo"]["cv"]


def test_inference_compatibility_and_scoring(client):
    train_df = synthetic_transactions(2500, prevalence=0.05, seed=21)
    ds = _setup(client, train_df)
    exp = _run(client, ds, mode="quick", algorithms=["logreg"], permutation_importance=False, calibrate=True)
    run_id = exp["runs"][0]["id"]

    new = synthetic_transactions(400, prevalence=0.05, seed=22).drop(columns=["Class"])
    nd = upload(client, new, "unlabeled.csv").json()["dataset"]
    chk = client.post("/api/inference/check", json={"run_id": run_id, "dataset_id": nd["id"]}).json()
    assert chk["status"] == "warnings" and any("PCA" in w for w in chk["warnings"])
    assert client.post("/api/inference/run", json={"run_id": run_id, "dataset_id": nd["id"]}).status_code == 409
    res = client.post("/api/inference/run", json={"run_id": run_id, "dataset_id": nd["id"], "acknowledge_warnings": True}).json()
    assert res["n"] == 400 and "calibrated_probability" in res["columns"] and "metric" not in json.dumps(res["top"][0])
    assert client.get(f"/api/inference/{res['prediction_id']}.csv").status_code == 200

    broken = new.drop(columns=["V2"])
    bd = upload(client, broken, "broken.csv").json()["dataset"]
    chk2 = client.post("/api/inference/check", json={"run_id": run_id, "dataset_id": bd["id"]}).json()
    assert chk2["status"] == "incompatible" and "V2" in chk2["errors"][0]
    assert client.post("/api/inference/run", json={"run_id": run_id, "dataset_id": bd["id"], "acknowledge_warnings": True}).status_code == 422


def test_report_generation_and_export(client, settings):
    ds = _setup(client, synthetic_transactions(2500, prevalence=0.05, seed=31))
    exp = _run(client, ds, mode="quick", algorithms=["dummy", "logreg"], permutation_importance=False)
    rep = client.post("/api/reports", json={"dataset_id": ds["id"], "experiment_id": exp["id"]}).json()
    paths = {f["path"] for f in rep["files"]}
    assert {"report.md", "report.html", "manifest.json", "tables/class_distribution.csv", "tables/model_results.csv"} <= paths
    assert any(p.endswith(".svg") for p in paths) and any(p.endswith(".png") for p in paths)
    md = client.get(f"/api/reports/{rep['id']}/files/report.md").text
    for section in ("1. Dataset description", "6. Class imbalance", "10. Comparative results", "12. Limitations", "Analysis coverage"):
        assert section in md
    assert client.get(f"/api/reports/{rep['id']}/files/../../ftad.sqlite3").status_code in (400, 404)
    assert client.get(f"/api/reports/{rep['id']}/files/..%2F..%2Fftad.sqlite3").status_code == 404
    assert client.get(f"/api/reports/{rep['id']}/bundle.zip").status_code == 200
    out = client.post(f"/api/reports/{rep['id']}/export").json()
    assert out["exported_to"].startswith(str(settings.export_dir)) and out["files"] >= 10


def test_no_leakage_preprocessing_fitted_on_train_only(client, settings):
    import joblib

    df = synthetic_transactions(2000, prevalence=0.05, seed=41)
    ds = _setup(client, df)
    exp = _run(client, ds, mode="quick", algorithms=["logreg"], permutation_importance=False)
    run_id = exp["runs"][0]["id"]
    bundle = joblib.load(settings.experiments_dir / exp["id"] / run_id / "model.joblib")
    split = np.load(settings.experiments_dir / exp["id"] / "split.npz")
    scaler = bundle["pipeline"].named_steps["columns"].named_transformers_["num"].named_steps["scale"]
    feats = bundle["meta"]["numeric_features"]
    train_mean = df.iloc[split["train"]][feats].mean().to_numpy()
    full_mean = df[feats].mean().to_numpy()
    assert np.allclose(scaler.mean_, train_mean)
    assert not np.allclose(scaler.mean_, full_mean)


def test_every_registered_algorithm_trains_and_evaluates(client):
    """Each adapter is individually validated: fit, score, threshold, metrics, persistence."""
    ds = _setup(client, synthetic_transactions(2500, prevalence=0.06, seed=51))
    keys = [s.key for s in SPECS]
    exp = _run(client, ds, mode="full", algorithms=keys, imbalance=["none"], permutation_importance=False,
               split={"method": "stratified", "train": 0.6, "validation": 0.2, "test": 0.2, "seed": 5})
    by = {r["algorithm"]: r for r in exp["runs"]}
    assert set(by) == set(keys)
    for key, r in by.items():
        assert r["status"] == "completed", (key, r["error"])
        m = r["metrics"]["test"]
        assert m["roc_auc"] is not None and 0.0 <= m["roc_auc"] <= 1.0, key
        assert sum(m["confusion"].values()) == exp["split"]["partitions"]["test"]["n"], key
    # informative models must beat the majority baseline's PR-AUC on this separable fixture
    base = by["dummy"]["metrics"]["test"]["pr_auc"]
    for key in ("logreg", "rf", "xgb", "lgbm", "catboost", "hgb", "svm", "mlp"):
        assert by[key]["metrics"]["test"]["pr_auc"] > base, key
