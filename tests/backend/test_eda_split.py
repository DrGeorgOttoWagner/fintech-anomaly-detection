"""Descriptive statistics, EDA and splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import confirm_suggested_schema, synthetic_mixed, synthetic_transactions, upload

from ftad.core.eda import AnalysisContext
from ftad.core.eda import analysis as A
from ftad.core.schema import suggest_schema
from ftad.core.splitting import SplitConfig, make_split, summarize


@pytest.fixture()
def ctx():
    df = synthetic_transactions(2000, prevalence=0.05, seed=5)
    df = pd.concat([df, df.iloc[:7]], ignore_index=True)  # 7 exact duplicates
    return AnalysisContext(df, suggest_schema(df))


def test_class_balance_matches_counts(ctx):
    cb = A.class_balance(ctx)
    y = ctx.df["Class"].to_numpy()
    assert cb["n_positive"] == y.sum() and cb["n_negative"] == len(y) - y.sum()
    assert cb["prevalence"] == pytest.approx(y.mean())
    assert cb["majority_baseline_accuracy"] == pytest.approx(1 - y.mean())
    assert cb["minority_to_majority"] == pytest.approx(y.sum() / (len(y) - y.sum()))


def test_overview_counts_duplicates_and_missing(ctx):
    ov = A.overview(ctx)
    assert ov["duplicate_rows"] == 7
    assert ov["total_missing_cells"] == 0
    amount = next(c for c in ov["columns"] if c["name"] == "Amount")
    v = ctx.df["Amount"]
    assert amount["mean"] == pytest.approx(v.mean()) and amount["median"] == pytest.approx(v.median())
    assert amount["q1"] == pytest.approx(v.quantile(0.25)) and amount["max"] == pytest.approx(v.max())
    assert amount["std"] == pytest.approx(v.std())


def test_value_analysis_by_class(ctx):
    res = A.value_analysis(ctx, "Amount")
    pos = ctx.df.loc[ctx.df.Class == 1, "Amount"]
    assert res["by_class"]["Fraud"]["median"] == pytest.approx(pos.median())
    assert res["box"]["Fraud"]["q3"] == pytest.approx(pos.quantile(0.75))
    h = res["histogram"]
    assert sum(h["series"]["Fraud"]["counts"]) == len(pos)  # full data, not a sample
    assert res["tests"]["available"]
    log = A.value_analysis(ctx, "Amount", log=True)
    assert log["histogram"]["log1p"] is True


def test_time_analysis_elapsed_seconds(ctx):
    t = A.time_analysis(ctx, bin_size=3600)
    assert t["kind"] == "elapsed_seconds" and "not a calendar timestamp" in t["interpretation_note"]
    assert sum(r["n"] for r in t["series"]) == len(ctx.df)
    assert sum(r["n_positive"] for r in t["series"]) == ctx.df.Class.sum()
    assert len(t["hour_of_cycle"]) == 24


def test_correlation_and_relevance(ctx):
    c = A.correlation(ctx)
    m = np.array(c["matrix"])
    assert np.allclose(np.diag(m), 1) and np.allclose(m, m.T)
    rel = A.feature_relevance(ctx)
    top = [r["feature"] for r in rel["features"][:3]]
    assert set(top) <= {"V1", "V2", "V3", "Amount"}  # the fixture shifts V1-V3 for positives


def test_generic_timestamp_analysis():
    df = synthetic_mixed(800)
    cfg = suggest_schema(df)
    ctx = AnalysisContext(df, cfg)
    t = A.time_analysis(ctx)
    assert t["kind"] == "timestamp" and t["origin"].startswith("2024-01-01")
    assert A.overview(ctx)["class_balance"]["positive_name"] == "Positive"


def test_stratified_split_disjoint_and_stratified(ctx):
    y = ctx.y
    split = make_split(ctx.df, y, SplitConfig(), None, None)
    allidx = np.concatenate([split.train, split.validation, split.test])
    assert len(np.unique(allidx)) == len(allidx) == len(y)
    s = summarize(split, y)
    prevs = [p["prevalence"] for p in s["partitions"].values()]
    assert max(prevs) - min(prevs) < 0.01
    again = make_split(ctx.df, y, SplitConfig(), None, None)
    assert np.array_equal(split.test, again.test)  # reproducible for the same seed


def test_chronological_split_preserves_order(ctx):
    split = make_split(ctx.df, ctx.y, SplitConfig(method="chronological"), "Time", None)
    t = ctx.df["Time"].to_numpy()
    assert t[split.train].max() <= t[split.validation].min() <= t[split.validation].max() <= t[split.test].min()


def test_group_split_keeps_entities_together():
    df = synthetic_mixed(1200)
    cfg = suggest_schema(df)
    ctx = AnalysisContext(df, cfg)
    split = make_split(df, ctx.y, SplitConfig(method="group"), None, "customer_id")
    g = df["customer_id"].to_numpy()
    assert not set(g[split.train]) & set(g[split.test])
    assert not set(g[split.validation]) & set(g[split.test])


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        SplitConfig(train=0.7, validation=0.2, test=0.2)


def test_split_preview_rejects_impossible_configs(client):
    df = synthetic_transactions(800, prevalence=0.02, seed=9)
    ds = upload(client, df).json()["dataset"]
    confirm_suggested_schema(client, ds["id"])
    r = client.post(f"/api/datasets/{ds['id']}/split/preview", json={"method": "stratified", "train": 0.96, "validation": 0.02, "test": 0.02, "seed": 1}).json()
    assert r["result"]["valid"] is False
    ok = client.post(f"/api/datasets/{ds['id']}/split/preview", json={"method": "stratified", "train": 0.6, "validation": 0.2, "test": 0.2, "seed": 1}).json()
    assert ok["result"]["valid"] and ok["result"]["warnings"]  # few positives → warning


def test_eda_endpoints(client):
    ds = upload(client, synthetic_transactions(1500, prevalence=0.05)).json()["dataset"]
    confirm_suggested_schema(client, ds["id"])
    base = f"/api/datasets/{ds['id']}/eda"
    for path in ("/overview", "/value", "/value?log=true", "/time", "/correlation?method=spearman", "/relevance", "/distribution?column=V1", "/pca", "/projection", "/findings"):
        r = client.get(base + path)
        assert r.status_code == 200, (path, r.text)
        assert r.json()["dataset_id"] == ds["id"]
    findings = client.get(base + "/findings").json()["result"]
    assert any("accuracy" in f for f in findings)
