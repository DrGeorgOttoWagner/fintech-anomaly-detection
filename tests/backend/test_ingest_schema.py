"""Dataset ingestion, validation and schema mapping."""

from __future__ import annotations

import io
import os
import stat

import pandas as pd
import pytest
from conftest import confirm_suggested_schema, synthetic_mixed, synthetic_transactions, upload

from ftad.core.data.ingest import IngestError, detect_format, sanitize_filename


def test_starts_without_dataset(client):
    assert client.get("/api/session").json() == {"active_dataset": None, "schema": None}
    assert client.get("/api/datasets").json() == []


@pytest.mark.parametrize("raw,expected", [("../../etc/passwd.csv", "passwd.csv"), ("C:\\x\\..\\data.CSV", "data.csv"), ("my data (1).csv", "my data _1_.csv")])
def test_sanitize_filename_strips_paths(raw, expected):
    assert sanitize_filename(raw) == expected


def test_rejects_unsupported_extension():
    with pytest.raises(IngestError):
        detect_format("payload.exe")


def test_csv_upload_preview_and_fingerprint(client, settings):
    df = synthetic_transactions(500)
    r = upload(client, df, "../../evil/transactions.csv")
    assert r.status_code == 200, r.text
    body = r.json()
    ds = body["dataset"]
    assert ds["n_rows"] == 500 and ds["n_cols"] == df.shape[1]
    assert len(ds["sha256"]) == 64 and body["duplicate"] is False
    raw = settings.data_dir / ds["raw_path"]
    assert raw.parent.name == ds["sha256"]  # content-addressed storage, client name cannot traverse
    assert not (os.stat(raw).st_mode & stat.S_IWUSR)  # raw upload is read-only
    assert client.get("/api/session").json()["active_dataset"]["id"] == ds["id"]
    prev = client.get(f"/api/datasets/{ds['id']}/preview?limit=5").json()
    assert prev["columns"] == list(df.columns) and len(prev["rows"]) == 5
    assert prev["rows"][0][0] == pytest.approx(df.iloc[0, 0])


def test_duplicate_upload_is_detected(client):
    df = synthetic_transactions(300)
    a = upload(client, df).json()
    b = upload(client, df, "copy.csv").json()
    assert b["duplicate"] is True and b["dataset"]["id"] == a["dataset"]["id"]
    assert len(client.get("/api/datasets").json()) == 1


def test_parquet_upload(client, tmp_path):
    df = synthetic_mixed(200)
    path = tmp_path / "x.parquet"
    df.to_parquet(path)
    r = client.post("/api/datasets/upload", files={"file": ("x.parquet", path.read_bytes(), "application/octet-stream")})
    assert r.status_code == 200, r.text
    assert r.json()["dataset"]["format"] == "parquet"


@pytest.mark.parametrize("name,content,fragment", [
    ("bad.csv", b"", "empty"),
    ("fake.parquet", b"not parquet at all", "not a valid Parquet"),
    ("bin.csv", b"a,b\n\x00\x01\x02,3\n", "binary"),
    ("one.csv", b"a,b\n1,2\n", "at least two rows"),
    ("dupe.csv", b"a,a\n1,2\n3,4\n", "Duplicate column"),
])
def test_invalid_files_give_useful_errors(client, name, content, fragment):
    r = client.post("/api/datasets/upload", files={"file": (name, io.BytesIO(content), "text/csv")})
    assert r.status_code == 422
    assert fragment.lower() in r.json()["detail"].lower()


def test_kaggle_like_schema_suggestion(client):
    ds = upload(client, synthetic_transactions(800, n_pca=8)).json()["dataset"]
    s = client.get(f"/api/datasets/{ds['id']}/inspect").json()["suggested"]
    assert s["target"] == "Class" and s["positive_label"] == "1"
    assert s["roles"]["Amount"] == "numeric" and s["amount_column"] == "Amount"
    assert s["roles"]["Time"] == "time" and s["time_kind"] == "elapsed_seconds"
    assert [c for c, r in s["roles"].items() if r == "pca"] == [f"V{i}" for i in range(1, 9)]
    assert s["domain"] == "fraud" and s["positive_name"] == "Fraud"


@pytest.mark.parametrize("n_pca", [8, 28, 60])
def test_any_number_of_pca_columns(client, n_pca):
    ds = upload(client, synthetic_transactions(400, n_pca=n_pca, seed=n_pca)).json()["dataset"]
    s = client.get(f"/api/datasets/{ds['id']}/inspect").json()["suggested"]
    assert sum(r == "pca" for r in s["roles"].values()) == n_pca


def test_generic_schema_with_different_names(client):
    ds = upload(client, synthetic_mixed()).json()["dataset"]
    s = client.get(f"/api/datasets/{ds['id']}/inspect").json()["suggested"]
    assert s["target"] == "is_anomaly" and s["positive_label"] == "yes"
    assert s["roles"]["event_id"] == "transaction_id"
    assert s["roles"]["customer_id"] == "group_id"
    assert s["roles"]["event_time"] == "time" and s["time_kind"] == "timestamp"
    assert s["roles"]["channel"] == "categorical"
    assert s["domain"] == "generic" and s["positive_name"] == "Positive"


def test_schema_validation_errors(client):
    ds = upload(client, synthetic_transactions(300)).json()["dataset"]
    s = client.get(f"/api/datasets/{ds['id']}/inspect").json()["suggested"]
    bad = {**s, "positive_label": "7"}
    v = client.post(f"/api/datasets/{ds['id']}/schema/validate", json=bad).json()
    assert not v["valid"] and any("does not occur" in e for e in v["errors"])
    no_features = {**s, "roles": {c: ("ignore" if r in ("pca", "numeric") else r) for c, r in s["roles"].items()}}
    assert client.put(f"/api/datasets/{ds['id']}/schema", json=no_features).status_code == 422
    user_choice = {**s, "roles": {**s["roles"], "Time": "numeric"}, "time_column": None, "time_kind": None}
    r = client.put(f"/api/datasets/{ds['id']}/schema", json=user_choice)
    assert r.status_code == 200  # suggestions are not enforced


def test_session_isolation_between_datasets(client):
    a = upload(client, synthetic_transactions(600, seed=3), "a.csv").json()["dataset"]
    confirm_suggested_schema(client, a["id"])
    ov_a = client.get(f"/api/datasets/{a['id']}/eda/overview").json()
    b = upload(client, synthetic_mixed(400), "b.csv").json()["dataset"]
    sess = client.get("/api/session").json()
    assert sess["active_dataset"]["id"] == b["id"] and sess["schema"] is None  # no stale schema carried over
    assert client.get(f"/api/datasets/{b['id']}/eda/overview").status_code == 409  # must confirm schema first
    confirm_suggested_schema(client, b["id"])
    ov_b = client.get(f"/api/datasets/{b['id']}/eda/overview").json()
    assert ov_b["dataset_id"] == b["id"] and ov_b["result"]["n_records"] == 400
    assert ov_a["result"]["n_records"] == 600
    client.post("/api/session/active", json={"dataset_id": a["id"]})
    assert client.get("/api/session").json()["schema"]["hash"] == ov_a["schema_hash"]


def test_delete_dataset(client):
    ds = upload(client, synthetic_transactions(200)).json()["dataset"]
    assert client.delete(f"/api/datasets/{ds['id']}").status_code == 200
    assert client.get("/api/session").json()["active_dataset"] is None
    assert client.get(f"/api/datasets/{ds['id']}").status_code == 404
