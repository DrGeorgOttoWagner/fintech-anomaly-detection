"""Shared pytest fixtures.

SYNTHETIC TEST FIXTURES ONLY: the data generated here exists to exercise software
behaviour. It is not empirical data and its results must never be reported as research.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND))

import ftad.modules  # noqa: E402,F401  (registers domain hint providers, as the app does)


def synthetic_transactions(n: int = 3000, n_pca: int = 8, prevalence: float = 0.03, seed: int = 0) -> pd.DataFrame:
    """Card-transaction-shaped synthetic fixture (Time, V1..Vk, Amount, Class)."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    cols = {"Time": np.sort(rng.integers(0, 172_800, n)).astype(float)}
    for i in range(1, n_pca + 1):
        shift = (1.5 if i <= 3 else 0.0) * y
        cols[f"V{i}"] = rng.normal(0, 1.0 / i ** 0.3, n) - shift
    cols["Amount"] = np.round(rng.lognormal(3.0, 1.2, n) * (1 + 0.5 * y), 2)
    cols["Class"] = y
    return pd.DataFrame(cols)


def synthetic_mixed(n: int = 1500, seed: int = 1) -> pd.DataFrame:
    """Generic fixture with different column names, categoricals, an entity ID and timestamps."""
    rng = np.random.default_rng(seed)
    label = rng.random(n) < 0.08
    return pd.DataFrame({
        "event_id": [f"E{i:06d}" for i in range(n)],
        "customer_id": rng.integers(0, 200, n).astype(str),
        "event_time": pd.date_range("2024-01-01", periods=n, freq="7min").astype(str),
        "channel": rng.choice(["web", "app", "branch"], n),
        "country": rng.choice(["DE", "FR", "NL", "PL"], n),
        "login_attempts": rng.poisson(1 + 3 * label),
        "session_value": np.round(rng.gamma(2.0, 30.0, n), 2),
        "risk_score_a": rng.normal(label * 1.2, 1),
        "is_anomaly": np.where(label, "yes", "no"),
    })


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


@pytest.fixture()
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("FTAD_DATA_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("FTAD_EXPORT_DIR", str(tmp_path / "export"))
    (tmp_path / "export").mkdir()
    from ftad.config import get_settings

    get_settings.cache_clear()
    s = get_settings()
    yield s
    get_settings.cache_clear()


@pytest.fixture()
def client(settings):
    from fastapi.testclient import TestClient

    from ftad.app import create_app

    app = create_app(settings, start_jobs=True)
    with TestClient(app) as c:
        yield c
    app.state.ftad.jobs.shutdown()


def upload(client, df: pd.DataFrame, name: str = "fixture.csv"):
    return client.post("/api/datasets/upload", files={"file": (name, io.BytesIO(to_csv_bytes(df)), "text/csv")})


def confirm_suggested_schema(client, dataset_id: str) -> dict:
    suggested = client.get(f"/api/datasets/{dataset_id}/inspect").json()["suggested"]
    r = client.put(f"/api/datasets/{dataset_id}/schema", json=suggested)
    assert r.status_code == 200, r.text
    return r.json()


def wait_for(client, exp_id: str, timeout: float = 240) -> dict:
    t = time.time()
    while time.time() - t < timeout:
        exp = client.get(f"/api/experiments/{exp_id}").json()
        if exp["status"] not in ("created", "queued", "running"):
            return exp
        time.sleep(0.5)
    raise TimeoutError(exp_id)
