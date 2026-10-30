"""Scoring of new (typically unlabeled) datasets with a persisted model."""

from __future__ import annotations

import json
import uuid

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ftad.api.state import AppState
from ftad.core.experiments.service import get_run
from ftad.core.inference import check_compatibility, load_model, score_frame
from ftad.core.jsonsafe import clean
from ftad.core.schema import SchemaConfig
from ftad.db import utcnow


class InferenceBody(BaseModel):
    run_id: str
    dataset_id: str
    acknowledge_warnings: bool = False


def build_router(state: AppState) -> APIRouter:
    r = APIRouter(prefix="/api/inference")
    pred_dir = state.settings.data_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    def _bundle(run_id: str) -> dict:
        try:
            run, exp = get_run(state.db, run_id)
        except KeyError:
            raise HTTPException(404, "Run not found.")
        if run["status"] != "completed":
            raise HTTPException(409, "Only completed runs can be used for inference.")
        return load_model(state.settings.experiments_dir / exp["id"] / run_id / "model.joblib")

    @r.post("/check")
    def check(body: InferenceBody):
        dataset = state.dataset(body.dataset_id)
        bundle = _bundle(body.run_id)
        return clean(check_compatibility(bundle["meta"], state.store.frame(body.dataset_id), dataset["sha256"]))

    @r.post("/run")
    def run(body: InferenceBody):
        dataset = state.dataset(body.dataset_id)
        bundle = _bundle(body.run_id)
        df = state.store.frame(body.dataset_id)
        report = check_compatibility(bundle["meta"], df, dataset["sha256"])
        if report["status"] == "incompatible":
            raise HTTPException(422, {"message": "Dataset is incompatible with the model.", **clean(report)})
        if report["status"] == "warnings" and not body.acknowledge_warnings:
            raise HTTPException(409, {"message": "Compatibility warnings must be acknowledged before scoring.", **clean(report)})
        result = score_frame(bundle, df)
        meta = bundle["meta"]
        sr = state.schema_row(body.dataset_id)
        id_col = SchemaConfig.model_validate_json(sr["config_json"]).id_column if sr else None
        out = pd.DataFrame({"row_index": np.arange(len(df))})
        if id_col and id_col in df.columns:
            out["record_id"] = df[id_col].astype(str).to_numpy()
        out["score"] = result["scores"]
        out["score_type"] = meta["score_type"]
        out["threshold"] = result["threshold"]
        out["predicted_label"] = np.where(result["predicted"] == 1, meta["positive_name"], meta["negative_name"])
        if "model_probability" in result:
            out["model_probability_uncalibrated"] = result["model_probability"]
        if "calibrated_probability" in result:
            out["calibrated_probability"] = result["calibrated_probability"]
        pred_id = "pred_" + uuid.uuid4().hex[:10]
        out.to_csv(pred_dir / f"{pred_id}.csv", index=False)
        flagged = int(result["predicted"].sum())
        hist, edges = np.histogram(result["scores"], bins=40)
        summary = {
            "prediction_id": pred_id,
            "created_at": utcnow(),
            "run_id": body.run_id,
            "dataset_id": body.dataset_id,
            "model": report["model"],
            "n": int(len(df)),
            "n_flagged": flagged,
            "flagged_rate": flagged / len(df) if len(df) else None,
            "score_histogram": {"counts": hist.tolist(), "edges": edges.tolist()},
            "compatibility": report,
            "columns": list(out.columns),
            "top": json.loads(out.sort_values("score", ascending=False).head(50).to_json(orient="records")),
            "note": "No ground-truth metrics are shown for inference: the scored data is treated as unlabeled.",
        }
        (pred_dir / f"{pred_id}.json").write_text(json.dumps(clean(summary)))
        return clean(summary)

    @r.get("/{pred_id}.csv")
    def download(pred_id: str):
        if not pred_id.startswith("pred_") or not pred_id[5:].isalnum():
            raise HTTPException(400, "Invalid prediction id.")
        path = pred_dir / f"{pred_id}.csv"
        if not path.exists():
            raise HTTPException(404, "Prediction not found.")
        return FileResponse(path, media_type="text/csv", filename=f"{pred_id}.csv")

    return r
