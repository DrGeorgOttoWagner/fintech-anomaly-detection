"""Dataset, schema, EDA and split-preview endpoints."""

from __future__ import annotations

import shutil
import uuid

import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from ftad.api.state import AppState
from ftad.core.data import IngestError
from ftad.core.eda import analysis as A
from ftad.core.jsonsafe import clean
from ftad.core.schema import SchemaConfig, profile_columns, suggest_schema, validate_schema
from ftad.core.splitting import SplitConfig, SplitError, make_split, summarize
from ftad.db import utcnow


class ActiveBody(BaseModel):
    dataset_id: str | None


def build_router(state: AppState) -> APIRouter:
    r = APIRouter(prefix="/api")

    # -- session --------------------------------------------------------------
    @r.get("/session")
    def session():
        active = state.store.get_active_id()
        record = None
        schema = None
        if active:
            try:
                record = state.store.get(active)
                row = state.schema_row(active)
                schema = {"id": row["id"], "hash": row["schema_hash"], "config": SchemaConfig.model_validate_json(row["config_json"]).model_dump()} if row else None
            except Exception:
                state.store.set_active(None)
                record = None
        return {"active_dataset": record, "schema": schema}

    @r.post("/session/active")
    def set_active(body: ActiveBody):
        if body.dataset_id:
            state.dataset(body.dataset_id)
        state.store.set_active(body.dataset_id)
        return session()

    # -- datasets ---------------------------------------------------------------
    @r.post("/datasets/upload")
    def upload(request: Request, file: UploadFile = File(...)):
        limit = state.settings.max_upload_mb * 1024 * 1024
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > limit + 1024 * 1024:
            raise HTTPException(413, f"File exceeds the upload limit of {state.settings.max_upload_mb} MB.")
        try:
            record, duplicate = state.store.ingest_upload(file.file, file.filename)
        except IngestError as exc:
            raise HTTPException(422, str(exc))
        state.store.set_active(record["id"])
        return {"dataset": record, "duplicate": duplicate, "has_schema": state.schema_row(record["id"]) is not None}

    @r.get("/datasets")
    def list_datasets():
        rows = state.store.list()
        active = state.store.get_active_id()
        for row in rows:
            sr = state.schema_row(row["id"])
            row["has_schema"] = sr is not None
            row["schema_hash"] = sr["schema_hash"] if sr else None
            row["active"] = row["id"] == active
            row["n_experiments"] = state.db.one("SELECT COUNT(*) AS n FROM experiments WHERE dataset_id=?", (row["id"],))["n"]
        return rows

    @r.get("/datasets/{dataset_id}")
    def get_dataset(dataset_id: str):
        return state.dataset(dataset_id)

    @r.delete("/datasets/{dataset_id}")
    def delete_dataset(dataset_id: str):
        state.dataset(dataset_id)
        running = state.db.one("SELECT id FROM experiments WHERE dataset_id=? AND status IN ('queued','running')", (dataset_id,))
        if running:
            raise HTTPException(409, "Cancel the running experiment for this dataset first.")
        exp_ids = [e["id"] for e in state.db.query("SELECT id FROM experiments WHERE dataset_id=?", (dataset_id,))]
        state.store.delete(dataset_id)  # experiments/runs rows cascade in SQLite
        for exp_id in exp_ids:
            shutil.rmtree(state.settings.experiments_dir / exp_id, ignore_errors=True)
        state.cache.drop_dataset(dataset_id)
        return {"deleted": dataset_id}

    @r.get("/datasets/{dataset_id}/preview")
    def preview(dataset_id: str, offset: int = 0, limit: int = 50):
        state.dataset(dataset_id)
        return {"dataset_id": dataset_id, **state.store.preview(dataset_id, offset, limit)}

    @r.get("/datasets/{dataset_id}/rows/{row}")
    def row_values(dataset_id: str, row: int):
        state.dataset(dataset_id)
        df = state.store.frame(dataset_id)
        if not 0 <= row < len(df):
            raise HTTPException(404, "Row out of range.")
        sr = state.schema_row(dataset_id)
        roles = SchemaConfig.model_validate_json(sr["config_json"]).roles if sr else {}
        values = df.iloc[row]
        return clean({"dataset_id": dataset_id, "row": row, "values": [{"column": c, "role": roles.get(c), "value": _py(values[c])} for c in df.columns]})

    # -- schema -------------------------------------------------------------------
    @r.get("/datasets/{dataset_id}/inspect")
    def inspect(dataset_id: str):
        state.dataset(dataset_id)

        def compute():
            df = state.store.frame(dataset_id)
            profiles = profile_columns(df)
            return clean({"profiles": profiles, "suggested": suggest_schema(df, profiles).model_dump()})

        return {"dataset_id": dataset_id, **state.cache.get_or_compute((dataset_id, "inspect"), compute)}

    @r.get("/datasets/{dataset_id}/schema")
    def get_schema(dataset_id: str):
        state.dataset(dataset_id)
        row = state.schema_row(dataset_id)
        if not row:
            return {"dataset_id": dataset_id, "schema": None}
        return {"dataset_id": dataset_id, "schema": {"id": row["id"], "hash": row["schema_hash"], "created_at": row["created_at"], "config": SchemaConfig.model_validate_json(row["config_json"]).model_dump()}}

    @r.get("/datasets/{dataset_id}/values")
    def column_values(dataset_id: str, column: str):
        state.dataset(dataset_id)
        df = state.store.frame(dataset_id)
        if column not in df.columns:
            raise HTTPException(404, f"Unknown column '{column}'.")
        counts = df[column].astype(str).value_counts().head(50)
        return {"column": column, "values": [{"value": _norm_label(k), "count": int(v)} for k, v in counts.items()], "n_unique": int(df[column].nunique())}

    @r.post("/datasets/{dataset_id}/schema/validate")
    def validate(dataset_id: str, cfg: SchemaConfig):
        state.dataset(dataset_id)
        return validate_schema(cfg, state.store.frame(dataset_id))

    @r.put("/datasets/{dataset_id}/schema")
    def save_schema(dataset_id: str, cfg: SchemaConfig):
        state.dataset(dataset_id)
        result = validate_schema(cfg, state.store.frame(dataset_id))
        if not result.valid:
            raise HTTPException(422, {"message": "Schema is invalid.", "errors": result.errors, "warnings": result.warnings})
        h = cfg.schema_hash()
        current = state.schema_row(dataset_id)
        if not current or current["schema_hash"] != h:
            state.db.execute(
                "INSERT INTO schemas(id,dataset_id,schema_hash,config_json,created_at) VALUES(?,?,?,?,?)",
                ("sc_" + uuid.uuid4().hex[:10], dataset_id, h, cfg.model_dump_json(), utcnow()),
            )
        return {"dataset_id": dataset_id, "schema_hash": h, "warnings": result.warnings, **get_schema(dataset_id)}

    # -- EDA ----------------------------------------------------------------------
    def eda(dataset_id: str, name: str, fn, *params):
        ctx, h = state.context(dataset_id)
        try:
            result = state.cache.get_or_compute((dataset_id, h, name, *params), lambda: clean(fn(ctx)))
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        return {"dataset_id": dataset_id, "schema_hash": h, "result": result}

    def require_numeric(ctx, column: str):
        if column not in ctx.df.columns:
            raise HTTPException(404, f"Unknown column '{column}'.")
        import pandas as pd

        if not pd.api.types.is_numeric_dtype(ctx.df[column]):
            raise HTTPException(422, f"Column '{column}' is not numeric.")

    @r.get("/datasets/{dataset_id}/eda/overview")
    def eda_overview(dataset_id: str):
        return eda(dataset_id, "overview", A.overview)

    @r.get("/datasets/{dataset_id}/eda/value")
    def eda_value(dataset_id: str, column: str | None = None, log: bool = False, bins: int = 60):
        ctx, _ = state.context(dataset_id)
        col = column or ctx.cfg.amount_column
        if not col:
            raise HTTPException(422, "No amount/value column configured; pass ?column=")
        require_numeric(ctx, col)
        bins = int(np.clip(bins, 10, 200))
        return eda(dataset_id, "value", lambda c: A.value_analysis(c, col, bins=bins, log=log), col, log, bins)

    @r.get("/datasets/{dataset_id}/eda/time")
    def eda_time(dataset_id: str, bin_size: float | None = None):
        return eda(dataset_id, "time", lambda c: A.time_analysis(c, bin_size), bin_size)

    @r.get("/datasets/{dataset_id}/eda/correlation")
    def eda_corr(dataset_id: str, method: str = "pearson"):
        if method not in ("pearson", "spearman"):
            raise HTTPException(422, "method must be pearson or spearman")
        return eda(dataset_id, "corr", lambda c: A.correlation(c, method), method)

    @r.get("/datasets/{dataset_id}/eda/relevance")
    def eda_relevance(dataset_id: str):
        return eda(dataset_id, "relevance", A.feature_relevance)

    @r.get("/datasets/{dataset_id}/eda/distribution")
    def eda_distribution(dataset_id: str, column: str, bins: int = 50):
        ctx, _ = state.context(dataset_id)
        require_numeric(ctx, column)
        bins = int(np.clip(bins, 10, 200))
        return eda(dataset_id, "dist", lambda c: A.class_conditional(c, column, bins), column, bins)

    @r.get("/datasets/{dataset_id}/eda/pca")
    def eda_pca(dataset_id: str):
        return eda(dataset_id, "pca", A.pca_exploration)

    @r.get("/datasets/{dataset_id}/eda/projection")
    def eda_projection(dataset_id: str):
        return eda(dataset_id, "projection", A.projection)

    @r.get("/datasets/{dataset_id}/eda/findings")
    def eda_findings(dataset_id: str):
        ctx, h = state.context(dataset_id)
        ov = eda_overview(dataset_id)["result"]
        val = eda_value(dataset_id)["result"] if ctx.cfg.amount_column else None
        tim = eda_time(dataset_id)["result"]
        rel = eda_relevance(dataset_id)["result"]
        return {"dataset_id": dataset_id, "schema_hash": h, "result": A.findings(ctx, ov, val, tim, rel)}

    # -- split preview ------------------------------------------------------------
    @r.post("/datasets/{dataset_id}/split/preview")
    def split_preview(dataset_id: str, cfg: SplitConfig):
        ctx, h = state.context(dataset_id)
        if ctx.y is None:
            raise HTTPException(422, "A target is required to plan train/validation/test partitions.")
        try:
            split = make_split(ctx.df, ctx.y, cfg, ctx.cfg.time_column, ctx.cfg.group_column)
        except SplitError as exc:
            return {"dataset_id": dataset_id, "schema_hash": h, "result": {"valid": False, "errors": [str(exc)], "warnings": [], "partitions": {}, "notes": []}}
        return {"dataset_id": dataset_id, "schema_hash": h, "result": clean(summarize(split, ctx.y, ctx.pos_name))}

    return r


def _norm_label(v: str) -> str:
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else v
    except ValueError:
        return v


def _py(v):
    if hasattr(v, "item"):
        try:
            return v.item()
        except (ValueError, AttributeError):
            pass
    return v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
