"""Dataset registry: immutable raw uploads, normalised Parquet frames, active session."""

from __future__ import annotations

import os
import shutil
import stat
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from ftad.config import Settings
from ftad.core.data import ingest
from ftad.db import Database, utcnow

ACTIVE_KEY = "active_dataset_id"


class DatasetNotFound(KeyError):
    pass


class DatasetStore:
    def __init__(self, db: Database, settings: Settings, cache_size: int = 3) -> None:
        self.db = db
        self.settings = settings
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    # -- registration ---------------------------------------------------
    def ingest_upload(self, stream: BinaryIO, filename: str | None) -> tuple[dict, bool]:
        """Persist an upload. Returns ``(dataset_record, is_duplicate)``."""
        safe_name = ingest.sanitize_filename(filename)
        fmt = ingest.detect_format(safe_name)
        tmp = self.settings.uploads_dir / f".incoming-{uuid.uuid4().hex}"
        stored = ingest.copy_stream(stream, tmp, self.settings.max_upload_mb * 1024 * 1024)
        try:
            existing = self.db.one("SELECT * FROM datasets WHERE sha256=?", (stored.sha256,))
            if existing:
                return existing, True
            ingest.validate_content_signature(tmp, fmt)
            df = ingest.parse_file(tmp, fmt, self.settings.max_columns)

            dataset_id = "ds_" + uuid.uuid4().hex[:12]
            raw_dir = self.settings.uploads_dir / stored.sha256
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path = raw_dir / safe_name
            shutil.move(str(tmp), raw_path)
            os.chmod(raw_path, stat.S_IRUSR | stat.S_IRGRP)  # immutable raw copy (read-only)

            frame_dir = self.settings.datasets_dir / dataset_id
            frame_dir.mkdir(parents=True, exist_ok=True)
            frame_path = frame_dir / "frame.parquet"
            _to_parquet(df, frame_path)

            record = {
                "id": dataset_id,
                "name": Path(safe_name).stem,
                "original_filename": safe_name,
                "sha256": stored.sha256,
                "format": fmt,
                "size_bytes": stored.size_bytes,
                "n_rows": int(df.shape[0]),
                "n_cols": int(df.shape[1]),
                "raw_path": str(raw_path.relative_to(self.settings.data_dir)),
                "frame_path": str(frame_path.relative_to(self.settings.data_dir)),
                "created_at": utcnow(),
            }
            self.db.execute(
                "INSERT INTO datasets(id,name,original_filename,sha256,format,size_bytes,n_rows,n_cols,raw_path,frame_path,created_at)"
                " VALUES(:id,:name,:original_filename,:sha256,:format,:size_bytes,:n_rows,:n_cols,:raw_path,:frame_path,:created_at)",
                record,
            )
            self._remember(dataset_id, df)
            return record, False
        finally:
            tmp.unlink(missing_ok=True)

    # -- lookup ------------------------------------------------------------
    def list(self) -> list[dict]:
        return self.db.query("SELECT * FROM datasets ORDER BY created_at DESC")

    def get(self, dataset_id: str) -> dict:
        row = self.db.one("SELECT * FROM datasets WHERE id=?", (dataset_id,))
        if not row:
            raise DatasetNotFound(dataset_id)
        return row

    def frame(self, dataset_id: str) -> pd.DataFrame:
        with self._lock:
            if dataset_id in self._cache:
                self._cache.move_to_end(dataset_id)
                return self._cache[dataset_id]
        record = self.get(dataset_id)
        df = pd.read_parquet(self.settings.data_dir / record["frame_path"])
        self._remember(dataset_id, df)
        return df

    def _remember(self, dataset_id: str, df: pd.DataFrame) -> None:
        with self._lock:
            self._cache[dataset_id] = df
            self._cache.move_to_end(dataset_id)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    def preview(self, dataset_id: str, offset: int = 0, limit: int = 50) -> dict:
        df = self.frame(dataset_id)
        limit = max(1, min(limit, 500))
        offset = max(0, min(offset, max(0, len(df) - 1)))
        page = df.iloc[offset : offset + limit]
        return {
            "columns": [str(c) for c in df.columns],
            "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
            "offset": offset,
            "limit": limit,
            "total": int(len(df)),
            "row_index": [int(i) for i in page.index],
            "rows": _rows(page),
        }

    def delete(self, dataset_id: str) -> None:
        record = self.get(dataset_id)
        self.db.execute("DELETE FROM datasets WHERE id=?", (dataset_id,))
        with self._lock:
            self._cache.pop(dataset_id, None)
        shutil.rmtree(self.settings.datasets_dir / dataset_id, ignore_errors=True)
        raw = self.settings.data_dir / record["raw_path"]
        if raw.exists():
            os.chmod(raw, stat.S_IRUSR | stat.S_IWUSR)
            shutil.rmtree(raw.parent, ignore_errors=True)
        if self.get_active_id() == dataset_id:
            self.set_active(None)

    # -- active dataset session -----------------------------------------
    def get_active_id(self) -> str | None:
        return self.db.get_state(ACTIVE_KEY)

    def set_active(self, dataset_id: str | None) -> None:
        if dataset_id is not None:
            self.get(dataset_id)
        self.db.set_state(ACTIVE_KEY, dataset_id)


def _to_parquet(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            # Mixed-type object columns are stored as strings so Parquet can hold them.
            out[col] = out[col].map(lambda v: None if pd.isna(v) else str(v))
    out.to_parquet(path, index=False)


def _rows(page: pd.DataFrame) -> list[list]:
    from ftad.core.jsonsafe import clean

    return clean([[_scalar(v) for v in row] for row in page.itertuples(index=False)])


def _scalar(v):
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if hasattr(v, "item"):
        try:
            return v.item()
        except (ValueError, AttributeError):
            return str(v)
    return v
