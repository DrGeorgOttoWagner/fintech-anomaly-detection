"""Process-wide application state shared by the API routers."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Callable

from fastapi import HTTPException

from ftad.config import Settings
from ftad.core.data import DatasetNotFound, DatasetStore
from ftad.core.eda import AnalysisContext
from ftad.core.experiments.jobs import JobManager
from ftad.core.schema import SchemaConfig
from ftad.db import Database


class ResultCache:
    """Small LRU cache for deterministic analysis results keyed by (dataset, schema hash, params)."""

    def __init__(self, size: int = 256) -> None:
        self._d: OrderedDict[tuple, Any] = OrderedDict()
        self._size = size
        self._lock = threading.Lock()

    def get_or_compute(self, key: tuple, fn: Callable[[], Any]) -> Any:
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        value = fn()
        with self._lock:
            self._d[key] = value
            while len(self._d) > self._size:
                self._d.popitem(last=False)
        return value

    def drop_dataset(self, dataset_id: str) -> None:
        with self._lock:
            for k in [k for k in self._d if k and k[0] == dataset_id]:
                self._d.pop(k, None)


class AppState:
    def __init__(self, settings: Settings, start_jobs: bool = True) -> None:
        self.settings = settings
        self.db = Database(settings.db_path)
        self.store = DatasetStore(self.db, settings)
        self.cache = ResultCache()
        self.jobs = JobManager(self.db, settings.data_dir) if start_jobs else None

    def dataset(self, dataset_id: str) -> dict:
        try:
            return self.store.get(dataset_id)
        except DatasetNotFound:
            raise HTTPException(404, f"Dataset '{dataset_id}' not found.")

    def schema_row(self, dataset_id: str) -> dict | None:
        return self.db.one("SELECT * FROM schemas WHERE dataset_id=? ORDER BY created_at DESC, rowid DESC LIMIT 1", (dataset_id,))

    def require_schema(self, dataset_id: str) -> tuple[dict, SchemaConfig]:
        self.dataset(dataset_id)
        row = self.schema_row(dataset_id)
        if not row:
            raise HTTPException(409, "Confirm the schema mapping for this dataset first.")
        return row, SchemaConfig.model_validate_json(row["config_json"])

    def context(self, dataset_id: str) -> tuple[AnalysisContext, str]:
        row, cfg = self.require_schema(dataset_id)
        return AnalysisContext(self.store.frame(dataset_id), cfg), row["schema_hash"]
