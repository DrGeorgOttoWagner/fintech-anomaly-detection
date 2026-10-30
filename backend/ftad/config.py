"""Runtime configuration.

All settings come from environment variables (optionally loaded from ``FTAD/.env``).
Defaults are chosen for safe, local-only operation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../FTAD


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines). Existing env vars take precedence."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    export_dir: Path | None
    host: str = "127.0.0.1"
    port: int = 8790
    max_upload_mb: int = 1024
    max_columns: int = 2000
    max_workers_per_model: int = field(default_factory=lambda: max(1, (os.cpu_count() or 2) - 2))
    frontend_dist: Path = PROJECT_ROOT / "frontend" / "dist"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def experiments_dir(self) -> Path:
        return self.data_dir / "experiments"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "ftad.sqlite3"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.uploads_dir, self.datasets_dir, self.experiments_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_dotenv(PROJECT_ROOT / ".env")
    data_dir = Path(os.environ.get("FTAD_DATA_DIR", PROJECT_ROOT / "var")).expanduser().resolve()
    export_raw = os.environ.get("FTAD_EXPORT_DIR", "").strip()
    export_dir = Path(export_raw).expanduser().resolve() if export_raw else None
    settings = Settings(
        data_dir=data_dir,
        export_dir=export_dir,
        host=os.environ.get("FTAD_HOST", "127.0.0.1"),
        port=int(os.environ.get("FTAD_PORT", "8790")),
        max_upload_mb=int(os.environ.get("FTAD_MAX_UPLOAD_MB", "1024")),
    )
    settings.ensure_dirs()
    return settings
