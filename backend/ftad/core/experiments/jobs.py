"""Asynchronous experiment execution in a separate worker process.

Training never runs in the API process, so the UI stays responsive. Only one
experiment trains at a time (bounded CPU/memory); further submissions are queued.
Cancellation terminates the worker process.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
from pathlib import Path

from ftad.db import Database, utcnow

_ctx = mp.get_context("spawn")


def _worker_main(db_path: str, data_dir: str, exp_id: str) -> None:
    import os

    os.environ["FTAD_DATA_DIR"] = data_dir
    from ftad.config import get_settings
    from ftad.core.experiments.runner import ExperimentRunner

    db = Database(Path(db_path))
    ExperimentRunner(db, get_settings(), exp_id).run()


class JobManager:
    def __init__(self, db: Database, data_dir: Path, poll_seconds: float = 0.5) -> None:
        self.db = db
        self.data_dir = data_dir
        self.poll = poll_seconds
        self._proc: mp.Process | None = None
        self._active: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._recover()
        self._thread = threading.Thread(target=self._loop, name="ftad-jobs", daemon=True)
        self._thread.start()

    def _recover(self) -> None:
        """Experiments that were running when the server stopped cannot be resumed."""
        self.db.execute(
            "UPDATE experiments SET status='interrupted', error='The server stopped while this experiment was running.', finished_at=? WHERE status='running'",
            (utcnow(),),
        )
        self.db.execute("UPDATE runs SET status='interrupted' WHERE status='running'")

    def submit(self, exp_id: str) -> None:
        self.db.execute("UPDATE experiments SET status='queued' WHERE id=?", (exp_id,))

    def cancel(self, exp_id: str) -> bool:
        with self._lock:
            row = self.db.one("SELECT status FROM experiments WHERE id=?", (exp_id,))
            if not row or row["status"] not in ("queued", "running"):
                return False
            if self._active == exp_id and self._proc is not None:
                self._proc.terminate()
                self._proc.join(timeout=10)
                self._proc, self._active = None, None
            self.db.execute("UPDATE experiments SET status='cancelled', finished_at=? WHERE id=?", (utcnow(), exp_id))
            self.db.execute("UPDATE runs SET status='cancelled' WHERE experiment_id=? AND status IN ('pending','running')", (exp_id,))
            return True

    def active(self) -> str | None:
        return self._active

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            if self._proc is not None and self._proc.is_alive():
                self._proc.terminate()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:  # keep the supervisor alive
                pass
            time.sleep(self.poll)

    def _tick(self) -> None:
        with self._lock:
            if self._proc is not None:
                if self._proc.is_alive():
                    return
                exit_code = self._proc.exitcode
                row = self.db.one("SELECT status FROM experiments WHERE id=?", (self._active,))
                if row and row["status"] in ("running", "queued"):
                    self.db.execute(
                        "UPDATE experiments SET status='failed', error=?, finished_at=? WHERE id=?",
                        (f"Worker process exited unexpectedly (code {exit_code}).", utcnow(), self._active),
                    )
                    self.db.execute("UPDATE runs SET status='failed', error='worker exited' WHERE experiment_id=? AND status IN ('pending','running')", (self._active,))
                self._proc, self._active = None, None
            nxt = self.db.one("SELECT id FROM experiments WHERE status='queued' ORDER BY created_at LIMIT 1")
            if not nxt:
                return
            exp_id = nxt["id"]
            self.db.execute("UPDATE experiments SET status='running', started_at=? WHERE id=?", (utcnow(), exp_id))
            # Non-daemonic so joblib may use worker processes/threads for parallel model fitting.
            proc = _ctx.Process(target=_worker_main, args=(str(self.db.path), str(self.data_dir), exp_id), daemon=False)
            proc.start()
            self._proc, self._active = proc, exp_id
