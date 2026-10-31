"""Research report generation, download and explicit export to the configured folder."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ftad.api.state import AppState
from ftad.core.experiments.service import get_experiment, run_detail
from ftad.core.jsonsafe import clean
from ftad.core.reporting.report import export_copy, generate_report, list_files, zip_dir

SAFE_ID = re.compile(r"^rep_[0-9a-f]{10}$")


class ReportBody(BaseModel):
    dataset_id: str
    experiment_id: str | None = None
    title: str | None = None


def build_router(state: AppState) -> APIRouter:
    r = APIRouter(prefix="/api/reports")
    root = state.settings.reports_dir

    def _dir(report_id: str):
        if not SAFE_ID.match(report_id):
            raise HTTPException(400, "Invalid report id.")
        d = root / report_id
        if not d.is_dir():
            raise HTTPException(404, "Report not found.")
        return d

    @r.get("/config")
    def config():
        ed = state.settings.export_dir
        return {"export_dir": str(ed) if ed else None, "export_dir_exists": bool(ed and ed.is_dir()),
                "how_to_configure": "Set FTAD_EXPORT_DIR in FTAD/.env to a local folder to enable one-click export."}

    @r.post("")
    def create(body: ReportBody):
        dataset = state.dataset(body.dataset_id)
        ctx, h = state.context(body.dataset_id)
        experiment, details = None, None
        if body.experiment_id:
            try:
                experiment = get_experiment(state.db, body.experiment_id)
            except KeyError:
                raise HTTPException(404, "Experiment not found.")
            if experiment["dataset_id"] != body.dataset_id:
                raise HTTPException(409, "The experiment belongs to a different dataset.")
            if experiment["schema_hash"] != h:
                raise HTTPException(409, "The experiment used a different schema version than the current one.")
            details = [run_detail(state.settings, state.db, run["id"]) for run in experiment["runs"] if run["status"] == "completed"]
        report_id = "rep_" + uuid.uuid4().hex[:10]
        out = root / report_id
        out.mkdir(parents=True)

        def analysis(name, fn):
            return state.cache.get_or_compute((body.dataset_id, h, "report", name), lambda: clean(fn()))

        manifest = generate_report(out, dataset, h, ctx, analysis, experiment, details)
        meta = {"id": report_id, "dataset_id": body.dataset_id, "dataset_name": dataset["name"], "experiment_id": body.experiment_id,
                "created_at": manifest["created_at"], "title": body.title or f"FTAD report — {dataset['name']}"}
        (out / "report_meta.json").write_text(json.dumps(meta))
        return meta | {"files": list_files(out), "evidence": manifest["evidence"]}

    @r.get("")
    def list_reports(dataset_id: str | None = None):
        items = []
        for d in sorted(root.glob("rep_*"), reverse=True):
            mf = d / "report_meta.json"
            if mf.exists():
                meta = json.loads(mf.read_text())
                if dataset_id and meta["dataset_id"] != dataset_id:
                    continue
                items.append(meta)
        return sorted(items, key=lambda m: m["created_at"], reverse=True)

    @r.get("/{report_id}")
    def get(report_id: str):
        d = _dir(report_id)
        return json.loads((d / "report_meta.json").read_text()) | {"files": list_files(d)}

    @r.get("/{report_id}/files/{path:path}")
    def file(report_id: str, path: str):
        d = _dir(report_id).resolve()
        target = (d / path).resolve()
        if d not in target.parents or not target.is_file():
            raise HTTPException(404, "File not found.")
        return FileResponse(target)

    @r.get("/{report_id}/bundle.zip")
    def bundle(report_id: str):
        d = _dir(report_id)
        z = root / f"{report_id}.zip"
        zip_dir(d, z)
        return FileResponse(z, media_type="application/zip", filename=f"ftad_{report_id}.zip")

    @r.post("/{report_id}/export")
    def export(report_id: str):
        d = _dir(report_id)
        ed = state.settings.export_dir
        if not ed:
            raise HTTPException(409, "No export folder configured. Set FTAD_EXPORT_DIR in FTAD/.env and restart FTAD.")
        if not ed.is_dir():
            raise HTTPException(409, "The configured export folder does not exist. Create it or update FTAD_EXPORT_DIR.")
        meta = json.loads((d / "report_meta.json").read_text())
        name = f"FTAD_{re.sub(r'[^A-Za-z0-9_-]', '_', meta['dataset_name'])[:40]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{report_id[-4:]}"
        try:
            dest = export_copy(d, ed / "ftad_exports", name) if (ed / "ftad_exports").is_dir() or _mk(ed / "ftad_exports") else None
        except FileExistsError as exc:
            raise HTTPException(409, str(exc))
        return {"exported_to": str(dest), "folder": dest.name, "files": len(list_files(dest))}

    return r


def _mk(p) -> bool:
    p.mkdir(parents=True, exist_ok=True)
    return True
