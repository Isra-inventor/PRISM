"""v3 API: sessions of several datasets, schema import, merge, audit (see prism/)."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


def _load(sid):
    from prism import store
    try:
        return store.load(sid)
    except store.SessionError as e:
        raise HTTPException(404, str(e))


def session_view(st):
    from backend import workflow
    out = dict(st.data)
    for d in out["datasets"]:
        if d.get("step0_session_id") and d["status"] == "wizard_in_progress":
            try:
                s = workflow.get_session(d["step0_session_id"])
                d["wizard_steps"] = (s.draft or {}).get("steps")
            except KeyError:
                d["wizard_steps"] = None
    return out


class NewSession(BaseModel):
    name: Optional[str] = ""


@router.post("/api/sessions")
def create_session(req: NewSession):
    from prism import store
    return session_view(store.create(req.name or ""))


@router.get("/api/sessions")
def list_sessions():
    from prism import store
    return store.list_sessions()


@router.get("/api/sessions/{sid}")
def get_session(sid: str):
    return session_view(_load(sid))


# ---------------------------------------------------------------- datasets: schema import (v3 §4)

from fastapi import File, Form, UploadFile  # noqa: E402

MAX_BYTES = 200 * 1024 * 1024


def _rejected(e):
    raise HTTPException(422, {"message": str(e), "errors": getattr(e, "errors", [])})


@router.post("/api/sessions/{sid}/datasets")
async def add_dataset(sid: str, data: UploadFile = File(...), schema_file: Optional[UploadFile] = File(None, alias="schema"),
                      metadata: Optional[UploadFile] = File(None)):
    """Data + schema.json (+ metadata): the import report. Without a schema, use /api/upload with
    study_session_id to start the wizard instead."""
    from starlette.concurrency import run_in_threadpool
    from prism.io import importer
    st = _load(sid)
    if schema_file is None:
        raise HTTPException(422, "Upload a schema.json to import; to use the wizard, upload the data alone.")
    raw = await data.read(MAX_BYTES + 1)
    sraw = await schema_file.read(MAX_BYTES + 1)
    mraw = await metadata.read(MAX_BYTES + 1) if metadata is not None else None
    if len(raw) > MAX_BYTES or (mraw and len(mraw) > MAX_BYTES):
        raise HTTPException(413, "File is too large.")

    def work():
        try:
            return importer.import_bytes(st, data.filename or "data.csv", raw, sraw,
                                         metadata.filename if metadata is not None else None, mraw)
        except importer.ImportRejected as e:
            _rejected(e)
    return await run_in_threadpool(work)


@router.get("/api/sessions/{sid}/datasets/{did}/import-report")
def import_report(sid: str, did: str):
    import json
    st = _load(sid)
    p = st.dataset_dir(did) / "import_report.json"
    if not p.exists():
        raise HTTPException(404, "No import report for this dataset.")
    rep = json.loads(p.read_text(encoding="utf-8"))
    rep["dataset"] = st.dataset(did)
    try:
        from backend import workflow
        s = workflow.get_session(rep["step0_session_id"])
        rep["questions"] = workflow.questions.public(s.draft)
    except KeyError:
        pass
    return rep


def _dataset_op(sid, did, fn):
    from prism import store
    st = _load(sid)
    try:
        st.dataset(did)
        res = fn(st, did)
    except store.SessionError as e:
        raise HTTPException(422, str(e))
    return {"result": res, "session": session_view(store.load(sid))}


@router.post("/api/sessions/{sid}/datasets/{did}/accept-import")
def accept_import(sid: str, did: str):
    from prism.io import importer
    return _dataset_op(sid, did, importer.accept)


@router.post("/api/sessions/{sid}/datasets/{did}/open-wizard")
def open_wizard(sid: str, did: str):
    from prism.io import importer
    return _dataset_op(sid, did, importer.open_in_wizard)


@router.post("/api/sessions/{sid}/datasets/{did}/reject")
def reject_import(sid: str, did: str):
    from prism.io import importer
    return _dataset_op(sid, did, importer.reject)


@router.get("/api/step0/{step0_sid}")
def step0_state(step0_sid: str):
    """Load an existing Step 0 dataset into the wizard (e.g. an import opened in the wizard)."""
    from backend import workflow
    from backend.main import session_payload
    try:
        s = workflow.get_session(step0_sid)
    except KeyError:
        raise HTTPException(404, "Unknown Step 0 session.")
    if s.draft is None:
        raise HTTPException(409, "No proposal yet.")
    return {"session": session_payload(s), "draft": workflow.public_draft(s), "digests": s.digests}
