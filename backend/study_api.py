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


# ---------------------------------------------------------------- merge (v3 §5)


def _merge(sid, fn):
    from prism.session import merge
    st = _load(sid)
    try:
        return fn(st, merge)
    except merge.MergeError as e:
        raise HTTPException(422, str(e))


@router.get("/api/sessions/{sid}/merge-report")
def merge_report(sid: str):
    return _merge(sid, lambda st, m: m.report(st))


class MergeDecision(BaseModel):
    item_id: str
    decision: str
    unified_id: Optional[str] = None


@router.post("/api/sessions/{sid}/merge/confirm-mapping")
def merge_confirm_mapping(sid: str, req: MergeDecision):
    """An ID suggestion: confirm (optionally with the unified ID to use), dismiss or reopen."""
    if not req.item_id.startswith("ms"):
        raise HTTPException(422, "Not an ID suggestion.")
    return _merge(sid, lambda st, m: m.decide(st, req.item_id, req.decision, "user", req.unified_id))


@router.post("/api/sessions/{sid}/merge/resolve-conflict")
def merge_resolve_conflict(sid: str, req: MergeDecision):
    """A value conflict (take:D1 / keep_both / drop / reopen) or a merge question (option or dismiss)."""
    if not req.item_id.startswith(("mc", "mq")):
        raise HTTPException(422, "Not a conflict or a merge question.")
    return _merge(sid, lambda st, m: m.decide(st, req.item_id, req.decision, "user"))


@router.post("/api/sessions/{sid}/merge/mapping-csv")
async def merge_mapping_csv(sid: str, file: UploadFile = File(...)):
    raw = await file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "File is too large.")
    return _merge(sid, lambda st, m: m.set_mapping_csv(st, raw, file.filename or "mapping.csv"))


@router.delete("/api/sessions/{sid}/merge/mapping-csv")
def merge_clear_mapping_csv(sid: str):
    return _merge(sid, lambda st, m: m.clear_mapping_csv(st))


# ---------------------------------------------------------------- audit (v3 §6, §7)


def _audit(sid, fn):
    from prism.audit import engine, overrides
    st = _load(sid)
    try:
        return fn(st, engine)
    except (engine.AuditError, overrides.OverrideError) as e:
        raise HTTPException(422, str(e))


class AuditRun(BaseModel):
    factors: Optional[list] = None
    datasets: Optional[list] = None
    params: Optional[dict] = None


@router.post("/api/sessions/{sid}/audit/run")
def audit_run(sid: str, req: AuditRun):
    return _audit(sid, lambda st, e: e.run(st, req.factors, req.datasets, req.params or {}, who="user"))


@router.get("/api/sessions/{sid}/audit")
def audit_runs(sid: str):
    def fn(st, e):
        out = []
        for r in e.list_runs(st):
            m = e.show(st, r)
            out.append({"run_id": r, "created_at": m["created_at"], "params_sha256": m["params_sha256"],
                        "audits": m["scope"]["audits"], "n_findings": len(m["findings"])})
        return out
    return _audit(sid, fn)


@router.get("/api/sessions/{sid}/audit/{run_id}")
def audit_show(sid: str, run_id: str):
    """The run's manifest and every finding (what the report page renders)."""
    from prism.audit import report
    return _audit(sid, lambda st, e: report.bundle(st, run_id))


@router.get("/api/sessions/{sid}/audit/{run_id}/compare/{other}")
def audit_compare(sid: str, run_id: str, other: str):
    return _audit(sid, lambda st, e: e.compare(st, other, run_id))


@router.get("/api/sessions/{sid}/audit/{run_id}/report.html")
def audit_report_html(sid: str, run_id: str):
    from fastapi.responses import HTMLResponse
    from prism.audit import report
    html = _audit(sid, lambda st, e: report.html(st, run_id))
    return HTMLResponse(html, headers={"Content-Disposition": f'attachment; filename="prism_audit_{run_id}.html"'})


@router.get("/api/audit/params")
def audit_params():
    from prism.audit import params
    return {"defaults": params.defaults(), "heuristic": list(params.HEURISTIC)}


@router.get("/api/sessions/{sid}/overrides")
def get_overrides(sid: str):
    from prism.audit import overrides
    return {"overrides": overrides.load(_load(sid)), "kinds": list(overrides.KINDS), "roles": list(overrides.ROLES)}


class OverrideIn(BaseModel):
    kind: str
    sample: Optional[str] = None
    role: Optional[str] = None
    column: Optional[str] = None
    dataset: Optional[str] = None
    key: Optional[str] = None
    value: Optional[object] = None
    reason: Optional[str] = ""


@router.post("/api/sessions/{sid}/overrides")
def add_override(sid: str, req: OverrideIn):
    from prism.audit import overrides
    return _audit(sid, lambda st, e: {"override": overrides.add(st, req.model_dump(), who="user"),
                                      "overrides": overrides.load(st)})


@router.delete("/api/sessions/{sid}/overrides/{oid}")
def delete_override(sid: str, oid: str):
    from prism.audit import overrides
    return _audit(sid, lambda st, e: (overrides.remove(st, oid, who="user"), {"overrides": overrides.load(st)})[1])


class AuditChat(BaseModel):
    message: str
    run_id: Optional[str] = None


@router.post("/api/sessions/{sid}/audit/chat")
def audit_chat(sid: str, req: AuditChat):
    """The thin AI operator (v3 §8): a reply and the results of its tool calls."""
    from prism.audit import operator
    if not req.message.strip():
        raise HTTPException(422, "Write a message.")
    return _audit(sid, lambda st, e: operator.chat(st, req.message.strip()[:4000], req.run_id))


# ---------------------------------------------------------------- saved-schemas library (v3 §4.8)


@router.get("/api/library/{file_sha}")
def library_matches(file_sha: str):
    from prism.io import library
    return library.matches(file_sha)


class UseSaved(BaseModel):
    schema_id: str


@router.post("/api/sessions/{sid}/datasets/{did}/use-saved-schema")
def use_saved_schema(sid: str, did: str, req: UseSaved):
    from prism import store
    from prism.io import importer, library
    st = _load(sid)
    try:
        rep = library.use(st, did, req.schema_id)
    except importer.ImportRejected as e:
        _rejected(e)
    except store.SessionError as e:
        raise HTTPException(422, str(e))
    return {"report": rep, "session": session_view(store.load(sid))}


@router.get("/api/sessions/{sid}/datasets/{did}/explain-differences")
def explain_differences(sid: str, did: str):
    """Deterministic sentences for the import report's differences (no AI in import)."""
    import json
    from prism.io import importer
    st = _load(sid)
    p = st.dataset_dir(did) / "import_report.json"
    if not p.exists():
        raise HTTPException(404, "No import report for this dataset.")
    return {"explanations": importer.explain(json.loads(p.read_text(encoding="utf-8")))}


# ---------------------------------------------------------------- the final report


@router.get("/api/sessions/{sid}/final-report")
def final_report(sid: str, run_id: Optional[str] = None):
    from prism.audit import report
    return _audit(sid, lambda st, e: report.final_bundle(st, run_id))


@router.get("/api/sessions/{sid}/final-report.html")
def final_report_html(sid: str, run_id: Optional[str] = None, download: int = 1):
    from fastapi.responses import HTMLResponse
    from prism.audit import report
    html = _audit(sid, lambda st, e: report.final_html(st, run_id))
    headers = {"Content-Disposition": f'attachment; filename="prism_report_{sid}.html"'} if download else {}
    return HTMLResponse(html, headers=headers)
