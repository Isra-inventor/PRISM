"""PRISM Step 0 -- General schema recognition + guided confirmation (API).

Run from the repository root:
    python -m uvicorn backend.main:app --reload
then open http://127.0.0.1:8000

Flow: upload (parse + profile + group, deterministic) -> propose (the AI labels
the groups, briefed by backend/briefing.md; every claim validated; without the AI
a known-format signature gives the manual starting point) -> the wizard confirms
each step -> finalize (schema.json + canonical tables). Everything is logged to
backend/logs/<session_id>.jsonl.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .envfile import DOTENV_REPORT, PROJECT_DIR, _load_dotenv

_load_dotenv()

from . import llm_providers as llm  # noqa: E402  (after .env is loaded)
from . import outputs, workflow  # noqa: E402
from .parsing import InputError  # noqa: E402
from .schema import vocabulary_payload  # noqa: E402
from .session_log import log_path  # noqa: E402

MAX_BYTES = int(os.environ.get("PRISM_MAX_UPLOAD_MB", "200")) * 1024 * 1024
PREVIEW_ROWS = 10
FRONTEND_DIR = PROJECT_DIR / "frontend"

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s - %(message)s")
app = FastAPI(title="PRISM Step 0 - Schema recognition")

_ok, _why = llm.available()
if _ok:
    llm.say(f"AI provider: {llm.provider_name()}; models tried in order: {', '.join(llm.model_list())}")
else:
    llm.say(f"AI fallback OFF: {_why} The wizard works in manual mode. What PRISM checked:")
    for _line in DOTENV_REPORT:
        llm.say("  - " + _line)


# ---------------------------------------------------------------- progress

PROGRESS: "OrderedDict[str, dict]" = OrderedDict()
_PROGRESS_LOCK = threading.Lock()


def set_progress(pid, **fields):
    if not pid:
        return
    now = time.time()
    with _PROGRESS_LOCK:
        p = PROGRESS.get(pid)
        if p is None:
            p = PROGRESS[pid] = {"stage": "starting", "percent": 0, "message": "", "batches_done": 0,
                                 "batches_total": 0, "log": [], "started": now}
            while len(PROGRESS) > 50:
                PROGRESS.popitem(last=False)
        p.update(fields)
        p["updated"] = now
        if fields.get("message"):
            p["log"] = (p["log"] + [{"t": round(now - p["started"], 1), "msg": fields["message"]}])[-8:]


def ai_progress(pid, start=10, end=99, stage="ai"):
    def cb(done, total, message):
        pct = start + (end - start) * done / total if total else start
        set_progress(pid, stage=stage, percent=int(pct), batches_done=done, batches_total=total, message=message)
    return cb


@app.get("/api/progress/{pid}")
def get_progress(pid: str):
    with _PROGRESS_LOCK:
        p = PROGRESS.get(pid)
        if p is None:
            raise HTTPException(404, "Unknown progress id.")
        out = dict(p, log=list(p["log"]))
    out["elapsed_s"] = round(time.time() - out["started"], 1)
    out["idle_s"] = round(time.time() - out["updated"], 1)
    return out


async def run(pid, fn, *args):
    try:
        result = await run_in_threadpool(fn, *args)
    except HTTPException as e:
        set_progress(pid, stage="error", message=str(e.detail))
        raise
    except Exception as e:
        set_progress(pid, stage="error", message=f"{type(e).__name__}: {e}")
        raise
    set_progress(pid, stage="done", percent=100, message="Done")
    return result


def session_or_404(sid):
    try:
        return workflow.get_session(sid)
    except KeyError:
        raise HTTPException(404, "Unknown or expired session. Please upload the file again.")


# ---------------------------------------------------------------- read-only endpoints

@app.get("/api/health")
def health():
    ok, why = llm.available()
    return {"status": "ok", "ai_available": ok, "ai_unavailable_reason": None if ok else why,
            "ai_provider": llm.provider_name(), "ai_models": llm.model_list() if ok else [],
            "example_values_sent": os.environ.get("AI_SEND_EXAMPLE_VALUES", "true")}


@app.get("/api/vocabulary")
def vocabulary():
    return vocabulary_payload()


@app.get("/api/sessions/{sid}/log")
def session_log(sid: str):
    s = session_or_404(sid)
    path = log_path(s.sid)
    if not path.exists():
        raise HTTPException(404, "No log for this session.")
    return JSONResponse([json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()])


# ---------------------------------------------------------------- upload

def _group_public(g):
    return {k: g.get(k) for k in ("group_id", "columns", "indices", "n_columns", "kind", "origin", "pattern", "type",
                                  "profile", "histogram", "sample_id_rule", "sample_names", "split_from",
                                  "merged_from")}


def session_payload(s):
    return {
        "session_id": s.sid, "filename": s.filename, "sha256": s.sha,
        "header": s.table["header"], "labels": s.cols.labels,
        "preview_rows": s.table["rows"][:PREVIEW_ROWS],
        "n_rows": len(s.table["rows"]), "n_columns": len(s.table["header"]),
        "parse_report": s.table["parse_report"], "layout_hints": s.hints,
        "groups": [_group_public(g) for g in s.groups],
    }


def _do_upload(filename, raw, pid):
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, f"File is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    set_progress(pid, stage="reading", percent=4, message="Parsing the table")
    try:
        s = workflow.create_session(filename, raw)
    except InputError as e:
        raise HTTPException(415, str(e))
    set_progress(pid, stage="detecting", percent=9, message=f"Profiled {len(s.table['header'])} columns")
    s.log("upload", {"filename": s.filename, "bytes": len(raw), "sha256": s.sha,
                     "n_rows": len(s.table["rows"]), "n_columns": len(s.table["header"]),
                     "header": s.table["header"]})
    s.log("parse_report", s.table["parse_report"])
    s.log("facts", {"layout_hints": s.hints, "shared_name_parts": [
        {"column": c, **a} for c, a in zip(s.cols.labels, s.affixes)]})
    return session_payload(s)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), progress_id: Optional[str] = Form(None)):
    set_progress(progress_id, stage="reading", percent=2, message="File received by the server")
    raw = await file.read(MAX_BYTES + 1)
    return await run(progress_id, _do_upload, file.filename or "upload", raw, progress_id)


# ---------------------------------------------------------------- propose / steps

class ProposeRequest(BaseModel):
    session_id: str
    ai: bool = True
    progress_id: Optional[str] = None


def _do_propose(req):
    s = session_or_404(req.session_id)
    with s.lock:
        set_progress(req.progress_id, stage="ai", percent=10,
                     message="Asking the AI to label the column groups" if req.ai else "Preparing manual mode")
        try:
            workflow.propose(s, ai_on=req.ai, on_progress=ai_progress(req.progress_id))
        except workflow.StepError as e:
            raise HTTPException(422, str(e))
        return {"session": session_payload(s), "draft": workflow.public_draft(s), "digests": s.digests}


@app.post("/api/propose")
async def propose(req: ProposeRequest):
    return await run(req.progress_id, _do_propose, req)


class StepRequest(BaseModel):
    session_id: str
    step_id: str
    decision: dict = {}
    progress_id: Optional[str] = None


def _do_step(req):
    s = session_or_404(req.session_id)
    with s.lock:
        if s.draft is None:
            raise HTTPException(409, "Run /api/propose first.")
        try:
            res = workflow.confirm_step(s, req.step_id, req.decision, on_progress=ai_progress(req.progress_id))
        except workflow.StepError as e:
            raise HTTPException(422, str(e))
        res["session"] = session_payload(s) if res["reproposed"] else None
        res["digests"] = s.digests if res["reproposed"] else None
        return res


@app.post("/api/confirm-step")
async def confirm_step(req: StepRequest):
    return await run(req.progress_id, _do_step, req)


class ReconsiderRequest(BaseModel):
    session_id: str
    group_ids: List[str] = []
    group_id: Optional[str] = None  # older clients: a single group
    user_hint: str = ""
    progress_id: Optional[str] = None


def _do_reconsider(req):
    s = session_or_404(req.session_id)
    with s.lock:
        try:
            res = workflow.reconsider(s, req.group_ids or [req.group_id], req.user_hint,
                                      on_progress=ai_progress(req.progress_id))
            res["session"] = session_payload(s)  # groups change when the AI's split is applied
            return res
        except workflow.StepError as e:
            raise HTTPException(422, str(e))


@app.post("/api/reconsider")
async def reconsider(req: ReconsiderRequest):
    return await run(req.progress_id, _do_reconsider, req)


class MergeRequest(BaseModel):
    session_id: str
    group_ids: List[str]
    user_hint: str = ""


class SplitRequest(BaseModel):
    session_id: str
    group_id: str
    columns: List[str]


class GroupColumnsRequest(BaseModel):
    session_id: str
    columns: List[str]
    reason: str = ""


def _structure_op(req, fn, *args):
    s = session_or_404(req.session_id)
    with s.lock:
        if s.draft is None:
            raise HTTPException(409, "Run /api/propose first.")
        try:
            res = fn(s, *args)
        except workflow.StepError as e:
            raise HTTPException(422, str(e))
        if "draft" in res:
            res["session"] = session_payload(s)
        return res


class SessionRequest(BaseModel):
    session_id: str


@app.post("/api/consolidate")
async def consolidate(req: SessionRequest):
    """Retry the cross-chunk consolidation call (e.g. after a quota error)."""
    return await run_in_threadpool(_structure_op, req, workflow.retry_consolidation)


class ChatRequest(BaseModel):
    session_id: str
    message: str = ""
    step: Optional[str] = None
    selection: List[str] = []
    progress_id: Optional[str] = None


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Talk to the AI. It replies and may propose patches; nothing is applied."""
    def work():
        set_progress(req.progress_id, stage="ai", percent=20, message="The AI is reading your message")
        return _structure_op(req, workflow.chat, req.message, req.step, req.selection)
    return await run(req.progress_id, work)


class SettingsRequest(BaseModel):
    session_id: str
    auto_apply: Optional[bool] = None


@app.post("/api/settings")
async def settings(req: SettingsRequest):
    """Per-session settings: 'apply what I ask for automatically' (default off; still logged and undoable)."""
    return await run_in_threadpool(_structure_op, req, workflow.set_settings, req.auto_apply)


class PatchRequest(BaseModel):
    session_id: str
    patch_ids: List[str]
    confirm_large: List[str] = []
    overrides: dict = {}


@app.post("/api/patch/apply")
async def patch_apply(req: PatchRequest):
    """Your click: apply these patches through the edit layer (per-patch applied / held / rejected)."""
    return await run_in_threadpool(_structure_op, req, workflow.apply_patches, req.patch_ids, req.confirm_large,
                                   req.overrides)


@app.post("/api/patch/dismiss")
async def patch_dismiss(req: PatchRequest):
    return await run_in_threadpool(_structure_op, req, workflow.dismiss_patches, req.patch_ids)


class QuestionRequest(BaseModel):
    session_id: str
    question_id: str
    option_ids: List[str] = []
    text: Optional[str] = None


@app.post("/api/question/answer")
async def question_answer(req: QuestionRequest):
    """Your answer to a queued question: its options' patches go through the edit layer."""
    return await run_in_threadpool(_structure_op, req, workflow.answer_question, req.question_id, req.option_ids, req.text)


@app.post("/api/question/dismiss")
async def question_dismiss(req: QuestionRequest):
    return await run_in_threadpool(_structure_op, req, workflow.dismiss_question, req.question_id, req.text)


@app.post("/api/merge")
async def merge(req: MergeRequest):
    """Your decision: merge these groups (applied by code only on this confirmation)."""
    return await run_in_threadpool(_structure_op, req, workflow.merge_groups, req.group_ids, req.user_hint)


@app.post("/api/split")
async def split(req: SplitRequest):
    return await run_in_threadpool(_structure_op, req, workflow.split_columns, req.group_id, req.columns)


@app.post("/api/group-columns")
async def group_columns(req: GroupColumnsRequest):
    return await run_in_threadpool(_structure_op, req, workflow.group_columns, req.columns, req.reason)


class UndoRequest(BaseModel):
    session_id: str
    edit_id: Optional[str] = None


@app.post("/api/undo")
async def undo(req: UndoRequest):
    """Undo the latest change, or every change back to and including edit_id."""
    def fn(s):
        return {"undone": workflow.edits.undo(s, req.edit_id), "draft": workflow.public_draft(s)}
    return await run_in_threadpool(_structure_op, req, fn)


@app.post("/api/metadata-upload")
async def metadata_upload(session_id: str = Form(...), file: UploadFile = File(...)):
    s = session_or_404(session_id)
    raw = await file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "Metadata file is too large.")

    def work():
        with s.lock:
            try:
                d = workflow.upload_metadata(s, file.filename or "metadata.csv", raw)
            except InputError as e:
                raise HTTPException(415, str(e))
            except workflow.StepError as e:
                raise HTTPException(422, str(e))
            (s.dir / "metadata_source.csv").write_bytes(raw)
            return {"draft": d}
    return await run_in_threadpool(work)


# ---------------------------------------------------------------- finalize / export

class FinalizeRequest(BaseModel):
    session_id: str


ARTIFACT_TYPES = {".json": "application/json", ".csv": "text/csv"}


def _do_finalize(req):
    s = session_or_404(req.session_id)
    with s.lock:
        try:
            schema, artifacts, flags = outputs.build(s)
        except outputs.OutputError as e:
            raise HTTPException(422, str(e))
        out = s.dir / "outputs"
        out.mkdir(parents=True, exist_ok=True)
        for name, text in artifacts.items():
            (out / name).write_bytes(text.encode("utf-8"))  # exact bytes; no newline translation
        (out / "schema.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
        s.draft["steps"]["review"] = "confirmed"
        s.draft["finalized"] = True
        s.save()
        names = ["schema.json"] + list(artifacts)
        s.log("finalize", {"artifacts": names, "integrity_flags": flags, "schema": schema})
        return {"schema": schema, "artifacts": names, "integrity_flags": flags}


@app.post("/api/finalize")
async def finalize(req: FinalizeRequest):
    return await run_in_threadpool(_do_finalize, req)


@app.get("/api/export/{sid}/{artifact}")
def export(sid: str, artifact: str):
    s = session_or_404(sid)
    if "/" in artifact or "\\" in artifact or artifact.startswith("."):
        raise HTTPException(400, "Bad artifact name.")
    path = s.dir / "outputs" / artifact
    if not path.exists():
        raise HTTPException(404, "Not found. Finish the wizard first.")
    ext = os.path.splitext(artifact)[1]
    return Response(path.read_bytes(), media_type=ARTIFACT_TYPES.get(ext, "application/octet-stream"),
                    headers={"Content-Disposition": f'attachment; filename="{s.sid}_{artifact}"'})


@app.get("/tool")
def tool_page():
    return FileResponse(FRONTEND_DIR / "tool.html")


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
