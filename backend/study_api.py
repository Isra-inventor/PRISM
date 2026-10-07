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
