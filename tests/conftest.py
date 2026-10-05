import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Every test: mock LLM, and sessions / logs / cache in a temp dir."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "PRISM_LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    from backend import ai, literature, session_log, workflow
    import fake_europepmc
    monkeypatch.setattr(literature, "_get", lambda url, kind: fake_europepmc.respond(url))  # never the network
    monkeypatch.setattr(literature, "CACHE_DIR", tmp_path / "litcache")
    monkeypatch.setattr(workflow, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(session_log, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(ai, "CACHE_DIR", tmp_path / "cache")
    workflow._sessions.clear()
    return tmp_path


def fixture_bytes(name):
    return (FIX / name).read_bytes()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from backend.main import app
    return TestClient(app)


class Flow:
    """Drive the API like the wizard does."""

    def __init__(self, client, name, ai=True, content=None):
        self.c = client
        r = client.post("/api/upload", files={"file": (name, content if content is not None else fixture_bytes(name))})
        assert r.status_code == 200, r.text
        self.upload = r.json()
        self.sid = self.upload["session_id"]
        r = client.post("/api/propose", json={"session_id": self.sid, "ai": ai})
        assert r.status_code == 200, r.text
        self.draft = r.json()["draft"]
        self.digests = r.json()["digests"]
        self.upload = r.json()["session"]   # groups may have been split on the AI's suggestion

    def group(self, column):
        for g in self.upload["groups"] if not getattr(self, "groups", None) else self.groups:
            if column in g["columns"]:
                return g["group_id"]
        raise KeyError(column)

    def step(self, step, decision, expect=200):
        r = self.c.post("/api/confirm-step", json={"session_id": self.sid, "step_id": step, "decision": decision})
        assert r.status_code == expect, r.text
        if r.status_code == 200:
            self.draft = r.json()["draft"]
            if r.json().get("session"):
                self.upload = r.json()["session"]
        return r.json()

    def confirm_all_as_proposed(self):
        d = self.draft
        self.step("layout", {"layout": d["layout"]["value"],
                             "assays": [{k: a.get(k) for k in ("assay_label", "omics_type", "source_software",
                                                                "in_supported_scope", "scope_reason")}
                                        for a in d["assays"]]})
        d = self.draft
        dec = {"feature_identity": {"group_ids": d["feature_identity"]["group_ids"]}}
        if d["sample_id_group"]["value"]:
            dec["sample_id_group"] = d["sample_id_group"]["value"]
        self.step("feature_id", dec)
        for st in ("annotations", "values", "samples"):
            if self.draft["steps"][st] != "not_applicable":
                dec = {}
                if st == "annotations":  # pick a flagged value where none was proposed
                    items = [{"group_id": gid, "flagged_values": [next(v for v in it["flag_values"] if v)]}
                             for gid, it in self.draft["groups"].items()
                             if it.get("marks_rows_as_suspect") and it.get("flag_values")
                             and not it.get("flagged_values") and any(it["flag_values"])]
                    if items:
                        dec["items"] = items
                self.step(st, dec)
        if d["layout"]["value"] == "samples_in_columns":
            self.step("sample_info", {"metadata": {"skip": True}})
        else:
            self.step("sample_info", {})
        self.step("history", {"processing_history": {q: {"answer": "not_sure"} for q in (
            "normalized", "log_transformed", "imputed", "batch_corrected",
            "features_or_samples_removed_before_upload")}})

    def finalize(self, expect=200):
        r = self.c.post("/api/finalize", json={"session_id": self.sid})
        assert r.status_code == expect, r.text
        return r.json()

    def export(self, name):
        r = self.c.get(f"/api/export/{self.sid}/{name}")
        assert r.status_code == 200, r.text
        return r.text


@pytest.fixture
def flow(client):
    return lambda name, **kw: Flow(client, name, **kw)
