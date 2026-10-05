"""v2.4 stage 4: the chat (one way to talk to the AI): context, history, auto-apply."""

import json

from backend import mock_llm

_ORIGINAL = mock_llm.MockLLM.respond


def post(f, path, expect=200, **body):
    r = f.c.post(path, json={"session_id": f.sid, **body})
    assert r.status_code == expect, r.text
    if expect == 200:
        f.draft = r.json()["draft"]
        if r.json().get("session"):
            f.upload = r.json()["session"]
    return r.json()


def test_context_has_summary_history_step_selection_and_no_rows(flow, monkeypatch):
    seen = []

    def respond(system, prompt):
        if prompt.startswith("Chat"):
            seen.append(json.loads(prompt.split("\n", 1)[1]))
            assert "Keep replies short" in system and "set_flag_values" in system
            return json.dumps({"reply": f"turn {len(seen)}", "patches": [], "questions": []})
        return _ORIGINAL(system, prompt)
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(respond))
    f = flow("C_mzmine_feature_table.csv")
    for k in range(12):
        post(f, "/api/chat", message=f"question {k} about formula", step="annotations", selection=["adduct"])
    ctx = seen[-1]
    assert ctx["step"] == "annotations" and ctx["selection"] == ["adduct"]
    assert len(ctx["history"]) == 10 and ctx["history"][-1] == {"user": "question 10 about formula", "assistant": "turn 11"}
    assert {c["column"] for c in ctx["digest_of_referred_columns"]} == {"adduct", "formula"}
    cols = {c["column"]: c for c in ctx["columns"]}
    assert cols["formula"]["role"] == "feature_annotation" and cols["formula"]["file"] == "main"
    assert any(g["role"] == "value" and len(g["columns"]) == 17 for g in ctx["groups"])
    assert ctx["settings"]["raw_rows_sent"] is False
    raw = json.dumps(ctx)
    assert "Glucose" not in raw and "694.1481" not in raw                  # never a raw row
    assert f.draft["last_chat_context"]["message"] == "question 11 about formula"   # 'What the AI saw'
    assert "context" not in f.draft["chat"][-1]


def test_auto_apply_only_what_is_safe(flow, monkeypatch):
    def respond(system, prompt):
        if prompt.startswith("Chat"):
            return json.dumps({"reply": "ok", "questions": [], "patches": [
                {"patch_id": "a", "op": "set_label", "target": {"selector": {"columns": ["formula"]}},
                 "args": {"label": "sum formula"}, "reason": "x"},
                {"patch_id": "b", "op": "set_block_keep", "target": {"selector": {"role": "value"}},
                 "args": {"keep": True}, "reason": "x"},
                {"patch_id": "c", "op": "set_keep", "target": {"selector": {"columns": ["HMDB_ID"]}},
                 "args": {"keep": False}, "reason": "x"}]})
        return _ORIGINAL(system, prompt)
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(respond))
    f = flow("C_mzmine_feature_table.csv")
    m = post(f, "/api/chat", message="relabel")["message"]
    assert m["auto_applied"] == [] and all(p["status"] == "pending" for p in f.draft["patches"])   # default off
    post(f, "/api/settings", auto_apply=True)
    assert f.draft["settings"]["auto_apply"] is True
    m = post(f, "/api/chat", message="relabel again")["message"]
    st = {p["ai_patch_id"]: p["status"] for p in f.draft["patches"] if p["message_id"] == m["message_id"]}
    assert st == {"a": "applied", "b": "pending", "c": "applied"}       # a block op always waits for a click
    assert len(m["auto_applied"]) == 2
    post(f, "/api/undo")                                               # still undoable
    g = next(x["group_id"] for x in f.upload["groups"] if x["columns"] == ["HMDB_ID"])
    assert f.draft["groups"][g]["keep"] is True
