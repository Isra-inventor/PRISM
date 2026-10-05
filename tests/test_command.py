"""'Ask the AI to do it': instruction -> previewed actions -> applied only on confirmation."""

import json

from backend import mock_llm

_ORIGINAL = mock_llm.MockLLM.respond


def plan(f, text, expect=200):
    r = f.c.post("/api/ai-command", json={"session_id": f.sid, "instruction": text})
    assert r.status_code == expect, r.text
    if expect == 200:
        f.draft = r.json()["draft"]
        return r.json()["command"]
    return r.json()


def apply(f, cmd, accept=None):
    r = f.c.post("/api/apply-command", json={"session_id": f.sid, "command_id": cmd["id"], "accept": accept})
    assert r.status_code == 200, r.text
    f.upload, f.draft = r.json()["session"], r.json()["draft"]
    return r.json()


def test_exclude_many_columns_with_one_instruction(flow):
    f = flow("A_maxquant_proteinGroups.txt")
    before = {g: it["keep"] for g, it in f.draft["groups"].items()}
    cmd = plan(f, "don't include the score, coverage and peptide count columns in the output")
    assert cmd["reply"] and len(cmd["actions"]) == 1 and cmd["actions"][0]["usable"]
    assert "leave out of the outputs" in cmd["actions"][0]["description"]
    assert {g: it["keep"] for g, it in f.draft["groups"].items()} == before       # preview only: nothing applied
    res = apply(f, cmd)
    assert res["applied"] and not f.draft.get("pending_command")
    dropped = {c for g in f.upload["groups"] if not f.draft["groups"][g["group_id"]]["keep"] for c in g["columns"]}
    assert {"Score", "Sequence coverage [%]", "Peptides"} <= dropped and "Protein IDs" not in dropped
    assert all(f.draft["groups"][g]["provenance"] == "ai_proposed_confirmed"
               for g in f.draft["groups"] if not f.draft["groups"][g]["keep"])
    f.confirm_all_as_proposed()
    out = f.finalize()
    excl = {x["column"] for x in out["schema"]["excluded_columns"]}
    assert "Score" in excl
    assert "Score" not in f.export("feature_metadata.csv").splitlines()[0]


def test_invalid_actions_are_checked_before_you_see_them(flow, monkeypatch):
    def wild(system, prompt):
        if not prompt.startswith("Command"):
            return _ORIGINAL(system, prompt)
        p = json.loads(prompt.split("\n", 1)[1])
        fid = next(g["group_id"] for g in p["groups"] if g["role"] == "feature_id")
        text = next(g["group_id"] for g in p["groups"] if g["columns"] == ["Gene names"])
        return json.dumps({"reply": "ok", "not_possible": "I cannot delete rows.", "actions": [
            {"action": "set_keep", "group_ids": [fid, "ghost"], "keep": False, "reason": "x"},
            {"action": "set_role", "group_ids": [text], "role": "value", "reason": "x"},
            {"action": "set_audit_kind", "group_ids": [text], "audit_kind": "foo", "reason": "x"}]})
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(wild))
    f = flow("A_maxquant_proteinGroups.txt")
    cmd = plan(f, "remove the decoy rows and exclude everything")
    assert cmd["not_possible"] == "I cannot delete rows."
    keep, role, kind = cmd["actions"]
    assert not keep["usable"] and any("identifier" in p for p in keep["problems"]) and any("ghost" in p for p in keep["problems"])
    assert not role["usable"] and any("non-numeric" in p for p in role["problems"])
    assert not kind["usable"]
    res = apply(f, cmd)
    assert res["applied"] == []                                      # nothing usable, nothing changed


def test_samples_and_discard(flow):
    f = flow("C_mzmine_feature_table.csv")
    cmd = plan(f, "mark S1* samples as non-study")
    assert cmd["actions"][0]["sample_ids"] == ["S10", "S11", "S12"]
    r = f.c.post("/api/discard-command", json={"session_id": f.sid})
    assert r.status_code == 200 and r.json()["draft"]["pending_command"] is None
    r = f.c.post("/api/apply-command", json={"session_id": f.sid, "command_id": cmd["id"]})
    assert r.status_code == 422                                      # discarded: cannot be applied
    cmd = plan(f, "mark S1* samples as non-study")
    apply(f, cmd)
    assert f.draft["samples"]["S11"]["is_study_sample"] is False and f.draft["samples"]["S01"]["is_study_sample"]


def test_needs_the_ai(flow):
    f = flow("C_mzmine_feature_table.csv", ai=False)
    assert "AI is off" in plan(f, "exclude formula", expect=422)["detail"]
