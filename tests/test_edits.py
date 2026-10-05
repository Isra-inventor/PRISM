"""v2.4 stage 1: one edit layer (validation, provenance, undo) and column accounting."""

import json

import pytest

from backend import edits, session_log, workflow

HISTORY = {"processing_history": {q: {"answer": "not_sure"} for q in (
    "normalized", "log_transformed", "imputed", "batch_corrected", "features_or_samples_removed_before_upload")}}


def gid_of(f, col):
    return next(g["group_id"] for g in f.upload["groups"] if col in g["columns"])


def undo(f, edit_id=None, expect=200):
    r = f.c.post("/api/undo", json={"session_id": f.sid, "edit_id": edit_id})
    assert r.status_code == expect, r.text
    if expect == 200:
        f.upload, f.draft = r.json()["session"], r.json()["draft"]
    return r.json()


def events(sid):
    return [json.loads(l) for l in session_log.log_path(sid).read_text(encoding="utf-8").splitlines()]


def test_manual_edit_is_logged_and_undoable(flow):
    f = flow("C_mzmine_feature_table.csv")
    g = gid_of(f, "formula")
    before = f.draft["groups"][g]
    f.step("layout", {"layout": f.draft["layout"]["value"]})
    f.step("feature_id", {"feature_identity": {"group_ids": f.draft["feature_identity"]["group_ids"]}})
    f.step("annotations", {"items": [{"group_id": g, "keep": False, "label": "sum formula"}]})
    it = f.draft["groups"][g]
    assert it["keep"] is False and it["provenance"] == "user_set"
    assert it["excluded"]["by"] == "user" and it["excluded"]["at"] and it["excluded"]["reason"]
    assert f.draft["changes"][0]["summary"].startswith("Confirmed step 3")
    assert any(x["column"] == "formula" and x["file"] == "main" for x in f.draft["excluded_columns"])
    assert f.draft["column_ledger"]["main"]["excluded"] == 1
    ev = [e for e in events(f.sid) if e["event"] == "edit_applied" and e["op"] == "confirm_step"][-1]
    assert ev["group_changes"][g]["keep"] == [True, False]           # before and after are logged
    undo(f)
    it = f.draft["groups"][g]
    assert it["keep"] is True and it["label"] == before["label"] and "excluded" not in it
    assert f.draft["steps"]["annotations"] == "pending"              # the confirmation is undone too
    assert any(e["event"] == "patch_undone" for e in events(f.sid))


def test_undo_restores_structure_after_merge_and_split(flow):
    f = flow("A_maxquant_proteinGroups.txt")
    a, b = gid_of(f, "Score"), gid_of(f, "Peptides")
    n_groups = len(f.upload["groups"])
    r = f.c.post("/api/merge", json={"session_id": f.sid, "group_ids": [a, b]})
    assert r.status_code == 200
    f.upload, f.draft = r.json()["session"], r.json()["draft"]
    assert len(f.upload["groups"]) == n_groups - 1
    lfq = gid_of(f, "LFQ intensity S01")
    r = f.c.post("/api/split", json={"session_id": f.sid, "group_id": lfq, "columns": ["LFQ intensity S01"]})
    f.upload, f.draft = r.json()["session"], r.json()["draft"]
    assert len(f.upload["groups"]) == n_groups
    first = f.draft["changes"][-1]["edit_id"]                         # the merge (oldest)
    undo(f, first)                                                    # undoes the split and then the merge
    assert len(f.upload["groups"]) == n_groups and gid_of(f, "Score") == a and gid_of(f, "Peptides") == b
    assert len(f.upload["groups"][[g["group_id"] for g in f.upload["groups"]].index(lfq)]["columns"]) > 1
    assert list(f.draft["groups"]) == [g["group_id"] for g in f.upload["groups"]]   # file order kept
    assert undo(f, expect=422)["detail"] == "Nothing to undo."


def test_failed_edit_changes_nothing(flow):
    f = flow("C_mzmine_feature_table.csv")
    s = workflow.get_session(f.sid)
    before = json.dumps(s.draft, sort_keys=True)
    n = len(s.undo)
    with pytest.raises(workflow.StepError):    # two valid edits, then a contradicted one: all rolled back
        workflow.confirm_step(s, "annotations", {"items": [
            {"group_id": gid_of(f, "formula"), "label": "x"}, {"group_id": gid_of(f, "adduct"), "keep": False},
            {"group_id": gid_of(f, "compound_name"), "role": "value"}]})
    assert json.dumps(s.draft, sort_keys=True) == before and len(s.undo) == n
    with pytest.raises(edits.EditError):
        edits.apply_edit(s, "set_processing_history", HISTORY["processing_history"], actor="ai_patch")
    with pytest.raises(edits.EditError):
        edits.apply_edit(s, "no_such_op", {})


def test_undo_stack_is_kept_and_persisted(flow):
    f = flow("C_mzmine_feature_table.csv")
    s = workflow.get_session(f.sid)
    g = gid_of(f, "formula")
    for k in range(edits.UNDO_MAX + 5):
        edits.apply_edit(s, "edit_group", {"group_id": g, "fields": {"label": f"label {k}"}})
    s.save()
    assert len(s.undo) == edits.UNDO_MAX >= 100
    workflow._sessions.clear()                                        # reload from disk
    s2 = workflow.get_session(f.sid)
    assert len(s2.undo) == edits.UNDO_MAX
    edits.undo(s2)
    assert s2.draft["groups"][g]["label"] == f"label {edits.UNDO_MAX + 3}"


def test_ledger_files_and_finalized_is_immutable(flow):
    f = flow("B_diann_pg_matrix.tsv")
    f.confirm_all_as_proposed()                                       # (skips the metadata file)
    meta = "sample,group,age,note\nS01,ctrl,40,a\nS02,ctrl,41,b\nS03,case,50,c\n"
    r = f.c.post("/api/metadata-upload", data={"session_id": f.sid}, files={"file": ("meta.csv", meta.encode())})
    assert r.status_code == 200, r.text
    f.step("sample_info", {"metadata": {"columns": [
        {"column": "group", "audit_kind": "group"}, {"column": "age", "audit_kind": "covariate"},
        {"column": "note", "audit_kind": "other", "keep": False}]}})
    out = f.finalize()["schema"]
    led = out["column_ledger"]
    n_main = len(f.upload["groups"]) and sum(g["n_columns"] for g in f.upload["groups"])
    assert led["main"]["total"] == n_main == sum(led["main"][k] for k in
                                                  ("feature_id", "annotation", "value", "sample_id", "sample_metadata",
                                                   "excluded"))
    assert led["metadata"] == {"feature_id": 0, "annotation": 0, "value": 0, "sample_id": 1, "sample_metadata": 2,
                               "excluded": 1, "unresolved": 0, "unaccounted": 0, "total": 4}
    roles = [x["file_role"] for x in out["files"]]
    assert roles == ["main", "metadata"]
    md = out["files"][1]
    assert md["parse_report"]["delimiter"] == "comma" and md["parse_report"] is not out["files"][0]["parse_report"]
    assert md["join"]["key_column"] == "sample" and md["join"]["matched"] == 3
    assert {"column": "note", "file": "metadata"}.items() <= next(
        x for x in out["excluded_columns"] if x["column"] == "note").items()
    assert all(x["reason"] and x["by"] for x in out["excluded_columns"])
    # finalized: no more edits, no undo
    f.step("history", HISTORY, expect=422)
    undo(f, expect=422)


def test_unaccounted_column_blocks_finalize(flow):
    f = flow("C_mzmine_feature_table.csv")
    f.confirm_all_as_proposed()
    s = workflow.get_session(f.sid)
    g = s.groups_by_id[gid_of(f, "HMDB_ID")]
    s.groups = [x for x in s.groups if x is not g]                    # simulate a column in no category
    s.draft["groups"].pop(g["group_id"])
    r = f.finalize(expect=422)
    assert "Column accounting" in r["detail"] and "'HMDB_ID' is in no category" in r["detail"]
