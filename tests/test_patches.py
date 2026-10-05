"""v2.4 stage 2: the AI acts through patches: selectors, closed ops, invariants, preview, apply, undo."""

import csv
import io
import json

from backend import mock_llm

_ORIGINAL = mock_llm.MockLLM.respond


def chat(f, text, expect=200, **kw):
    r = f.c.post("/api/chat", json={"session_id": f.sid, "message": text, **kw})
    assert r.status_code == expect, r.text
    if expect == 200:
        f.draft = r.json()["draft"]
        f.upload = r.json()["session"]
        return r.json()["message"]
    return r.json()


def patches_of(f, msg):
    by = {p["patch_id"]: p for p in f.draft["patches"]}
    return [by[p] for p in msg["patch_ids"]]


def apply(f, ids, confirm_large=(), overrides=None):
    r = f.c.post("/api/patch/apply", json={"session_id": f.sid, "patch_ids": ids, "confirm_large": list(confirm_large),
                                           "overrides": overrides or {}})
    assert r.status_code == 200, r.text
    f.upload, f.draft = r.json()["session"], r.json()["draft"]
    return r.json()["results"]


def fake_chat(patches, reply="I've prepared a change."):
    def respond(system, prompt):
        if not prompt.startswith("Chat"):
            return _ORIGINAL(system, prompt)
        return json.dumps({"reply": reply, "patches": patches, "questions": []})
    return staticmethod(respond)


def annotation_cols(f):
    return {c for g in f.upload["groups"] if f.draft["groups"][g["group_id"]]["role"] == "feature_annotation"
            for c in g["columns"]}


def test_exclude_all_annotations_except_one(flow):
    f = flow("A_maxquant_proteinGroups.txt")
    ann = annotation_cols(f) - {"Gene names"}
    msg = chat(f, "exclude all annotation columns except Gene names")
    assert "prepared" in msg["reply"]
    (p,) = patches_of(f, msg)
    assert p["status"] == "pending" and p["op"] == "set_keep"
    assert set(p["resolved"]["columns"]) == ann and p["n_columns"] == len(ann)       # the card shows the count
    assert p["preview"]["ledger_after"]["main"]["excluded"] == len(ann)
    assert p["consequences"] and not any(it["keep"] is False for it in f.draft["groups"].values())  # nothing applied
    res = apply(f, [p["patch_id"]])
    assert res[0]["status"] == "applied"
    excl = {x["column"]: x for x in f.draft["excluded_columns"]}
    assert set(excl) == ann and all(x["by"] == "ai_patch" and x["reason"] == p["reason"] for x in excl.values())
    led = f.draft["column_ledger"]["main"]
    assert led["excluded"] == len(ann) and led["unaccounted"] == 0 and led["annotation"] == 1
    assert all(f.draft["groups"][g]["provenance"] == "ai_proposed_confirmed"
               for g in f.draft["groups"] if f.draft["groups"][g]["keep"] is False)
    fid = f.draft["feature_identity"]["group_ids"][0]
    assert f.draft["groups"][fid]["keep"] is True
    p = next(x for x in f.draft["patches"] if x["patch_id"] == p["patch_id"])
    assert p["status"] == "applied" and p["edit_id"] == f.draft["changes"][0]["edit_id"]
    r = f.c.post("/api/undo", json={"session_id": f.sid, "edit_id": p["edit_id"]})
    f.upload, f.draft = r.json()["session"], r.json()["draft"]
    assert not f.draft["excluded_columns"] and f.draft["column_ledger"]["main"]["annotation"] == len(ann) + 1
    assert next(x for x in f.draft["patches"] if x["patch_id"] == p["patch_id"])["status"] == "pending"


def test_rejections_are_shown_with_reasons(flow, monkeypatch):
    monkeypatch.setattr(mock_llm.MockLLM, "respond", fake_chat([
        {"patch_id": "a", "op": "set_keep", "target": {"selector": {"role": "feature_id"}}, "args": {"keep": False},
         "reason": "x"},
        {"patch_id": "b", "op": "set_label", "target": {"selector": {"columns": ["Protein IDz"]}},
         "args": {"label": "y"}, "reason": "x"},
        {"patch_id": "c", "op": "set_label", "target": {"selector": {"columns": ["processing_history"]}},
         "args": {"label": "normalized: yes"}, "reason": "x"},
        {"patch_id": "d", "op": "delete_rows", "target": {"selector": {"role": "value"}}, "args": {}, "reason": "x"},
        {"patch_id": "e", "op": "set_block_keep", "target": {"selector": {"role": "value"}}, "args": {"keep": False},
         "reason": "x"},
        {"patch_id": "f", "op": "set_role", "target": {"selector": {"columns": ["Gene names"]}}, "args": {"role": "value"},
         "reason": "x"},
    ]))
    f = flow("A_maxquant_proteinGroups.txt")
    ps = {p["ai_patch_id"]: p for p in patches_of(f, chat(f, "do things"))}
    assert all(p["status"] == "rejected" for p in ps.values())
    assert "feature ID column 'Protein IDs' cannot be excluded" in ps["a"]["problems"][0]
    assert "'Protein IDz'" in ps["b"]["problems"][0] and "does not exist" in ps["b"]["problems"][0].replace("do(es)", "does")
    assert "answered by you only" in ps["c"]["problems"][0]
    assert "not an allowed operation" in ps["d"]["problems"][0]
    assert "At least one value block must stay kept" in ps["e"]["problems"][0]
    assert "non-numeric" in ps["f"]["problems"][0]
    r = apply(f, [p["patch_id"] for p in ps.values()])
    assert all(x["status"] == "rejected" for x in r)                  # nothing changed
    assert not f.draft["changes"]


def test_large_ops_need_an_explicit_click(flow, monkeypatch):
    monkeypatch.setattr(mock_llm.MockLLM, "respond", fake_chat([
        {"patch_id": "a", "op": "set_label", "target": {"selector": {"name_starts_with": "LFQ intensity"}},
         "args": {"label": "label-free quantification"}, "reason": "x"}]))
    header = ["id"] + [f"LFQ intensity S{k:02d}" for k in range(30)]
    rows = [[f"P{r}"] + [str(1000 + r * k) for k in range(30)] for r in range(40)]
    buf = io.StringIO()
    csv.writer(buf).writerows([header] + rows)
    f = flow("lfq.csv", content=buf.getvalue().encode())
    (p,) = patches_of(f, chat(f, "rename"))
    assert p["status"] == "pending" and p["large"] and p["n_columns"] == 30
    (r,) = apply(f, [p["patch_id"]])
    assert r["status"] == "held" and "30 columns" in r["reason"]
    assert not f.draft["changes"]
    (r,) = apply(f, [p["patch_id"]], confirm_large=[p["patch_id"]])
    assert r["status"] == "applied"
    g = next(g for g in f.upload["groups"] if "LFQ intensity S01" in g["columns"])
    assert f.draft["groups"][g["group_id"]]["label"] == "label-free quantification"


def test_partial_group_is_isolated_and_edited_args_are_corrected(flow, monkeypatch):
    monkeypatch.setattr(mock_llm.MockLLM, "respond", fake_chat([
        {"patch_id": "a", "op": "set_block_keep", "target": {"selector": {"columns": ["LFQ intensity S01", "LFQ intensity S02"]}},
         "args": {"keep": False}, "reason": "outliers"},
        {"patch_id": "b", "op": "set_label", "target": {"selector": {"columns": ["Gene names"]}},
         "args": {"label": "gene"}, "reason": "x"}]))
    f = flow("A_maxquant_proteinGroups.txt")
    a, b = patches_of(f, chat(f, "x"))
    assert a["status"] == "pending" and "taken out of their group" in a["warnings"][0]
    n_before = len(f.upload["groups"])
    res = apply(f, [a["patch_id"], b["patch_id"]], overrides={b["patch_id"]: {"label": "HGNC gene symbol"}})
    assert [x["status"] for x in res] == ["applied", "applied"]
    assert len(f.upload["groups"]) == n_before + 1
    g = next(g for g in f.upload["groups"] if "LFQ intensity S01" in g["columns"])
    assert g["columns"] == ["LFQ intensity S01", "LFQ intensity S02"] and f.draft["groups"][g["group_id"]]["keep"] is False
    gn = next(g for g in f.upload["groups"] if g["columns"] == ["Gene names"])["group_id"]
    assert f.draft["groups"][gn]["label"] == "HGNC gene symbol"
    assert f.draft["groups"][gn]["provenance"] == "ai_proposed_corrected"


def test_samples_flags_and_assays(flow, monkeypatch):
    f = flow("C_mzmine_feature_table.csv")
    (p,) = patches_of(f, chat(f, "mark S1* samples as non-study"))
    assert p["resolved"]["samples"] == ["S10", "S11", "S12"]
    apply(f, [p["patch_id"]])
    assert f.draft["samples"]["S11"]["is_study_sample"] is False and f.draft["samples"]["S01"]["is_study_sample"]
    assert f.draft["samples"]["S11"]["provenance"] == "ai_proposed_confirmed"
    # flags: several values of one column
    monkeypatch.setattr(mock_llm.MockLLM, "respond", fake_chat([
        {"patch_id": "a", "op": "set_flag_values", "target": {"selector": {"columns": ["adduct"]}},
         "args": {"flagged_values": ["[M+H]+", "[M+Na]+"]}, "reason": "x"},
        {"patch_id": "b", "op": "set_flag_values", "target": {"selector": {"columns": ["adduct"]}},
         "args": {"flagged_values": ["nope"]}, "reason": "x"}]))
    a, b = patches_of(f, chat(f, "flag"))
    assert b["status"] == "rejected" and "do not occur" in b["problems"][0]
    apply(f, [a["patch_id"]])
    g = next(g for g in f.upload["groups"] if g["columns"] == ["adduct"])["group_id"]
    it = f.draft["groups"][g]
    assert it["marks_rows_as_suspect"] and it["flagged_values"] == ["[M+H]+", "[M+Na]+"]
    assert it["n_flagged"] == sum(it["flag_values"][v] for v in ("[M+H]+", "[M+Na]+"))


def test_merge_assays_and_derive_feature_annotation(flow, monkeypatch):
    header = ["ID"] + [f"{c}_{k}" for c in ("Amino Acid", "Lipid", "") for k in (10, 20, 30)]
    rows = [[f"S{r:02d}"] + [str(1 + (r + k) % 7 / 10) for k in range(9)] for r in range(6)]
    buf = io.StringIO()
    csv.writer(buf).writerows([header] + rows)
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(lambda system, prompt: json.dumps({
        "layout": {"value": "samples_in_rows", "confidence": 0.9, "evidence": "x"},
        "assays": [{"assay_label": "aa", "omics_type": "metabolomics", "in_supported_scope": "yes", "confidence": 1,
                    "evidence": "x"}, {"assay_label": "lip", "omics_type": "lipidomics", "in_supported_scope": "yes",
                                       "confidence": 1, "evidence": "x"}],
        "groups": [{"group_id": "i", "columns": ["ID"], "role": "sample_id", "label": "sample", "confidence": 1, "evidence": "x"},
                   {"group_id": "a", "columns": header[1:4], "role": "value", "assay_label": "aa", "label": "aa",
                    "confidence": 1, "evidence": "x"},
                   {"group_id": "b", "columns": header[4:], "role": "value", "assay_label": "lip", "label": "lip",
                    "confidence": 1, "evidence": "x"}]}) if prompt.startswith("Digest") else json.dumps({
        "reply": "ok", "questions": [], "patches": [
            {"patch_id": "m", "op": "merge_assays", "args": {"assay_labels": ["aa", "lip"], "label": "metabolon"}, "reason": "x"},
            {"patch_id": "d", "op": "derive_feature_annotation", "reason": "x",
             "args": {"source": "column_headers", "rule": {"delimiter": "_", "occurrence": "first", "parts": [
                 {"name": "feature_class", "label": "metabolite class"},
                 {"name": "feature_numeric_id", "label": "compound ID"}]}}}]})))
    f = flow("metab.csv", content=buf.getvalue().encode())
    assert [a["assay_label"] for a in f.draft["assays"]] == ["aa", "lip"]
    m, d = patches_of(f, chat(f, "one measurement; split the names"))
    assert m["preview"]["assays"] == [["aa", "lip"], ["metabolon"]]
    der = d["derivation"]
    assert der["coverage"] == 1.0
    assert {v["label"]: v["count"] for v in der["parts"]["feature_class"]["values"]} == {"Amino Acid": 3, "Lipid": 3, "(empty)": 3}
    apply(f, [m["patch_id"], d["patch_id"]])
    assert [a["assay_label"] for a in f.draft["assays"]] == ["metabolon"]
    assert {it["assay_label"] for it in f.draft["groups"].values() if it["role"] == "value"} == {"metabolon"}
    fr = next(q for q in f.draft["questions"] if q["kind"] == "fragmentation")     # still two blocks
    assert fr["text"].startswith("2 blocks in 1 assay have the same samples")
    r = f.c.post("/api/question/answer", json={"session_id": f.sid, "question_id": fr["question_id"], "option_ids": ["o1"]})
    assert r.status_code == 200, r.text
    f.draft = r.json()["draft"]
    assert sum(1 for it in f.draft["groups"].values() if it["role"] == "value") == 1
    q = next(q for q in f.draft["questions"] if q["kind"] == "unclassified_part")
    assert q["status"] == "open" and q["text"].startswith("3 feature name(s) have an empty feature class")
    r = f.c.post("/api/question/answer", json={"session_id": f.sid, "question_id": q["question_id"], "option_ids": ["o1"]})
    assert r.status_code == 200, r.text
    f.draft = r.json()["draft"]
    assert f.draft["derived_feature_annotations"][0]["display_labels"] == {"": "unclassified"}
    f.confirm_all_as_proposed()
    out = f.finalize()
    fm = list(csv.reader(io.StringIO(f.export("feature_metadata.csv"))))
    assert fm[0][-2:] == ["feature_class", "feature_numeric_id"]
    assert fm[1][0] == "Amino Acid_10" and fm[1][-2:] == ["Amino Acid", "10"] and fm[-1][-2:] == ["", "30"]
    ann = {x["column"]: x for x in out["schema"]["feature_annotations"]}
    assert ann["feature_class"]["derived_from"] == "feature_names" and ann["feature_class"]["rule"]["delimiter"] == "_"
    assert ann["feature_class"]["display_labels"] == {"": "unclassified"}
    assert "_10" in fm[-3][0]                                         # original names unchanged


def test_chat_needs_the_ai(flow):
    f = flow("C_mzmine_feature_table.csv", ai=False)
    assert "AI is off" in chat(f, "exclude formula", expect=422)["detail"]
