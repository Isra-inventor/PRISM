"""v2.3: deterministic consistency checks. Flags that need your answer, never silent fixes."""

import csv
import io
import json
import re

from backend import mock_llm, workflow
from conftest import fixture_bytes

J = "J_subject_code_blocks.csv"
HISTORY = {"processing_history": {q: {"answer": "not_sure"} for q in (
    "normalized", "log_transformed", "imputed", "batch_corrected", "features_or_samples_removed_before_upload")}}


def rows_of(text):
    return list(csv.reader(io.StringIO(text)))


def per_code(system, prompt):
    """The over-fragmenting answer seen in a real run: one value block per subject code."""
    if not prompt.startswith("Digest"):
        return json.dumps({"cross_chunk_merges": []})
    digest = json.loads(prompt.split("\n", 1)[1])
    names = [c["column"] for c in digest["columns"]]
    by_code = {}
    for n in names:
        m = re.match(r"^([A-Z0-9]{4})_\d+$", n)
        if m:
            by_code.setdefault(m.group(1), []).append(n)
    groups = [{"group_id": "fid", "columns": ["SeqId"], "role": "feature_id", "label": "aptamer id", "confidence": 0.9,
               "evidence": "x"},
              {"group_id": "tgt", "columns": ["Target"], "role": "feature_annotation", "label": "target", "confidence": 0.9,
               "evidence": "x"}]
    groups += [{"group_id": f"v_{code}", "columns": cols, "role": "value", "assay_label": "SomaScan",
                "label": f"{code} RFU", "confidence": 0.8, "evidence": f"columns start with {code}"}
               for code, cols in by_code.items()]
    return json.dumps({"layout": {"value": "samples_in_columns", "confidence": 0.9, "evidence": "x"},
                       "assays": [{"assay_label": "SomaScan", "omics_type": "proteomics", "in_supported_scope": "yes",
                                   "confidence": 0.8, "evidence": "x", "feature_identity": {"group_ids": ["fid"]}}],
                       "groups": groups})


def confirm_rest(f):
    f.step("layout", {"layout": "samples_in_columns", "assays": [{"assay_label": "SomaScan", "omics_type": "proteomics"}]})
    f.step("feature_id", {"feature_identity": {"group_ids": f.draft["feature_identity"]["group_ids"]}})
    for st in ("annotations", "values", "samples"):
        f.step(st, {})
    f.step("sample_info", {"metadata": {"skip": True}})
    f.step("history", HISTORY)


def answer(f, kind, label_start):
    q = next(q for q in f.draft["questions"] if q["kind"] == kind and q["status"] == "open")
    opt = next(o for o in q["options"] if o["label"].startswith(label_start))
    r = f.c.post("/api/question/answer", json={"session_id": f.sid, "question_id": q["question_id"],
                                                "option_ids": [opt["option_id"]]})
    assert r.status_code == 200, r.text
    f.upload, f.draft = r.json()["session"], r.json()["draft"]
    return q


def test_near_identical_subject_blocks_are_flagged_not_merged(flow, monkeypatch):
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(per_code))
    f = flow(J)
    values = [g for g in f.upload["groups"] if f.draft["groups"][g["group_id"]]["role"] == "value"]
    assert len(values) == 10                                       # the AI's block count is kept as proposed ...
    flags = f.draft["consistency"]["block_flags"]
    assert len(flags) == 1 and sorted(flags[0]["group_ids"]) == sorted(g["group_id"] for g in values)
    assert "look statistically identical" in flags[0]["message"]  # ... but flagged for you
    assert set(flags[0]["codes"]) >= {"DEAB", "GA43", "T623", "M88A"}
    confirm_rest(f)
    r = f.finalize(expect=422)                                     # a flag needs an answer before finishing
    assert "statistically identical" in r["detail"]


def test_one_block_parses_subject_and_suffix_into_sample_information(flow, monkeypatch):
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(per_code))
    f = flow(J)
    q = answer(f, "near_identical_blocks", "One measurement")
    assert next(x for x in f.draft["questions"] if x["question_id"] == q["question_id"])["status"] == "answered"
    values = [g for g in f.upload["groups"] if f.draft["groups"][g["group_id"]]["role"] == "value"]
    assert len(values) == 1 and values[0]["n_columns"] == 27 and values[0]["origin"] == "merged_consistency"
    assert not f.draft["consistency"]["block_flags"] and not f.draft["consistency"]["sample_collisions"]
    assert {"DEAB_0", "T623_0", "M88A_4"} <= set(f.draft["sample_list"]["ids"])   # full names: no collision
    confirm_rest(f)
    out = f.finalize()
    a = out["schema"]["assays"][0]
    assert a["n_samples"] == a["n_value_columns"] == 27 == len(out["schema"]["samples"])
    sm = {r[0]: r for r in rows_of(f.export("sample_metadata.csv"))}
    head = sm["sample_id"]
    assert sm["DEAB_136"][head.index("subject_code")] == "DEAB" and sm["DEAB_136"][head.index("name_suffix")] == "136"
    kinds = {x["column"]: x["audit_kind"] for x in out["schema"]["sample_metadata"]}
    assert kinds["subject_code"] == "subject_id" and kinds["name_suffix"] == "timepoint"


def test_global_sample_id_collision_is_caught(flow, monkeypatch):
    """Dismissing the flag keeps 10 blocks: their stripped IDs ('0', '4', ...) collide
    across blocks and must be told apart before finishing (never merged silently)."""
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(per_code))
    f = flow(J)
    answer(f, "near_identical_blocks", "Separate measurements")
    assert not f.draft["consistency"]["block_flags"]
    col = f.draft["consistency"]["sample_collisions"]
    assert len(col) == 1 and "'0' appears in more than one block" in col[0]["message"]
    assert "0" in col[0]["ids"] and len(col[0]["group_ids"]) >= 2
    confirm_rest(f)
    assert "appears in more than one block" in f.finalize(expect=422)["detail"]
    answer(f, "sample_id_collision", "Use the full column names")
    assert not f.draft["consistency"]["sample_collisions"]
    f.step("samples", {})
    out = f.finalize()
    a = out["schema"]["assays"][0]
    assert a["sample_structure"] == "disjoint" and a["n_samples"] == 27 == len(out["schema"]["samples"])


def test_collision_with_descriptive_names_and_block_labels(flow, monkeypatch):
    """Partial overlap is a collision even with descriptive names; a per-block label fixes it."""
    header = ["id", "Liver_S1", "Liver_S2", "Kidney_S1", "Kidney_S3"]
    rows = [[f"F{k}", "1.5", "2.5", f"{100 + k}", "300"] for k in range(30)]
    buf = io.StringIO()
    csv.writer(buf).writerows([header] + rows)

    def two_blocks(system, prompt):
        return json.dumps({"layout": {"value": "samples_in_columns", "confidence": 0.9, "evidence": "x"},
                           "assays": [{"assay_label": "a", "omics_type": "x", "in_supported_scope": "yes",
                                       "confidence": 0.5, "evidence": "x", "feature_identity": {"group_ids": ["i"]}}],
                           "groups": [{"group_id": "i", "columns": ["id"], "role": "feature_id", "label": "id",
                                       "confidence": 1, "evidence": "x"},
                                      {"group_id": "l", "columns": ["Liver_S1", "Liver_S2"], "role": "value",
                                       "assay_label": "a", "label": "liver", "confidence": 1, "evidence": "x"},
                                      {"group_id": "k", "columns": ["Kidney_S1", "Kidney_S3"], "role": "value",
                                       "assay_label": "a", "label": "kidney", "confidence": 1, "evidence": "x"}]})
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(two_blocks))
    f = flow("organs.csv", content=buf.getvalue().encode())
    col = f.draft["consistency"]["sample_collisions"]
    assert len(col) == 1 and col[0]["ids"] == ["S1"]
    gids = col[0]["group_ids"]
    answer(f, "sample_id_collision", "Put each block's label")
    assert sorted(f.draft["sample_list"]["ids"]) == ["kidney_S1", "kidney_S3", "liver_S1", "liver_S2"]


def test_parallel_measurements_are_not_a_collision(flow):
    """MaxQuant: LFQ / raw / iBAQ / peptides blocks of the SAME samples share IDs by design."""
    f = flow("A_maxquant_proteinGroups.txt")
    rep = f.draft["consistency"]
    assert not rep["sample_collisions"] and rep["sample_structure"]["proteomics"] == "parallel"
    assert not rep["block_flags"]                                  # descriptive names: not identifier codes


def test_finalization_blocks_on_unreconciled_counts(flow):
    f = flow("C_mzmine_feature_table.csv")
    f.confirm_all_as_proposed()
    s = workflow.get_session(f.sid)
    del s.draft["samples"]["S05"]                                   # samples[] no longer matches the columns
    r = f.finalize(expect=422)
    assert "do not reconcile" in r["detail"] and "17 raw value columns" in r["detail"]
    assert "16 matching entries in samples[]" in r["detail"]
