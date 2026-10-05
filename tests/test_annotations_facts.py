"""v2.4 stage 7: column-specific labels, near-duplicate and profile facts, omics_family,
processing history as a state machine, feature-name derivation on the metabolomics shape."""

import csv
import io
import json

from backend import mock_llm, session_log, workflow
from backend.parsing import parse_bytes
from backend.profiling import Columns, feature_facts, near_duplicate_pairs

_ORIGINAL = mock_llm.MockLLM.respond
HIST = ("normalized", "log_transformed", "imputed", "batch_corrected", "features_or_samples_removed_before_upload")


def events(sid, kind=None):
    ev = [json.loads(l) for l in session_log.log_path(sid).read_text(encoding="utf-8").splitlines()]
    return [e for e in ev if kind is None or e.get("kind") == kind]


def test_repeated_annotation_labels_get_one_retry(flow):
    f = flow("K_somascan_nhp.csv")            # the default mock labels many numeric annotations identically
    retries = f.draft["grouping"]["label_retries"]
    assert retries and retries[0]["n_columns"] > 5 and retries[0]["relabelled"] == retries[0]["n_columns"]
    assert len(events(f.sid, "relabel")) == len({r["label"] for r in retries})   # exactly one retry per label
    labels = [it["label"] for it in f.draft["groups"].values() if it["role"] == "feature_annotation"]
    assert max(labels.count(x) for x in labels) <= 5
    it = next(it for it in f.draft["groups"].values() if it.get("family") == retries[0]["label"])
    assert it["label"] != it["family"]                       # own label, the shared descriptor as family


def test_label_warning_stays_when_the_retry_does_not_help(flow, monkeypatch):
    def stubborn(system, prompt):
        if prompt.startswith("Relabel"):
            p = json.loads(prompt.split("\n", 1)[1])
            return json.dumps({"labels": [{"column": c["column"], "label": p["shared_label"]} for c in p["columns"]]})
        return _ORIGINAL(system, prompt)
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(stubborn))
    f = flow("K_somascan_nhp.csv")
    r = f.draft["grouping"]["label_retries"][0]
    assert r["still_shared"] == r["n_columns"]
    warned = [it for it in f.draft["groups"].values() if any("share the label" in m for m in it["validation"]["messages"])]
    assert len(warned) == r["n_columns"] and all(it["validation"]["status"] == "warning" for it in warned)


def test_near_duplicate_pairs_are_a_fact():
    rows = []
    for k in range(120):
        rows.append([f"F{k}", f"A{k % 40}", f"A{k % 40}", f"B{k % 30}", f"B{k % 30}" if k != 7 else "zz", f"C{k % 11}",
                     "" if k % 10 == 0 else f"A{k % 40}"])
    buf = io.StringIO()
    csv.writer(buf).writerows([["id", "a", "a_copy", "b", "b_near", "c", "a_sparse"]] + rows)
    cols = Columns(parse_bytes("x.csv", buf.getvalue().encode()))
    pairs, skipped = near_duplicate_pairs(cols, range(7))
    got = {tuple(p["columns"]): p for p in pairs}
    assert skipped is None and got[("a", "a_copy")]["exact"] is True
    assert got[("b", "b_near")]["match_share"] >= 0.99 and not got[("b", "b_near")]["exact"]
    assert got[("a", "a_sparse")]["match_share"] == 1.0 and got[("a", "a_sparse")]["n_compared"] == 108  # missing skipped
    assert not any("c" in k for k in got)
    wide = Columns(parse_bytes("w.csv", (",".join(f"c{k}" for k in range(301)) + "\n" + ",".join("x" for _ in range(301))).encode()))
    pairs, skipped = near_duplicate_pairs(wide, range(301))
    assert pairs == [] and "not computed" in skipped


def test_profile_facts_are_reported_and_answer_nothing(flow):
    header = ["sample", "f1", "f2", "f3"]
    data = [["S1", "1", "10", "5"], ["S2", "2", "20", "5"], ["S3", "3", "30", "5"], ["S4", "4", "40", "7"]]
    buf = io.StringIO()
    csv.writer(buf).writerows([header] + data)
    cols = Columns(parse_bytes("x.csv", buf.getvalue().encode()))
    ff = feature_facts(cols, [1, 2, 3], "samples_in_rows")      # per-feature medians: 2.5, 25, 5
    assert ff["per_feature_median"] == {"p5": 2.5, "p50": 5, "p95": 25}
    assert ff["ties_at_minimum"]["n_features_with_2_or_more_at_min"] == 1 and ff["ties_at_minimum"]["share_max"] == 0.75
    f = flow("L_metabolon_like.csv")
    facts = f.draft["feature_facts"]
    assert facts
    ties = [x["ties_at_minimum"] for x in facts.values()]
    assert any(t["n_features_with_2_or_more_at_min"] >= 1 and t["share_max"] >= 14 / 27 for t in ties)
    ph = f.draft["processing_history"]
    assert all(ph[q]["answer"] is None and ph[q]["provenance"] == "unanswered" for q in HIST)   # nothing preselected
    assert "median" in ph["normalized"].get("ai_hint", "")                                     # a hint raises it
    assert any(u["step"] == "history" for u in f.draft["unresolved"])


def test_omics_family_is_closed_and_history_is_yours(flow):
    f = flow("A_maxquant_proteinGroups.txt")
    a = f.draft["assays"][0]
    assert a["omics_family"] == "proteomics"
    f.step("layout", {"layout": "samples_in_columns", "assays": [dict(a, omics_family="proteinomics")]}, expect=422)
    f.confirm_all_as_proposed()
    out = f.finalize()["schema"]
    assert out["schema_version"] == "0.4.0" and out["omics_family"]["value"] == "proteomics"
    assert out["assays"][0]["omics_family"] == {"value": "proteomics", "provenance": "ai_proposed_confirmed"}
    ph = out["processing_history"]["normalized"]
    assert ph["answer"] == "not_sure" and ph["provenance"] == "user_set" and ph["answered_at"]


def test_feature_class_derivation_on_the_metabolomics_shape(flow, monkeypatch):
    def respond(system, prompt):
        if prompt.startswith("Chat"):
            return json.dumps({"reply": "I've prepared a change.", "questions": [], "patches": [
                {"patch_id": "d", "op": "derive_feature_annotation", "reason": "class and compound ID are in the names",
                 "args": {"source": "column_headers", "rule": {"delimiter": "_", "occurrence": "first", "parts": [
                     {"name": "feature_class", "label": "metabolite class"},
                     {"name": "feature_numeric_id", "label": "compound ID"}]}}}]})
        return _ORIGINAL(system, prompt)
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(respond))
    f = flow("L_metabolon_like.csv")
    r = f.c.post("/api/chat", json={"session_id": f.sid, "message": "split the names"})
    f.draft = r.json()["draft"]
    p = next(p for p in f.draft["patches"] if p["op"] == "derive_feature_annotation")
    vals = {v["label"]: v["count"] for v in p["derivation"]["parts"]["feature_class"]["values"]}
    assert len(vals) == 10 and vals["(empty)"] == 160 and vals["Lipid"] == 450 and p["derivation"]["coverage"] == 1.0
    r = f.c.post("/api/patch/apply", json={"session_id": f.sid, "patch_ids": [p["patch_id"]]})
    f.draft = r.json()["draft"]
    q = next(q for q in f.draft["questions"] if q["kind"] == "unclassified_part")
    assert q["text"].startswith("160 feature name(s) have an empty feature class")
    s = workflow.get_session(f.sid)
    assert "Amino Acid_100010863" in s.cols.labels                   # names never change
