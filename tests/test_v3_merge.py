"""v3 stage 3: multi-dataset sessions and merge (M1 to M7)."""

import csv
import io
import json
from pathlib import Path

import pytest

from prism import store
from prism.session import merge

DATA = Path(__file__).parent / "data"


def output_folder(path, samples, columns=None, kinds=None, design=None, family="proteomics"):
    """A minimal Step 0 output folder: one value matrix and a sample table."""
    columns = columns or {}
    kinds = kinds or {}
    path.mkdir(parents=True)
    with open(path / "value_matrix_A1_B1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature_key"] + samples)
        for i in range(3):
            w.writerow([f"f{i}"] + [str(i + j) for j in range(len(samples))])
    names = list(columns)
    with open(path / "sample_metadata.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "sample_label", "is_study_sample"] + names)
        for j, s in enumerate(samples):
            w.writerow([s, "study sample", "true"] + [columns[c][j] for c in names])
    schema = {"schema_version": "0.4.1", "source_file": path.name + ".csv", "file_sha256": "0" * 64,
              "omics_family": {"value": family},
              "assays": [{"assay_id": "A1", "assay_label": family, "omics_type": family, "n_features": 3,
                          "n_samples": len(samples),
                          "value_blocks": [{"block_id": "B1", "group_id": "g1", "file": "value_matrix_A1_B1.csv"}]}],
              "sample_metadata": [{"column": c, "audit_kind": kinds.get(c, "covariate")} for c in names],
              "design": design or {"subject": {"source": "none"}, "time": {"source": "none"}}}
    (path / "schema.json").write_text(json.dumps(schema))
    return path


def add(st, tmp_path, name, samples, **kw):
    out = output_folder(tmp_path / name, samples, **kw)
    d = st.add_dataset(name, "output_folder", "imported_awaiting_confirm")
    store.publish_output(st, d["dataset_id"], out, {"mode": "output_folder"})
    return d["dataset_id"]


DERIVED = {"subject": {"source": "derived_from_sample_names"},
           "time": {"source": "derived_from_sample_names", "unit": {"value": "weeks"}}}


def subj_time(samples):
    return {"derived_subject": [s.split("_")[0] for s in samples], "derived_time": [s.split("_")[1] for s in samples]}


@pytest.fixture
def st(isolated):
    return store.create("merge")


def test_exact_ids_overlap_and_outputs(st, tmp_path):
    a = ["P1_0", "P1_4", "P2_0", "P3_0"]
    b = ["P1_0", "P1_4", "P2_0", "P4_8"]
    add(st, tmp_path, "soma", a, columns=dict(subj_time(a), sex=["M", "M", "F", "F"]),
        kinds={"derived_subject": "subject_id", "derived_time": "timepoint", "sex": "covariate"}, design=DERIVED)
    add(st, tmp_path, "metab", b, columns=dict(subj_time(b), sex=["M", "M", "F", ""]),
        kinds={"derived_subject": "subject_id", "derived_time": "timepoint", "sex": "covariate"}, design=DERIVED,
        family="metabolomics")
    doc = merge.report(st)
    x = doc["cross_dataset"]
    (p,) = x["sample_overlap"]["pairs"]
    assert (p["n_shared"], p["n_only_a"], p["n_only_b"]) == (3, 1, 1) and p["jaccard"] == 0.6
    assert p["only_a"] == ["P3_0"] and p["only_b"] == ["P4_8"]
    assert x["sample_overlap"]["presence"]["P3_0"] == ["D1"]
    assert x["id_mapping"]["suggestions"] == [] and x["id_mapping"]["entries"] == []
    assert x["conflicts"] == [] and doc["ready_for_audit"], doc["blocking"]
    da = x["design_agreement"][0]
    assert da["subject"]["agree"] and da["time"]["agree"] and da["time_unit"]["match"]
    # outputs on disk; the sex column merged (an empty cell on one side is not a disagreement)
    assert json.loads((st.dir / "session_schema.json").read_text())["datasets"][1]["omics_family"] == "metabolomics"
    rows = list(csv.DictReader(io.StringIO((st.dir / "session_sample_table.csv").read_text())))
    assert [r["unified_id"] for r in rows] == ["P1_0", "P1_4", "P2_0", "P3_0", "P4_8"]
    assert rows[0]["present_in"] == "D1;D2" and rows[3]["in_D2"] == "0" and rows[4]["sample_id@D1"] == ""
    assert "sex" in rows[0] and "sex@D1" not in rows[0]
    # one-dataset view: the unified table restricted to that dataset's samples
    _, only = merge.sample_table(st, only="D2")
    assert [r["unified_id"] for r in only] == ["P1_0", "P1_4", "P2_0", "P4_8"]


def test_near_misses_are_suggested_never_applied(st, tmp_path):
    add(st, tmp_path, "a", ["S01-A", "s2_b", " S3 C", "X9"])
    add(st, tmp_path, "b", ["s1_a", "S2-B", "S3_C", "X9"])
    doc = merge.report(st)
    sug = doc["cross_dataset"]["id_mapping"]["suggestions"]
    pairs = {(s["a"]["sample_id"], s["b"]["sample_id"]): s for s in sug}
    assert set(pairs) == {("S01-A", "s1_a"), ("s2_b", "S2-B"), (" S3 C", "S3_C")}
    assert pairs[("S01-A", "s1_a")]["rules"] == ["case", "separator", "leading_zeros"]
    assert pairs[(" S3 C", "S3_C")]["rules"] == ["whitespace", "separator"]
    assert all(s["status"] == "open" for s in sug)
    ov = doc["cross_dataset"]["sample_overlap"]["pairs"][0]
    assert ov["n_shared"] == 1                                    # nothing mapped automatically
    assert not doc["ready_for_audit"] and "3 ID suggestion(s)" in doc["blocking"][0]
    s1 = pairs[("S01-A", "s1_a")]["suggestion_id"]
    doc = merge.decide(st, s1, "confirm")
    assert doc["cross_dataset"]["sample_overlap"]["pairs"][0]["n_shared"] == 2
    (e,) = doc["cross_dataset"]["id_mapping"]["entries"]
    assert e == {"dataset_id": "D2", "sample_id": "s1_a", "unified_id": "S01-A", "source": "confirmed_suggestion",
                 "suggestion_id": s1}
    for s in sug[1:]:
        doc = merge.decide(st, s["suggestion_id"], "dismiss")
    assert doc["ready_for_audit"]
    rows = list(csv.DictReader(io.StringIO((st.dir / "session_sample_table.csv").read_text())))
    r = next(r for r in rows if r["unified_id"] == "S01-A")
    assert r["sample_id@D1"] == "S01-A" and r["sample_id@D2"] == "s1_a"   # original IDs kept
    # the values in the output folders are untouched
    assert (st.output_dir("D2") / "value_matrix_A1_B1.csv").read_text().splitlines()[0].startswith("feature_key,s1_a")
    log = (st.dir / "logs" / "session.jsonl").read_text()
    assert "merge_decision" in log


def test_mapping_csv(st, tmp_path):
    add(st, tmp_path, "a", ["A1", "A2", "A3"])
    add(st, tmp_path, "b", ["B1", "B2", "B3"])
    bad = b"dataset_id,sample_id,unified_id\nD2,B1,A1\nD2,B2,A1\n"
    with pytest.raises(merge.MergeError, match="not valid.*'B1', 'B2' would all become 'A1'"):
        merge.set_mapping_csv(st, bad)
    with pytest.raises(merge.MergeError, match="'Q' is not a sample of D2"):
        merge.set_mapping_csv(st, b"dataset_id,sample_id,unified_id\nD2,Q,A1\n")
    with pytest.raises(merge.MergeError, match="would all become 'A2'"):    # collides with an unmapped sample
        merge.set_mapping_csv(st, b"dataset_id,sample_id,unified_id\nD1,A1,A2\n")
    assert store.load(st.sid).data["merge"].get("mapping_csv") is None       # nothing applied
    doc = merge.set_mapping_csv(st, b"dataset_id,sample_id,unified_id\nD2,B1,A1\nD2,B2,A2\nD1,A1,A1\n")
    assert doc["cross_dataset"]["sample_overlap"]["pairs"][0]["n_shared"] == 2
    assert [e["sample_id"] for e in doc["cross_dataset"]["id_mapping"]["entries"]] == ["B1", "B2"]
    assert doc["ready_for_audit"]


def test_metadata_disagreement_is_a_conflict(st, tmp_path, client):
    s = ["P1_0", "P1_4", "P2_0"]
    add(st, tmp_path, "a", s, columns={"batch": ["1", "1", "2"], "site": ["x", "y", "z"]},
        kinds={"batch": "batch", "site": "covariate"})
    add(st, tmp_path, "b", s, columns={"batch": ["1", "2", "2.0"], "site": ["x", "y", "z"], "location": ["x", "y", "z"]},
        kinds={"batch": "batch", "site": "covariate"})
    doc = client.get(f"/api/sessions/{st.sid}/merge-report").json()
    (c,) = doc["cross_dataset"]["conflicts"]
    assert c["column"] == "batch" and c["n_disagree"] == 1
    assert c["disagreements"][0] == {"unified_id": "P1_4", "values": {"D1": "1", "D2": "2"}}   # 2 == 2.0
    assert c["options"] == ["take:D1", "take:D2", "keep_both", "drop"]
    cols = {x["column"]: x for x in doc["cross_dataset"]["columns"]}
    assert cols["site"]["status"] == "merged" and cols["batch"]["status"] == "conflict_open"
    (cs,) = doc["cross_dataset"]["column_suggestions"]                     # different names, same values
    assert (cs["a"]["column"], cs["b"]["column"]) == ("site", "location")
    assert "location" in [c["column"] for c in doc["cross_dataset"]["columns"]]   # not merged into 'site'
    head = (st.dir / "session_sample_table.csv").read_text().splitlines()[0].split(",")
    assert "batch@D1" in head and "batch@D2" in head                       # unresolved: both kept, nothing lost
    assert not doc["ready_for_audit"]
    r = client.post(f"/api/sessions/{st.sid}/merge/resolve-conflict", json={"item_id": c["conflict_id"],
                                                                           "decision": "take:D2"})
    assert r.status_code == 200, r.text
    assert r.json()["ready_for_audit"]
    rows = list(csv.DictReader(io.StringIO((st.dir / "session_sample_table.csv").read_text())))
    assert [x["batch"] for x in rows] == ["1", "2", "2.0"]
    r = client.post(f"/api/sessions/{st.sid}/merge/resolve-conflict", json={"item_id": c["conflict_id"],
                                                                           "decision": "keep_both"})
    head = (st.dir / "session_sample_table.csv").read_text().splitlines()[0].split(",")
    assert "batch@D1" in head and "batch" not in head
    r = client.post(f"/api/sessions/{st.sid}/merge/resolve-conflict", json={"item_id": c["conflict_id"],
                                                                           "decision": "nonsense"})
    assert r.status_code == 422


def test_audit_kind_mismatch_is_a_question(st, tmp_path):
    s = ["A", "B", "C"]
    add(st, tmp_path, "a", s, columns={"group": ["x", "x", "y"]}, kinds={"group": "group"})
    add(st, tmp_path, "b", s, columns={"group": ["1", "1", "2"]}, kinds={"group": "batch"})
    doc = merge.report(st)
    (q,) = doc["cross_dataset"]["questions"]
    assert q["kind"] == "audit_kind_mismatch" and q["column"] == "group"
    assert [o["option_id"] for o in q["options"]] == ["same:batch", "same:group", "different"]
    assert doc["cross_dataset"]["conflicts"] == []
    doc = merge.decide(st, q["question_id"], "same:group")    # one variable: the values disagree now
    (c,) = doc["cross_dataset"]["conflicts"]
    assert c["audit_kind"] == "group" and c["n_disagree"] == 3
    doc = merge.decide(st, q["question_id"], "different")
    assert doc["cross_dataset"]["conflicts"] == [] and doc["ready_for_audit"]


def test_subject_mismatch_is_possible_id_reuse(st, tmp_path):
    a = ["P1_0", "P1_4", "P2_0", "P2_4"]
    add(st, tmp_path, "a", a, columns=subj_time(a), kinds={"derived_subject": "subject_id"}, design=DERIVED)
    # labels differ but split the samples the same way: agreement
    cols = {"subj": ["NHP-P1", "NHP-P1", "NHP-P2", "NHP-P2"], "tp": ["0", "4", "0", "4"]}
    meta_design = {"subject": {"source": "metadata_column", "column": "subj"},
                   "time": {"source": "metadata_column", "column": "tp", "unit": {"value": "weeks"}}}
    add(st, tmp_path, "b", a, columns=cols, kinds={"subj": "subject_id", "tp": "timepoint"}, design=meta_design)
    doc = merge.report(st)
    da = doc["cross_dataset"]["design_agreement"][0]
    assert da["subject"]["agree"] and not da["subject"]["same_labels"] and da["time"]["agree"]
    assert doc["cross_dataset"]["possible_id_reuse"] == []


def test_subject_or_time_disagreement_flags_reuse(st, tmp_path):
    a = ["P1_0", "P1_4", "P2_0", "P2_4", "P3_0", "P3_4"]
    add(st, tmp_path, "a", a, columns=subj_time(a), kinds={"derived_subject": "subject_id"}, design=DERIVED)
    cols = {"subj": ["P1", "P1", "P2", "P3", "P3", "P3"], "tp": ["0", "4", "0", "4", "0", "8"]}
    meta_design = {"subject": {"source": "metadata_column", "column": "subj"},
                   "time": {"source": "metadata_column", "column": "tp", "unit": {"value": "weeks"}}}
    add(st, tmp_path, "b", a, columns=cols, kinds={"subj": "subject_id", "tp": "timepoint"}, design=meta_design)
    doc = merge.report(st)
    reuse = doc["cross_dataset"]["possible_id_reuse"]
    assert {(r["unified_id"], r["field"]) for r in reuse} == {("P2_4", "subject"), ("P3_4", "time")}
    (q,) = [q for q in doc["cross_dataset"]["questions"] if q["kind"] == "possible_id_reuse"]
    assert "possibly the same ID used for different samples" in q["text"] and not doc["ready_for_audit"]
    doc = merge.decide(st, q["question_id"], "flag")
    assert doc["ready_for_audit"]


def test_time_unit_mismatch_is_a_question(st, tmp_path):
    a = ["P1_0", "P1_4"]
    add(st, tmp_path, "a", a, columns=subj_time(a), design=DERIVED)
    days = {"subject": {"source": "derived_from_sample_names"},
            "time": {"source": "derived_from_sample_names", "unit": {"value": "days"}}}
    add(st, tmp_path, "b", a, columns=subj_time(a), design=days)
    doc = merge.report(st)
    (q,) = doc["cross_dataset"]["questions"]
    assert q["kind"] == "time_unit_mismatch" and "'weeks' in D1 and 'days' in D2" in q["text"]
    da = doc["cross_dataset"]["design_agreement"][0]
    assert da["time_unit"]["match"] is False and da["time"] is None        # values not compared
    assert not doc["ready_for_audit"]
    doc = merge.decide(st, q["question_id"], "dismiss")
    assert doc["ready_for_audit"]


def test_unconfirmed_dataset_blocks_and_cli(st, tmp_path, capsys):
    from prism import cli
    add(st, tmp_path, "a", ["A", "B"])
    st.add_dataset("pending", "schema_import", "imported_awaiting_confirm")
    assert cli.main(["session", "merge-report", "--session", st.sid]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert not doc["ready_for_audit"] and "D2 (pending) is not confirmed" in doc["blocking"][0]
    assert doc["unconfirmed_datasets"][0]["dataset_id"] == "D2"
    st2 = store.create("cli")
    add(st2, tmp_path / "x", "a", ["S1", "S2"])
    add(st2, tmp_path / "x", "b", ["s1", "S2"])
    doc = merge.report(st2)
    sid = doc["cross_dataset"]["id_mapping"]["suggestions"][0]["suggestion_id"]
    assert cli.main(["session", "merge-decide", "--session", st2.sid, sid, "confirm"]) == 0
    assert "ready for audit" in capsys.readouterr().out
    p = tmp_path / "m.csv"
    p.write_text("dataset_id,sample_id,unified_id\nD2,s1,S2\n")
    assert cli.main(["session", "merge-mapping", "--session", st2.sid, str(p)]) == 2
    assert "would all become 'S2'" in capsys.readouterr().err


def test_wizard_outputs_merge(client, flow, isolated, monkeypatch):
    """Two datasets made by the wizard (same samples, with metadata): 27 of 27 shared."""
    import test_design
    st = store.create("wizard")
    for _ in range(2):
        f = test_design.one_block_with_metadata(flow, monkeypatch)
        f.confirm_all_as_proposed()
        f.finalize()
        d = st.add_dataset("K", "wizard", "wizard_in_progress")
        store.publish_output(st, d["dataset_id"], isolated / "sessions" / f.sid / "outputs", {"mode": "wizard"})
    doc = client.get(f"/api/sessions/{st.sid}/merge-report").json()
    x = doc["cross_dataset"]
    assert x["sample_overlap"]["pairs"][0]["n_shared"] == 27 and x["sample_overlap"]["n_in_all"] == 27
    assert x["id_mapping"]["suggestions"] == [] and x["conflicts"] == [] and doc["ready_for_audit"], doc["blocking"]
    da = x["design_agreement"][0]
    assert da["subject"]["agree"] and da["time"]["agree"] and da["time_unit"]["match"]


@pytest.mark.skipif(not (DATA / "SomaExpr_common.csv").exists() or not (DATA / "MetaboExpr_common.csv").exists(),
                    reason="real data files not present in tests/data")
def test_golden_real_files_share_all_27_samples(client, isolated):
    import real_data
    st = store.create("golden")
    for name in ("SomaExpr_common.csv", "MetaboExpr_common.csv"):
        out = real_data.wizard_output(client, isolated, name)
        d = st.add_dataset(name, "wizard", "wizard_in_progress")
        store.publish_output(st, d["dataset_id"], out, {"mode": "wizard"})
    doc = merge.report(st)
    x = doc["cross_dataset"]
    p = x["sample_overlap"]["pairs"][0]
    assert (p["n_shared"], p["n_only_a"], p["n_only_b"]) == (27, 0, 0) and p["jaccard"] == 1.0
    assert x["id_mapping"]["suggestions"] == [] and doc["ready_for_audit"], doc["blocking"]
