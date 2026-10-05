"""v2.4 stage 6: metadata through the AI, the join record, and the study design (SomaScan NHP shape)."""

import json
import re

from backend import mock_llm, session_log
from conftest import fixture_bytes

_ORIGINAL = mock_llm.MockLLM.respond
_SAMPLE = re.compile(r"^[A-Z0-9]{4}_\d+$")


def one_value_block(system, prompt):
    """Like the real SomaScan run: all 27 <animal>_<number> columns are one value block."""
    out = json.loads(_ORIGINAL(system, prompt))
    if not prompt.startswith("Digest"):
        return json.dumps(out)
    digest = json.loads(prompt.split("\n", 1)[1])
    names = [n for n in mock_llm.member_names(mock_llm.digest_entries(digest)) if _SAMPLE.match(n)]
    tpls = {t["template"] for t in digest["templates"] if all(_SAMPLE.match(c) for c in t["columns"])}
    keep = [g for g in out["groups"] if not (set(g.get("templates") or []) & tpls)
            and not any(_SAMPLE.match(c) for c in g.get("columns") or [])]
    keep.append({"group_id": "rfu", "columns": names, "role": "value", "assay_label": "proteomics",
                 "label": "SomaScan RFU", "confidence": 0.9, "evidence": "27 numeric columns <animal>_<number>"})
    for g in keep:
        if g.get("columns") == ["SeqId"]:
            g["role"], fid = "feature_id", g["group_id"]
        elif g["role"] == "unresolved":
            g.update(role="feature_annotation", label=f"{(g.get('columns') or g.get('templates'))[0]} (per aptamer)")
    out["groups"] = keep
    out["assays"] = [dict(out["assays"][0], assay_label="proteomics", feature_identity={"group_ids": [fid]})]
    return json.dumps(out)

K = "K_somascan_nhp.csv"
KM = "K_somascan_nhp_metadata.csv"


def post(f, path, expect=200, **body):
    r = f.c.post(path, json={"session_id": f.sid, **body})
    assert r.status_code == expect, r.text
    if expect == 200 and "draft" in r.json():
        f.draft = r.json()["draft"]
        if r.json().get("session"):
            f.upload = r.json()["session"]
    return r.json()


def one_block_with_metadata(flow, monkeypatch):
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(one_value_block))
    f = flow(K)
    q = next((q for q in f.draft["questions"] if q["kind"] == "fragmentation"), None)
    if q:
        f.answer(q, "One measurement")
    r = f.c.post("/api/metadata-upload", data={"session_id": f.sid}, files={"file": (KM, fixture_bytes(KM))})
    assert r.status_code == 200, r.text
    f.draft = r.json()["draft"]
    return f


def test_metadata_columns_go_through_the_ai_and_the_join_is_recorded(flow, monkeypatch):
    f = one_block_with_metadata(flow, monkeypatch)
    reqs = [json.loads(l) for l in session_log.log_path(f.sid).read_text(encoding="utf-8").splitlines()]
    meta_req = [r for r in reqs if r["event"] == "ai_request" and r.get("kind") == "metadata"]
    assert meta_req, "the metadata columns reached the AI"
    sent = {c["column"] for c in meta_req[0]["digest"]["columns"]}
    assert {"nhp_id", "time_point", "TimePoint", "study_group", "SubjectID"} <= sent
    assert all(c["file"] == "metadata" and "join" in c for c in meta_req[0]["digest"]["columns"])
    meta = f.draft["metadata"]
    assert meta["id_column"] == "SampleId" and meta["join_key_source"] == "ai"
    assert meta["report"]["n_matched"] == 27 and not meta["report"]["only_in_data"]
    tp = next(c for c in meta["columns"] if c["column"] == "time_point")
    assert tp["source"] == "ai" and tp["audit_kind"] == "timepoint" and tp["confidence"] > 0 and tp["evidence"]
    labels = [c["label"] for c in meta["columns"] if c["role"] == "sample_metadata"]
    assert len(set(labels)) == len(labels)                         # no generic label reused across columns
    f.confirm_all_as_proposed()
    out = f.finalize()["schema"]
    md = next(x for x in out["files"] if x["file_role"] == "metadata")
    assert md["parse_report"]["n_columns"] == 33 if "n_columns" in md["parse_report"] else md["n_columns"] == 33
    assert md["join"]["key_column"] == "SampleId" and md["join"]["matched"] == 27
    assert out["column_ledger"]["metadata"]["total"] == 33 and out["column_ledger"]["metadata"]["sample_id"] == 1
    smd = {x["column"]: x for x in out["sample_metadata"]}
    assert smd["time_point"]["provenance"] == "ai_proposed_confirmed" and smd["time_point"]["file"] == "metadata"
    assert smd["study_group"]["varies_within_subject"] is False         # constant per animal
    assert smd["viral_load"]["varies_within_subject"] is True
    assert "nhp_week_index" in smd                                      # nothing disappears silently


def test_design_facts_for_the_nhp_study(flow, monkeypatch):
    f = one_block_with_metadata(flow, monkeypatch)
    des = f.draft["design"]
    assert des["subject"]["source"] == des["time"]["source"] == "derived_from_sample_names"
    assert des["derivation"] == {"delimiter": "_", "occurrence": "last", "left": "subject", "right": "time"}
    rep = f.draft["design_report"]
    rm = rep["repeated_measures"]
    assert rep["n_subjects"] == 8 and sorted(rm["per_subject"].values()) == [1, 1, 1, 4, 4, 5, 5, 6]
    assert rm["samples_per_subject"]["counts"] == {"1": 3, "4": 2, "5": 2, "6": 1}
    assert rm["detected"] is True and rm["balanced"] is False and rm["subjects_with_single_sample"] == 3
    assert rep["time_kind"] == "numeric" and rep["n_distinct_time"] == 6
    assert rep["label"] == "longitudinal, unbalanced (1 to 6 samples per subject)"
    checks = {c["check"]: c for c in rep["cross_checks"]}
    sub = next(c for k, c in checks.items() if k.startswith("derived_subject_vs_metadata:SubjectID"))
    assert sub["agree"] == 27 and sub["disagree"] == 0                  # DEAB vs NHP-DEAB: same partition
    tim = next(c for k, c in checks.items() if k.startswith("derived_time_vs_metadata:TimePoint"))
    assert tim["agree"] == 27                                           # 0 vs W0
    # the time unit is asked, never inferred, and blocks finishing until answered or dismissed
    q = next(q for q in f.draft["questions"] if q["kind"] == "time_unit")
    assert q["status"] == "open" and q["step"] == "design" and "weeks" in [o["label"] for o in q["options"]]
    assert des["time"]["unit"]["value"] is None and des["time"]["unit"]["provenance"] == "unanswered"
    f.step("layout", {"layout": "samples_in_columns"})
    for st in ("feature_id", "annotations", "values", "samples", "sample_info", "design"):
        if f.draft["steps"][st] != "not_applicable":
            f.step(st, {"feature_identity": {"group_ids": f.draft["feature_identity"]["group_ids"]}} if st == "feature_id" else {})
    f.step("history", {"processing_history": {k: {"answer": "not_sure"} for k in (
        "normalized", "log_transformed", "imputed", "batch_corrected", "features_or_samples_removed_before_upload")}})
    assert "What unit is the time in" in f.finalize(expect=422)["detail"]
    f.answer(q, "weeks")
    out = f.finalize()["schema"]
    d = out["design"]
    assert d["time"]["unit"]["value"] == "weeks" and d["time"]["unit"]["provenance"] == "user_set"
    assert d["subject"]["n_subjects"] == 8 and d["repeated_measures"]["detected"] is True
    assert d["subject"]["provenance"] == "ai_proposed_confirmed"
    smd = {x["column"]: x for x in out["sample_metadata"]}
    assert smd["derived_subject"]["audit_kind"] == "subject_id" and smd["derived_time"]["audit_kind"] == "timepoint"


def test_partial_derivation_becomes_a_question_and_ai_cannot_set_the_unit(flow, monkeypatch):
    f = one_block_with_metadata(flow, monkeypatch)
    p = post(f, "/api/design/preview", rule={"delimiter": "A", "occurrence": "first", "left": "subject", "right": "time"})
    assert 0 < p["coverage"] < 1 and p["n_failures"] and p["rows"][0]["sample"]
    f.step("design", {"design": {"derivation": {"delimiter": "A", "occurrence": "first", "left": "subject", "right": "time"}}})
    q = next(q for q in f.draft["questions"] if q["kind"] == "design_derivation_coverage")
    assert q["status"] == "open" and f"parsed {p['n_parsed']} of 27" in q["text"]
    f.answer(q, "Do not derive")
    assert f.draft["design"]["subject"]["source"] == "none" and f.draft["design"]["time"]["source"] == "none"
    f.step("design", {"design": {"subject": {"source": "metadata_column", "column": "nhp_id"},
                                 "time": {"source": "metadata_column", "column": "time_point"}}})
    rep = f.draft["design_report"]
    assert rep["n_subjects"] == 8 and f.draft["design"]["subject"]["provenance"] == "user_set"
    from backend import edits, workflow
    s = workflow.get_session(f.sid)
    try:
        edits.apply_edit(s, "set_design", {"time": {"unit": "days"}}, actor="ai_patch")
        raise AssertionError("an AI patch must not set the time unit")
    except workflow.StepError as e:
        assert "never inferred" in str(e)
