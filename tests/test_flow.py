"""End-to-end API flows with the mocked LLM (acceptance criteria A-F and more)."""

import csv
import io
import json

import pytest

from backend import mock_llm, session_log
from conftest import fixture_bytes

HISTORY = {"processing_history": {q: {"answer": "not_sure"} for q in (
    "normalized", "log_transformed", "imputed", "batch_corrected", "features_or_samples_removed_before_upload")}}


def rows_of(text):
    return list(csv.reader(io.StringIO(text)))


def gid_of(f, col):
    return next(g["group_id"] for g in f.upload["groups"] if col in g["columns"])


def events(tmp, sid):
    return [json.loads(l)["event"] for l in (tmp / "logs" / f"{sid}.jsonl").read_text().splitlines()]


# ---------------------------------------------------------------- A: MaxQuant

def test_A_maxquant(flow, isolated):
    f = flow("A_maxquant_proteinGroups.txt")
    d = f.draft
    assert d["signature_hint"] == "headers match the known MaxQuant proteinGroups.txt pattern"
    assert d["layout"]["provenance"] == "ai_proposed_confirmed"          # the hint is not a provenance
    g = d["groups"]
    lfq, inten, pep = gid_of(f, "LFQ intensity S01"), gid_of(f, "Intensity S01"), gid_of(f, "Peptides S01")
    assert all(g[x]["role"] == "value" and g[x]["keep"] for x in (lfq, inten, pep))   # no ranking
    assert g[pep]["role"] == "value" and g[pep]["validation"]["status"] == "ok"
    assert len({g[x]["assay_label"] for x in (lfq, inten, pep)}) == 1    # one assay, several blocks
    rev, con = gid_of(f, "Reverse"), gid_of(f, "Potential contaminant")
    for x, n in ((rev, 3), (con, 2)):
        assert g[x]["role"] == "feature_annotation" and g[x]["marks_rows_as_suspect"] is True
        assert g[x]["flag_values"] == {"": 60 - n, "+": n} and g[x]["flagged_values"] == ["+"] and g[x]["n_flagged"] == n
    f.confirm_all_as_proposed()
    out = f.finalize()
    schema = out["schema"]
    assert schema["signature_hint"] == d["signature_hint"]
    blocks = schema["assays"][0]["value_blocks"]
    assert len(blocks) == 4 and all(b["label"] and "block_role" not in b for b in blocks)
    assert len([a for a in out["artifacts"] if a.startswith("value_matrix_A1_")]) == 4   # one matrix per kept block
    rev_ann = next(a for a in schema["feature_annotations"] if a["column"] == "Reverse")
    assert rev_ann["marks_rows_as_suspect"] is True and rev_ann["flagged_values"] == ["+"] and rev_ann["flag_counts"] == {"+": 3}
    assert rev_ann["n_flagged"] == 3 and rev_ann["provenance"] == "ai_proposed_confirmed" and rev_ann["label"]
    lfq_file = next(b["file"] for b in blocks if b["group_id"] == lfq)
    vm = rows_of(f.export(lfq_file))
    assert vm[0] == ["feature_key", "S01", "S02", "S03", "S04", "S05", "S06"]
    assert len(vm) == 61                                      # all 60 proteins kept, incl. decoys/contaminants
    src = list(csv.reader(io.StringIO(fixture_bytes("A_maxquant_proteinGroups.txt").decode()), delimiter="\t"))
    lfq_i = src[0].index("LFQ intensity S01")
    assert vm[1][1] == src[1][lfq_i] and vm[60][0].startswith(src[60][0][:3])  # values copied exactly
    fm = rows_of(f.export("feature_metadata.csv"))
    assert "Reverse" in fm[0] and len(fm) == 61
    ev = events(isolated, f.sid)
    for e in ("upload", "parse_report", "proposal", "step_confirmed", "finalize"):
        assert e in ev


def test_A_manual_mode_starts_from_signature(flow):
    f = flow("A_maxquant_proteinGroups.txt", ai=False)
    d = f.draft
    assert d["ai"]["used"] is False and d["layout"]["value"] == "samples_in_columns"
    assert d["layout"]["provenance"] == "computed"
    g = d["groups"]
    assert g[gid_of(f, "LFQ intensity S01")]["label"] == "LFQ intensity"
    assert g[gid_of(f, "Reverse")]["marks_rows_as_suspect"] is True
    f.confirm_all_as_proposed()
    schema = f.finalize()["schema"]
    assert schema["assays"][0]["value_blocks"][0]["provenance"] == "computed"


def test_suspect_flag_value_is_chosen_by_the_user(flow):
    f = flow("A_maxquant_proteinGroups.txt")
    rev = gid_of(f, "Reverse")
    f.step("annotations", {"items": [{"group_id": rev, "flagged_values": [""]}]})   # user says: empty = flagged
    assert f.draft["groups"][rev]["n_flagged"] == 57
    f.step("annotations", {"items": [{"group_id": rev, "marks_rows_as_suspect": False}]})
    assert "n_flagged" not in f.draft["groups"][rev]


# ---------------------------------------------------------------- B: DIA-NN

def test_B_diann(flow):
    f = flow("B_diann_pg_matrix.tsv")
    g = f.draft["groups"]
    for c in ("PG.ProteinNames", "PG.Genes"):
        assert g[gid_of(f, c)]["role"] == "feature_annotation" and g[gid_of(f, c)]["label"]
    block = g[gid_of(f, "S01.raw")]
    assert block["role"] == "value" and block["validation"]["status"] == "ok"
    assert f.draft["sample_list"]["ids"][:2] == ["S01", "S02"]      # '.raw' stripped
    f.confirm_all_as_proposed()
    vm = rows_of(f.export("value_matrix_A1_B1.csv")) if f.finalize() else None
    assert "" in [c for r in vm[1:] for c in r[1:]]              # 'NaN' written as empty


# ---------------------------------------------------------------- C: MZmine (AI path)

def test_C_mzmine_ai_path(flow):
    f = flow("C_mzmine_feature_table.csv")
    d = f.draft
    assert d["signature_hint"] is None and d["ai"]["provider"] == "mock"
    for c in ("row m/z", "row retention time", "compound_name", "formula", "adduct", "HMDB_ID"):
        it = d["groups"][gid_of(f, c)]
        assert it["role"] == "feature_annotation" and it["label"] and it["marks_rows_as_suspect"] is False
    st = d["samples"]
    assert st["QC_01"]["is_study_sample"] is False and st["blank_02"]["is_study_sample"] is False
    assert st["S01"]["is_study_sample"] is True
    # composite key m/z + RT is accepted
    f.step("layout", {"layout": "samples_in_columns"})
    f.step("feature_id", {"feature_identity": {"group_ids": [gid_of(f, "row m/z"), gid_of(f, "row retention time")]}})
    assert f.draft["feature_identity"]["composite"] is True
    assert f.draft["groups"][gid_of(f, "row ID")]["role"] == "feature_annotation"
    for st_ in ("annotations", "values", "samples"):
        f.step(st_, {})
    f.step("sample_info", {"metadata": {"skip": True}})
    f.confirm_design()
    f.step("history", HISTORY)
    out = f.finalize()
    assert any(fl["flag"] == "no_sample_metadata" for fl in out["integrity_flags"])
    sm = rows_of(f.export("sample_metadata.csv"))
    assert sm[0][:3] == ["sample_id", "sample_label", "is_study_sample"]
    assert len(sm) == 18 and ["QC_01", "false"] in [[r[0], r[2]] for r in sm]     # QC kept, labelled
    samples = {x["sample"]: x for x in out["schema"]["samples"]}
    assert samples["QC_01"]["is_study_sample"] is False and samples["QC_01"]["provenance"] == "ai_proposed_confirmed"
    vm = rows_of(f.export("value_matrix_A1_B1.csv"))
    assert "|" in vm[1][0]


# ---------------------------------------------------------------- D: samples in rows, two assays

def test_D_samples_in_rows(flow):
    f = flow("D_samples_in_rows_multiomics.csv")
    d = f.draft
    assert d["layout"]["value"] == "samples_in_rows"
    g = d["groups"]
    for c in ("age", "CD4_count", "iron"):             # iron was split out of the metabolite block
        assert g[gid_of(f, c)]["role"] == "sample_metadata" and g[gid_of(f, c)]["audit_kind"] == "covariate"
    visit = g[gid_of(f, "visit")]
    assert visit["audit_kind"] == "timepoint" and visit["detail"].startswith("ordinal label")
    assert g[gid_of(f, "subject_id")]["audit_kind"] == "subject_id"
    assert g[gid_of(f, "batch")]["audit_kind"] == "batch"
    assert g[gid_of(f, "severity_group")]["audit_kind"] == "group"
    values = [it for it in g.values() if it["role"] == "value"]
    assert len({it["assay_label"] for it in values}) == 2 and len(d["assays"]) == 2
    assert all(it["keep"] for it in values)
    f.confirm_all_as_proposed()
    out = f.finalize()
    assert len(out["schema"]["assays"]) == 2
    names = out["artifacts"]
    assert "value_matrix_A1_B1.csv" in names and "value_matrix_A2_B1.csv" in names
    m = rows_of(f.export("value_matrix_A1_B1.csv"))
    assert m[0][1:4] == ["S001", "S002", "S003"] and len(m[0]) == 41     # transposed: features x samples
    sm = rows_of(f.export("sample_metadata.csv"))
    assert {"age", "CD4_count", "iron", "visit"} <= set(sm[0])
    assert len(sm) == 41
    kinds = {x["column"]: x["audit_kind"] for x in out["schema"]["sample_metadata"]}
    assert kinds["visit"] == "timepoint" and kinds["subject_id"] == "subject_id"


# ---------------------------------------------------------------- E: SomaScan-like

def test_E_somascan(flow):
    f = flow("E_somascan_adat_like.csv")
    g = f.draft["groups"]
    for c in ("PlateScale_Scalar", "HybControlNormScale", "NormScale_20"):
        assert g[gid_of(f, c)]["role"] == "sample_metadata"            # technical: describes samples
    st = f.draft["samples"]
    assert st["SAM001"]["is_study_sample"] is True
    for sid in ("QC031", "CAL035", "BUF039"):
        assert st[sid]["is_study_sample"] is False
    assert len({v["label"] for v in st.values()}) >= 4


# ---------------------------------------------------------------- F: long format

def test_F1_long_pivots(flow):
    f = flow("F1_long_unique.csv")
    assert f.draft["layout"]["value"] == "long"
    f.confirm_all_as_proposed()
    f.finalize()
    m = rows_of(f.export("value_matrix_A1_B1.csv"))
    assert m[0] == ["feature_key", "S01", "S02", "S03", "S04", "S05", "S06"] and len(m) == 13


def test_F2_long_duplicates_refuse(flow):
    f = flow("F2_long_duplicates.csv")
    f.confirm_all_as_proposed()
    assert "aggregation is required" in f.draft["long_duplicates"]["message"]
    r = f.finalize(expect=422)
    assert "aggregation" in r["detail"]


# ---------------------------------------------------------------- AI robustness

_ORIGINAL_MOCK = mock_llm.MockLLM.respond


def _wrong(system, prompt):
    base = json.loads(_ORIGINAL_MOCK(system, prompt))
    for g in base["groups"]:
        if g["role"] == "feature_annotation" and "adduct" in g["label"]:
            g["role"] = "value"                                   # value role on a text column
        if "formula" in g["label"]:
            g["role"], g["audit_kind"] = "sample_metadata", "foo"  # invalid closed-field value
    base["groups"].append({"group_id": "g999", "columns": ["no such column"], "role": "value", "label": "x",
                           "confidence": 0.9, "evidence": "made up"})      # hallucinated column / group
    base["groups"] = [g for g in base["groups"] if g["columns"] != ["row m/z"]]   # a column left out
    return json.dumps(base)


def test_wrong_ai_proposal_is_downgraded(flow, monkeypatch):
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(_wrong))
    f = flow("C_mzmine_feature_table.csv")
    d = f.draft
    adduct = d["groups"][gid_of(f, "adduct")]
    assert adduct["role"] == "unresolved" and adduct["validation"]["status"] == "contradicted"
    assert adduct["claimed"]["role"] == "value"
    formula = d["groups"][gid_of(f, "formula")]
    assert formula["role"] == "unresolved" and formula["validation"]["status"] == "contradicted"
    assert any(r.get("column") == "no such column" for r in d["rejected"])
    mz = d["groups"][gid_of(f, "row m/z")]                                 # never silently dropped
    assert mz["role"] == "unresolved" and "Not placed" in mz["evidence"]
    assert any("unresolved" in u["what"] for u in d["unresolved"])
    # the user cannot confirm the contradicted claim either
    f.step("annotations", {"items": [{"group_id": gid_of(f, "adduct"), "role": "value"}]}, expect=422)
    f.step("annotations", {"items": [{"group_id": gid_of(f, "adduct"), "role": "feature_annotation",
                                      "label": "adduct type"}]})
    assert f.draft["groups"][gid_of(f, "adduct")]["provenance"] == "user_set"


def test_invalid_json_retries_then_manual(flow, monkeypatch):
    calls = []

    def broken(system, prompt):
        calls.append(1)
        return "{not json"
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(broken))
    f = flow("C_mzmine_feature_table.csv")
    assert len(calls) == 2                                  # retried once
    assert all(it["role"] == "unresolved" for it in f.draft["groups"].values())
    assert "invalid JSON" in f.draft["ai"]["error"]


def test_ai_off_full_manual_flow(flow):
    f = flow("C_mzmine_feature_table.csv", ai=False)
    d = f.draft
    assert d["layout"]["value"] == "unresolved" and "features usually outnumber" in d["layout"]["hint"]
    assert all(it["role"] == "unresolved" for it in d["groups"].values())
    f.step("layout", {"layout": "samples_in_columns",
                      "assays": [{"assay_label": "LC-MS untargeted", "omics_type": "metabolomics",
                                  "source_software": "MZmine", "in_supported_scope": "yes"}]})
    # nothing is grouped for you: the shared name parts are offered, you decide
    assert len(f.upload["groups"]) == f.upload["n_columns"]
    part = next(p for p in f.draft["name_parts"] if p["text"].strip() == "Peak area")
    assert part["n"] == 17
    r = f.c.post("/api/group-columns", json={"session_id": f.sid, "columns": part["columns"], "reason": "one sample each"})
    assert r.status_code == 200, r.text
    f.upload, f.draft = r.json()["session"], r.json()["draft"]
    block = next(g for g in f.upload["groups"] if g["n_columns"] == 17)
    assert block["sample_names"][:2] == ["QC_01", "QC_02"] and block["origin"] == "grouped_user"
    items = []
    for g in f.upload["groups"]:
        c = g["columns"][0]
        if g["kind"] == "numeric_block":
            items.append({"group_id": g["group_id"], "role": "value", "label": "peak area",
                          "assay_label": "LC-MS untargeted"})
        elif c != "row ID":
            items.append({"group_id": g["group_id"], "role": "feature_annotation", "label": c})
    f.step("feature_id", {"feature_identity": {"group_ids": [gid_of(f, "row ID")]}})
    f.step("annotations", {"items": [i for i in items if i["role"] == "feature_annotation"]})
    f.step("values", {"items": [i for i in items if i["role"] == "value"]})
    f.step("samples", {"samples": {"QC_01": {"label": "pooled QC", "is_study_sample": False}}})
    f.step("sample_info", {"metadata": {"skip": True}})
    f.confirm_design()
    f.step("history", HISTORY)
    out = f.finalize()
    prov = {a["column"]: a["provenance"] for a in out["schema"]["feature_annotations"]}
    assert prov["formula"] == "user_set"
    assert out["schema"]["layout"]["provenance"] == "user_set"
    assert out["schema"]["ai"]["enabled"] is False
    assert out["schema"]["assays"][0]["source_software"] == "MZmine"
    qc = next(x for x in out["schema"]["samples"] if x["sample"] == "QC_01")
    assert qc["is_study_sample"] is False and qc["label"] == "pooled QC"


def test_provenance_confirmed_vs_corrected(flow):
    f = flow("C_mzmine_feature_table.csv")
    f.confirm_all_as_proposed()
    f.step("annotations", {"items": [{"group_id": gid_of(f, "adduct"), "label": "ion species"}]})
    for st in ("values", "samples", "sample_info"):
        if f.draft["steps"][st] == "pending":
            f.step(st, {} if st != "sample_info" else {"metadata": {"skip": True}})
    schema = f.finalize()["schema"]
    prov = {a["column"]: a["provenance"] for a in schema["feature_annotations"]}
    assert prov["adduct"] == "user_set" and prov["formula"] == "ai_proposed_confirmed"


def test_out_of_scope_is_a_notice_not_a_block(flow, monkeypatch):
    def other(system, prompt):
        base = json.loads(_ORIGINAL_MOCK(system, prompt))
        for a in base["assays"]:
            a.update(omics_type="lipidomics", in_supported_scope="unsure", scope_reason="lipid species names")
        return json.dumps(base)
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(other))
    f = flow("C_mzmine_feature_table.csv")
    assert f.draft["assays"][0]["in_supported_scope"] == "unsure"
    f.confirm_all_as_proposed()
    out = f.finalize()
    fl = next(x for x in out["integrity_flags"] if x["flag"] == "outside_supported_scope")
    assert "lipidomics" in fl["detail"] and "not yet supported" in fl["detail"]


def test_history_must_be_answered_and_no_default(flow):
    f = flow("B_diann_pg_matrix.tsv")
    assert all(v["answer"] is None for v in f.draft["processing_history"].values())
    f.step("history", {"processing_history": {"normalized": {"answer": "yes"}}}, expect=422)


def test_values_step_keep_and_exclude(flow):
    f = flow("A_maxquant_proteinGroups.txt")
    blocks = [gid_of(f, c) for c in ("LFQ intensity S01", "Intensity S01", "iBAQ S01", "Peptides S01")]
    f.step("values", {"items": [{"group_id": g, "keep": False} for g in blocks]}, expect=422)   # keep at least one
    f.confirm_all_as_proposed()
    f.step("values", {"items": [{"group_id": g, "keep": False} for g in blocks[1:]]})
    for st in ("samples", "sample_info"):
        if f.draft["steps"][st] == "pending":
            f.step(st, {} if st == "samples" else {"metadata": {"skip": True}})
    out = f.finalize()
    assert [a for a in out["artifacts"] if a.startswith("value_matrix")] == ["value_matrix_A1_B1.csv"]
    excl = {x["column"] for x in out["schema"]["excluded_columns"] if x["file"] == "main" and x["by"] == "user" and x["at"]}
    assert "iBAQ S01" in excl and "LFQ intensity S01" not in excl


def test_layout_change_triggers_reproposal(flow):
    f = flow("C_mzmine_feature_table.csv")
    r = f.step("layout", {"layout": "samples_in_rows"})
    assert r["reproposed"] is True
    assert f.draft["layout"]["value"] == "samples_in_rows"
    assert f.draft["steps"]["feature_id"] in ("pending", "not_applicable")


def test_disagree_reconsiders_a_whole_step(flow, monkeypatch):
    f = flow("C_mzmine_feature_table.csv")
    seen = {}

    def answer(system, prompt):
        digest = json.loads(prompt.split("\n", 1)[1])
        seen["columns"] = sorted(c["column"] for c in digest["columns"])
        seen["feedback"] = digest["already_confirmed"].get("user_feedback")
        seen["current"] = sorted(g["columns"] for g in digest["already_confirmed"]["current_grouping_of_these_columns"])
        return json.dumps({"groups": [{"group_id": f"x{k}", "columns": [c], "role": "feature_annotation",
                                       "label": "ion form", "confidence": 0.9, "evidence": "user feedback"}
                                      for k, c in enumerate(seen["columns"])]})
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(answer))
    gids = sorted([gid_of(f, "adduct"), gid_of(f, "formula")])
    r = f.c.post("/api/reconsider", json={"session_id": f.sid, "group_ids": gids, "user_hint": "these are ion forms"})
    assert r.status_code == 200, r.text
    assert seen == {"columns": ["adduct", "formula"], "feedback": "these are ion forms",
                    "current": [["adduct"], ["formula"]]}
    # same columns -> same group ids: the wizard keeps its place
    assert all(r.json()["draft"]["groups"][g]["label"] == "ion form" for g in gids)
    # older single-group form still works
    r = f.c.post("/api/reconsider", json={"session_id": f.sid, "group_id": gids[0], "user_hint": "x"})
    assert r.status_code == 200, r.text


def test_reconsider_needs_the_ai(flow):
    f = flow("C_mzmine_feature_table.csv", ai=False)
    r = f.c.post("/api/reconsider", json={"session_id": f.sid, "group_ids": ["g1"], "user_hint": "x"})
    assert r.status_code == 422


def test_metadata_upload_matching(flow):
    f = flow("B_diann_pg_matrix.tsv")
    meta = "sample,group,age\nS01,ctrl,40\ns02,ctrl,41\nS03,case,50\nS99,case,60\n"
    r = f.c.post("/api/metadata-upload", data={"session_id": f.sid}, files={"file": ("meta.csv", meta.encode())})
    assert r.status_code == 200, r.text
    rep = r.json()["draft"]["metadata"]["report"]
    assert rep["n_matched"] == 2 and "S99" in rep["only_in_metadata"]
    assert rep["near_misses"] == [{"data_id": "S02", "metadata_id": "s02",
                                   "reason": "differs only in case, spaces/underscores or leading zeros"}]
    d = f.draft
    f.step("layout", {"layout": "samples_in_columns"})
    f.step("feature_id", {"feature_identity": {"group_ids": d["feature_identity"]["group_ids"]}})
    for st in ("annotations", "values", "samples"):
        f.step(st, {})
    f.step("sample_info", {"metadata": {"accept_near_misses": [["S02", "s02"]],
                                        "columns": [{"column": "group", "audit_kind": "group", "label": "arm"},
                                                    {"column": "age", "audit_kind": "covariate"}]}})
    f.confirm_design()
    f.step("history", HISTORY)
    f.finalize()
    rows = rows_of(f.export("sample_metadata.csv"))
    gi = rows[0].index("group")
    sm = {r[0]: r for r in rows}
    assert sm["S02"][gi] == "ctrl"                      # near miss accepted by the user
    assert sm["S04"][gi] == ""                          # no metadata row: kept, empty


def test_example_values_can_be_withheld(flow, monkeypatch):
    monkeypatch.setenv("AI_SEND_EXAMPLE_VALUES", "false")
    f = flow("D_samples_in_rows_multiomics.csv")
    text = json.dumps(f.digests)
    assert "severe" not in text and "S001" not in text
    assert f.digests[0]["settings"] == {"example_values_sent": False, "raw_rows_sent": False}


def test_digest_never_contains_raw_identifiers(flow):
    f = flow("D_samples_in_rows_multiomics.csv")
    text = json.dumps(f.digests)
    assert "S001" not in text                             # high-cardinality ids never sent
    assert "PT01" not in text                             # identifier-like (>= 50% distinct) never sent
    assert "severe" in text                               # low-cardinality categories are


def test_cache_reuses_proposal(flow, monkeypatch):
    f1 = flow("C_mzmine_feature_table.csv")
    calls = []
    def counting(system, prompt):
        calls.append(1)
        return _ORIGINAL_MOCK(system, prompt)
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(counting))
    f2 = flow("C_mzmine_feature_table.csv")
    assert calls == [] and f2.draft["ai"]["cached"] is True


def test_upload_limits_and_filename_sanitized(client):
    r = client.post("/api/upload", files={"file": ("../../etc/passwd.csv", b"a,b\n1,2\n3,4\n")})
    assert r.status_code == 200 and r.json()["filename"] == "passwd.csv"
    r = client.get(f"/api/export/{r.json()['session_id']}/..%2Fstate.json")
    assert r.status_code in (400, 404)


def test_assay_rename_keeps_block_provenance(flow):
    f = flow("A_maxquant_proteinGroups.txt")
    a = f.draft["assays"][0]
    f.step("layout", {"layout": "samples_in_columns", "assays": [dict(a, assay_label="LFQ proteomics")]})
    lfq = f.draft["groups"][gid_of(f, "LFQ intensity S01")]
    assert lfq["assay_label"] == "LFQ proteomics" and lfq["provenance"] == "ai_proposed_confirmed"
    assert f.draft["assays"][0]["provenance"] == "user_set"


def test_disagree_can_split_a_column_out(flow, monkeypatch):
    """The AI put iron among the metabolites; the user says 'iron is a clinical value'
    and the AI's suggest_split is applied by code on reconsider."""
    def iron_inside(system, prompt):
        base = json.loads(_ORIGINAL_MOCK(system, prompt))
        if prompt.startswith("Digest"):
            iron = next(g for g in base["groups"] if g["columns"] == ["iron"])
            met = next(g for g in base["groups"] if "Glucose" in g["columns"])
            met["columns"] = ["iron"] + met["columns"]
            base["groups"].remove(iron)
        return json.dumps(base)
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(iron_inside))
    f = flow("D_samples_in_rows_multiomics.csv")
    block = gid_of(f, "iron")
    assert f.draft["groups"][block]["role"] == "value" and gid_of(f, "Glucose") == block

    def with_split(system, prompt):
        digest = json.loads(prompt.split("\n", 1)[1])
        assert digest["already_confirmed"]["user_feedback"] == "iron is a clinical value"
        names = [c["column"] for c in digest["columns"]]
        return json.dumps({"groups": [{"group_id": "g1", "columns": names, "role": "value", "assay_label": "metabolomics",
                                       "label": "metabolites", "confidence": 0.9, "evidence": "n",
                                       "suggest_split": ["iron"], "suggest_split_role": "sample_metadata",
                                       "suggest_split_audit_kind": "covariate", "suggest_split_label": "serum iron"}]})
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(with_split))
    r = f.c.post("/api/reconsider", json={"session_id": f.sid, "group_ids": [block], "user_hint": "iron is a clinical value"})
    assert r.status_code == 200, r.text
    body = r.json()
    iron = next(g for g in body["session"]["groups"] if g["columns"] == ["iron"])
    it = body["draft"]["groups"][iron["group_id"]]
    assert it["role"] == "sample_metadata" and it["audit_kind"] == "covariate" and it["label"] == "serum iron"
    met = next(g for g in body["session"]["groups"] if "Glucose" in g["columns"])
    assert "iron" not in met["columns"] and met["n_columns"] == 20
