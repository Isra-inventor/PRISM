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
    assert d["signature"]["name"] == "maxquant_proteinGroups"
    assert d["layout"]["provenance"] == "signature"
    g = d["groups"]
    lfq, inten, pep = gid_of(f, "LFQ intensity S01"), gid_of(f, "Intensity S01"), gid_of(f, "Peptides S01")
    assert g[lfq]["block_role"] == "primary" and g[inten]["block_role"] == "auxiliary"
    assert g[pep]["measurement_type"] == "count" and g[pep]["validation"]["status"] == "ok"
    rev, con = gid_of(f, "Reverse"), gid_of(f, "Potential contaminant")
    assert g[rev]["kind"] == "flag_decoy" and g[con]["kind"] == "flag_contaminant"
    assert d["flags"][rev] == 3 and d["flags"][con] == 2
    f.confirm_all_as_proposed()
    out = f.finalize()
    schema = out["schema"]
    assert [b["block_role"] for b in schema["assays"][0]["value_blocks"]].count("primary") == 1
    rev_ann = next(a for a in schema["feature_annotations"] if a["column"] == "Reverse")
    assert rev_ann["n_flagged"] == 3 and rev_ann["provenance"] == "signature"
    vm = rows_of(f.export("value_matrix_A1.csv"))
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


# ---------------------------------------------------------------- B: DIA-NN

def test_B_diann_log_scale(flow):
    f = flow("B_diann_pg_matrix.tsv")
    g = f.draft["groups"]
    assert g[gid_of(f, "PG.ProteinNames")]["kind"] == "protein_name"
    assert g[gid_of(f, "PG.Genes")]["kind"] == "gene_symbol"
    block = g[gid_of(f, "S01.raw")]
    assert block["scale"] == "log2" and block["validation"]["status"] == "ok"
    assert f.draft["samples"]["ids"][:2] == ["S01", "S02"]      # '.raw' stripped
    f.confirm_all_as_proposed()
    vm = rows_of(f.export("value_matrix_A1.csv")) if f.finalize() else None
    assert "" in [c for r in vm[1:] for c in r[1:]]              # 'NaN' written as empty


# ---------------------------------------------------------------- C: MZmine (AI path)

def test_C_mzmine_ai_path(flow):
    f = flow("C_mzmine_feature_table.csv")
    d = f.draft
    assert d["signature"] is None and d["ai"]["provider"] == "mock"
    kinds = {c: d["groups"][gid_of(f, c)]["kind"] for c in
             ("row m/z", "row retention time", "compound_name", "formula", "adduct", "HMDB_ID")}
    assert kinds == {"row m/z": "mz", "row retention time": "retention_time", "compound_name": "metabolite_name",
                     "formula": "molecular_formula", "adduct": "adduct", "HMDB_ID": "metabolite_db_id"}
    st = d["sample_types"]
    assert st["QC_01"]["type"] == "qc" and st["blank_02"]["type"] == "blank" and st["S01"]["type"] == "study"
    # composite key m/z + RT is accepted
    f.step("layout", {"layout": "samples_in_columns", "omics_type": "metabolomics"})
    f.step("feature_id", {"feature_identity": {"group_ids": [gid_of(f, "row m/z"), gid_of(f, "row retention time")]}})
    assert f.draft["feature_identity"]["composite"] is True
    assert f.draft["groups"][gid_of(f, "row ID")]["role"] == "feature_annotation"
    for st_ in ("annotations", "values", "samples"):
        f.step(st_, {})
    f.step("sample_info", {"metadata": {"skip": True}})
    f.step("history", HISTORY)
    out = f.finalize()
    assert any(fl["flag"] == "no_sample_metadata" for fl in out["integrity_flags"])
    sm = rows_of(f.export("sample_metadata.csv"))
    assert len(sm) == 18 and ["QC_01", "qc"] in [r[:2] for r in sm]     # QC kept, flagged
    vm = rows_of(f.export("value_matrix_A1.csv"))
    assert "|" in vm[1][0]


# ---------------------------------------------------------------- D: samples in rows, two assays

def test_D_samples_in_rows(flow):
    f = flow("D_samples_in_rows_multiomics.csv")
    d = f.draft
    assert d["layout"]["value"] == "samples_in_rows"
    g = d["groups"]
    for c in ("age", "CD4_count", "iron"):
        assert g[gid_of(f, c)]["role"] == "sample_metadata" and g[gid_of(f, c)]["kind"] == "covariate_numeric"
    visit = g[gid_of(f, "visit")]
    assert visit["kind"] == "timepoint" and visit["detail"] == "ordinal_label"
    assert g[gid_of(f, "severity_group")]["kind"] == "group"
    values = [it for it in g.values() if it["role"] == "value"]
    assert sorted(it["omics_type"] for it in values) == ["metabolomics", "proteomics"]
    assert all(it["block_role"] == "primary" for it in values)
    f.confirm_all_as_proposed()
    out = f.finalize()
    assert [a["omics_type"]["value"] for a in out["schema"]["assays"]] in (["proteomics", "metabolomics"],
                                                                           ["metabolomics", "proteomics"])
    names = out["artifacts"]
    assert "value_matrix_A1.csv" in names and "value_matrix_A2.csv" in names
    m = rows_of(f.export("value_matrix_A1.csv"))
    assert m[0][1:4] == ["S001", "S002", "S003"] and len(m[0]) == 41     # transposed: features x samples
    sm = rows_of(f.export("sample_metadata.csv"))
    assert {"age", "CD4_count", "iron", "visit"} <= set(sm[0])
    assert len(sm) == 41


# ---------------------------------------------------------------- E: SomaScan-like

def test_E_somascan(flow):
    f = flow("E_somascan_adat_like.csv")
    g = f.draft["groups"]
    for c in ("PlateScale_Scalar", "HybControlNormScale", "NormScale_20"):
        assert g[gid_of(f, c)]["kind"] == "technical_numeric"
    st = f.draft["sample_types"]
    types = {v["type"] for v in st.values()}
    assert {"study", "qc", "calibrator", "blank"} <= types


# ---------------------------------------------------------------- F: long format

def test_F1_long_pivots(flow):
    f = flow("F1_long_unique.csv")
    assert f.draft["layout"]["value"] == "long"
    f.confirm_all_as_proposed()
    f.finalize()
    m = rows_of(f.export("value_matrix_A1.csv"))
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
        if g["role"] == "value":
            g["measurement_type"] = "count"          # non-integer data: must be downgraded
    base["groups"].append({"group_id": "g999", "role": "value", "confidence": 0.9, "evidence": "made up"})
    base["groups"] = [g for g in base["groups"] if g["group_id"] != "g2"]   # a group left out
    return json.dumps(base)


def test_wrong_ai_proposal_is_downgraded(flow, monkeypatch):
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(_wrong))
    f = flow("C_mzmine_feature_table.csv")
    d = f.draft
    block = d["groups"][gid_of(f, "S01 Peak area")]
    assert block["role"] == "unresolved" and block["validation"]["status"] == "contradicted"
    assert block["claimed"]["measurement_type"] == "count"
    assert any(r["group_id"] == "g999" for r in d["rejected"])
    assert d["groups"]["g2"]["role"] == "unresolved"
    assert any("unresolved" in u["what"] for u in d["unresolved"])


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
    f.step("layout", {"layout": "samples_in_columns", "omics_type": "metabolomics", "source_software": "MZmine"})
    items = []
    for g in f.upload["groups"]:
        c = g["columns"][0]
        if g["kind"] == "numeric_block":
            items.append({"group_id": g["group_id"], "role": "value", "measurement_type": "intensity",
                          "scale": "linear", "block_role": "primary", "omics_type": "metabolomics"})
        elif c != "row ID":
            items.append({"group_id": g["group_id"], "role": "feature_annotation", "kind": "other_annotation"})
    f.step("feature_id", {"feature_identity": {"group_ids": [gid_of(f, "row ID")]}})
    f.step("annotations", {"items": [i for i in items if i["role"] == "feature_annotation"]})
    f.step("values", {"items": [i for i in items if i["role"] == "value"]})
    f.step("samples", {"sample_types": {"QC_01": "qc"}})
    f.step("sample_info", {"metadata": {"skip": True}})
    f.step("history", HISTORY)
    out = f.finalize()
    prov = {a["column"]: a["provenance"] for a in out["schema"]["feature_annotations"]}
    assert prov["formula"] == "user_set"
    assert out["schema"]["layout"]["provenance"] == "user_set"
    assert out["schema"]["ai"]["enabled"] is False


def test_provenance_confirmed_vs_corrected(flow):
    f = flow("C_mzmine_feature_table.csv")
    f.confirm_all_as_proposed()
    f.step("annotations", {"items": [{"group_id": gid_of(f, "adduct"), "kind": "other_annotation"}]})
    f.step("history", HISTORY)
    for st in ("values", "samples", "sample_info"):
        if f.draft["steps"][st] == "pending":
            f.step(st, {} if st != "sample_info" else {"metadata": {"skip": True}})
    schema = f.finalize()["schema"]
    prov = {a["column"]: a["provenance"] for a in schema["feature_annotations"]}
    assert prov["adduct"] == "ai_proposed_corrected" and prov["formula"] == "ai_proposed_confirmed"


def test_history_must_be_answered_and_no_default(flow):
    f = flow("B_diann_pg_matrix.tsv")
    assert all(v["answer"] is None for v in f.draft["processing_history"].values())
    f.step("history", {"processing_history": {"normalized": {"answer": "yes"}}}, expect=422)


def test_values_step_enforces_primary(flow):
    f = flow("A_maxquant_proteinGroups.txt")
    f.step("values", {"items": [{"group_id": gid_of(f, "Intensity S01"), "block_role": "primary"}]}, expect=422)
    f.step("values", {"items": [{"group_id": gid_of(f, "LFQ intensity S01"), "block_role": "auxiliary"}]}, expect=422)


def test_layout_change_triggers_reproposal(flow):
    f = flow("C_mzmine_feature_table.csv")
    r = f.step("layout", {"layout": "samples_in_rows", "omics_type": "metabolomics"})
    assert r["reproposed"] is True
    assert f.draft["layout"]["value"] == "samples_in_rows"
    assert f.draft["steps"]["feature_id"] in ("pending", "not_applicable")


def test_reconsider_one_group(flow, monkeypatch):
    f = flow("C_mzmine_feature_table.csv")
    seen = {}

    def answer(system, prompt):
        digest = json.loads(prompt.split("\n", 1)[1])
        seen["groups"] = [g["group_id"] for g in digest["groups"]]
        seen["hint"] = digest["already_confirmed"].get("user_hint_for_this_group")
        return json.dumps({"groups": [{"group_id": seen["groups"][0], "role": "feature_annotation",
                                       "kind": "other_annotation", "confidence": 0.9, "evidence": "user hint"}]})
    monkeypatch.setattr(mock_llm.MockLLM, "respond", staticmethod(answer))
    gid = gid_of(f, "adduct")
    r = f.c.post("/api/reconsider", json={"session_id": f.sid, "group_id": gid, "user_hint": "these are ion forms"})
    assert r.status_code == 200, r.text
    assert seen == {"groups": [gid], "hint": "these are ion forms"}
    assert r.json()["draft"]["groups"][gid]["kind"] == "other_annotation"


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
    f.step("layout", {"layout": "samples_in_columns", "omics_type": "proteomics"})
    f.step("feature_id", {"feature_identity": {"group_ids": d["feature_identity"]["group_ids"]}})
    for st in ("annotations", "values", "samples"):
        f.step(st, {})
    f.step("sample_info", {"metadata": {"accept_near_misses": [["S02", "s02"]],
                                        "columns": [{"column": "group", "kind": "group"},
                                                    {"column": "age", "kind": "covariate_numeric"}]}})
    f.step("history", HISTORY)
    f.finalize()
    sm = {r[0]: r for r in rows_of(f.export("sample_metadata.csv"))}
    assert sm["S02"][2] == "ctrl"                       # near miss accepted by the user
    assert sm["S04"][2] == ""                           # no metadata row: kept, empty


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
