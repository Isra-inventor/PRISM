import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import llm_fallback, main, session_log
from backend.format_detect import detect, match_signatures
from backend.models import AIProposalBatch, ColumnProposal

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture(autouse=True)
def tmp_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(session_log, "LOG_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def client():
    return TestClient(main.app)


def upload(client, name, content=None):
    data = content if content is not None else (EXAMPLES / name).read_bytes()
    return client.post("/api/upload", files={"file": (name, data)})


# ------------------------------------------------------------ detection

def test_signature_requires_all_columns():
    assert match_signatures(["Protein IDs", "Reverse", "x"]) == ["maxquant_proteinGroups"]
    assert match_signatures(["Protein IDs", "x"]) == []            # partial = no match
    assert match_signatures(["protein ids", "reverse"]) == []      # no fuzzy matching


def test_diann_value_columns_follow_pg_columns():
    header = ["PG.ProteinGroups", "PG.Genes", "run1.raw", "run2.raw"]
    det = detect(header, [["P1", "G", "1", "2"]])
    assert [header[i] for i in det.value_column_indices] == ["run1.raw", "run2.raw"]


def test_fragpipe_prefers_maxlfq():
    header = ["Protein", "A Intensity", "A MaxLFQ Intensity", "A Unique Intensity", "Indistinguishable Proteins"]
    det = detect(header, [])
    assert [header[i] for i in det.value_column_indices] == ["A MaxLFQ Intensity"]


def test_maxquant_upload_is_deterministic(client, monkeypatch, tmp_logs):
    monkeypatch.setattr(llm_fallback, "propose_roles", lambda *a: pytest.fail("AI must not be called"))
    r = upload(client, "maxquant_proteinGroups.txt")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["method"] == "signature"
    assert d["detection"]["platform"] == "MaxQuant proteinGroups.txt"
    values = [c["column"] for c in d["columns"] if c["role"] == "feature_value"]
    assert values and all(v.startswith("LFQ intensity ") for v in values)

    # confirm unchanged
    cols = [{"index": c["index"], "column": c["column"], "role": c["role"]} for c in d["columns"]]
    s = client.post(f"/api/sessions/{d['session_id']}/confirm", json={"columns": cols}).json()
    assert s["sample_count"] == 6 and s["feature_count"] == 20
    assert s["omics_type"] == "proteomics" and s["changes_from_proposal"] == 0

    events = [json.loads(l)["event"] for l in (tmp_logs / f"{d['session_id']}.jsonl").read_text().splitlines()]
    assert events == ["upload", "signature_match", "confirmation"]


def test_metabolomics_feature_table(client):
    d = upload(client, "xcms_feature_table.csv").json()
    assert d["detection"]["signature"] == "generic_feature_table"
    values = [c["column"] for c in d["columns"] if c["role"] == "feature_value"]
    assert len(values) == 10 and "mzmin" not in values


# ------------------------------------------------------------ input validation

@pytest.mark.parametrize("name,content", [
    ("run.raw", b"\x00\x01binary"),
    ("data.xlsx", b"PK\x03\x04"),
    ("spectra.mzML", b"<mzML/>"),
    ("fake.txt", b"no tabs here\nat all\n"),
])
def test_rejects_non_tables(client, name, content):
    r = upload(client, name, content)
    assert r.status_code == 415
    assert "quantified" in r.json()["detail"]


# ------------------------------------------------------------ AI fallback

def test_reconcile_never_defaults_unresolved():
    header = ["a", "b", "c", "d"]
    batch = AIProposalBatch(columns=[
        ColumnProposal(column="a", proposed_role="sample_id", confidence=0.9, evidence="x"),
        ColumnProposal(column="b", proposed_role="made_up_role", confidence=0.9, evidence="x"),
        ColumnProposal(column="c", proposed_role="unresolved", confidence=0.2, evidence="unclear"),
    ])
    out = llm_fallback._reconcile(header, [0, 1, 2, 3], batch)
    assert [o["role"] for o in out] == ["sample_id", "unresolved", "unresolved", "unresolved"]


def test_no_api_key_means_manual(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    d = upload(client, "cohort_metabolites_wide.csv").json()
    assert d["method"] == "manual"
    assert all(c["role"] == "unresolved" for c in d["columns"])


def test_ai_flow_and_confirmation_log(client, monkeypatch, tmp_logs):
    def fake(header, rows, n):
        roles = {"sample_code": "sample_id", "patient_code": "subject_id", "visit": "timepoint",
                 "plate": "batch", "diagnosis": "group_or_outcome", "injection_order": "ignore",
                 "age": "group_or_outcome", "operator_note": "unresolved"}
        cols = [{"index": i, "column": h, "role": roles.get(h, "feature_value"),
                 "confidence": 0.8, "evidence": "test"} for i, h in enumerate(header)]
        return {"status": "ok", "model": "fake", "error": None, "calls": [], "columns": cols}

    monkeypatch.setattr(llm_fallback, "propose_roles", fake)
    d = upload(client, "cohort_metabolites_wide.csv").json()
    assert d["method"] == "ai"
    cols = [{"index": c["index"], "column": c["column"], "role": c["role"]} for c in d["columns"]]

    # unresolved cannot be confirmed
    r = client.post(f"/api/sessions/{d['session_id']}/confirm", json={"columns": cols})
    assert r.status_code == 422

    cols[-1]["role"] = "ignore"          # user resolves operator_note
    cols[6]["role"] = "feature_annotation"  # user corrects 'age'
    s = client.post(f"/api/sessions/{d['session_id']}/confirm", json={"columns": cols}).json()
    assert s["platform"] == "generic/AI-assisted"
    assert s["layout"] == "samples_as_rows"
    assert s["sample_count"] == 12 and s["feature_count"] == 8
    assert s["changes_from_proposal"] == 2

    log = [json.loads(l) for l in (tmp_logs / f"{d['session_id']}.jsonl").read_text().splitlines()]
    conf = log[-1]
    assert conf["event"] == "confirmation" and conf["timestamp"]
    decisions = {x["column"]: x for x in conf["decisions"]}
    assert decisions["operator_note"]["decision"] == "assigned"
    assert decisions["age"] == {**decisions["age"], "proposed_role": "group_or_outcome",
                                "final_role": "feature_annotation", "decision": "corrected"}


def test_confirm_rejects_mismatched_columns(client):
    d = upload(client, "xcms_feature_table.csv").json()
    cols = [{"index": c["index"], "column": c["column"], "role": c["role"]} for c in d["columns"]]
    cols[0]["column"] = "renamed"
    assert client.post(f"/api/sessions/{d['session_id']}/confirm", json={"columns": cols}).status_code == 422
    assert client.post(f"/api/sessions/{d['session_id']}/confirm", json={"columns": cols[1:]}).status_code == 422


def test_values_are_not_modified(client):
    raw = (EXAMPLES / "xcms_feature_table.csv").read_text().splitlines()
    d = upload(client, "xcms_feature_table.csv").json()
    first = raw[1].split(",")
    assert d["preview_rows"][0] == first


def test_gemini_call_is_parsed_strictly(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    answer = {"columns": [
        {"column": "a", "proposed_role": "sample_id", "confidence": 0.93, "evidence": "unique ids"},
        {"column": "b", "proposed_role": "unresolved", "confidence": 0.3, "evidence": "unclear"},
    ]}
    seen = {}

    def fake_call(model, key, system, prompt):
        seen.update(model=model, key=key)
        return json.dumps(answer), "STOP"

    monkeypatch.setattr(llm_fallback, "call_gemini", fake_call)
    r = llm_fallback.propose_roles(["a", "b", "c"], [["S1", "?", "x"]], 1)
    assert r["status"] == "ok" and seen["key"] == "test-key"
    assert [c["role"] for c in r["columns"]] == ["sample_id", "unresolved", "unresolved"]


def test_gemini_truncated_answer_is_unresolved(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(llm_fallback, "call_gemini", lambda *a: ('{"columns": [', "MAX_TOKENS"))
    r = llm_fallback.propose_roles(["a"], [["S1"]], 1)
    assert r["columns"][0]["role"] == "unresolved"
