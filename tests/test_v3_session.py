"""v3 stage 1: sessions of datasets, the output-folder loader, ledger, manifest, CLI."""

import json
import math

import numpy as np

from prism import cli, ledger, manifest, store
from prism.io.loader import load_output
from prism.util import schema_sha256


def wizard_dataset(client, flow_cls, name="C_mzmine_feature_table.csv", **kw):
    st = store.create("test")
    from conftest import fixture_bytes
    r = client.post("/api/upload", files={"file": (name, fixture_bytes(name))}, data={"study_session_id": st.sid})
    assert r.status_code == 200, r.text
    return st, r.json()


def test_wizard_dataset_is_published_into_the_session(client, flow):
    st = store.create("t")
    f = flow("C_mzmine_feature_table.csv")
    # attach through the upload form field instead (same flow as the frontend)
    from conftest import Flow, fixture_bytes
    r = client.post("/api/upload", files={"file": ("C.csv", fixture_bytes("C_mzmine_feature_table.csv"))},
                    data={"study_session_id": st.sid})
    sid = r.json()["session_id"]
    assert r.json()["study"]["dataset_id"] == "D1"
    st = store.load(st.sid)
    assert st.datasets[0]["status"] == "wizard_in_progress" and st.datasets[0]["step0_session_id"] == sid
    assert (st.dataset_dir("D1") / "upload" / "C.csv").exists()
    f.sid = sid
    r = client.post("/api/propose", json={"session_id": sid, "ai": True})
    f.draft, f.upload = r.json()["draft"], r.json()["session"]
    f.confirm_all_as_proposed()
    f.finalize()
    st = store.load(st.sid)
    d = st.dataset("D1")
    assert d["status"] == "confirmed" and d["confirmed_at"]
    out = st.output_dir("D1")
    names = sorted(p.name for p in out.iterdir())
    assert names == ["feature_metadata.csv", "import_manifest.json", "sample_metadata.csv", "schema.json",
                     "value_matrix_A1_B1.csv"]
    man = json.loads((out / "import_manifest.json").read_text())
    schema = json.loads((out / "schema.json").read_text())
    assert man["mode"] == "wizard" and man["schema_sha256"] == schema_sha256(schema) == d["schema_sha256"]
    assert (st.dataset_dir("D1") / "schema.json").exists()
    o = load_output(out)
    (a,) = o.assays
    (b,) = a["blocks"]
    assert b.X.shape == (a["meta"]["n_features"], a["meta"]["n_samples"]) and len(b.sample_ids) == 17
    assert np.isnan(b.X).any() and b.n_unparsable == 0
    assert o.sample_table["columns"][:3] == ["sample_id", "sample_label", "is_study_sample"]
    assert len(o.features_of("A1")) == a["meta"]["n_features"]


def test_schema_fingerprint_ignores_timestamps_and_ai():
    s = {"schema_version": "0.4.1", "x": [{"at": "2026-01-01", "v": 1}], "ai": {"model": "a"}}
    t = {"schema_version": "0.4.1", "x": [{"at": "2027-05-05", "v": 1}], "ai": {"model": "b"}}
    assert schema_sha256(s) == schema_sha256(t) != schema_sha256(dict(t, x=[{"v": 2}]))


def test_loader_parses_missing_and_bad_cells(tmp_path):
    (tmp_path / "schema.json").write_text(json.dumps({"assays": [{"assay_id": "A1", "value_blocks": [
        {"block_id": "B1", "file": "value_matrix_A1_B1.csv"}]}]}))
    (tmp_path / "value_matrix_A1_B1.csv").write_text("feature_key,S1,S2\nf1,1.5,\nf2,abc,2,5\n".replace("2,5", '"2,5"'))
    o = load_output(tmp_path)
    X = o.assays[0]["blocks"][0].X
    assert X[0, 0] == 1.5 and math.isnan(X[0, 1]) and math.isnan(X[1, 0]) and X[1, 1] == 2.5
    assert o.assays[0]["blocks"][0].n_unparsable == 1


def test_ledger_manifest_and_cli(tmp_path, capsys):
    lg = ledger.Ledger(tmp_path / "ledger.jsonl")
    lg.record("run_audit", {"factor": "A4"}, who="ai", output={"a": 1})
    (rec,) = lg.read()
    assert rec["who"] == "ai" and rec["output_sha256"] and rec["params"] == {"factor": "A4"}
    m = manifest.build("r1", "s1", {"seed": 0, "permutations": 999}, {"f": "x"}, None, {}, {})
    assert m["seed"] == 0 and m["params_sha256"] == manifest.params_hash({"permutations": 999, "seed": 0})
    assert cli.main(["session", "new", "--name", "demo"]) == 0
    sid = capsys.readouterr().out.strip()
    assert cli.main(["session", "list"]) == 0 and sid in capsys.readouterr().out
    assert cli.main(["session", "show", "nope"]) == 2
    assert store.load(sid).data["name"] == "demo"


def test_session_api(client):
    r = client.post("/api/sessions", json={"name": "two omics"})
    assert r.status_code == 200 and r.json()["session_id"].startswith("s")
    sid = r.json()["session_id"]
    assert client.get(f"/api/sessions/{sid}").json()["name"] == "two omics"
    assert client.get("/api/sessions/s00000000000").status_code == 404
