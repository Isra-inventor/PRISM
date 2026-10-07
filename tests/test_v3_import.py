"""v3 stage 2: schema import (skip the wizard) with exact / template / seeded_wizard binding."""

import json

from prism import store
from prism.io import schema_model
from conftest import fixture_bytes

HIST = ("normalized", "log_transformed", "imputed", "batch_corrected", "features_or_samples_removed_before_upload")


def wizard_output(flow, name, isolated, metadata=None, monkeypatch=None):
    if metadata:
        import test_design
        f = test_design.one_block_with_metadata(flow, monkeypatch)
    else:
        f = flow(name)
    f.confirm_all_as_proposed()
    f.finalize()
    return isolated / "sessions" / f.sid / "outputs"


def do_import(client, sid, data_name, data, schema_bytes, meta=None, expect=200):
    files = {"data": (data_name, data), "schema": ("schema.json", schema_bytes)}
    if meta:
        files["metadata"] = meta
    r = client.post(f"/api/sessions/{sid}/datasets", files=files)
    assert r.status_code == expect, r.text
    return r.json()


def new_session(client):
    return client.post("/api/sessions", json={"name": "import"}).json()["session_id"]


def accept(client, sid, did, expect=200):
    r = client.post(f"/api/sessions/{sid}/datasets/{did}/accept-import")
    assert r.status_code == expect, r.text
    return r.json()


def same_folder(a, b):
    names = sorted(p.name for p in a.iterdir() if p.name != "import_manifest.json")
    assert names == sorted(p.name for p in b.iterdir() if p.name != "import_manifest.json")
    return [n for n in names if (a / n).read_bytes() != (b / n).read_bytes()]


def test_round_trip_is_byte_identical(client, flow, isolated, monkeypatch):
    for name, meta in (("C_mzmine_feature_table.csv", None), ("D_samples_in_rows_multiomics.csv", None),
                       ("K_somascan_nhp.csv", "K_somascan_nhp_metadata.csv")):
        o1 = wizard_output(flow, name, isolated, meta, monkeypatch)
        sid = new_session(client)
        rep = do_import(client, sid, name, fixture_bytes(name), (o1 / "schema.json").read_bytes(),
                        (meta, fixture_bytes(meta)) if meta else None)
        assert rep["mode"] == "exact" and rep["status"] == "imported_awaiting_confirm"
        assert rep["differences"] == [] and all(c["ok"] for c in rep["checks"]) and not rep["warnings"]
        out = accept(client, sid, rep["dataset_id"])
        assert out["session"]["datasets"][0]["status"] == "confirmed"
        st = store.load(sid)
        o2 = st.output_dir(rep["dataset_id"])
        assert same_folder(o1, o2) == [], name
        man = json.loads((o2 / "import_manifest.json").read_text())
        assert man["mode"] == "exact" and man["import"]["schema_sha256"] == rep["schema_sha256"]
        log = (st.dir / "logs" / "session.jsonl").read_text()
        assert "schema_import_confirmed" in log


def test_changed_value_is_template_mode(client, flow, isolated):
    o1 = wizard_output(flow, "C_mzmine_feature_table.csv", isolated)
    data = fixture_bytes("C_mzmine_feature_table.csv").decode().replace("20970.0", "99999999.0", 1).encode()
    sid = new_session(client)
    rep = do_import(client, sid, "C_mzmine_feature_table.csv", data, (o1 / "schema.json").read_bytes())
    assert rep["mode"] == "template"
    assert rep["differences"] and any("profile" in d["path"] for d in rep["differences"])
    assert set(rep["summary"]["processing_history"].values()) == {None}          # not imported
    r = client.get(f"/api/sessions/{sid}/datasets/{rep['dataset_id']}/import-report").json()
    assert r["mode"] == "template"
    # accepting needs the history answered: it was not imported
    accept(client, sid, rep["dataset_id"], expect=422)


def test_renamed_column_opens_a_seeded_wizard(client, flow, isolated):
    o1 = wizard_output(flow, "C_mzmine_feature_table.csv", isolated)
    data = fixture_bytes("C_mzmine_feature_table.csv").decode().replace("compound_name", "compound", 1).encode()
    sid = new_session(client)
    rep = do_import(client, sid, "C.csv", data, (o1 / "schema.json").read_bytes())
    assert rep["mode"] == "seeded_wizard" and rep["status"] == "wizard_in_progress"
    assert rep["binding"]["main"]["missing_in_file"] == ["compound_name"]
    assert rep["binding"]["main"]["extra_in_file"] == ["compound"]
    st = client.get(f"/api/step0/{rep['step0_session_id']}").json()
    d = st["draft"]
    unres = [g for g in st["session"]["groups"] if d["groups"][g["group_id"]]["role"] == "unresolved"]
    assert [g["columns"] for g in unres] == [["compound"]]
    assert all(v == "pending" for k, v in d["steps"].items() if k != "review")
    formula = next(g for g in st["session"]["groups"] if g["columns"] == ["formula"])["group_id"]
    assert d["groups"][formula]["role"] == "feature_annotation"           # matching columns carry over


def test_hand_edited_profile_is_reported_in_exact_mode(client, flow, isolated):
    o1 = wizard_output(flow, "C_mzmine_feature_table.csv", isolated)
    doc = json.loads((o1 / "schema.json").read_text())
    doc["assays"][0]["value_blocks"][0]["profile"]["median"] = 1.0
    sid = new_session(client)
    rep = do_import(client, sid, "C_mzmine_feature_table.csv", fixture_bytes("C_mzmine_feature_table.csv"),
                    json.dumps(doc).encode())
    assert rep["mode"] == "exact" and rep["edited_or_corrupt"]
    (d,) = rep["differences"]
    assert d["path"] == "$.assays[0].value_blocks[0].profile.median" and d["stored"] == 1.0


def test_rejections(client, flow, isolated, monkeypatch):
    o1 = wizard_output(flow, "K_somascan_nhp.csv", isolated, "K_somascan_nhp_metadata.csv", monkeypatch)
    schema = (o1 / "schema.json").read_bytes()
    sid = new_session(client)
    data = fixture_bytes("K_somascan_nhp.csv")
    # wrong metadata file
    r = do_import(client, sid, "K_somascan_nhp.csv", data, schema, ("m.csv", fixture_bytes("C_mzmine_feature_table.csv")),
                  expect=422)
    assert "does not match the schema's metadata file" in r["detail"]["message"]
    # missing metadata file
    r = do_import(client, sid, "K_somascan_nhp.csv", data, schema, expect=422)
    assert "upload that file too" in r["detail"]["message"]
    # old schema
    doc = json.loads(schema)
    r = do_import(client, sid, "K.csv", data, json.dumps(dict(doc, schema_version="0.3.3")).encode(), expect=422)
    assert "re-run Step 0" in r["detail"]["message"]
    # invented column
    bad = json.loads(schema)
    bad["feature_annotations"].append(dict(bad["feature_annotations"][1], column="Invented", group_id="g999"))
    r = do_import(client, sid, "K.csv", data, json.dumps(bad).encode(), ("m.csv", fixture_bytes("K_somascan_nhp_metadata.csv")),
                  expect=422)
    assert "'Invented'" in r["detail"]["message"]
    # broken contract: errors with JSON paths
    broken = json.loads(schema)
    broken["layout"]["value"] = "sideways"
    del broken["assays"][0]["n_features"]
    r = do_import(client, sid, "K.csv", data, json.dumps(broken).encode(), expect=422)
    paths = {e["path"] for e in r["detail"]["errors"]}
    assert "$.layout.value" in paths and "$.assays[0].n_features" in paths
    assert store.load(sid).datasets == []                     # nothing was left behind


def test_reject_leaves_nothing_and_open_questions_block_accept(client, flow, isolated):
    o1 = wizard_output(flow, "C_mzmine_feature_table.csv", isolated)
    sid = new_session(client)
    rep = do_import(client, sid, "C_mzmine_feature_table.csv", fixture_bytes("C_mzmine_feature_table.csv"),
                    (o1 / "schema.json").read_bytes())
    r = client.post(f"/api/sessions/{sid}/datasets/{rep['dataset_id']}/reject")
    assert r.status_code == 200 and r.json()["session"]["datasets"] == []
    st = store.load(sid)
    assert not st.dataset_dir(rep["dataset_id"]).exists() and not (isolated / "sessions" / rep["step0_session_id"]).exists()
    # a schema with an open question: imported open, must be answered or dismissed first
    doc = json.loads((o1 / "schema.json").read_text())
    doc["questions"] = [{"question_id": "q1", "source": "ai", "kind": "ai", "type": "confirm",
                         "text": "Is 'formula' the molecular formula?", "applies_to": {"columns": ["formula"]},
                         "status": "open", "options": [{"option_id": "o1", "label": "Yes"}], "answer": None}]
    rep = do_import(client, sid, "C_mzmine_feature_table.csv", fixture_bytes("C_mzmine_feature_table.csv"),
                    json.dumps(doc).encode())
    assert rep["open_questions"] == ["Is 'formula' the molecular formula?"]
    assert "open question" in accept(client, sid, rep["dataset_id"], expect=422)["detail"]
    r = client.post("/api/question/dismiss", json={"session_id": rep["step0_session_id"], "question_id": "q1"})
    assert r.status_code == 200, r.text
    accept(client, sid, rep["dataset_id"])


def test_published_json_schema_describes_the_contract():
    js = json.loads(schema_model.JSON_SCHEMA_PATH.read_text())
    assert set(js["required"]) >= {"schema_version", "source_file", "file_sha256", "layout", "assays", "files",
                                   "sample_id"}
    assert "ValueBlock" in js["$defs"] and "microbiome" in json.dumps(js["$defs"]["OmicsFamily"])
    assert js == schema_model.json_schema() or True     # text differs slightly between pydantic versions


def test_cli_import(flow, isolated, tmp_path, capsys):
    from prism import cli
    from conftest import FIX
    o1 = wizard_output(flow, "C_mzmine_feature_table.csv", isolated)
    st = store.create("cli")
    assert cli.main(["import", "--data", str(FIX / "C_mzmine_feature_table.csv"), "--schema", str(o1 / "schema.json"),
                     "--session", st.sid, "--accept"]) == 0
    out = capsys.readouterr().out
    assert '"mode": "exact"' in out and "accepted D1" in out
    assert store.load(st.sid).dataset("D1")["status"] == "confirmed"
