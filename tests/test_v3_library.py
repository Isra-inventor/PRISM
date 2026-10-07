"""v3 stage 9 (could-haves): the saved-schemas library and "explain differences"."""

from prism import store
from prism.io import importer, library
from conftest import fixture_bytes
from test_v3_import import do_import, new_session, wizard_output

C = "C_mzmine_feature_table.csv"


def test_confirmed_schemas_are_saved_and_suggested(client, flow, isolated):
    sid = new_session(client)
    r = client.post("/api/upload", files={"file": (C, fixture_bytes(C))}, data={"study_session_id": sid})
    assert r.status_code == 200 and r.json()["library_matches"] == []          # never seen
    o1 = wizard_output(flow, C, isolated)
    rep = do_import(client, sid, C, fixture_bytes(C), (o1 / "schema.json").read_bytes())
    r = client.post(f"/api/sessions/{sid}/datasets/{rep['dataset_id']}/accept-import")
    assert r.status_code == 200, r.text
    sha = rep["binding"]["main"]["uploaded_sha256"]
    (entry,) = library.matches(sha)
    assert entry["source_file"] == C and entry["session_id"] == sid and entry["mode"] == "exact"
    assert client.get(f"/api/library/{sha}").json() == [entry]
    # the same file uploaded for the wizard: the saved schema is suggested, nothing is applied
    sid2 = new_session(client)
    r = client.post("/api/upload", files={"file": (C, fixture_bytes(C))}, data={"study_session_id": sid2})
    body = r.json()
    assert [m["schema_id"] for m in body["library_matches"]] == [entry["schema_id"]]
    st2 = store.load(sid2)
    assert st2.datasets[0]["status"] == "wizard_in_progress"
    # used: it goes through the import review (exact mode); the wizard attempt is replaced
    did = body["study"]["dataset_id"]
    r = client.post(f"/api/sessions/{sid2}/datasets/{did}/use-saved-schema", json={"schema_id": entry["schema_id"]})
    assert r.status_code == 200, r.text
    rep2 = r.json()["report"]
    assert rep2["mode"] == "exact" and rep2["status"] == "imported_awaiting_confirm"
    assert [d["dataset_id"] for d in r.json()["session"]["datasets"]] == [rep2["dataset_id"]]
    assert not store.load(sid2).dataset_dir(did).exists()
    # a saved schema of another file is refused, and the wizard dataset stays
    sid3 = new_session(client)
    other = fixture_bytes("D_samples_in_rows_multiomics.csv")
    r = client.post("/api/upload", files={"file": ("D.csv", other)}, data={"study_session_id": sid3})
    did3 = r.json()["study"]["dataset_id"]
    r = client.post(f"/api/sessions/{sid3}/datasets/{did3}/use-saved-schema", json={"schema_id": entry["schema_id"]})
    assert r.status_code == 422 and "different file" in r.json()["detail"]
    assert store.load(sid3).dataset(did3)["status"] == "wizard_in_progress"
    # saving is idempotent per schema fingerprint
    assert len(library.matches(sha)) == 1


def test_explain_differences(client, flow, isolated):
    o1 = wizard_output(flow, C, isolated)
    data = fixture_bytes(C).decode().replace("20970.0", "99999999.0", 1).encode()
    sid = new_session(client)
    rep = do_import(client, sid, C, data, (o1 / "schema.json").read_bytes())
    r = client.get(f"/api/sessions/{sid}/datasets/{rep['dataset_id']}/explain-differences")
    assert r.status_code == 200
    ex = r.json()["explanations"]
    kinds = [x["kind"] for x in ex]
    assert kinds[0] == "summary" and "template mode these are expected" in ex[0]["text"]
    assert "file" in kinds and "values" in kinds
    mx = next(x for x in ex if x.get("path", "").endswith("profile.max"))
    assert "largest value was 13701600.0, it is now 100000000.0" in mx["text"]
    # a renamed column (seeded wizard): the binding is explained
    data = fixture_bytes(C).decode().replace("compound_name", "compound", 1).encode()
    rep = do_import(client, sid, "C.csv", data, (o1 / "schema.json").read_bytes())
    texts = " ".join(x["text"] for x in importer.explain(rep))
    assert "absent from the file: compound_name" in texts and "does not know: compound" in texts
    # nothing to explain
    assert importer.explain({"differences": [], "binding": {"main": {}}})[0]["kind"] == "none"


def test_file_name_difference_is_explained():
    ex = importer.explain({"differences": [{"path": "$.files[0].name", "stored": "a.csv", "recomputed": "b.csv"}]})
    assert ex[0]["kind"] == "file" and "made from 'a.csv'" in ex[0]["text"]
