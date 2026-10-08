"""v3 stage 7: audit API, overrides API, the report bundle and the self-contained HTML export."""

import json
import re

import pytest

from prism import store
from prism.audit import report
import audit_fixtures as af


@pytest.fixture
def st(isolated, tmp_path):
    s = store.create("report")
    X, samples, cols, kinds = af.planted(shift=True, outliers=True)
    af.add(s, tmp_path / "planted", X=X, samples=samples, columns=cols, kinds=kinds, design=af.SUBJ_DESIGN,
           history={"imputed": {"answer": "no"}})
    return s


def test_audit_api(client, st):
    base = f"/api/sessions/{st.sid}"
    assert client.get(f"{base}/audit").json() == []
    r = client.post(f"{base}/audit/run", json={"factors": ["A4", "A8", "A5"]})
    assert r.status_code == 200, r.text
    a = r.json()
    assert {i["audit_id"] for i in a["findings"]} == {"A4", "A8", "A5"} and a["who"] == "user"
    r = client.post(f"{base}/audit/run", json={"factors": ["A4"], "params": {"permutations": 99}})
    b = r.json()
    runs = client.get(f"{base}/audit").json()
    assert [x["run_id"] for x in runs] == [a["run_id"], b["run_id"]]
    shown = client.get(f"{base}/audit/{b['run_id']}").json()
    assert shown["previous_run"] == a["run_id"] and shown["manifest"]["params"]["permutations"] == 99
    assert shown["findings"][0]["audit_id"] == "A4" and "heuristic" in shown["params_meta"]
    cmp = client.get(f"{base}/audit/{b['run_id']}/compare/{a['run_id']}").json()
    assert cmp["params_changed"] == {"permutations": {"a": 999, "b": 99}}
    assert client.post(f"{base}/audit/run", json={"factors": ["A42"]}).status_code == 422
    assert client.post(f"{base}/audit/run", json={"params": {"seed": "x"}}).status_code == 422
    assert client.get(f"{base}/audit/r9999_x").status_code == 422
    p = client.get("/api/audit/params").json()
    assert p["defaults"]["permutations"] == 999 and "modified_z_cutoff" in p["heuristic"]


def test_overrides_api(client, st):
    base = f"/api/sessions/{st.sid}"
    r = client.post(f"{base}/overrides", json={"kind": "sample_role", "sample": "A00_0", "role": "qc", "reason": "pool"})
    assert r.status_code == 200, r.text
    assert r.json()["override"]["override_id"] == "o1" and r.json()["override"]["by"] == "user"
    assert client.post(f"{base}/overrides", json={"kind": "sample_role", "sample": "A00_0", "role": "x"}).status_code == 422
    assert client.post(f"{base}/overrides", json={"kind": "param", "key": "nope", "value": 1}).status_code == 422
    man = client.post(f"{base}/audit/run", json={"factors": ["A6"]}).json()
    assert man["scope"]["units"][0]["non_study_samples"] == ["A00_0"]
    r = client.delete(f"{base}/overrides/o1")
    assert r.status_code == 200 and r.json()["overrides"] == []
    assert client.delete(f"{base}/overrides/o1").status_code == 422


def test_static_html_export_is_self_contained(client, st, tmp_path):
    from prism.audit import engine
    man = engine.run(st)
    html = report.html(st, man["run_id"])
    assert html.startswith("<!doctype html>") and "window.PRISM_REPORT.render" in html
    assert not re.search(r"<script[^>]+src=|<link[^>]+href=|@import|url\(http", html)       # nothing fetched
    data = re.search(r'<script id="prism-report-data" type="application/json">(.*?)</script>', html, re.S).group(1)
    b = json.loads(data)
    assert b["manifest"]["run_id"] == man["run_id"] and len(b["findings"]) == len(man["findings"])
    assert {f["audit_id"] for f in b["findings"]} >= {"A1", "A4", "A7", "A11"}
    r = client.get(f"/api/sessions/{st.sid}/audit/{man['run_id']}/report.html")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    # CLI: report into the run folder (html) and to a path (json)
    from prism import cli
    assert cli.main(["audit", "report", "--session", st.sid]) == 0
    assert (st.dir / "audit" / man["run_id"] / "report.html").exists()
    out = tmp_path / "r.json"
    assert cli.main(["audit", "report", "--session", st.sid, "--format", "json", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["manifest"]["run_id"] == man["run_id"]


def test_script_json_is_safe():
    s = report._script_json({"x": "</script><!--"})
    assert "</script>" not in s and "<!--" not in s and json.loads(s) == {"x": "</script><!--"}


def test_final_report(client, st, tmp_path):
    from prism.audit import engine
    base = f"/api/sessions/{st.sid}"
    fb = client.get(f"{base}/final-report").json()               # before any audit: datasets only
    assert fb["audit"] is None and fb["datasets"][0]["n_samples"] == 36
    engine.run(st)
    fb = client.get(f"{base}/final-report").json()
    d = fb["datasets"][0]
    assert d["omics_family"] == "proteomics" and d["n_features"] == 400 and d["design"]["subject"] == "column 'animal'"
    assert d["design"]["time_unit"] == "weeks" and d["processing_history"]["imputed"] == "no"
    assert fb["audit"]["manifest"]["run_id"] and fb["merge"]["ready_for_audit"]
    r = client.get(f"{base}/final-report.html")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    html = r.text
    assert "window.PRISM_REPORT.renderFinal" in html and "prefers-color-scheme" in html and "@media print" in html
    assert not re.search(r"<script[^>]+src=|<link[^>]+href=|@import|url\(http", html)
    assert "content-disposition" not in {k.lower() for k in client.get(f"{base}/final-report.html?download=0").headers}
    from prism import cli
    out = tmp_path / "final.html"
    assert cli.main(["report", "--session", st.sid, "--out", str(out)]) == 0
    assert out.read_text().startswith("<!doctype html>")
    assert cli.main(["report", "--session", st.sid, "--format", "json"]) == 0
    assert json.loads((st.dir / "final_report.json").read_text())["session"]["session_id"] == st.sid
