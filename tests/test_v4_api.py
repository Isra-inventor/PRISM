"""v4 stage 12: API endpoints and report additions."""

import json
import re

import pytest

from prism import ports
from prism.audit import report
from test_v4_plan import batch_session


def test_v4_api_flow(client, isolated, tmp_path):
    st = batch_session(tmp_path, "api")
    base = f"/api/sessions/{st.sid}"
    reg = client.get("/api/registry").json()
    assert len(reg["steps"]) == 17 and "global|action" in reg["grid"]
    r = client.post(f"{base}/intake", json={"contrast_variable": "nope"})
    assert r.status_code == 422
    r = client.post(f"{base}/intake", json={"contrast_variable": "group"})
    assert r.status_code == 200 and r.json()["intake"]["contrast_variable"] == "group"
    got = client.get(f"{base}/intake").json()
    assert "group" in got["columns"] and got["version"] == 1
    plan = client.get(f"{base}/plan").json()
    g5 = next(e for e in plan["entries"] if e.get("group") == "G5")
    assert g5["status"] == "Recommended"
    assert client.post(f"{base}/plan/D1:A1:G6/approve", json={}).status_code == 422
    r = client.post(f"{base}/plan/{g5['entry_id']}/approve", json={"candidate": 0, "user": "ana"})
    assert r.status_code == 200, r.text
    out = r.json()
    states = client.get(f"{base}/states").json()
    assert [s["label"] for s in states] == ["S0", "S1"] and states[1]["head"]
    b = client.get(f"{base}/states/{out['state']['state_id'][:10]}/audit/{out['run_id']}").json()
    assert b["manifest"]["run_id"] == out["run_id"]
    assert client.get(f"{base}/states/{states[0]['state_id'][:10]}/audit/{out['run_id']}").status_code == 422
    assert client.post(f"{base}/plan/D1:A1:G6/reject", json={"reason": "no QC"}).json()["decision"] == "rejected"
    assert [d["decision"] for d in client.get(f"{base}/plan/decisions").json()] == ["approved", "rejected"]
    r = client.post(f"{base}/overrides", json={"kind": "processing_state", "item": "normalized", "value": "no"})
    assert r.status_code == 200 and r.json()["override"]["item"] == "normalized"
    r = client.post(f"{base}/overrides", json={"kind": "control_feature", "feature": "f3"})
    assert r.status_code == 200
    r = client.post(f"{base}/audit/run", json={"state": out["state"]["state_id"]})
    assert r.status_code == 200 and r.json()["scope"]["states"]["D1"]["checkpoint"] == "C1"


def test_report_v4_sections(client, isolated, tmp_path):
    st = batch_session(tmp_path, "rep")
    from prism.plan import engine as pe, intake
    intake.set_intake(st, {"contrast_variable": "group"})
    pe.build(st)
    from prism.audit import engine
    man = engine.run(st)
    b = report.bundle(st, man["run_id"])
    v4 = b["v4"]
    assert v4["grid"] and v4["plan"]["entries"] and v4["states"]
    assert {f["audit_id"] for f in b["findings"]} >= {"A12", "A13"}
    html = report.final_html(st)
    assert "renderFinal" in html and not re.search(r"<script[^>]+src=|<link[^>]+href=", html)
    data = json.loads(re.search(r'<script id="prism-report-data" type="application/json">(.*?)</script>', html, re.S).group(1))
    assert data["v4"]["plan"]["entries"]
    for e in data["v4"]["plan"]["entries"]:
        for c in e.get("candidates", []):
            ids = (c.get("params") or {}).get("ids")
            assert not isinstance(ids, list) or len(ids) <= 12        # trimmed in the report
