"""Long requests run as background jobs; errors always come back as JSON."""

import time

from test_v4_plan import batch_session


def wait(client, v, limit=120):
    t = time.time()
    while v["status"] == "running" and time.time() - t < limit:
        time.sleep(0.2)
        v = client.get(f"/api/jobs/{v['job_id']}").json()
    return v


def test_audit_and_plan_jobs(client, isolated, tmp_path):
    st = batch_session(tmp_path, "jobs")
    base = f"/api/sessions/{st.sid}"
    v = client.post(f"{base}/audit/start", json={}).json()
    assert v["status"] in ("running", "done") and v["job_id"].startswith("j")
    v = wait(client, v)
    assert v["status"] == "done" and v["result"]["run_id"] and v["progress"] == "finished"
    assert client.post(f"{base}/intake", json={"contrast_variable": "group"}).status_code == 200
    v = wait(client, client.post(f"{base}/plan/start").json())
    g5 = next(e for e in v["result"]["entries"] if e.get("group") == "G5")
    v = wait(client, client.post(f"{base}/plan/{g5['entry_id']}/approve/start", json={"candidate": 0}).json())
    assert v["status"] == "done" and v["result"]["state"]["checkpoint"] == "C1"
    v = wait(client, client.post(f"{base}/plan/D1:A1:G6/approve/start", json={}).json())
    assert v["status"] == "error" and "cannot be approved" in v["error"]          # a refusal, in words
    v = wait(client, client.post(f"{base}/audit/start", json={"factors": ["A42"]}).json())
    assert v["status"] == "error" and "Unknown audit" in v["error"]
    assert client.get("/api/jobs/jnope").status_code == 404


def test_unexpected_errors_are_json(client, isolated, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from backend import main
    from prism.audit import engine
    st = batch_session(tmp_path, "boom")

    def boom(*a, **k):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(engine, "list_runs", boom)
    c = TestClient(main.app, raise_server_exceptions=False)
    r = c.get(f"/api/sessions/{st.sid}/audit")
    assert r.status_code == 500 and "synthetic failure" in r.json()["detail"]


def test_server_answers_while_a_job_runs_and_a_killed_job_is_reported(client, isolated, tmp_path):
    import os
    import signal
    from backend import jobs
    st = batch_session(tmp_path, "kill")
    v = client.post(f"/api/sessions/{st.sid}/audit/start", json={}).json()
    j = jobs.get(v["job_id"])
    assert client.get("/api/health").json()["status"] == "ok"
    t = time.time()
    while j.progress == "starting" and j.status == "running" and time.time() - t < 60:
        time.sleep(0.05)                                 # the child is working before it is killed
    assert j.status == "running"
    os.kill(j.pid, signal.SIGKILL)                       # what an out-of-memory kill looks like
    v = wait(client, v)
    assert v["status"] == "error" and "signal 9" in v["error"] and "memory" in v["error"]
