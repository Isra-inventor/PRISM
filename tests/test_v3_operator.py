"""v3 stage 8: the thin AI operator (mock model). Tools run the same deterministic engine;
parameters change only when the user names them; overrides are proposals; numbers are checked;
every call is in the ledger; the model never sees matrix rows."""

import json

import numpy as np
import pytest

from prism import store
from prism.audit import engine, operator
import audit_fixtures as af


@pytest.fixture
def st(isolated, tmp_path):
    s = store.create("operator")
    X, samples, cols, kinds = af.planted(shift=True, outliers=True)
    af.add(s, tmp_path / "planted", X=X, samples=samples, columns=cols, kinds=kinds, design=af.SUBJ_DESIGN)
    s.X = X
    return s


def chat(client, st, message, run_id=None):
    r = client.post(f"/api/sessions/{st.sid}/audit/chat", json={"message": message, "run_id": run_id})
    assert r.status_code == 200, r.text
    return r.json()


def test_run_explain_plot_and_ledger(client, st):
    r = chat(client, st, "Please run A4 and A7 with permutations = 199")
    (t,) = r["tool_results"]
    assert t["tool"] == "run_audit" and t["ok"]
    man = engine.show(st, t["result"]["run_id"], full=True)["manifest"]
    assert man["params"]["permutations"] == 199 and man["who"] == "ai" and man["scope"]["audits"] == ["A4", "A7"]
    run_led = [json.loads(l) for l in (st.dir / "audit" / man["run_id"] / "ledger.jsonl").read_text().splitlines()]
    assert all(x["who"] == "ai" for x in run_led)
    # explain: the reply uses the finding's own sentences; its numbers are all found
    r = chat(client, st, "explain A4")
    assert r["tool_results"][0]["tool"] == "show_finding" and r["tool_results"][0]["ok"]
    assert "floor" in r["reply"] and r["unverified_numbers"] == []
    # a number the findings do not contain is flagged
    r = chat(client, st, "explain A4 with a made-up number")
    assert "87.123%" in r["unverified_numbers"]
    # plot view: validated against the plot's variables
    r = chat(client, st, "colour the PCA by run_day")
    (t,) = r["tool_results"]
    assert t["ok"] and t["result"]["view"]["color_by"] == "run_day" and t["result"]["view"]["pcs"] == [1, 2]
    r = chat(client, st, "colour the PCA by nonsense")
    assert not r["tool_results"][0]["ok"] and "not a variable" in r["tool_results"][0]["error"]
    # rerun keeps the run's own parameters
    r = chat(client, st, "rerun please")
    new = engine.show(st, r["tool_results"][0]["result"]["run_id"], full=True)["manifest"]
    assert new["params"]["permutations"] == 199 and new["scope"]["audits"] == ["A4", "A7"]
    led = [json.loads(l) for l in (st.dir / "audit" / "ledger.jsonl").read_text().splitlines()]
    tools = [x["tool"] for x in led]
    assert tools.count("ai.chat") == 6 and "ai.run_audit" in tools and "ai.plot" in tools and "ai.rerun" in tools
    assert all(x["who"] == "ai" and x["output_sha256"] for x in led)


def test_no_silent_parameter_change(client, st):
    r = chat(client, st, "run A4 and invent something")
    (t,) = r["tool_results"]
    assert not t["ok"] and "never changes a parameter silently" in t["error"] and "seed" in t["error"]
    assert engine.list_runs(st) == []


def test_override_is_a_proposal(client, st):
    r = chat(client, st, "mark A00_0 as qc")
    (t,) = r["tool_results"]
    assert t["ok"] and t["result"]["applied"] is False
    assert t["result"]["proposal"] == {"kind": "sample_role", "sample": "A00_0", "role": "qc", "reason": "named in chat"}
    from prism.audit import overrides
    assert overrides.load(st) == []                          # nothing applied by the AI
    r = client.post(f"/api/sessions/{st.sid}/overrides", json=t["result"]["proposal"])
    assert r.status_code == 200 and overrides.load(st)[0]["by"] == "user"


def test_request_addition_writes_a_file(client, st):
    r = chat(client, st, "can you add a wavelet decomposition of the injection sequence?")
    (t,) = r["tool_results"]
    assert t["ok"] and t["result"]["request"] == "requests/001.md"
    text = (st.dir / "requests" / "001.md").read_text()
    for h in ("## Inputs", "## Outputs", "## Algorithm", "## Tests", "## Citations"):
        assert h in text
    assert "Nothing was run" in text


def test_unknown_tool_and_bad_args(st):
    with pytest.raises(operator.ToolRejected, match="request_addition"):
        operator.execute(st, {"tool": "exec_python", "args_json": "{}"}, "", None)
    with pytest.raises(operator.ToolRejected, match="JSON object"):
        operator.execute(st, {"tool": "plot", "args_json": "[1]"}, "", None)


def test_the_model_never_sees_matrix_rows(st):
    from backend import ai
    seen = {}
    real = ai.audit_chat

    def spy(payload, sha, log=None, mock_fn=None):
        seen["payload"] = json.dumps(payload)
        return real(payload, sha, log, mock_fn)
    engine.run(st)
    ai.audit_chat, saved = spy, ai.audit_chat
    try:
        operator.chat(st, "explain A11")
    finally:
        ai.audit_chat = saved
    text = seen["payload"]
    assert "plot_data" not in text
    rng = np.random.default_rng(0)
    cells = st.X[np.isfinite(st.X)]
    sample = rng.choice(cells, 200, replace=False)
    leaked = [v for v in sample if repr(float(v)) in text or f"{v:.10g}" in text]
    # a few cells can appear as finding numbers (a feature's max, a flagged cell); rows never do
    assert len(leaked) <= 2
    # no value-column list either
    assert "A00_0" not in json.dumps(json.loads(text)["context"]["schemas"])


def test_unverified_numbers_tolerance():
    ctx = {"findings": [{"measures": {"share": 0.3381601363, "n": 397}, "indicators": [{"text": "33.8% of features"}]}]}
    assert operator.unverified_numbers("397 features, 33.8% (0.338)", ctx, "") == []
    assert operator.unverified_numbers("398 features", ctx, "") == ["398"]
    assert operator.unverified_numbers("A4 and PC1", ctx, "") == []
