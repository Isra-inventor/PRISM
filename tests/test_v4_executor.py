"""v4 stage 3: DataState, the executor, invariants, replay and checkpoints in the audit runner."""

import numpy as np
import pytest

from prism import cli, store
from prism.audit import engine, overrides
from prism.executor import ExecutorError, OPS, OpSpec, apply, register, replay, run_op
from prism.executor import state as S
from prism.util import sha256_file
import audit_fixtures as af

APPROVAL = {"user": "tester", "at": "2026-10-21T10:00:00+00:00", "plan_entry": "p1"}


def appr(entry):
    return dict(APPROVAL, plan_entry=entry)


@pytest.fixture
def st(isolated, tmp_path):
    s = store.create("exec")
    X, samples, cols, kinds = af.planted(shift=True)
    af.add(s, tmp_path / "planted", X=X, samples=samples, columns=cols, kinds=kinds, design=af.SUBJ_DESIGN,
           history={"imputed": {"answer": "no"}})
    return s


def hashes(folder):
    return {p.name: sha256_file(p) for p in sorted(folder.iterdir()) if p.is_file()}


def test_root_state_is_computed_without_writing(st):
    rid = S.root_id(st, "D1")
    assert not S.states_dir(st, "D1").exists()
    m = S.meta(st, "D1", rid)
    assert m["label"] == "S0" and m["checkpoint"] == "C0" and m["counts"]["n_features"] == 400
    man = engine.run(st, factors=["A4"])
    f = engine.finding(st, man["run_id"], man["findings"][0]["file"])
    assert f["state_id"] == rid and f["checkpoint"] == "C0"
    assert man["scope"]["states"]["D1"]["state_id"] == rid
    assert not S.states_dir(st, "D1").exists()           # an audit writes nothing outside audit/


def test_nothing_runs_without_approval(st):
    with pytest.raises(ExecutorError, match="approval"):
        apply(st, "D1", "remove_features", {"ids": ["f1"], "reason": "x"}, approval=None)
    with pytest.raises(ExecutorError, match="approval"):
        apply(st, "D1", "remove_features", {"ids": ["f1"], "reason": "x"}, approval={"user": "u"})


def test_remove_features_keeps_parents_and_nan_pattern(st):
    rid = S.root_id(st, "D1")
    m1 = apply(st, "D1", "remove_features", {"ids": ["f1", "f2"], "reason": "decoys"}, appr("g3a"), step="G3")
    root_dir = S.states_dir(st, "D1") / rid
    before_root = hashes(root_dir)
    before_out = hashes(st.output_dir("D1"))
    rec = m1["op_record"]
    assert rec["features_removed"] == 2 and rec["cells_changed"] == 0 and rec["approval"]["plan_entry"] == "g3a"
    assert m1["parent"] == rid and m1["checkpoint"] == "C0" and S.head(st, "D1") == m1["state_id"]
    d1, _ = S.load_data(st, "D1", m1["state_id"])
    assert d1.blocks[0].X.shape == (398, 36) and "f1" not in d1.blocks[0].feature_keys
    assert not any(r["feature_key"] == "f1" for r in d1.features["rows"])
    m2 = apply(st, "D1", "remove_samples", {"ids": ["A00_0"], "reason": "test"}, appr("s4a"), step="S4")
    assert hashes(root_dir) == before_root and hashes(st.output_dir("D1")) == before_out
    assert hashes(S.states_dir(st, "D1") / m1["state_id"])["value_matrix_A1_B1.csv"] == \
        m1["matrices"]["value_matrix_A1_B1.csv"]
    d2, _ = S.load_data(st, "D1", m2["state_id"])
    assert d2.blocks[0].X.shape == (398, 35) and m2["checkpoint"] == "C2"
    # NaN pattern of the remaining cells is identical
    X0 = d1.blocks[0].X[:, 1:]
    assert (np.isnan(X0) == np.isnan(d2.blocks[0].X)).all()


def test_preconditions_refuse_with_the_failing_one(st):
    with pytest.raises(ExecutorError, match="confirmed processing state"):
        apply(st, "D1", "log_transform", {}, appr("g5"))
    with pytest.raises(ExecutorError, match="every feature id exists"):
        apply(st, "D1", "remove_features", {"ids": ["nope"], "reason": "x"}, appr("g3"))
    with pytest.raises(ExecutorError, match="unknown parameter"):
        apply(st, "D1", "remove_features", {"ids": ["f1"], "reason": "x", "bogus": 1}, appr("g3"))
    with pytest.raises(ExecutorError, match="log2 scale"):
        apply(st, "D1", "normalize_median", {}, appr("s1"))


def test_log_then_median_checkpoints_and_replay(st):
    overrides.add(st, {"kind": "processing_state", "item": "log_transformed", "value": "no"})
    assert S.meta(st, "D1", S.root_id(st, "D1"))["scale"] == "linear"
    m1 = apply(st, "D1", "log_transform", {"offset_rule": "half_min"}, appr("g5"))
    assert m1["scale"] == "log2" and m1["checkpoint"] == "C1"
    with pytest.raises(ExecutorError, match="double transform"):
        apply(st, "D1", "log_transform", {}, appr("g5b"))
    m2 = apply(st, "D1", "normalize_median", {}, appr("s1"))
    d, _ = S.load_data(st, "D1", m2["state_id"])
    med = np.nanmedian(d.blocks[0].X, axis=0)
    assert np.allclose(med, med[0])
    d0, _ = S.load_data(st, "D1", S.root_id(st, "D1"))
    assert (np.isnan(d0.blocks[0].X) == np.isnan(d.blocks[0].X)).all()
    # the audit at that state: the C1 battery, Y = X (already log2)
    man = engine.run(st, state=m2["state_id"][:10])
    assert {i["audit_id"] for i in man["findings"]} <= {"A3", "A4", "A6", "A7", "A8", "A12"}
    assert man["scope"]["states"]["D1"]["checkpoint"] == "C1"
    f = engine.finding(st, man["run_id"], man["findings"][0]["file"])
    assert f["checkpoint"] == "C1" and f["state_id"] == m2["state_id"]
    a3 = next(engine.finding(st, man["run_id"], i["file"]) for i in man["findings"] if i["audit_id"] == "A3")
    assert a3["method"]["transform"]["transform"] == "none"
    # replay from S0 reproduces the final state id
    r = replay(st, "D1")
    assert r["identical"] and r["replayed"] == m2["state_id"] and len(r["steps"]) == 2
    # before / after
    engine.run(st, factors=["A4", "A7"])
    cmp = engine.compare_states(st, S.root_id(st, "D1"), m2["state_id"])
    assert cmp["state_diff"]["blocks"][0]["nan_pattern_identical"] and cmp["table"]


def test_invariant_violations_refuse(st, monkeypatch):
    def fill(data, p, ctx):
        X = data.blocks[0].X
        data.blocks[0].X = np.where(np.isnan(X), 0.0, X)
    monkeypatch.setitem(OPS, "fill_test", OpSpec("fill_test", fill, ("S1",), {}))
    with pytest.raises(ExecutorError, match="NaN pattern"):
        apply(st, "D1", "fill_test", {}, appr("x"))

    def outside(data, p, ctx):
        data.blocks[0].X = data.blocks[0].X * 2
        return {"_touched": {("A1", "B1"): {"f0"}}}
    monkeypatch.setitem(OPS, "outside_test", OpSpec("outside_test", outside, ("S1",), {}))
    with pytest.raises(ExecutorError, match="outside the features"):
        apply(st, "D1", "outside_test", {}, appr("y"))
    assert S.head(st, "D1") == S.root_id(st, "D1")       # nothing was written


def test_recode_zero_is_the_only_nan_exception(st, tmp_path):
    X = np.array([[1.0, 0.0, 3.0, 4.0], [0.0, 2.0, np.nan, 5.0], [1.5, 2.5, 3.5, 4.5]])
    s2 = store.create("zeros")
    af.add(s2, tmp_path / "z", X=X, samples=["a", "b", "c", "d"])
    m = apply(s2, "D1", "recode_zero_as_missing", {"answer": "yes"}, appr("g7"))
    d, _ = S.load_data(s2, "D1", m["state_id"])
    assert np.isnan(d.blocks[0].X).sum() == 3 and m["op_record"]["details"]["cells_recoded"] == 2
    with pytest.raises(ExecutorError, match="answered"):
        apply(s2, "D1", "recode_zero_as_missing", {"answer": "no"}, appr("g7b"), parent=S.root_id(s2, "D1"))


def test_no_imputation_op_exists():
    assert not [o for o in OPS if "imput" in o.lower() or "fill" in o.lower()]
    with pytest.raises(ExecutorError, match="no imputation"):
        register(OpSpec("impute_knn", lambda *a: None, ("S6",), {}))


def test_state_cli(st, capsys):
    apply(st, "D1", "remove_features", {"ids": ["f3"], "reason": "x"}, appr("g3"))
    assert cli.main(["state", "list", "--session", st.sid]) == 0
    out = capsys.readouterr().out
    assert "S0" in out and "S1" in out and "HEAD" in out
    h = S.head(st, "D1")
    assert cli.main(["state", "show", "--session", st.sid, h[:8]]) == 0
    assert cli.main(["state", "diff", "--session", st.sid, S.root_id(st, "D1")[:8], h[:8]]) == 0
    assert '"features_removed": 1' in capsys.readouterr().out
