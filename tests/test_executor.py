"""Data states and the executor (v4 sections 2 and 11): invariants, approval, preconditions, replay."""
import numpy as np
import pytest

import prism
from prism import executor as ex
from prism.executor import state as st
from test_audit import finished

N = 12
APPROVAL = {"user": "tester", "at": "2026-11-01T10:00:00", "plan_entry": "E1"}


def cells(M):
    return [["" if np.isnan(v) else f"{v:.6g}" for v in row] for row in M]


def store_for(tmp_path, X=None, **kw):
    rng = np.random.default_rng(0)
    if X is None:
        X = rng.lognormal(8, 1.2, (40, 1)) * rng.lognormal(0, 0.3, (40, N + 1))
        X[3, 2] = np.nan
    s = finished(tmp_path, cells(X), **kw)
    root = st.from_schema(s, "linear")
    return s, root, st.StateStore(root, tmp_path / "work")


def test_nothing_runs_without_approval(tmp_path):
    _, _, store = store_for(tmp_path)
    with pytest.raises(ex.ExecutorError, match="approval"):
        ex.apply(store, "log_transform", {}, approval={})


def test_log_then_median_parent_unchanged_and_nan_kept(tmp_path):
    s, root, store = store_for(tmp_path)
    root_files = store.file_hashes(store.root_id)
    m1 = ex.apply(store, "log_transform", {"offset_rule": "half_min"}, APPROVAL)
    assert store.data().scale == "log2" and m1["op_record"]["nan_after"] == m1["op_record"]["nan_before"]
    m2 = ex.apply(store, "normalize_median", {}, dict(APPROVAL, plan_entry="E2"))
    assert store.file_hashes(store.root_id) == root_files                    # S0 never modified
    assert store.file_hashes(m1["state_id"]) == store.file_hashes(m1["state_id"])
    X0, X2 = root.block("A1B1").X, store.data().block("A1B1").X
    assert (np.isnan(X0) == np.isnan(X2)).all()
    med = np.nanmedian(X2, axis=0)
    assert np.allclose(med, med[0])                                          # sample medians aligned
    assert len(store.chain()) == 2


def test_preconditions_refuse(tmp_path):
    _, _, store = store_for(tmp_path)
    with pytest.raises(ex.ExecutorError, match="log2 scale"):
        ex.apply(store, "normalize_median", {}, APPROVAL)                    # still linear
    ex.apply(store, "log_transform", {}, APPROVAL)
    with pytest.raises(ex.ExecutorError, match="double transform"):
        ex.apply(store, "log_transform", {}, dict(APPROVAL, plan_entry="E2"))
    batch = {f"S{j}": ("a" if j <= 6 else "b") for j in range(1, N + 1)}
    with pytest.raises(ex.ExecutorError, match="no missing values"):
        ex.apply(store, "batch_combat", {"batch": batch}, dict(APPROVAL, plan_entry="E3"))


def test_remove_features_changes_only_shape(tmp_path):
    _, root, store = store_for(tmp_path)
    keep = root.block("A1B1").feature_keys[2:]
    ex.apply(store, "remove_features", {"ids": root.block("A1B1").feature_keys[:2], "reason": "G3 test"}, APPROVAL)
    b = store.data().block("A1B1")
    assert b.feature_keys == keep and np.allclose(b.X, root.block("A1B1").X[2:], equal_nan=True)
    assert len(store.data().features) == len(root.features) - 2


def test_recode_zero_is_the_only_nan_exception(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.lognormal(8, 1, (20, N + 1))
    X[0, :3] = 0
    _, _, store = store_for(tmp_path, X)
    with pytest.raises(ex.ExecutorError, match="answered"):
        ex.apply(store, "recode_zero_as_missing", {"answer": "no"}, APPROVAL)
    m = ex.apply(store, "recode_zero_as_missing", {"answer": "yes"}, APPROVAL)
    assert m["op_record"]["nan_after"] == m["op_record"]["nan_before"] + 3


def test_batch_center_and_replay(tmp_path):
    _, root, store = store_for(tmp_path)
    batch = {f"S{j}": ("a" if j <= 6 else "b") for j in range(1, N + 1)}
    ex.apply(store, "log_transform", {}, APPROVAL)
    ex.apply(store, "batch_center", {"batch": batch, "stat": "median"}, dict(APPROVAL, plan_entry="E2"))
    r = ex.replay(store, root)
    assert r["identical"] and len(r["steps"]) == 2


def test_no_imputation_op():
    assert not any("imput" in n for n in ex.op_names())
    with pytest.raises(ex.ExecutorError, match="no imputation"):
        ex.register(ex.OpSpec("impute_knn", lambda *a: None, ("S6",), {}))


def test_audit_on_a_state(tmp_path):
    s, root, store = store_for(tmp_path)
    ex.apply(store, "log_transform", {}, APPROVAL)
    a = prism.Audit(s, state=store.data(), state_id=store.head, checkpoint="C1", root=False)
    a.integrity()
    assert a.scale().measures["A1B1"]["class"] != "continuous_linear"
    assert a.processing_state().status == "not_applicable"
    assert a.diagnostic("A1B1")[1]["transform"] == "none"
