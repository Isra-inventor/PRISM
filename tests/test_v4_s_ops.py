"""v4 stage 9: G6 QC drift, S1/S2 ops through the backend, and the D2 joint node."""

import os
import subprocess
import sys

import numpy as np
import pytest

from prism import store, ports
from prism.audit import overrides
from prism.executor import ExecutorError, apply, replay
from prism.executor import state as S
from prism.plan.joint import resolve_d2
import audit_fixtures as af

needs_port = pytest.mark.skipif(not ports.enabled(), reason="the proBatch port is disabled")


def appr(e):
    return {"user": "t", "at": "2026-10-21T00:00:00Z", "plan_entry": e}


def make(tmp_path, name, X, samples, cols=None, kinds=None, is_study=None):
    st = store.create(name)
    af.add(st, tmp_path / name, X=X, samples=samples, columns=cols or {}, kinds=kinds or {}, is_study=is_study)
    overrides.add(st, {"kind": "processing_state", "item": "log_transformed", "value": "yes"})
    return st


def drift_matrix(seed=0, p=60, n=40, nan=True):
    rng = np.random.default_rng(seed)
    order = np.arange(1, n + 1, dtype=float)
    batch = np.array(["b1"] * (n // 2) + ["b2"] * (n // 2))
    L = rng.normal(16, 1.5, size=(p, 1)) + rng.normal(0, 0.1, size=(p, n))
    drift = np.where(batch == "b1", 0.05 * order, 0.04 * (order - n / 2))
    L[:40] += drift[None, :]                     # features 40+ have no drift
    L[:, batch == "b2"] += 0.7
    if nan:
        L[5, [3, 30]] = np.nan
    return L, order, batch


def slope(order, med):
    return abs(np.polyfit(order, med, 1)[0])


@needs_port
def test_drift_sample_loess_reduces_slope_keeps_nan(isolated, tmp_path):
    L, order, batch = drift_matrix()
    samples = [f"s{j:02d}" for j in range(len(order))]
    st = make(tmp_path, "drift", L, samples)
    m = apply(st, "D1", "drift_sample_loess", {"batch": dict(zip(samples, batch)), "run_order": dict(zip(samples, order)),
                                               "span": 0.75}, appr("s2a"), step="S2")
    d, _ = S.load_data(st, "D1", m["state_id"])
    Y = d.blocks[0].X
    assert (np.isnan(Y) == np.isnan(L)).all()
    for b in ("b1", "b2"):
        ix = np.where(batch == b)[0]
        assert slope(order[ix], np.nanmedian(Y[:40, ix], 0)) < 0.2 * slope(order[ix], np.nanmedian(L[:40, ix], 0))
    # unaffected features (noise SD 0.1) stay within tolerance: LOESS follows some noise, nothing more
    dev = np.abs(Y[40:] - L[40:])
    assert np.nanmean(dev) < 0.05 and np.nanmax(dev) < 0.3
    rec = m["op_record"]
    assert rec["details"]["backend"] == "probatch_port" and rec["details"]["source"]["source_verified"] is True
    assert m["checkpoint"] == "C2"
    m2 = apply(st, "D1", "batch_center", {"batch": dict(zip(samples, batch))}, appr("s2b"), step="S2")
    d2, _ = S.load_data(st, "D1", m2["state_id"])
    assert np.allclose(np.nanmedian(d2.blocks[0].X[:, batch == "b1"], 1), np.nanmedian(d2.blocks[0].X[:, batch == "b2"], 1))
    assert replay(st, "D1")["identical"]


@needs_port
def test_combat_refuses_incomplete_coverage(isolated, tmp_path):
    L, order, batch = drift_matrix()
    samples = [f"s{j:02d}" for j in range(len(order))]
    st = make(tmp_path, "combat", L, samples)
    with pytest.raises(ExecutorError, match="no missing values in the features ComBat corrects"):
        apply(st, "D1", "batch_combat", {"batch": dict(zip(samples, batch))}, appr("s2c"), step="S2")
    d2 = resolve_d2(True, "batch_combat", 0.98)
    assert d2["status"] == "Do not automate" and "batch_center" in d2["alternatives"]
    L2, _, _ = drift_matrix(nan=False)
    st2 = make(tmp_path, "combat2", L2, samples)
    m = apply(st2, "D1", "batch_combat", {"batch": dict(zip(samples, batch))}, appr("s2d"), step="S2")
    d, _ = S.load_data(st2, "D1", m["state_id"])
    X = d.blocks[0].X
    assert abs(X[:, batch == "b1"].mean() - X[:, batch == "b2"].mean()) < 0.05


def test_qc_drift_removes_drift_and_needs_five_qc(isolated, tmp_path):
    rng = np.random.default_rng(1)
    n, p = 40, 50
    order = np.arange(1, n + 1, dtype=float)
    qc_idx = [0, 7, 14, 21, 28, 35, 39]
    L = rng.normal(14, 1, size=(p, 1)) + rng.normal(0, 0.05, size=(p, n)) + 0.03 * order[None, :]
    samples = [("QC%02d" % j) if j in qc_idx else f"s{j:02d}" for j in range(n)]
    st = make(tmp_path, "qc", L, samples, is_study=[j not in qc_idx for j in range(n)])
    qcs = [samples[j] for j in qc_idx]
    m = apply(st, "D1", "drift_qc_loess", {"qc_samples": qcs, "run_order": dict(zip(samples, order))}, appr("g6"))
    d, _ = S.load_data(st, "D1", m["state_id"])
    assert slope(order, np.median(d.blocks[0].X, 0)) < 0.1 * slope(order, np.median(L, 0))
    assert m["op_record"]["details"]["evidence_level"] == "unverified" and m["checkpoint"] == "C1"
    with pytest.raises(ExecutorError, match="at least 5 QC samples"):
        apply(st, "D1", "drift_qc_loess", {"qc_samples": qcs[:4], "run_order": dict(zip(samples, order))}, appr("g6b"),
              parent=S.root_id(st, "D1"))


def test_normalize_quantile_op(isolated, tmp_path):
    L, order, batch = drift_matrix(nan=False, p=20, n=10)
    samples = [f"s{j}" for j in range(10)]
    st = make(tmp_path, "quant", L, samples)
    m = apply(st, "D1", "normalize_quantile", {}, appr("s1q"))
    X = S.load_data(st, "D1", m["state_id"])[0].blocks[0].X
    assert np.allclose(np.sort(X, 0), np.sort(X, 0)[:, [0]])


def test_s2_ops_refuse_with_ports_disabled(tmp_path):
    code = ("import os; os.environ['PRISM_DISABLE_PORTS']='1'\n"
            "from prism.executor.ops import _backend_or_refuse\n"
            "from prism.executor import ExecutorError\n"
            "try:\n    _backend_or_refuse()\nexcept ExecutorError as e:\n    print('refused:', e)\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert "refused: This op needs the proBatch port" in r.stdout, r.stderr


def test_d2_cases():
    assert resolve_d2(False, "batch_combat")["case"] == "D2.no_missing"
    r = resolve_d2(True, "drift_sample_loess")
    assert r["case"] == "D2.tolerant_first" and "hand_off_to_method_selection" in r["order"]
