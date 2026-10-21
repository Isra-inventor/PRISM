"""v4 stage 4: diagnostic_matrix policies, the fallback, the sensitivity block, and A7/A8/A11 retrofits."""

import numpy as np
import pytest

from prism import store
from prism.audit import engine
from prism.audit import diagnostic_matrix as dmx
from prism.audit.params import defaults
import audit_fixtures as af


def test_policies_never_fill():
    rng = np.random.default_rng(0)
    Y = rng.normal(10, 1, size=(30, 8))
    Y[0, 1] = np.nan
    Y[1, :6] = np.nan                 # sample-heavy pattern
    with pytest.raises(ValueError):
        dmx.diagnostic_matrix(Y, "all_observed")
    d = dmx.diagnostic_matrix(Y, "complete_features", 100)
    assert d.n_features == 28 and len(d.cols) == 8 and not np.isnan(d.Y).any()
    assert np.array_equal(d.Y, Y[d.rows][:, d.cols])          # the values are the input values
    floor = np.zeros(30, bool)
    floor[5:10] = True
    e = dmx.diagnostic_matrix(Y, "exclude_floor", 100, floor_features=floor)
    assert e.n_features == 23
    S = np.zeros_like(Y, bool)
    S[20, 3] = True
    c = dmx.diagnostic_matrix(Y, "floor_censored", 100, floor_cell_mask=S)
    assert c.n_features == 27 and not np.isnan(Y[20, 3])     # the input is untouched
    R, N = dmx.pairwise_corr(Y)
    assert N[0, 1] == 28 and N[0, 2] == 29 and np.isfinite(R[0, 1])
    rep = d.report()
    assert rep["bias_warning"] == dmx.BIAS_WARNING and rep["share"] == pytest.approx(28 / 30)


def test_fallback_to_lower_q_is_reported():
    rng = np.random.default_rng(1)
    Y = rng.normal(10, 1, size=(100, 10))
    Y[:95, 9] = np.nan                # one bad sample makes 95 of 100 features incomplete
    P = defaults()
    d = dmx.default_matrix(Y, P)
    assert d.q == 75 and d.n_features == 100 and d.samples_dropped == 1
    assert [t["q"] for t in d.report()["fallback"]] == [100, 75]


def _session(tmp_path, X, samples, cols, kinds, name):
    s = store.create(name)
    af.add(s, tmp_path / name, X=X, samples=samples, columns=cols, kinds=kinds)
    return s


def _batch_data(sparse_shift, seed=3):
    """36 samples, batch b1/b2. Three samples miss 60% of the features (the sparse ones)."""
    rng = np.random.default_rng(seed)
    n, p = 36, 200
    batch = ["b1" if j % 2 == 0 else "b2" for j in range(n)]
    b2 = np.array([b == "b2" for b in batch])
    L = rng.normal(12, 1, size=(p, 1)) + rng.normal(0, 0.3, size=(p, n))
    sparse = np.arange(80, 200)
    if sparse_shift:
        L[np.ix_(sparse, np.where(b2)[0])] += 2.0
    else:
        L[:, b2] += 2.0
    X = 2.0 ** L
    for j in (3, 10, 21):             # the sparse features are missing in three samples
        X[sparse, j] = np.nan
    samples = [f"s{j:02d}" for j in range(n)]
    return X, samples, {"batch": batch}, {"batch": "batch"}


def test_shift_on_complete_features_gives_one_conclusion(isolated, tmp_path):
    X, samples, cols, kinds = _batch_data(sparse_shift=False)
    st = _session(tmp_path, X, samples, cols, kinds, "complete")
    man = engine.run(st, factors=["A7", "A11"])
    a7 = engine.finding(st, man["run_id"], next(i["file"] for i in man["findings"] if i["audit_id"] == "A7"))
    sens = a7["measures"]["sensitivity"]
    assert {v["variant"] for v in sens["variants"]} >= {"base", "q75", "q50"}
    assert sens["same_conclusion"] and not sens["depends"]
    assert all(v["conclusion"]["strongest_batch_variable"] == "batch" for v in sens["variants"] if v["computable"])
    assert "diagnostic_conclusion_depends_on_policy" not in [i["code"] for i in a7["indicators"]]
    diag = a7["measures"]["diagnostic"]
    assert diag["policy"] == "complete_features" and diag["q"] == 100 and diag["n_features_used"] == 80
    assert diag["bias_warning"].startswith("the retained features are biased")


def test_shift_only_in_sparse_features_depends_on_policy(isolated, tmp_path):
    X, samples, cols, kinds = _batch_data(sparse_shift=True)
    st = _session(tmp_path, X, samples, cols, kinds, "sparse")
    man = engine.run(st, factors=["A7"])
    a7 = engine.finding(st, man["run_id"], man["findings"][0]["file"])
    sens = a7["measures"]["sensitivity"]
    assert sens["depends"]
    by = {v["variant"]: v for v in sens["variants"]}
    assert by["base"]["conclusion"]["batch_variables_p_le_0_05"] == []
    assert by["q75"]["conclusion"]["batch_variables_p_le_0_05"] == ["batch"]
    assert "diagnostic_conclusion_depends_on_policy" in [i["code"] for i in a7["indicators"]]


def test_floor_variants_and_a8_pairwise(isolated, tmp_path):
    st = store.create("floors")
    X, samples, cols, kinds = af.planted(shift=True, outliers=True)
    af.add(st, tmp_path / "planted", X=X, samples=samples, columns=cols, kinds=kinds, design=af.SUBJ_DESIGN)
    man = engine.run(st, factors=["A7", "A8", "A11"])
    get = {i["audit_id"]: engine.finding(st, man["run_id"], i["file"]) for i in man["findings"]}
    for aid in ("A7", "A11"):
        names = {v["variant"] for v in get[aid]["measures"]["sensitivity"]["variants"]}
        assert {"exclude_floor", "floor_censored"} <= names
        assert get[aid]["measures"]["sensitivity"]["n_floor_tied_features"] == 20
    sc = get["A8"]["measures"]["sample_correlation"]
    assert sc["policy"] == "pairwise" and sc["shared_features_per_pair"]["min"] < 400
    assert "A05_8" in get["A11"]["measures"]["samples"]["flagged"]
