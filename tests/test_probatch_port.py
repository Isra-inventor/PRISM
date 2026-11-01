"""The proBatch port (v4 section 7). Unit tests (including one per confirmed proBatch bug), license
isolation. The oracle comparisons with R from the v4 build need R; they are not run in this package yet."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest


from prism import ports
from prism.batch import combat as combat_mod, loess, normalize

if not ports.enabled():
    pytest.skip("the proBatch port is disabled (PRISM_DISABLE_PORTS=1)", allow_module_level=True)

from prism.ports.probatch import correct, diagnostics, drift, qc, transform  # noqa: E402

ORACLE_LOG = {}


def drift_data(seed=0, p=30, n=24, nan=True):
    rng = np.random.default_rng(seed)
    order = np.arange(1, n + 1, dtype=float)
    batch = np.array(["b1"] * (n // 2) + ["b2"] * (n // 2))
    base = rng.normal(15, 1, size=(p, 1))
    trend = np.where(batch == "b1", 0.08 * order, -0.06 * (order - n / 2)) + 0.002 * (order - n / 2) ** 2
    X = base + trend[None, :] + rng.normal(0, 0.1, size=(p, n))
    X[:, batch == "b2"] += 0.8
    if nan:
        X[3, [2, 15]] = np.nan
        X[7, 1:20:2] = np.nan          # too few consecutive values: left unchanged
    return X, batch, order


# ------------------------------------------------------------------ confirmed proBatch bugs (one test each)


def test_bug_median_centering_never_returns_a_column_name():
    """proBatch normalize_sample_medians_df: ifelse(flag, NA, measure_col) put the string 'Intensity'
    into flagged cells. The port excludes flagged cells from the statistic and keeps numbers."""
    X = np.array([[1.0, 2.0, 3.0, 9.0], [2.0, 3.0, 4.0, 1.0], [5.0, 4.0, 3.0, 2.0]])
    flags = np.zeros_like(X, bool)
    flags[0, 3] = True
    out, _ = correct.center_feature_batch(X, ["a", "a", "b", "b"], "median", exclude=flags)
    assert out.dtype == float and np.isfinite(out).all()
    assert out[0, 3] == pytest.approx(9.0 + (np.median([1, 2, 3]) - 3.0))   # median of b is 3 (9 excluded)


def test_bug_cv_filter_filters():
    """proBatch: if (any(df_long$n) > 2) never filters. The port drops features with < 3 values."""
    X = np.log2(np.array([[10.0, 12.0, np.nan, np.nan], [10.0, 11.0, 12.0, 13.0]]) + 1)
    cv = qc.feature_cv(X, offset=1)
    assert np.isnan(cv["total"][0]) and np.isfinite(cv["total"][1])


def test_bug_no_default_log_offset():
    """proBatch applies offset 1 by default everywhere; the port needs an explicit offset."""
    with pytest.raises(ValueError):
        transform.log_transform(np.ones((2, 2)))
    assert transform.log_transform(np.array([[1.0]]), offset=1)[0, 0] == pytest.approx(1.0)


def test_bug_unlog_is_the_inverse_of_log():
    """unlog_df computed base^(x - offset), unlog_dm base^x - offset; only the second inverts log(x + offset)."""
    X = np.array([[3.0, 100.0], [0.5, 7.0]])
    assert np.allclose(transform.unlog(transform.log_transform(X, 2, 1.0), 2, 1.0), X)
    buggy = 2.0 ** (transform.log_transform(X, 2, 1.0) - 1.0)
    assert not np.allclose(buggy, X)


def test_bug_correct_batch_effects_returns_a_matrix():
    """correct_batch_effects_dm hard-coded keep_all = FALSE (R's switch then returns NULL)."""
    X, batch, order = drift_data(nan=False)
    out, info = correct.correct_batch_effects(X, batch, order, continuous="loess_regression",
                                              discrete="MedianCentering")
    assert isinstance(out, np.ndarray) and out.shape == X.shape and info["trend"]["fits"] > 0


def test_bug_rle_uses_the_fitted_values():
    """rle_func hard-coded df$Intensity; the port counts consecutive measured values of the fitted series."""
    y = np.array([1, 2, np.nan, 4, 5, 6, np.nan, 8], float)
    assert drift.max_consecutive_measured(y) == 3
    fit, why = drift.fit_nonlinear(np.arange(8.0), y, min_measurements=4)
    assert fit is None and "consecutive" in why


# ------------------------------------------------------------------ behaviour


def test_drift_reduces_run_order_slope_and_keeps_nan():
    X, batch, order = drift_data()
    out, info = drift.adjust_batch_trend(X, batch, order, span=0.75, min_measurements=8)
    assert (np.isnan(out) == np.isnan(X)).all()
    assert np.allclose(out[7], X[7], equal_nan=True) and info["left_unchanged"]   # row 7 left unchanged
    for b in ("b1", "b2"):
        ix = np.where(batch == b)[0]
        before = abs(np.polyfit(order[ix], np.nanmedian(X[:, ix], 0), 1)[0])
        after = abs(np.polyfit(order[ix], np.nanmedian(out[:, ix], 0), 1)[0])
        assert after < 0.2 * before


def test_center_and_combat():
    rng = np.random.default_rng(4)
    batch = np.array(["b1"] * 12 + ["b2"] * 12)
    X = rng.normal(15, 1, size=(30, 1)) + rng.normal(0, 0.2, size=(30, 24))
    X[:, batch == "b2"] += 0.8
    c, _ = correct.center_feature_batch(X, batch)
    assert np.allclose(np.median(c[:, batch == "b1"], 1), np.median(c[:, batch == "b2"], 1))
    cb, info = combat_mod.combat(X, batch)
    d0 = np.abs(X[:, batch == "b1"].mean(1) - X[:, batch == "b2"].mean(1)).mean()
    d1 = np.abs(cb[:, batch == "b1"].mean(1) - cb[:, batch == "b2"].mean(1)).mean()
    assert d1 < 0.1 * d0 and info["model"] == "~1"
    Xn = X.copy()
    Xn[0, 0] = np.nan
    with pytest.raises(combat_mod.ComBatError, match="complete"):
        combat_mod.combat(Xn, batch)


def test_quantile_no_nan_equals_standard():
    X = np.array([[5.0, 4.0, 3.0], [2.0, 1.0, 4.0], [3.0, 4.0, 6.0], [4.0, 2.0, 8.0]])
    out, info = normalize.quantile(X)
    # hand-computed: reference = mean of sorted columns = [2, 3, 4.6667, 5.6667]
    assert np.allclose(out[:, 0], [5.6667, 2.0, 3.0, 4.6667], atol=1e-4)
    assert np.allclose(out[:, 2], [2.0, 3.0, 4.6667, 5.6667], atol=1e-4)
    Xn = X.copy()
    Xn[1, 1] = np.nan
    o2, info2 = normalize.quantile(Xn)
    assert np.isnan(o2[1, 1]) and np.isfinite(o2).sum() == 11 and "NA-aware" in info2["form"]


def test_pvca_finds_the_batch():
    X, batch, order = drift_data(nan=False, p=200, n=24)
    group = np.array(["g1", "g2"] * 12)
    X[:, group == "g2"] += 0.05
    res = diagnostics.pvca(X, {"batch": batch.tolist(), "group": group.tolist()}, ["batch", "group"])
    assert abs(sum(res["weights"].values()) - 1) < 1e-9 and res["pc_n"] >= 3
    assert res["weights"]["batch"] > res["weights"]["group"]


def test_core_runs_with_ports_disabled(tmp_path):
    code = ("import os; os.environ['PRISM_DISABLE_PORTS']='1'\n"
            "from prism.batch import backend\n"
            "b = backend.get()\n"
            "assert b.name == 'none'\n"
            "try:\n    b.center(None, None)\nexcept backend.BackendUnavailable as e:\n    print('refused:', e)\n"
            "import sys; assert 'prism.ports.probatch' not in sys.modules\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert r.returncode == 0, r.stderr
    assert "refused: per-batch centering needs the proBatch port" in r.stdout


def test_license_files_exist():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lic = open(os.path.join(root, "LICENSING.md")).read()
    assert "undecided" in lic and "GPL-3" in lic and "nothing is to be published" in lic.lower()
    notice = open(os.path.join(root, "prism", "ports", "probatch", "NOTICE.md")).read()
    assert "Jelena Cuklina" in notice and "Yuliya Burankova" in notice and "GPL-3" in notice
    for f in ("transform.py", "drift.py", "correct.py", "diagnostics.py", "qc.py"):
        head = open(os.path.join(root, "prism", "ports", "probatch", f)).read(600)
        assert "Derived from proBatch" in head and "GPL-3" in head and "Adapted to Python by PRISM" in head


# ------------------------------------------------------------------ oracles against R

