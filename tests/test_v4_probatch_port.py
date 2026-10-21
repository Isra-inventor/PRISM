"""v4 stage 8: the proBatch port. Unit tests (including one per confirmed proBatch bug), license
isolation, and oracle comparisons with R (skipped when R, a package or PROBATCH_SRC is missing)."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

import r_oracle as ro
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


def _record(name, diff, tol):
    ORACLE_LOG[name] = {"max_abs_diff": float(diff), "tolerance": tol}
    print(f"ORACLE {name}: max abs diff {diff:.3g} (tolerance {tol})")


def test_oracle_log2_offset(tmp_path):
    ro.need()
    X = np.abs(np.random.default_rng(1).normal(100, 30, size=(10, 6)))
    ro.write_matrix(tmp_path / "x.csv", X)
    r = ro.run("x <- as.matrix(read.csv('x.csv', row.names=1)); write.csv(log2(x + 1), 'y.csv')", tmp_path)
    assert r.returncode == 0, r.stderr
    d = np.abs(ro.read_matrix(tmp_path / "y.csv") - transform.log_transform(X, 2, 1.0)).max()
    _record("log2(x + 1)", d, 1e-12)
    assert d < 1e-12


def test_oracle_median_centering_probatch(tmp_path):
    ro.need("dplyr", "tidyr", "purrr", probatch=True)
    X, batch, order = drift_data()
    ro.write_matrix(tmp_path / "x.csv", X)
    r = ro.run("x <- as.matrix(read.csv('x.csv', row.names=1)); y <- normalize_sample_medians_dm(x);"
               "write.csv(y[rownames(x), colnames(x)], 'y.csv')", tmp_path, probatch=True)
    assert r.returncode == 0, r.stderr
    out, _ = normalize.median_center(X)
    d = np.nanmax(np.abs(ro.read_matrix(tmp_path / "y.csv") - out))
    _record("median centering (proBatch normalize_sample_medians_dm)", d, 1e-10)
    assert d < 1e-10


def test_oracle_quantile_preprocesscore(tmp_path):
    ro.need("preprocessCore")
    rng = np.random.default_rng(2)
    X = np.round(rng.normal(10, 2, size=(50, 6)), 1)      # rounded: includes ties
    ro.write_matrix(tmp_path / "x.csv", X)
    r = ro.run("x <- as.matrix(read.csv('x.csv', row.names=1)); y <- normalize.quantiles(x);"
               "rownames(y) <- rownames(x); colnames(y) <- colnames(x); write.csv(y, 'y.csv')", tmp_path,
               packages=("preprocessCore",))
    assert r.returncode == 0, r.stderr
    out, _ = normalize.quantile(X)
    d = np.abs(ro.read_matrix(tmp_path / "y.csv") - out).max()
    _record("quantile (preprocessCore normalize.quantiles, with ties)", d, 1e-8)
    assert d < 1e-8


def test_oracle_loess_r(tmp_path):
    ro.need()
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0, 30, 25))
    y = np.sin(x / 5) + rng.normal(0, 0.1, 25)
    ro.write_matrix(tmp_path / "xy.csv", np.vstack([x, y]))
    r = ro.run("m <- as.matrix(read.csv('xy.csv', row.names=1)); x <- m[1,]; y <- m[2,];"
               "fit <- loess(y ~ x, span=0.75, degree=2, surface='direct');"
               "write.csv(t(as.matrix(predict(fit, newdata=x))), 'p.csv')", tmp_path)
    assert r.returncode == 0, r.stderr
    rp = ro.read_matrix(tmp_path / "p.csv")[0]
    py, eng = loess.fit_predict(x, y, x, span=0.75, degree=2)
    rel = np.max(np.abs(py - rp) / np.maximum(np.abs(rp), 1e-12))
    _record(f"LOESS ({eng}) vs R loess, surface direct", np.abs(py - rp).max(), "rel 1e-6")
    assert eng == "skmisc" and rel < 1e-6
    fb = loess._fallback(x, y, x, 0.75, 2)
    _record("LOESS (PRISM fallback) vs R loess", np.abs(fb - rp).max(), "documented")


def test_oracle_drift_probatch(tmp_path):
    ro.need("dplyr", "tidyr", "purrr", probatch=True)
    X, batch, order = drift_data()
    ro.write_matrix(tmp_path / "x.csv", X)
    with open(tmp_path / "ann.csv", "w") as f:
        f.write("FullRunName,MS_batch,order\n" + "".join(f"s{j},{batch[j]},{int(order[j])}\n" for j in range(len(order))))
    r = ro.run("x <- as.matrix(read.csv('x.csv', row.names=1)); a <- read.csv('ann.csv');"
               "y <- suppressWarnings(adjust_batch_trend_df(matrix_to_long(x), a, span=0.75, min_measurements=8));"
               "m <- long_to_matrix(y); write.csv(m[rownames(x), colnames(x)], 'y.csv')", tmp_path, probatch=True)
    assert r.returncode == 0, r.stderr
    out, info = drift.adjust_batch_trend(X, batch, order, span=0.75, degree=2, min_measurements=8)
    d = np.nanmax(np.abs(ro.read_matrix(tmp_path / "y.csv") - out))
    _record(f"LOESS drift (proBatch adjust_batch_trend_df), engine {info['loess_engine']}", d, 1e-6)
    assert d < 1e-6


def test_oracle_batch_median_centering_probatch(tmp_path):
    ro.need("dplyr", "tidyr", "purrr", probatch=True)
    X, batch, order = drift_data()
    ro.write_matrix(tmp_path / "x.csv", X)
    with open(tmp_path / "ann.csv", "w") as f:
        f.write("FullRunName,MS_batch\n" + "".join(f"s{j},{batch[j]}\n" for j in range(len(batch))))
    r = ro.run("x <- as.matrix(read.csv('x.csv', row.names=1)); a <- read.csv('ann.csv');"
               "y <- suppressWarnings(center_feature_batch_medians_dm(x, a));"
               "write.csv(y[rownames(x), colnames(x)], 'y.csv')", tmp_path, probatch=True)
    assert r.returncode == 0, r.stderr
    out, _ = correct.center_feature_batch(X, batch)
    d = np.nanmax(np.abs(ro.read_matrix(tmp_path / "y.csv") - out))
    _record("per-batch median centering (proBatch center_feature_batch_medians_dm)", d, 1e-10)
    assert d < 1e-10


def test_oracle_combat_sva(tmp_path):
    ro.need("sva")
    X, batch, order = drift_data(nan=False, p=60, n=24)
    batch3 = np.array(["b1"] * 8 + ["b2"] * 8 + ["b3"] * 8)
    X[:, batch3 == "b3"] *= 1.3
    ro.write_matrix(tmp_path / "x.csv", X)
    with open(tmp_path / "b.csv", "w") as f:
        f.write("batch\n" + "".join(f"{b}\n" for b in batch3))
    r = ro.run("x <- as.matrix(read.csv('x.csv', row.names=1)); b <- read.csv('b.csv')$batch;"
               "a <- data.frame(batch=b); y <- ComBat(dat=x, batch=b, mod=model.matrix(~1, data=a), par.prior=TRUE);"
               "write.csv(y, 'y.csv')", tmp_path, packages=("sva",))
    assert r.returncode == 0, r.stderr
    out, _ = combat_mod.combat(X, batch3)
    d = np.abs(ro.read_matrix(tmp_path / "y.csv") - out).max()
    _record("ComBat (sva::ComBat, ~1, parametric)", d, 1e-6)
    assert d < 1e-6


def test_bug_keep_all_false_fails_in_r(tmp_path):
    """Confirms the proBatch bug: correct_batch_effects_dm cannot return a matrix."""
    ro.need("dplyr", "tidyr", "purrr", probatch=True)
    X, batch, order = drift_data(nan=False)
    ro.write_matrix(tmp_path / "x.csv", X)
    with open(tmp_path / "ann.csv", "w") as f:
        f.write("FullRunName,MS_batch,order\n" + "".join(f"s{j},{batch[j]},{int(order[j])}\n" for j in range(len(order))))
    r = ro.run("x <- as.matrix(read.csv('x.csv', row.names=1)); a <- read.csv('ann.csv');"
               "y <- try(suppressWarnings(correct_batch_effects_dm(x, a, discrete_func='MedianCentering')), silent=TRUE);"
               "cat(if (inherits(y, 'try-error') || is.null(y)) 'FAILED' else 'OK')", tmp_path, probatch=True)
    assert "FAILED" in r.stdout, r.stdout + r.stderr


PVCA_R = r"""
x <- as.matrix(read.csv('x.csv', row.names=1)); a <- read.csv('ann.csv', stringsAsFactors=TRUE)
xc <- x - rowMeans(x); C <- cor(xc); e <- eigen(C); ev <- e$values; pct <- ev / sum(ev)
cnt <- 0; s <- 1; for (i in length(ev):1) { s <- s - pct[i]; if (s <= 0.6) cnt <- cnt + 1 }
pc_n <- max(3, cnt)
f <- c('batch', 'group')
a$batch__group <- interaction(a$batch, a$group)
M <- c()
for (i in 1:pc_n) {
  d <- data.frame(y = e$vectors[, i], batch = a$batch, group = a$group, bg = a$batch__group)
  m <- suppressWarnings(suppressMessages(lmer(y ~ (1|batch) + (1|group) + (1|bg), data = d, REML = TRUE)))
  vc <- as.data.frame(VarCorr(m)); v <- c(vc$vcov[vc$grp=='batch'], vc$vcov[vc$grp=='group'], vc$vcov[vc$grp=='bg'],
                                          vc$vcov[vc$grp=='Residual'])
  M <- rbind(M, v / sum(v) * pct[i])
}
w <- colSums(M); w <- w / sum(w)
write.csv(t(as.matrix(w)), 'w.csv')
"""


def test_oracle_pvca_lme4(tmp_path):
    """The Bioconductor pvca package is not installable here (network policy); the oracle is an lme4
    re-implementation of pvcaBatchAssess's steps (stated in docs/probatch_port.md)."""
    ro.need("lme4")
    pytest.importorskip("statsmodels")
    # an identifiable crossed design (3 batches x 2 groups); over seeds 0 to 2 the largest difference was
    # 0.011. Singular fits (nested or near-empty components) differ more: 0.043 was seen on a drift design.
    rng = np.random.default_rng(0)
    n = 36
    batch = np.array(["b1", "b2", "b3"] * 12)
    group = np.array(["g1"] * 18 + ["g2"] * 18)
    X = rng.normal(10, 1, size=(300, 1)) + rng.normal(0, 0.5, size=(300, n))
    X += rng.normal(0, 0.6, size=(300, 3))[:, [int(b[1]) - 1 for b in batch]]
    X += rng.normal(0, 0.4, size=(300, 2))[:, [int(g[1]) - 1 for g in group]]
    ro.write_matrix(tmp_path / "x.csv", X)
    with open(tmp_path / "ann.csv", "w") as f:
        f.write("batch,group\n" + "".join(f"{b},{g}\n" for b, g in zip(batch, group)))
    r = ro.run(PVCA_R, tmp_path, packages=("lme4",))
    assert r.returncode == 0, r.stderr
    rw = ro.read_matrix(tmp_path / "w.csv")[0]
    res = diagnostics.pvca(X, {"batch": batch.tolist(), "group": group.tolist()}, ["batch", "group"])
    pw = np.array([res["weights"][k] for k in ("batch", "group", "batch:group", "resid")])
    d = np.abs(pw - rw).max()
    _record(f"PVCA (statsmodels {set(res['method_per_pc'])}) vs lme4 re-implementation", d, 0.02)
    assert d < 0.02
