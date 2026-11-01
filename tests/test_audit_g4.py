"""G4 part 1: the diagnostic matrix (v4 section 3), A3 distribution, A6 noise and QC, with planted answers."""
import numpy as np

import prism
from prism.audit import diagnostic as dmx
from prism.audit.params import defaults
from prism.audit.quality import QC_NAME
from test_audit import finished

N_STUDY, N_QC = 12, 4
STUDY = [f"S{j}" for j in range(1, N_STUDY + 1)]
QCS = [f"QC_{j}" for j in range(1, N_QC + 1)]
PLATES = {s: ("A" if j < 6 else "B") for j, s in enumerate(STUDY)}


def cells(M):
    return [["" if np.isnan(v) else f"{v:.6g}" for v in row] for row in M]


def raw(rng, n_p=150, noise=0.3):                     # multiplicative noise: SD proportional to the mean
    level = rng.lognormal(8, 1.5, n_p)
    return level[:, None] * rng.lognormal(0, noise, (n_p, N_STUDY + N_QC))


def test_diagnostic_matrix_policy_and_fallback():
    rng = np.random.default_rng(0)
    Y = rng.normal(20, 1, (100, 10))
    Y[:95, 0] = np.nan                                 # sample 0 misses almost everything
    d = dmx.default_matrix(Y, defaults())
    r = d.report()
    assert r["q"] < 100 and r["fallback"] and "biased toward higher-abundance" in r["bias_warning"]
    assert d.n_features >= 20 and r["n_samples_dropped"] == 1
    assert not np.isnan(d.Y).any()                     # nothing filled, rows / samples dropped instead


def test_mean_sd_slope_and_median_shift_by_plate(tmp_path):
    rng = np.random.default_rng(1)
    X = raw(rng)
    plate_b = [j for j, s in enumerate(STUDY) if PLATES[s] == "B"]
    X[:, plate_b] *= 2.0                               # plate B reads twice as high
    a = prism.Audit(finished(tmp_path, cells(X), plates=PLATES, n_qc=N_QC))
    f = a.distribution()
    m = f.measures["A1B1"]
    assert 0.8 < m["mean_sd"]["slope"] < 1.2 and m["mean_sd"]["ci95"][0] > 0.5
    ms = m["median_shift"]
    assert ms["medians_differ"] and ms["follows_technical"] == ["Plate"]
    # A13 now cites the A3 slope as evidence for the log question
    a.processing_state()
    row = next(r for r in a.findings["A13"].measures["items"] if r["item"] == "log_transformed")
    assert any("A3 slope" in t for t in row["observed"])


def test_log_scale_data_slope_near_zero(tmp_path):
    rng = np.random.default_rng(2)
    X = np.log2(raw(rng, noise=0.2)) - 8               # signed log values, additive noise
    m = prism.Audit(finished(tmp_path, cells(X), n_qc=N_QC)).distribution().measures["A1B1"]
    assert m["diagnostic_copy"]["transform"] == "none"     # not logged twice


def test_qc_repeatability_and_drift(tmp_path):
    rng = np.random.default_rng(3)
    X = raw(rng)
    qc = list(range(N_STUDY, N_STUDY + N_QC))
    X[:, qc] = X[:, [qc[0]]] * rng.lognormal(0, 0.05, (X.shape[0], N_QC))     # QC: ~5% RSD
    order = {s: j + 1 for j, s in enumerate(STUDY + QCS)}
    trend = np.array([2 ** (0.08 * order[s]) for s in STUDY + QCS])            # signal rises over the run
    X = X * trend
    a = prism.Audit(finished(tmp_path, cells(X), n_qc=N_QC, run_order=order))
    f = a.quality()
    m = f.measures["A1B1"]
    assert m["qc_rsd"]["n_qc"] == N_QC and m["qc_rsd"]["summary"]["median"] < 15
    assert m["run_order"]["spearman_median"] > 0.9
    assert m["run_order"]["qc_protein_trend"]["share_abs_rho_gt_0_5"] > 0.5
    text = " ".join(t for _, t in f.sentences)
    assert "drift" in text
    assert len(m["per_sample"]) == N_STUDY


def test_no_run_order_or_qc(tmp_path):
    rng = np.random.default_rng(4)
    f = prism.Audit(finished(tmp_path, cells(raw(rng)[:, :N_STUDY + 1]))).quality()
    statuses = [s for s, _ in f.sentences]
    assert statuses.count("not_applicable") == 2


def test_vendor_qc_column_names():
    for name in ("CV Plasma", "LoDB", "ColCheck", "QC_flag"):
        assert QC_NAME.search(name), name
    for name in ("Target", "Dilution", "Organism"):
        assert not QC_NAME.search(name), name
