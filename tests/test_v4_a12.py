"""v4 stage 6: A12 missingness profile with planted ground truth (seeded)."""

import json
import re

import numpy as np
import pytest

from prism import store
from prism.audit import engine, overrides
from prism.audit.a12_missingness_profile import detection_curve
from prism.audit.params import defaults
import audit_fixtures as af


def expit(x):
    return 1 / (1 + np.exp(-x))


def simulate(b1, p=1500, n=30, seed=5, b0=None, batch_rates=None, absent=0):
    rng = np.random.default_rng(seed)
    mu = rng.normal(20, 2, size=p)
    V = mu[:, None] + rng.normal(0, 0.3, size=(p, n))
    prob = expit((b0 if b0 is not None else -b1 * 18.5) + b1 * mu) if b1 else np.full(p, 0.7)
    det = rng.random((p, n)) < prob[:, None]
    batch = ["b1" if j < n // 2 else "b2" for j in range(n)]
    if batch_rates:                          # NaN rate by batch for the first 300 features
        for j in range(n):
            r = batch_rates[batch[j]]
            det[:300, j] = rng.random(300) >= r
    if absent:                               # features absent from every b2 sample
        det[300:300 + absent, n // 2:] = False
        det[300:300 + absent, : n // 2] = True
    X = np.where(det, 2.0 ** V, np.nan)
    return X, [f"s{j:02d}" for j in range(n)], {"batch": batch}, {"batch": "batch"}


def run_a12(tmp_path, name, X, samples, cols, kinds):
    st = store.create(name)
    af.add(st, tmp_path / name, X=X, samples=samples, columns=cols, kinds=kinds)
    man = engine.run(st, factors=["A12"])
    return st, engine.finding(st, man["run_id"], man["findings"][0]["file"])


def test_detection_curve_recovers_b1():
    P = defaults()
    X, *_ = simulate(0.8)
    dc = detection_curve(np.log2(X), P)
    assert dc["status"] == "computed" and dc["converged"]
    # tolerance tuned on this simulation: over seeds 5 to 14 the estimate ranged 0.783 to 0.811 (true 0.8);
    # the test allows 10% of b1
    assert abs(dc["b1"] - 0.8) < 0.08
    lo, hi = dc["b1_ci95"]
    assert lo < 0.8 < hi and lo > 0
    assert 0 < dc["plateau_missing_share"] < 0.5 and dc["p_detect_at_p90"] > dc["p_detect_at_p10"]


def test_constant_detection_gives_a_ci_containing_zero():
    X, *_ = simulate(0.0)
    dc = detection_curve(np.log2(X), defaults())
    lo, hi = dc["b1_ci95"]
    assert lo < 0 < hi


def test_batch_wide_absence_is_recovered_exactly(isolated, tmp_path):
    X, samples, cols, kinds = simulate(0.0, p=600, absent=15)
    st, f = run_a12(tmp_path, "absent", X, samples, cols, kinds)
    bwa = f["measures"]["batch_wide_absence"]["variables"][0]
    assert bwa["n_features"] == 15 and bwa["features"] == [f"f{i}" for i in range(300, 315)]
    assert bwa["per_level"][1] == {"level": "b2", "n_samples": 15, "n_features_absent": 15}
    assert f["measures"]["feature_classes"]["counts"]["batch_wide_absent"] == 15
    assert f["measures"]["pattern"]["indicators"]["batch_association"] == "yes"


def test_batch_associated_missingness(isolated, tmp_path):
    X, samples, cols, kinds = simulate(0.0, p=600, batch_rates={"b1": 0.02, "b2": 0.45})
    st, f = run_a12(tmp_path, "batchnan", X, samples, cols, kinds)
    feat = f["measures"]["design_association"]["per_feature"][0]
    assert feat["n_significant"] >= 100 and feat["n_tested"] >= 300
    assert f["measures"]["pattern"]["pattern_class"] in ("batch_associated", "mixed")
    ps = f["measures"]["design_association"]["per_sample"]["rows"]
    assert any(r["variable"] == "batch" and r["q"] <= 0.05 for r in ps)
    cands = f["measures"]["imputation_candidates"]
    assert cands["ranked"] is False and "nothing is imputed" in cands["hand_off"]
    assert {e["id"] for e in cands["entries"] if e["fits_profile"]} >= {"within_batch", "prefer_no_imputation"}


def test_no_nan_floor_and_zero_question(isolated, tmp_path):
    X, samples, cols, kinds = af.planted()
    X = np.where(np.isnan(X), X[np.isfinite(X)].min(), X)          # no NaN, floors stay
    X[5, 3] = 0.0
    st, f = run_a12(tmp_path, "nonan", X, samples, cols, kinds)
    M = f["measures"]
    assert M["cell_encoding"]["nan"] == 0 and M["cell_encoding"]["zero"] == 1
    assert M["cell_encoding"]["n_floor_tied_features"] >= 20
    assert M["detection_curve"]["status"] == "not_applicable" and "A4 item 4" in M["detection_curve"]["substitute"]
    assert M["pattern"]["pattern_class"] == "none" and M["pattern"]["indicators"]["floor_pattern"] == "yes"
    assert M["questions"][0]["kind"] == "zeros_meaning"
    overrides.add(st, {"kind": "zeros_are_missing", "value": "no"})
    man = engine.run(st, factors=["A12"])
    assert engine.finding(st, man["run_id"], man["findings"][0]["file"])["measures"]["questions"] == []


def test_wording_rule_and_nothing_filled(isolated, tmp_path):
    X, samples, cols, kinds = simulate(0.8, p=400, batch_rates={"b1": 0.05, "b2": 0.3})
    st, f = run_a12(tmp_path, "words", X, samples, cols, kinds)
    g = dict(f)
    g["measures"] = {k: v for k, v in f["measures"].items() if k != "imputation_candidates"}
    text = json.dumps(g)
    assert not re.search(r"\b(MCAR|MAR|MNAR)\b", text)
    assert "Molenberghs" in f["measures"]["pattern"]["mechanism_note"]
    assert f["measures"]["pattern"]["wording"] == "Pattern quantified, plausible mechanisms assessed."
    assert "simplified" in f["measures"]["detection_curve"]["limitation"]
    # the output folder is untouched (nothing written outside audit/)
    from prism.io.loader import load_output
    out = load_output(st.output_dir("D1"))
    assert np.isnan(out.assays[0]["blocks"][0].X).sum() == np.isnan(X).sum()
