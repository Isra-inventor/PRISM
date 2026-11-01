"""G2: scale (A2) and processing state (A13), with planted answers."""
import numpy as np

import prism
from prism import registry
from test_audit import finished

RNG = np.random.default_rng(0)


def cells(M):
    return [[f"{v:.6g}" for v in row] for row in M]


def raw(n=40):                       # linear, right-skewed intensities, unequal sample medians
    return RNG.lognormal(8, 1.2, (n, 1)) * RNG.lognormal(0, 0.8, (n, 6)) * np.array([1, 1.6, 0.7, 1.3, 0.8, 1])


def test_registry_is_valid_and_honest():
    r = registry.check()
    assert r["ok"], r["problems"]
    built = [s["id"] for s in registry.steps() if s["implemented"]]
    assert {"G1", "G2", "G4", "G7"} <= set(built)
    assert registry.step("G2")["audits"] == ["A2", "A13"]


def test_linear_data_declared_not_log(tmp_path):
    a = prism.Audit(finished(tmp_path, cells(raw())))
    assert a.scale().measures["A1B1"]["class"] == "continuous_linear"
    rows = {r["item"]: r for r in a.processing_state().measures["items"]}
    assert rows["log_transformed"]["match"] == "matches"
    assert rows["normalized"]["match"] == "matches"
    assert rows["batch_corrected"]["match"] == "not_assessed"
    assert a.findings["A13"].measures["gate"]["blocks"] == []


def test_log_data_declared_not_log_is_a_mismatch_until_confirmed(tmp_path):
    a = prism.Audit(finished(tmp_path, cells(np.log2(raw()) - 10)))      # negatives: signed, log-like
    assert a.scale().measures["A1B1"]["class"] == "continuous_signed"
    f = a.processing_state()
    assert f.measures["gate"]["blocks"] == ["G5", "S1"]
    assert "log_transformed" in a.questions()["item"].tolist()
    a.confirm_processing_state("log_transformed", "yes")
    f = a.findings["A13"]
    assert "log_transformed" not in f.measures["gate"]["unconfirmed_mismatches"]
    for item in f.measures["gate"]["unconfirmed_mismatches"]:      # anything else the random data showed
        a.confirm_processing_state(item, "unsure")
    f = a.findings["A13"]
    assert f.measures["gate"]["blocks"] == []
    row = next(r for r in f.measures["items"] if r["item"] == "log_transformed")
    assert row["confirmed"] == "yes" and row["effective"] == "yes"
    a.save_overrides(tmp_path / "overrides.json")
    b = prism.Audit(a.schema, overrides=tmp_path / "overrides.json")
    assert b.processing_state().measures["gate"]["blocks"] == []


def test_normalized_signature(tmp_path):
    X = raw()
    X = X / np.median(X, axis=0) * 1000                                     # equal sample medians
    a = prism.Audit(finished(tmp_path, cells(X)))
    rows = {r["item"]: r for r in a.processing_state().measures["items"]}
    assert rows["normalized"]["match"] == "does_not_match"
    assert "SomaScan" in str(a.findings["A13"].measures["platform_notes"])     # SeqId column -> SomaScan note


def test_floor_ties_declared_not_imputed(tmp_path):
    X = raw()
    X[:20, :3] = X[:20].min(axis=1, keepdims=True) * 0.5                     # 20 proteins with a floor value
    a = prism.Audit(finished(tmp_path, cells(X)))
    rows = {r["item"]: r for r in a.processing_state().measures["items"]}
    assert rows["imputed"]["match"] == "does_not_match"
    assert a.findings["A13"].measures["observed"]["A1B1"]["n_floor_tied"] == 20


def test_count_like_is_out_of_scope(tmp_path):
    X = RNG.poisson(30, (40, 6)).astype(float)
    f = prism.Audit(finished(tmp_path, cells(X))).scale()
    assert f.measures["A1B1"]["class"] == "count_like"
    assert any("outside what PRISM" in t for _, t in f.sentences)
