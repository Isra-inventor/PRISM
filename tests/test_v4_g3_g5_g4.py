"""v4 stage 7: G3 cleaning candidates and ops, G5 transform, the G4 measures added to A7."""

import numpy as np
import pytest

from prism import store
from prism.audit import engine, overrides
from prism.executor import ExecutorError, apply
from prism.executor import state as S
import audit_fixtures as af

APPR = {"user": "t", "at": "2026-10-21T00:00:00Z", "plan_entry": "e"}


@pytest.fixture
def st(isolated, tmp_path):
    s = store.create("g3")
    X, samples, cols, kinds = af.planted(shift=True)
    X[0] = np.nan                       # all missing
    X[1] = 7.0                          # SD 0
    X[2, 2:] = np.nan                   # 2 observed
    ann = {"decoy": {"values": ["yes" if i in (3, 4) else "" for i in range(X.shape[0])], "marks_rows_as_suspect": True,
                     "flagged_values": ["yes"]}}
    af.add(s, tmp_path / "planted", X=X, samples=samples, columns=cols, kinds=kinds, design=af.SUBJ_DESIGN,
           annotations=ann)
    return s


def test_g3_candidates_by_class_and_removal(st):
    man = engine.run(st, factors=["A1"])
    g3 = engine.finding(st, man["run_id"], man["findings"][0]["file"])["measures"]["g3_candidates"]
    by = {c["class"]: c for c in g3["classes"]}
    assert by["all_missing"]["ids"] == ["f0"] and by["sd_zero"]["ids"] == ["f1"]
    assert by["fewer_than_3_observed"]["ids"] == ["f2"] and by["suspect:decoy"]["ids"] == ["f3", "f4"]
    assert "blank_dominated" not in by
    m = apply(st, "D1", "remove_features", {"ids": by["suspect:decoy"]["ids"], "reason": "suspect:decoy"},
              dict(APPR, plan_entry="g3-suspect"), step="G3")
    assert m["checkpoint"] == "C0" and m["counts"]["n_features"] == 398


def test_blank_dominated_needs_blanks(isolated, tmp_path):
    s = store.create("blank")
    X, samples, cols, kinds = af.planted()
    X[:, :2] = X[:, 2:4] * 0.01        # two blanks with low signal
    X[10, :2] = X[10, 2:].mean() * 2    # feature 10 dominated by blank signal
    af.add(s, tmp_path / "b", X=X, samples=samples, columns=cols, kinds=kinds, is_study=[j >= 2 for j in range(36)])
    for smp in samples[:2]:
        overrides.add(s, {"kind": "sample_role", "sample": smp, "role": "blank"})
    man = engine.run(s, factors=["A1"])
    by = {c["class"]: c for c in engine.finding(s, man["run_id"], man["findings"][0]["file"])["measures"]
          ["g3_candidates"]["classes"]}
    assert by["blank_dominated"]["ids"] == ["f10"]


def test_g5_log_and_arcsinh(st):
    overrides.add(st, {"kind": "processing_state", "item": "log_transformed", "value": "no"})
    with pytest.raises(ExecutorError, match="cofactor"):
        apply(st, "D1", "arcsinh_transform", {"cofactor": None, "justification": ""}, APPR)
    m = apply(st, "D1", "arcsinh_transform", {"cofactor": 150, "justification": "test"}, APPR)
    assert m["scale"] == "arcsinh"
    m2 = apply(st, "D1", "log_transform", {"offset_rule": "proBatch_compat"}, dict(APPR, plan_entry="g5"),
               parent=S.root_id(st, "D1"))
    assert m2["op_record"]["details"]["offset_used"] == {"A1": 1.0}
    m3 = apply(st, "D1", "log_transform", {"offset_rule": "fixed", "offset": 2.5}, dict(APPR, plan_entry="g5b"),
               parent=S.root_id(st, "D1"))
    assert m3["op_record"]["details"]["offset_used"] == {"A1": 2.5}


def test_log_offset_none_needs_positive_values(isolated, tmp_path):
    X = np.array([[0.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 5.0], [2.0, 3.0, 4.0, 6.0]])
    s = store.create("zero")
    af.add(s, tmp_path / "z", X=X, samples=["a", "b", "c", "d"])
    overrides.add(s, {"kind": "processing_state", "item": "log_transformed", "value": "no"})
    with pytest.raises(ExecutorError, match="every observed value > 0"):
        apply(s, "D1", "log_transform", {"offset_rule": "none"}, APPR)


def test_a7_g4_measures(st):
    overrides.add(st, {"kind": "control_feature", "feature": "f50"})
    man = engine.run(st, factors=["A7"])
    M = engine.finding(st, man["run_id"], man["findings"][0]["file"])["measures"]
    spb = {x["variable"]: x for x in M["samples_per_batch"]}
    assert spb["plate"]["min"] == 9 and spb["plate"]["below_suggestion"] == ["P1", "P2", "P3", "P4"]
    cov = {x["variable"]: x for x in M["batch_coverage"]}
    assert 0 < cov["run_day"]["share_observed_ge1_every_level"] <= 1
    rc = M["relationship_correlation"]
    assert rc["policy"] == "pairwise" and rc["same_subject_median_r"] > rc["different_subject_median_r"]
    day = next(b for b in rc["by_batch"] if b["variable"] == "run_day")
    assert day["batch_effect_on_correlation"] > 0 and "within_same_subject" in day
    assert M["control_features"]["status"] == "computed"
    assert M["control_features"]["features"][0]["status"] == "insufficient_metadata"   # no run order column
