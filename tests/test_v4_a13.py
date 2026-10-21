"""v4 stage 5: A13 processing_state, the confirm_processing_state question, platform notes, the gate."""

import numpy as np
import pytest

from prism import store
from prism.audit import engine, overrides
from prism.executor import ExecutorError, apply
from prism.executor import state as S
import audit_fixtures as af

APPR = {"user": "t", "at": "2026-10-21T00:00:00Z", "plan_entry": "g5"}


def a13(st, man, did="D1"):
    i = next(i for i in man["findings"] if i["audit_id"] == "A13" and i["dataset"] == did)
    return engine.finding(st, man["run_id"], i["file"])


@pytest.fixture
def st(isolated, tmp_path):
    s = store.create("a13")
    X, samples, cols, kinds = af.planted(shift=True)
    af.add(s, tmp_path / "planted", X=X, samples=samples, columns=cols, kinds=kinds, design=af.SUBJ_DESIGN,
           history={"imputed": {"answer": "no"}, "log_transformed": {"answer": "not_sure"}})
    return s


def test_declared_vs_observed_and_questions(st):
    man = engine.run(st, factors=["A13"])
    f = a13(st, man)
    items = {r["item"]: r for r in f["measures"]["items"]}
    assert items["imputed"]["declared"] == "no" and items["imputed"]["match"] == "does_not_match"   # 20/400 floor-tied
    assert items["log_transformed"]["declared"] == "unsure" and items["log_transformed"]["match"] == "not_assessed"
    assert items["batch_corrected"]["match"] == "not_assessed" and items["removed_before_upload"]["match"] == "not_assessed"
    assert all(r["confirmed"] is None for r in items.values())
    assert any("floor ties" in s for s in items["imputed"]["observed_evidence"])
    qs = {q["item"]: q for q in f["measures"]["questions"]}
    assert set(qs) == {"imputed", "log_transformed", "normalized"}
    assert qs["imputed"]["kind"] == "confirm_processing_state" and qs["imputed"]["match"] == "does_not_match"
    assert f["measures"]["gate"]["blocks"] == ["G5", "S1"]
    assert "processing_state_mismatch" in [i["code"] for i in f["indicators"]]


def test_confirmation_clears_the_gate_and_sets_the_scale(st):
    overrides.add(st, {"kind": "processing_state", "item": "imputed", "value": "yes", "reason": "min imputation"})
    overrides.add(st, {"kind": "processing_state", "item": "log_transformed", "value": "no"})
    man = engine.run(st, factors=["A13"])
    f = a13(st, man)
    items = {r["item"]: r for r in f["measures"]["items"]}
    assert items["imputed"]["confirmed"] == "yes" and items["imputed"]["effective"] == "yes"
    assert f["measures"]["unconfirmed_mismatches"] == [] and f["measures"]["gate"]["blocks"] == []
    assert {q["item"] for q in f["measures"]["questions"]} == {"normalized"}
    assert S.root_scale(st, "D1") == "linear"
    apply(st, "D1", "log_transform", {}, APPR)


def test_matching_declaration_is_effective_without_a_question(isolated, tmp_path):
    s = store.create("decl")
    X, samples, cols, kinds = af.planted()
    af.add(s, tmp_path / "p2", X=X, samples=samples, columns=cols, kinds=kinds,
           history={"log_transformed": {"answer": "no"}, "normalized": {"answer": "no"}})
    man = engine.run(s, factors=["A13"])
    items = {r["item"]: r for r in a13(s, man)["measures"]["items"]}
    assert items["log_transformed"]["match"] == "matches" and items["log_transformed"]["effective"] == "no"
    assert S.root_scale(s, "D1") == "linear"
    s2 = store.create("decl2")
    af.add(s2, tmp_path / "p3", X=np.log2(X), samples=samples, columns=cols, kinds=kinds,
           history={"log_transformed": {"answer": "no"}})
    man = engine.run(s2, factors=["A13"])
    items = {r["item"]: r for r in a13(s2, man)["measures"]["items"]}
    assert items["log_transformed"]["match"] == "does_not_match"
    assert S.root_scale(s2, "D1") == "unknown"
    with pytest.raises(ExecutorError, match="confirmed processing state"):
        apply(s2, "D1", "log_transform", {}, APPR)


def test_somascan_platform_note(isolated, tmp_path):
    s = store.create("soma")
    X, samples, cols, kinds = af.planted()
    ann = {"SeqId": {"values": [f"{i}-1" for i in range(X.shape[0])]}}
    af.add(s, tmp_path / "soma", X=X, samples=samples, columns=cols, kinds=kinds, annotations=ann,
           history={"normalized": {"answer": "no"}})
    f = a13(s, engine.run(s, factors=["A13"]))
    items = {r["item"]: r for r in f["measures"]["items"]}
    note = items["normalized"]["platform_notes"]
    assert note and note[0]["platform"] == "SomaScan" and note[0]["verify_with_vendor_documentation"] is True
    assert items["normalized"]["confirmed"] is None          # a note never overrides
