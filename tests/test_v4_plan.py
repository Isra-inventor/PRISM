"""v4 stage 10: intake, the metadata-sufficiency gate, the rule base, the plan engine, approval, replay."""

import json

import numpy as np
import pytest

from prism import cli, registry, store, ports
from prism.audit import overrides
from prism.executor import state as S
from prism.plan import engine as pe, intake
import audit_fixtures as af


def batch_session(tmp_path, name="flow", nan=False, seed=11):
    rng = np.random.default_rng(seed)
    n, p = 24, 200
    batch = ["b1" if j < 12 else "b2" for j in range(n)]
    group = ["g1" if j % 2 == 0 else "g2" for j in range(n)]
    L = rng.normal(12, 1.2, size=(p, 1)) + rng.normal(0, 0.15, size=(p, n))
    b2 = np.array([b == "b2" for b in batch])
    L[: p // 2, b2] += 1.5                       # half the features shift in batch b2
    X = 2.0 ** L
    if nan:
        X[5, 3] = np.nan
    st = store.create(name)
    af.add(st, tmp_path / name, X=X, samples=[f"s{j:02d}" for j in range(n)], columns={"batch": batch, "group": group},
           kinds={"batch": "batch", "group": "group"},
           history={"log_transformed": {"answer": "no"}, "normalized": {"answer": "no"}, "imputed": {"answer": "no"}})
    return st


def entry(plan, group):
    return next(e for e in plan["entries"] if e.get("group") == group)


def test_intake_validation(isolated, tmp_path):
    st = batch_session(tmp_path)
    with pytest.raises(intake.IntakeError, match="not a column"):
        intake.set_intake(st, {"contrast_variable": "nope"})
    with pytest.raises(intake.IntakeError, match="question_type"):
        intake.set_intake(st, {"question_type": "everything"})
    doc = intake.set_intake(st, {"contrast_variable": "group", "downstream_family": "differential_abundance"})
    assert doc["version"] == 1 and intake.resolved(st)["protected_variables"] == ["group"]
    doc = intake.set_intake(st, {"contrast_variable": "none"})
    assert doc["version"] == 2 and len(doc["history"]) == 2


def test_gate_lists_insufficient_metadata_first(isolated, tmp_path):
    st = batch_session(tmp_path)
    plan = pe.build(st)
    s2 = entry(plan, "S2")
    assert s2["status"] == "Insufficient metadata" and "contrast_variable" in s2["because"][0]
    assert plan["insufficient_metadata_first"][0] == plan["entries"][[e["entry_id"] for e in plan["entries"]].index(
        plan["insufficient_metadata_first"][0])]["entry_id"]
    g = {x["step"]: x for x in plan["gate"][0]["steps"]}
    assert g["S1"]["unmet"] == ["contrast_variable"] and g["G1"]["needs_satisfied"]
    assert g["G6"]["unmet"] == ["qc_samples", "run_order"]
    assert set(plan["statuses"]) == {"Recommended", "Alternative", "Do not automate", "Insufficient metadata",
                                     "Not applicable"}
    for e in plan["entries"]:
        assert e.get("checkpoint") or e["status"] in plan["statuses"]


def test_rules_are_typed_data_with_levels():
    R = pe.rules()
    O = pe.order()
    ids = registry.step_ids()
    for r in R["rules"]:
        step = O.get("steps_of_groups", {}).get(r["decision_group"], r["decision_group"])
        assert step in ids, r["id"]
        assert r["evidence_level"] in registry.LEVELS
        assert set(r.get("evidence", [])) <= set(registry.evidence())
    assert registry.check()["counted"]["rules/rules_v0.yaml"] == len(R["rules"])


def test_indeterminate_input_never_recommended():
    F = {"scale.class": "continuous_linear", "a3.slope_ci_excludes_zero": None, "processing_state.log_transformed": "no"}
    R = pe.rules()
    g5 = [r for r in R["rules"] if r["decision_group"] == "G5" and r["effect"] == "decide"]
    first = next(r for r in g5 if pe.holds(r["when"], F))
    assert first["id"] == "G-R2b" and first["then"]["status"] == "Alternative"
    F["a3.slope_ci_excludes_zero"] = "indeterminate"
    assert next(r for r in g5 if pe.holds(r["when"], F))["id"] == "G-R2b"


@pytest.mark.skipif(not ports.enabled(), reason="S2 ops need the proBatch port")
def test_full_flow_approve_checkpoints_replay(isolated, tmp_path):
    st = batch_session(tmp_path)
    intake.set_intake(st, {"contrast_variable": "group", "downstream_family": "differential_abundance"})
    plan = pe.build(st)
    g5 = entry(plan, "G5")
    assert g5["status"] == "Recommended" and g5["candidates"][0]["op"] == "log_transform"
    with pytest.raises(pe.PlanError, match="cannot be approved"):
        pe.approve(st, entry(plan, "G6")["entry_id"])
    out = pe.approve(st, g5["entry_id"], 0, user="ana")
    assert out["state"]["checkpoint"] == "C1" and out["run_id"]
    with pytest.raises(pe.PlanError, match="rebuild the plan"):
        pe.approve(st, g5["entry_id"], 0)                       # the dataset moved on
    plan = pe.build(st)
    s1 = entry(plan, "S1")
    assert s1["status"] == "Recommended" and s1["rule_ids"][0] == "R2"
    pe.approve(st, s1["entry_id"], 0)
    plan = pe.build(st)
    s2 = entry(plan, "S2")
    assert s2["status"] == "Recommended" and s2["candidates"][0]["op"] == "batch_center"
    assert any("covariate" in c["label"] for c in s2["candidates"])          # R12
    rej = pe.reject(st, entry(plan, "G6")["entry_id"], reason="no QC samples in this study")
    assert rej["decision"] == "rejected"
    out = pe.approve(st, s2["entry_id"], 0)
    assert out["state"]["checkpoint"] == "C2"
    r = pe.replay(st)
    assert r["identical"] and len(r["datasets"]["D1"]["steps"]) == 3
    dec = pe.decisions(st)
    assert [d["decision"] for d in dec if d["decision"].startswith("approved")] == ["approved"] * 3
    assert all(d["approval"]["plan_entry"] for d in dec if d["decision"] == "approved")
    assert json.loads((st.dir / "design_constraints.json").read_text())["constraints"]
    assert cli.main(["plan", "replay", "--session", st.sid]) == 0


@pytest.mark.skipif(not ports.enabled(), reason="S2 ops need the proBatch port")
def test_combat_do_not_automate_with_missing_values(isolated, tmp_path):
    st = batch_session(tmp_path, "nan", nan=True)
    overrides.add(st, {"kind": "processing_state", "item": "log_transformed", "value": "yes"})
    intake.set_intake(st, {"contrast_variable": "group"})
    plan = pe.build(st)
    s2 = entry(plan, "S2")
    combat = next(c for c in s2["candidates"] if c["op"] == "batch_combat")
    assert combat["status"] == "Do not automate" and combat["d2"]["case"] == "D2.combat_incomplete"
    with pytest.raises(pe.PlanError, match="not approvable"):
        pe.approve(st, s2["entry_id"], s2["candidates"].index(combat))


def test_nesting_blocks_batch_correction(isolated, tmp_path):
    st = store.create("nest")
    X, samples, cols, kinds = af.planted(shift=True)
    af.add(st, tmp_path / "p", X=X, samples=samples, columns=cols, kinds=kinds, design=af.SUBJ_DESIGN)
    intake.set_intake(st, {"contrast_variable": "week"})
    plan = pe.build(st)
    s2 = entry(plan, "S2")
    assert s2["status"] == "Do not automate" and s2["rule_ids"][0] == "S5-nesting"
    assert "'subject' is constant within 'plate'" in s2["because"][0]
    dc = plan["design_constraints"][0]
    assert dc["subject_variable"] == "animal" and dc["nesting_in_batch"]


def test_cli_intake_and_plan(isolated, tmp_path, capsys):
    st = batch_session(tmp_path, "cli")
    assert cli.main(["intake", "--session", st.sid, "--contrast", "group"]) == 0
    assert cli.main(["plan", "--session", st.sid]) == 0
    out = capsys.readouterr().out
    assert "D1:A1:G5" in out and "plan.json" in out
    assert cli.main(["plan", "approve", "D1:A1:G6", "--session", st.sid]) == 2
