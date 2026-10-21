"""v4 golden expectations on the two real files (section 11). The v4 numbers were stated as
expected-but-unverified; differences found on the first run are recorded in reports/v4_build.md."""

import pytest

from prism import store
from prism.audit import engine
from test_v3_audit import DATA, find


@pytest.mark.skipif(not (DATA / "SomaExpr_common.csv").exists() or not (DATA / "MetaboExpr_common.csv").exists(),
                    reason="real data files not present in tests/data")
def test_v4_golden_real_files(client, isolated):
    import real_data
    st = store.create("golden4")
    for name in (real_data.SOMA, real_data.METAB):
        out = real_data.wizard_output(client, isolated, name)
        d = st.add_dataset(name, "wizard", "wizard_in_progress")
        store.publish_output(st, d["dataset_id"], out, {"mode": "wizard"})
    man = engine.run(st)
    # A13: a confirm_processing_state question for both files
    soma, metab = find(man, st, "A13", "D1")["measures"], find(man, st, "A13", "D2")["measures"]
    mi = {r["item"]: r for r in metab["items"]}
    assert mi["imputed"]["match"] == "does_not_match" and mi["normalized"]["match"] == "does_not_match"
    assert {q["item"] for q in metab["questions"]} >= {"imputed", "normalized"}
    assert metab["gate"]["blocks"] == ["G5", "S1"]
    si = {r["item"]: r for r in soma["items"]}
    assert si["imputed"]["match"] == "matches"                       # 20/11,083 floor-tied: below the 1% indicator
    assert soma["questions"], "SomaScan: log_transformed and normalized were declared not_sure"
    assert si["normalized"]["platform_notes"] and si["normalized"]["platform_notes"][0]["platform"] == "SomaScan"
    # A12: metabolomics has 0 NaN and 397 floor-tied features; the detection curve is not applicable
    a12 = find(man, st, "A12", "D2")["measures"]
    assert a12["cell_encoding"]["nan"] == 0 and a12["cell_encoding"]["n_floor_tied_features"] == 397
    assert a12["detection_curve"]["status"] == "not_applicable" and "A4 item 4" in a12["detection_curve"]["substitute"]
    assert a12["pattern"]["pattern_class"] == "none" and a12["pattern"]["indicators"]["floor_pattern"] == "yes"
    assert find(man, st, "A12", "D1")["measures"]["cell_encoding"]["n_floor_tied_features"] == 20
    # A7/A11 sensitivity on the metabolomics file (33.8% floor-tied features)
    a11 = find(man, st, "A11", "D2")["measures"]
    assert {"exclude_floor", "floor_censored"} <= {v["variant"] for v in a11["sensitivity"]["variants"]}
    a7m = find(man, st, "A7", "D2")
    assert a7m["status"] == "insufficient_metadata" and "batch variable" in a7m["needs"]
    # the plan: G5 and S1 wait for the processing-state confirmation; metabolomics S2 needs a batch variable;
    # SomaScan's animals are nested in plates and no batch correction is proposed
    from prism.plan import engine as pe
    plan = pe.build(st)
    ent = {(e["dataset"], e.get("group")): e for e in plan["entries"] if not e.get("checkpoint")}
    for did in ("D1", "D2"):
        assert ent[(did, "G5")]["status"] == "Insufficient metadata"
        assert ent[(did, "S1")]["status"] == "Insufficient metadata"
    assert ent[("D2", "G5")]["because"][0] == "Insufficient metadata: confirm processing state."
    assert ent[("D2", "S2")]["status"] == "Insufficient metadata" and "needs batch variable" in ent[("D2", "S2")]["because"][0]
    s2 = ent[("D1", "S2")]
    assert s2["status"] in ("Insufficient metadata", "Do not automate")
    assert not [c for c in s2["candidates"] if c["op"] in ("batch_center", "batch_combat", "drift_sample_loess")]
    # Difference from the v4 expectation (recorded in reports/v4_build.md): the SomaScan session as uploaded
    # has no plate column (the sample metadata step was skipped), so A7 has no batch candidate and the plate
    # nesting cannot be reported; S2 is Insufficient metadata: needs batch variable, like metabolomics.
    a7s = find(man, st, "A7", "D1")
    assert a7s["status"] == "insufficient_metadata" and "needs batch variable" in s2["because"][0]
