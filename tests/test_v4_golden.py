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
    # A7/A11 sensitivity on the metabolomics file (33.8% floor-tied features)
    a11 = find(man, st, "A11", "D2")["measures"]
    assert {"exclude_floor", "floor_censored"} <= {v["variant"] for v in a11["sensitivity"]["variants"]}
    a7m = find(man, st, "A7", "D2")
    assert a7m["status"] == "insufficient_metadata" and "batch variable" in a7m["needs"]
