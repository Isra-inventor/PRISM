"""The plan (v4 sections 8 and 9): intake, gate, rules, approval, checkpoints, verification, replay."""
import inspect
import subprocess
import sys

import numpy as np
import pytest

import prism
from prism.audit import verification
from prism.plan import engine
from test_audit import finished

N = 16
STUDY = [f"S{j}" for j in range(1, N + 1)]
PLATES = {s: ("A" if j < 8 else "B") for j, s in enumerate(STUDY)}


def make(tmp_path, groups=None, shift=True, nan_cell=False):
    rng = np.random.default_rng(1)
    X = rng.lognormal(8, 1.5, 120)[:, None] * rng.lognormal(0, 0.15, (120, N + 1))
    if shift:
        X[:, 8:N] *= rng.lognormal(0.8, 0.1, (120, 1))
    if nan_cell:
        X[0, 0] = np.nan
    types = ["Protein"] * 120
    types[5] = "Control"
    groups = groups or {s: ("case" if j % 2 else "ctrl") for j, s in enumerate(STUDY)}
    s = finished(tmp_path, [["" if np.isnan(v) else f"{v:.6g}" for v in r] for r in X], types=types,
                 info={"Plate": ("batch", PLATES), "Group": ("group", groups)})
    st = prism.Study(s, folder=tmp_path / "work")
    st.audit()
    return st


def status(plan, entry):
    return plan.entry(entry)["status"]


def test_full_flow(tmp_path):
    st = make(tmp_path)
    st.set_intake(contrast_variable="Group", downstream_family="differential_abundance")
    p = st.build_plan()
    assert status(p, "A1B1:G3:flagged:Type") == "Recommended"
    assert status(p, "A1B1:G5") == "Recommended"
    assert status(p, "A1B1:G6") == "Insufficient metadata"                  # no run order
    with pytest.raises(engine.PlanError, match="cannot be approved"):
        st.approve("A1B1:G6")
    st.approve("A1B1:G3:flagged:Type")
    with pytest.raises(engine.PlanError, match="changed since this plan"):
        st.approve("A1B1:G5")                                             # stale plan
    st.build_plan()
    st.approve("A1B1:G5")
    p = st.build_plan()
    assert status(p, "A1B1:G5") == "Do not automate"                      # already log: no double transform
    assert status(p, "A1B1:S2") == "Recommended"
    st.approve("A1B1:S2", 0)
    f = st.verify()
    m = f.measures["A1B1"]
    assert m["negative_controls"]["no_batch_clustering"]
    table = {r["measure"]: r for r in m["comparisons"]["S0_vs_C2"]["table"]}
    assert table["permanova_r2[Plate]"]["after"] < 0.2 * table["permanova_r2[Plate]"]["before"]
    assert st.replay()["identical"]
    assert [d["decision"] for d in st.decisions] == ["approved"] * 3
    out = st.save()
    assert (out / "plan.json").exists() and (out / "decisions.jsonl").read_text().count("\n") == 3


def test_no_intake_means_insufficient_metadata(tmp_path):
    st = make(tmp_path)
    p = st.build_plan()
    s1 = p.entry("A1B1:S1")
    assert s1["status"] == "Insufficient metadata" and "contrast_variable" in s1["requires"]


def test_confounded_contrast_is_not_automated(tmp_path):
    st = make(tmp_path, groups={s: ("case" if j < 8 else "ctrl") for j, s in enumerate(STUDY)})
    st.set_intake(contrast_variable="Group")
    st.build_plan()
    st.approve("A1B1:G5")
    p = st.build_plan()
    assert status(p, "A1B1:S2") == "Do not automate"
    assert any("confounded with the contrast" in b for b in p.entry("A1B1:S2")["because"])


def test_verification_never_ranks():
    names = [n for n, _ in inspect.getmembers(verification, inspect.isfunction)]
    assert not any(w in n for n in names for w in ("rank", "select", "best", "choose"))


def test_phase2_is_not_reachable_from_the_plan():
    code = ("import prism, sys\nfrom prism.plan import engine, study\n"
            "assert 'prism.phase2' not in sys.modules and 'prism.phase2.imputation' not in sys.modules\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_experimental_imputation_writes_only_missing_cells(tmp_path):
    from prism.phase2 import imputation
    st = make(tmp_path, nan_cell=True)
    assert np.isnan(st.store.data().blocks[0].X).sum() == 1
    m, rec = imputation.impute(st.store, "min_prob", approval={"user": "you"})
    assert rec["experimental"] and rec["nan_after"] == 0
