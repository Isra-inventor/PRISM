"""G4 part 2 and the design audits: A7 batch structure, A8 repeated measures, A5, A11, G3 (planted answers)."""
import numpy as np

import prism
from test_audit import finished

N = 16
STUDY = [f"S{j}" for j in range(1, N + 1)]


def cells(M):
    return [["" if np.isnan(v) else f"{v:.6g}" for v in row] for row in M]


def base(rng, n_p=120):
    return rng.lognormal(8, 1.5, n_p)[:, None] * rng.lognormal(0, 0.15, (n_p, N + 1))


def plates():
    return {s: ("A" if j < 8 else "B") for j, s in enumerate(STUDY)}


def test_batch_shift_is_found_and_robust(tmp_path):
    rng = np.random.default_rng(1)
    X = base(rng)
    X[:, 8:N] *= rng.lognormal(0.8, 0.1, (X.shape[0], 1))          # plate B shifted, protein by protein
    groups = {s: ("case" if j % 2 else "ctrl") for j, s in enumerate(STUDY)}
    a = prism.Audit(finished(tmp_path, cells(X), info={"Plate": ("batch", plates()), "Group": ("group", groups)}))
    f = a.batch_structure()
    m = f.measures["A1B1"]
    perm = [r for r in m["associations"]["rows"] if r["test"] == "permanova" and r["variable"] == "Plate"][0]
    assert perm["q"] <= 0.05 and perm["r2"] > 0.3
    assert m["sensitivity"]["same_conclusion"]
    rc = m["relationship_correlation"]["by_batch"][0]
    assert rc["batch_effect_on_correlation"] > 0
    st = [s for s in m["structure"] if s["design"] == "Group"][0]
    assert st["code"] in ("crossed_balanced", "partially_confounded")
    assert m["samples_per_batch"][0]["below_suggestion"] == ["A", "B"]      # 8 < 25 (heuristic)


def test_confounded_batch_and_group(tmp_path):
    rng = np.random.default_rng(2)
    groups = {s: ("case" if j < 8 else "ctrl") for j, s in enumerate(STUDY)}      # group == plate
    a = prism.Audit(finished(tmp_path, cells(base(rng)), info={"Plate": ("batch", plates()),
                                                                "Group": ("group", groups)}))
    m = a.batch_structure().measures["A1B1"]
    assert [s["code"] for s in m["structure"] if s["design"] == "Group"] == ["confounded"]


def test_repeated_measures_icc_and_effective_n(tmp_path):
    rng = np.random.default_rng(3)
    X = base(rng)
    subj = {s: f"P{j // 2}" for j, s in enumerate(STUDY)}                    # 8 subjects x 2 samples
    for k in range(8):
        X[:, [2 * k, 2 * k + 1]] *= rng.lognormal(0, 0.6, (X.shape[0], 1))   # strong subject effect
    a = prism.Audit(finished(tmp_path, cells(X), info={"Subject": ("subject_id", subj)}))
    m = a.repeated_measures().measures["A1B1"]
    assert m["design"]["paired"] and m["design"]["n_subjects"] == 8
    assert m["icc"]["raw"]["median"] > 0.5
    assert m["sample_correlation"]["difference"] > 0
    d = a.dimensionality().measures["A1B1"]["design_effect"]
    assert d["n_eff"] < N and d["deff"] > 1


def test_outlier_sample_flagged(tmp_path):
    rng = np.random.default_rng(4)
    X = base(rng)
    X[:, 3] *= rng.lognormal(0, 1.5, X.shape[0])                              # S4 scrambled
    m = prism.Audit(finished(tmp_path, cells(X))).outliers().measures["A1B1"]
    assert m["samples"]["flagged"] == ["S4"]
    assert "same_conclusion" in m["sensitivity"]


def test_cleaning_candidates(tmp_path):
    rng = np.random.default_rng(5)
    X = base(rng)
    X[0, :N] = np.nan
    X[1, :N] = 7.0
    types = ["Protein"] * X.shape[0]
    types[2] = "Control"
    a = prism.Audit(finished(tmp_path, cells(X), types=types))
    c = {k["class"]: k for k in a.cleaning_candidates()["A1B1"]["classes"]}
    assert c["all_missing"]["ids"] == ["1000-1"] and c["sd_zero"]["ids"] == ["1001-1"]
    assert c["flagged:Type"]["ids"] == ["1002-1"]


def test_run_all(tmp_path):
    rng = np.random.default_rng(6)
    a = prism.Audit(finished(tmp_path, cells(base(rng)), info={"Plate": ("batch", plates())}))
    summary = a.run_all()
    assert set(summary["audit"]) == {"A1", "A2", "A13", "A3", "A6", "A4", "A12", "A7", "A8", "A5", "A11"}
