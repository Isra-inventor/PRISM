"""G7 part 1: A4 missingness and A12 sections 5.1-5.3, with planted answers."""
import numpy as np

import prism
from test_audit import finished

RNG = np.random.default_rng(3)
N_STUDY = 12
PLATES = {f"S{j}": ("A" if j <= 6 else "B") for j in range(1, N_STUDY + 1)}


def cells(M):
    return [["" if np.isnan(v) else f"{v:.6g}" for v in row] for row in M]


def base(n=60):
    abundance = np.sort(RNG.lognormal(8, 1.5, n))[::-1]       # protein 0 most abundant
    return abundance[:, None] * RNG.lognormal(0, 0.2, (n, N_STUDY + 1))


def test_complete_data(tmp_path):
    a = prism.Audit(finished(tmp_path, cells(base()), plates=PLATES))
    f4 = a.missingness()
    assert f4.measures["A1B1"]["missing"]["n"] == 0
    f12 = a.missingness_profile()
    enc = f12.measures["A1B1"]["cell_encoding"]
    assert enc["missing"] == 0 and enc["observed"] + enc["floor_tied"] + enc["zero"] == enc["n_cells"]
    assert f12.measures["A1B1"]["batch_wide_absence"]["Plate"]["n_proteins_batch_wide_absent"] == 0
    assert f12.measures["A1B1"]["batch_wide_absence"]["Plate"]["coverage_at_least_2_every_level"] == 1.0


def test_detection_limit_pattern(tmp_path):
    X = base()
    for i in range(30, 60):                       # the 30 least abundant proteins lose values, more the lower
        k = (i - 25) // 5
        X[i, RNG.choice(N_STUDY, k, replace=False)] = np.nan
    f = prism.Audit(finished(tmp_path, cells(X), plates=PLATES)).missingness()
    dep = f.measures["A1B1"]["abundance_dependence"]["missing"]
    assert dep["rho"] < -0.5
    assert any("detection limit" in t for _, t in f.sentences)


def test_batch_wide_absence_and_rates(tmp_path):
    X = base()
    plate_b = [j for j in range(N_STUDY) if PLATES[f"S{j + 1}"] == "B"]
    X[np.ix_([5, 6, 7], plate_b)] = np.nan       # three proteins never seen on plate B
    a = prism.Audit(finished(tmp_path, cells(X), plates=PLATES))
    f = a.missingness_profile()
    ab = f.measures["A1B1"]["batch_wide_absence"]["Plate"]
    assert ab["n_proteins_batch_wide_absent"] == 3 and ab["per_level"] == {"A": 0, "B": 3}
    rates = {r["level"]: r["missing_rate"] for r in f.measures["A1B1"]["rates_by_variable"]["Plate"]["levels"]}
    assert rates["A"] == 0 and rates["B"] > 0
    assert any("breaks both" in t for _, t in f.sentences)


def test_zeros_and_floor_counted_apart(tmp_path):
    X = base()
    X[0, :4] = 0
    X[10:20, :3] = X[10:20].min(axis=1, keepdims=True) * 0.5     # 10 proteins with a floor
    a = prism.Audit(finished(tmp_path, cells(X), plates=PLATES))
    f4 = a.missingness()
    m = f4.measures["A1B1"]
    assert m["zeros"]["n"] == 4 and m["missing"]["n"] == 0
    assert m["floor_ties"]["n_proteins"] >= 10
    enc = a.missingness_profile().measures["A1B1"]["cell_encoding"]
    assert enc["zero"] == 4 and enc["missing"] == 0 and enc["floor_tied"] >= 30
    assert sum(enc[k] for k in ("observed", "missing", "zero", "floor_tied")) == enc["n_cells"]


def test_no_batch_variable(tmp_path):
    f = prism.Audit(finished(tmp_path, cells(base()))).missingness_profile()
    assert any(st == "not_applicable" for st, _ in f.sentences)
