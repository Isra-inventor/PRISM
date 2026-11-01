"""G7 part 2: A12 sections 5.4-5.8 with planted answers (Build Request v4, section 11)."""
import numpy as np

import prism
from prism.audit.missingness import _expit, detection_curve
from prism.audit.params import defaults
from test_audit import finished

N_STUDY = 16
PLATES = {f"S{j}": ("A" if j <= 8 else "B") for j in range(1, N_STUDY + 1)}


def cells(M):
    return [["" if np.isnan(v) else f"{v:.6g}" for v in row] for row in M]


def abundances(rng, n_p):
    mu = rng.normal(20, 2, n_p)
    return mu, 2 ** (mu[:, None] + rng.normal(0, 0.3, (n_p, N_STUDY + 1)))


def test_detection_curve_recovers_planted_slope():
    """Planted logit-linear detection with b1 = 1 per log2 unit; tolerance 0.15 (observed error ~0.03)."""
    rng = np.random.default_rng(5)
    mu = rng.normal(20, 2, 400)
    Y = mu[:, None] + rng.normal(0, 0.3, (400, 24))
    Y[rng.random((400, 24)) > _expit(-20 + mu)[:, None]] = np.nan
    d = detection_curve(Y, defaults())
    assert d["status"] == "computed" and abs(d["b1"] - 1) < 0.15
    lo, hi = d["b1_ci95"]
    assert lo < 1 < hi + 0.1


def test_mcar_ci_contains_zero_and_reads_no():
    rng = np.random.default_rng(6)
    Y = rng.normal(20, 2, 400)[:, None] + rng.normal(0, 0.3, (400, 24))
    Y[rng.random((400, 24)) < 0.3] = np.nan
    lo, hi = detection_curve(Y, defaults())["b1_ci95"]
    assert lo < 0 < hi


def test_intensity_dependent_pattern_end_to_end(tmp_path):
    rng = np.random.default_rng(7)
    mu, X = abundances(rng, 200)
    X[rng.random(X.shape) > _expit(-20 + mu)[:, None]] = np.nan
    a = prism.Audit(finished(tmp_path, cells(X), plates=PLATES))
    f = a.missingness_profile()
    m = f.measures["A1B1"]
    assert m["pattern"]["indicators"]["intensity_dependence"] == "yes"
    assert m["pattern"]["pattern_class"] in ("mostly_intensity_dependent", "mixed")
    ids = {e["id"]: e for e in m["imputation_candidates"]["entries"]}
    assert ids["left_censored"]["fits_profile"] and m["imputation_candidates"]["ranked"] is False
    text = " ".join(t for _, t in f.sentences)
    assert "Molenberghs" in text and "MNAR" not in text and "MCAR" not in text


def test_batch_associated_missingness(tmp_path):
    rng = np.random.default_rng(8)
    _, X = abundances(rng, 120)
    plate_b = [j for j in range(N_STUDY) if PLATES[f"S{j + 1}"] == "B"]
    for i in range(40):                                 # 40 proteins lose half their plate-B values
        X[i, rng.choice(plate_b, 4, replace=False)] = np.nan
    X[np.ix_([50, 51], plate_b)] = np.nan               # two proteins absent from plate B entirely
    a = prism.Audit(finished(tmp_path, cells(X), plates=PLATES))
    m = a.missingness_profile().measures["A1B1"]
    rows = [r for r in m["design_association"]["per_sample"] if r["variable"] == "Plate"
            and r["response"] == "per-sample missing rate"]
    assert rows and rows[0]["q"] <= 0.05
    assert m["pattern"]["indicators"]["batch_association"] == "yes"
    assert m["pattern"]["pattern_class"] in ("batch_associated", "mixed")
    assert m["protein_classes"]["counts"]["batch_wide_absent"] == 2
    ids = {e["id"]: e for e in m["imputation_candidates"]["entries"]}
    assert ids["borrowing_across_batches"]["fits_profile"] and ids["within_batch"]["fits_profile"]


def test_nothing_is_filled(tmp_path):
    rng = np.random.default_rng(9)
    _, X = abundances(rng, 80)
    X[rng.random(X.shape) < 0.2] = np.nan
    a = prism.Audit(finished(tmp_path, cells(X), plates=PLATES))
    before = a.audit_matrix("A1B1").isna().sum().sum()
    a.missingness_profile()
    assert a.audit_matrix("A1B1").isna().sum().sum() == before
    assert a.findings["A12"].measures["A1B1"]["detection_curve"]["status"] in ("computed", "insufficient_data")


def test_no_missing_values(tmp_path):
    rng = np.random.default_rng(10)
    _, X = abundances(rng, 30)
    m = prism.Audit(finished(tmp_path, cells(X), plates=PLATES)).missingness_profile().measures["A1B1"]
    assert m["detection_curve"]["status"] == "not_applicable" and m["pattern"]["pattern_class"] == "none"
    assert m["imputation_candidates"]["entries"] == []
