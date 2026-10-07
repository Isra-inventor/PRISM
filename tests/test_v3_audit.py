"""v3 stage 4: audit core (conventions, permutations, findings, parameters, overrides, manifest)
and A4 missingness, A8 repeated measures, A5 dimensionality."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from prism import store
from prism.audit import engine, overrides, params as params_mod, stats
from prism.audit.a04_missingness import floor_ties
from prism.audit.a08_repeated import icc1
import audit_fixtures as af

DATA = Path(__file__).parent / "data"


@pytest.fixture
def st(isolated):
    return store.create("audit")


def planted_session(st, tmp_path, **kw):
    X, samples, cols, kinds = af.planted(**kw)
    af.add(st, tmp_path / "planted", X=X, samples=samples, columns=cols, kinds=kinds, design=af.SUBJ_DESIGN)
    return X, samples, cols


def find(man, st, audit_id, dataset="D1"):
    i = next(i for i in man["findings"] if i["audit_id"] == audit_id and i["dataset"] == dataset)
    return engine.finding(st, man["run_id"], i["file"])


# ------------------------------------------------------------------ statistics


def test_ranks_spearman_kruskal_bh():
    assert stats.rankdata([3, 1, 2, 2]).tolist() == [4.0, 1.0, 2.5, 2.5]
    rho, n = stats.spearman([1, 2, 3, 4, 5], [5, 6, 7, 8, 7])
    assert n == 5 and abs(rho - 0.8207826816681233) < 1e-12
    r = stats.rankdata([1, 2, 3, 4, 5, 6])
    assert abs(stats.kruskal_h(r, np.array([0, 0, 0, 1, 1, 1]), 2) - 3.857142857142857) < 1e-12
    P = np.array([[0, 0, 0, 1, 1, 1], [1, 0, 1, 0, 1, 0]])
    assert np.allclose(stats.kruskal_h_rows(r, P, 2), [stats.kruskal_h(r, P[0], 2), stats.kruskal_h(r, P[1], 2)])
    assert stats.bh([0.01, 0.04, None, 0.03]) == pytest.approx([0.03, 0.04, None, 0.04])
    z = stats.modified_z([1, 1, 1, 1, 10])
    assert z[-1] > 3.5 and abs(z[0]) < 1                        # MAD = 0: mean-absolute-deviation form


def test_restricted_permutations():
    # constant within subject: whole subjects are permuted; exhaustive with few arrangements
    codes = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    subj = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    p = stats.Permuter(codes, subj)
    assert p.mode == "subjects" and p.exhaustive and p.n_distinct == 6
    M = p.matrix()
    assert len(M) == 6 and len({tuple(r) for r in M}) == 6
    assert all(len(set(r[[0, 1]])) == 1 and len(set(r[[2, 3]])) == 1 for r in M)   # subjects stay whole
    pv, pmin = p.p_value(np.arange(6.0), 5.0)
    assert pv == 1 / 6 and pmin == 1 / 6
    # varies within subject: permuted within subject only
    codes = np.array([0, 1, 0, 1, 0, 1])
    subj = np.array(["a", "a", "b", "b", "c", "c"])
    p = stats.Permuter(codes, subj)
    assert p.mode == "within" and p.n_distinct == 8
    assert all(sorted(r[[0, 1]].tolist()) == [0, 1] for r in p.matrix())
    # no subject: free; Monte Carlo when there are too many, seeded
    codes = np.array([0, 1] * 15)
    p = stats.Permuter(codes, None, n_perm=999, seed=0, key="k")
    assert p.mode == "free" and not p.exhaustive
    a, b = p.matrix(), stats.Permuter(codes, None, n_perm=999, seed=0, key="k").matrix()
    assert a.shape == (999, 30) and (a == b).all()
    assert p.p_value(np.zeros(999), 1.0) == (1 / 1000, 1 / 1000)


def test_floor_ties_and_icc_formulas():
    X = np.array([[1, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6], [2, 2, np.nan, np.nan, np.nan, 3.0]])
    tied, t, n_obs, S = floor_ties(X, 5)
    assert tied.tolist() == [True, False, False] and t[0] == 2 and S.sum() == 2   # row 3: < 5 observed
    # ICC(1) on a hand example: a = 3 subjects of sizes 2, 2, 3
    y = np.array([[1.0, 2.0, 5.0, 6.0, 9.0, 10.0, 11.0]])
    codes = np.array([0, 0, 1, 1, 2, 2, 2])
    icc, a, N = icc1(y, codes)
    means = [1.5, 5.5, 10.0]
    grand = (3 + 11 + 30) / 7
    msb = (2 * (1.5 - grand) ** 2 + 2 * (5.5 - grand) ** 2 + 3 * (10 - grand) ** 2) / 2
    msw = (0.25 * 2 + 0.25 * 2 + 1 + 0 + 1) / 4
    k0 = (7 - (4 + 4 + 9) / 7) / 2
    assert a[0] == 3 and N[0] == 7 and abs(icc[0] - (msb - msw) / (msb + (k0 - 1) * msw)) < 1e-12
    assert means


def test_params_yaml_and_overrides(st):
    d = params_mod.defaults()
    assert d["seed"] == 0 and d["permutations"] == 999 and d["rsd_levels"] == [20, 30] and d["near_duplicate_r"] == 0.9999
    p, changed = params_mod.resolve({"permutations": 199}, [{"kind": "param", "key": "seed", "value": 3}])
    assert p["permutations"] == 199 and p["seed"] == 3 and changed["seed"]["source"] == "override"
    with pytest.raises(params_mod.ParamError, match="Unknown audit parameter"):
        params_mod.resolve({"nope": 1})
    with pytest.raises(params_mod.ParamError, match="must be a number"):
        params_mod.resolve({"seed": "x"})
    with pytest.raises(overrides.OverrideError, match="role must be"):
        overrides.add(st, {"kind": "sample_role", "sample": "S1", "role": "boss"})
    e = overrides.add(st, {"kind": "sample_role", "sample": "S1", "role": "qc", "reason": "pooled QC"})
    assert e["override_id"] == "o1" and e["by"] == "user" and e["at"]
    assert overrides.load(st)[0]["role"] == "qc"


# ------------------------------------------------------------------ planted ground truth


def test_planted_ground_truth(st, tmp_path):
    X, samples, cols = planted_session(st, tmp_path)
    man = engine.run(st)
    a4 = find(man, st, "A4")
    m = a4["measures"]
    assert a4["status"] == "computed" and m["floor_ties"]["n_features"] == 20 and m["floor_ties"]["of"] == 400
    assert m["missing"]["n"] == int(np.isnan(X).sum())
    assert m["abundance_dependence"]["missing"]["rho"] < -0.3
    assert "abundance_dependent_missingness" in [i["code"] for i in a4["indicators"]]
    rows = {(r["variable"], r["response"]): r for r in m["associations"]["rows"]}
    day = rows[("run_day", "per-sample missing rate")]
    assert day["status"] == "computed" and day["q"] < 0.05 and day["scheme"] == "permuted within subject"
    plate = rows[("plate", "per-sample missing rate")]
    assert plate["scheme"] == "whole subjects permuted"          # plate is constant within subject
    assert plate["p_min_attainable"] > 0 and m["associations"]["n_tests"] == len(
        [r for r in m["associations"]["rows"] if r["status"] == "computed"])
    assert {t["variable"] for t in m["per_batch"]} == {"plate", "run_day"}

    a8 = find(man, st, "A8")
    m8 = a8["measures"]
    assert m8["design"]["n_subjects"] == 12 and m8["design"]["subjects_per_cluster_size"] == {"3": 12}
    assert m8["design"]["balanced"] and not m8["design"]["paired"]
    assert abs(m8["icc"]["raw"]["median"] - 0.8) < 0.1           # planted ICC
    assert m8["constant_within_subject"]["plate"] is True and m8["constant_within_subject"]["run_day"] is False
    assert {"inner": "animal", "outer": "plate"} in m8["nested"]
    assert "variable_constant_within_subject" in [i["code"] for i in a8["indicators"]]
    assert m8["time_grid"]["within_subject_gaps"] == [4] * 24 and m8["time_grid"]["regular"]

    a5 = find(man, st, "A5")
    m5 = a5["measures"]
    deff = 1 + (36 / 12 - 1) * m8["icc"]["raw"]["median"]
    assert m5["n"] == 36 and m5["p"] == 400 and m5["n_subjects"] == 12
    assert abs(m5["design_effect"]["deff"] - deff) < 1e-9 and abs(m5["design_effect"]["n_eff"] - 36 / deff) < 1e-9
    assert m5["smallest_cell"]["n"] == 9                          # week has 12, plate 9, run_day 18
    assert abs(m5["sparsity"]["share_missing"] - np.isnan(X).mean()) < 1e-9
    for f in (a4, a8, a5):
        for i in f["indicators"]:
            assert "diagnos" not in i["text"].lower() and "confirmed" not in i["text"].lower()


# ------------------------------------------------------------------ statuses


def test_statuses(st, tmp_path):
    rng = np.random.default_rng(0)
    samples = [f"S{i}" for i in range(8)]
    # integer counts: out of scope for Tier 1
    af.add(st, tmp_path / "counts", X=rng.integers(0, 50, size=(30, 8)).astype(float), samples=samples)
    # continuous, no subject: A8 needs a subject
    af.add(st, tmp_path / "nosubj", X=2.0 ** rng.normal(10, 1, size=(30, 8)), samples=samples)
    # two subjects only: association guard
    cols = {"animal": ["a", "a", "a", "a", "b", "b", "b", "b"], "batch": ["x", "y"] * 4}
    X = 2.0 ** rng.normal(10, 1, size=(30, 8))
    X[0:5, 1] = np.nan
    af.add(st, tmp_path / "twosubj", X=X, samples=samples, columns=cols, kinds={"animal": "subject_id", "batch": "batch"},
           design={"subject": {"source": "metadata_column", "column": "animal"}, "time": {"source": "none"}})
    man = engine.run(st)
    for aid in ("A4", "A8", "A5"):
        f = find(man, st, aid, "D1")
        assert f["status"] == "not_applicable" and "count" in f["status_reason"]
    a8 = find(man, st, "A8", "D2")
    assert a8["status"] == "insufficient_metadata" and a8["needs"] == ["subject"]
    a4 = find(man, st, "A4", "D3")
    (row,) = [r for r in a4["measures"]["associations"]["rows"] if r["variable"] == "batch"]
    assert row["status"] == "insufficient_data" and row["reason"] == "fewer than 3 subjects"
    assert find(man, st, "A8", "D3")["measures"]["icc"]["status"] == "insufficient_data"


# ------------------------------------------------------------------ contract


def _output_hashes(st):
    from prism.util import sha256_file
    return {str(p): sha256_file(p) for d in st.datasets for p in sorted(st.output_dir(d["dataset_id"]).iterdir())}


def test_contract_determinism_params_and_read_only(st, tmp_path):
    planted_session(st, tmp_path)
    before = _output_hashes(st)
    a = engine.run(st)
    b = engine.run(st)
    assert _output_hashes(st) == before                                 # no audit modifies output/
    assert a["findings_sha256"] == b["findings_sha256"] and a["params_sha256"] == b["params_sha256"]
    for i in a["findings"]:
        fa = (st.dir / "audit" / a["run_id"] / "findings" / i["file"]).read_bytes()
        assert fa == (st.dir / "audit" / b["run_id"] / "findings" / i["file"]).read_bytes()
    c = engine.run(st, params={"permutations": 199})
    assert c["params_sha256"] != a["params_sha256"] and c["scope"]["params_changed"]["permutations"]["value"] == 199
    cmp = engine.compare(st, a["run_id"], c["run_id"])
    assert cmp["params_changed"] == {"permutations": {"a": 999, "b": 199}} and not cmp["findings_identical"]
    assert engine.compare(st, a["run_id"], b["run_id"])["findings"] == []
    man = a
    assert set(man) >= {"prism_version", "git_hash", "params", "seed", "inputs", "overrides_sha256", "timings_s"}
    assert any(k.endswith("output/value_matrix_A1_B1.csv") for k in man["inputs"])
    led = [json.loads(l) for l in (st.dir / "audit" / a["run_id"] / "ledger.jsonl").read_text().splitlines()]
    assert [r["tool"] for r in led][-1] == "audit.run" and all(r["who"] == "user" and r["output_sha256"] for r in led)
    # factor selection
    d = engine.run(st, factors=["missingness"])
    assert {i["audit_id"] for i in d["findings"]} == {"A4"}
    with pytest.raises(engine.AuditError, match="Unknown audit"):
        engine.run(st, factors=["A12"])


def test_overrides_change_what_is_used(st, tmp_path):
    X, samples, cols = planted_session(st, tmp_path)
    overrides.add(st, {"kind": "exclude_from_audit", "sample": samples[0], "reason": "hemolysed"})
    overrides.add(st, {"kind": "sample_role", "sample": samples[1], "role": "qc"})
    overrides.add(st, {"kind": "param", "key": "permutations", "value": 99})
    man = engine.run(st)
    u = man["scope"]["units"][0]
    assert u["n_samples"] == 34 and u["excluded_by_override"] == [samples[0]] and u["non_study_samples"] == [samples[1]]
    assert man["params"]["permutations"] == 99 and len(man["scope"]["overrides_applied"]) == 3
    assert man["overrides_sha256"]
    assert find(man, st, "A5")["measures"]["n"] == 34


def test_audit_runs_with_the_uploads_deleted(client, flow, isolated):
    """A wizard dataset of a session: delete the upload copy and the Step 0 session; the audit
    still runs from the output folder alone."""
    r = client.post("/api/sessions", json={"name": "u"})
    sid = r.json()["session_id"]
    from conftest import fixture_bytes
    r = client.post("/api/upload", files={"file": ("C.csv", fixture_bytes("C_mzmine_feature_table.csv"))},
                    data={"study_session_id": sid})
    assert r.status_code == 200, r.text
    step0 = r.json()["session_id"]
    from conftest import Flow
    f = Flow.__new__(Flow)
    f.c, f.sid = client, step0
    r = client.post("/api/propose", json={"session_id": step0, "ai": True})
    f.draft, f.upload = r.json()["draft"], r.json()["session"]
    f.confirm_all_as_proposed()
    f.finalize()
    st = store.load(sid)
    assert st.datasets[0]["status"] == "confirmed"
    shutil.rmtree(str(st.dataset_dir("D1") / "upload"), ignore_errors=True)
    shutil.rmtree(str(isolated / "sessions" / step0))
    man = engine.run(st)
    assert {i["status"] for i in man["findings"]} <= {"computed", "insufficient_metadata", "not_applicable"}


def test_not_ready_session_is_refused(st, tmp_path):
    rng = np.random.default_rng(0)
    af.add(st, tmp_path / "a", X=2.0 ** rng.normal(5, 1, (5, 3)), samples=["S1", "S2", "S3"])
    af.add(st, tmp_path / "b", X=2.0 ** rng.normal(5, 1, (5, 3)), samples=["s1", "S2", "S3"])
    with pytest.raises(engine.AuditError, match="not ready for audit.*ID suggestion"):
        engine.run(st)


def test_cli_audit(st, tmp_path, capsys):
    from prism import cli
    planted_session(st, tmp_path)
    assert cli.main(["audit", "override", "--session", st.sid, "--kind", "param", "--key", "permutations",
                     "--value", "99"]) == 0
    assert cli.main(["audit", "run", "--session", st.sid, "--factor", "A4", "--param", "seed=1"]) == 0
    out = capsys.readouterr().out
    run_a = out.split()[1]
    assert "A4" in out and "floor_ties_present" in out
    assert cli.main(["audit", "run", "--session", st.sid, "--factor", "A4"]) == 0
    run_b = capsys.readouterr().out.split()[0]
    assert cli.main(["audit", "show", "--session", st.sid]) == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == run_b
    assert cli.main(["audit", "compare", "--session", st.sid, run_a, run_b]) == 0
    assert json.loads(capsys.readouterr().out)["params_changed"]["seed"] == {"a": 1, "b": 0}
    assert cli.main(["audit", "run", "--session", st.sid, "--factor", "A99"]) == 2


# ------------------------------------------------------------------ golden: the real files


@pytest.mark.skipif(not (DATA / "SomaExpr_common.csv").exists() or not (DATA / "MetaboExpr_common.csv").exists(),
                    reason="real data files not present in tests/data")
def test_golden_real_files(client, isolated):
    import time
    import real_data
    st = store.create("golden")
    for name in (real_data.SOMA, real_data.METAB):
        out = real_data.wizard_output(client, isolated, name)
        d = st.add_dataset(name, "wizard", "wizard_in_progress")
        store.publish_output(st, d["dataset_id"], out, {"mode": "wizard"})
    t = time.time()
    man = engine.run(st)
    assert time.time() - t < 60
    soma, metab = find(man, st, "A4", "D1")["measures"], find(man, st, "A4", "D2")["measures"]
    assert (soma["floor_ties"]["n_features"], soma["floor_ties"]["of"]) == (20, 11083)
    assert round(100 * soma["floor_ties"]["share"], 2) == 0.18
    assert (metab["floor_ties"]["n_features"], metab["floor_ties"]["of"]) == (397, 1174)
    assert round(100 * metab["floor_ties"]["share"], 1) == 33.8
    by = {s["stratum"]: (s["n_floor_tied"], s["n_features"]) for s in metab["by_stratum"]["strata"]}
    assert by == {"Amino Acid": (52, 220), "Cofactors and Vitamins": (12, 34), "Lipid": (150, 528), "Energy": (3, 10),
                  "Carbohydrate": (9, 26), "Nucleotide": (14, 39), "Peptide": (11, 35), "Xenobiotics": (65, 108),
                  "Partially Characterized Molecules": (5, 12), "unclassified": (76, 162)}   # 'Unnamed'
    for did in ("D1", "D2"):
        m8 = find(man, st, "A8", did)["measures"]
        assert m8["design"]["n_subjects"] == 8
        assert m8["design"]["subjects_per_cluster_size"] == {"1": 3, "4": 2, "5": 2, "6": 1}
        assert m8["time_grid"]["samples_per_time"] == {"0": 7, "4": 3, "8": 5, "16": 1, "72": 3, "128": 2, "136": 3,
                                                       "248": 2, "336": 1}
        assert m8["time_grid"]["within_subject_gaps"] == [4, 4, 4, 4, 4, 8, 8, 8, 56, 56, 64, 64, 64, 64, 88, 120,
                                                          120, 120, 128]
        assert find(man, st, "A5", did)["measures"]["n_subjects"] == 8
    a11 = find(man, st, "A11", "D2")["measures"]
    cof = next(s for s in a11["by_stratum"]["strata"] if s["stratum"] == "Cofactors and Vitamins")
    assert cof["flagged"] and round(cof["max"], 1) == 12301.6
    # reported back (v3 §9, "expected but unverified"): the median-scaling signature
    assert find(man, st, "A2", "D2")["measures"]["median_scaling"]["signature"] is True
    assert find(man, st, "A2", "D1")["measures"]["median_scaling"]["signature"] is False
    # declared vs observed: metabolomics (imputed: no, normalized: no) emits both; SomaScan no imputation one
    dvo = {(r["dataset"], r["item"]): r["status"] for r in man["declared_vs_observed"]}
    assert dvo[("D2", "imputed")] == "mismatch" and dvo[("D2", "normalized")] == "mismatch"
    assert dvo[("D1", "imputed")] == "consistent"
    assert "declared_not_imputed_floor_ties" not in [i["code"] for i in find(man, st, "A4", "D1")["indicators"]]
    a10 = engine.finding(st, man["run_id"], "A10__session.json")["measures"]
    assert a10["overlap"]["pairs"][0]["n_shared"] == 27 and a10["overlap"]["open_id_suggestions"] == 0


# ------------------------------------------------------------------ stage 5: A7, A11, A2, A3


def test_planted_batch_outliers_scale_distribution(st, tmp_path):
    X, samples, cols = planted_session(st, tmp_path, shift=True, outliers=True)
    man = engine.run(st, factors=["A2", "A3", "A7", "A11"])
    a7 = find(man, st, "A7")
    m7 = a7["measures"]
    s = {(x["batch"], x["design"]): x for x in m7["structure"]}
    assert s[("plate", "subject")]["code"] == "nested" and "'subject' is constant within 'plate'" in s[("plate", "subject")]["nesting"]
    assert s[("run_day", "time")]["code"] == "crossed_balanced"
    assert s[("plate", "time")]["code"] == "crossed_balanced"
    rows = m7["associations"]["rows"]
    perm_day = next(r for r in rows if r["test"] == "permanova" and r["variable"] == "run_day")
    assert perm_day["q"] < 0.05 and perm_day["r2"] > 0.1 and perm_day["scheme"] == "permuted within subject"
    pc_day = [r for r in rows if r["test"] == "eta_squared" and r["variable"] == "run_day" and r.get("q") is not None]
    assert min(r["q"] for r in pc_day) < 0.05
    perm_plate = next(r for r in rows if r["test"] == "permanova" and r["variable"] == "plate")
    assert perm_plate["scheme"] == "whole subjects permuted"
    codes = [i["code"] for i in a7["indicators"]]
    assert "batch_nested" in codes and "permanova_association" in codes
    assert len(m7["associations"]["rows"]) >= m7["associations"]["n_tests"] > 0
    assert len(a7["plot_data"]["pca"]["samples"]) == 36

    a11 = find(man, st, "A11")
    m11 = a11["measures"]
    # the outlier sample, and the sample holding the planted outlier cell (one huge cell moves it away
    # from the median profile)
    assert set(m11["samples"]["flagged"]) == {"A05_8", samples[5]}
    assert next(r for r in m11["samples"]["per_sample"] if r["sample"] == "A05_8")["flags"] == ["distance", "pc1"]
    top = m11["cells"]["top"][0]
    assert (top["feature"], top["sample"]) == ("f100", samples[5])
    (stratum,) = m11["by_stratum"]["strata"]
    assert stratum["feature"] == "f100" and stratum["flagged"] and stratum["max"] == pytest.approx(X[100, 5])

    a2 = find(man, st, "A2")
    assert a2["measures"]["classification"] == "continuous_linear" and not a2["measures"]["median_scaling"]["signature"]
    a3 = find(man, st, "A3")
    assert 0.8 < a3["measures"]["mean_sd_x"]["slope"] < 1.3                      # lognormal: SD grows with the mean
    lo, hi = a3["measures"]["mean_sd_x"]["ci95"]
    assert lo < a3["measures"]["mean_sd_x"]["slope"] < hi
    assert a3["measures"]["skewness"]["x"]["median"] > a3["measures"]["skewness"]["y"]["median"]


def test_scale_classes(st, tmp_path):
    rng = np.random.default_rng(2)
    s = [f"S{i}" for i in range(10)]
    af.add(st, tmp_path / "counts", X=rng.poisson(20, size=(40, 10)).astype(float), samples=s)
    P = rng.random((40, 10))
    af.add(st, tmp_path / "props", X=P, samples=s)
    C = rng.random((40, 10))
    af.add(st, tmp_path / "comp", X=C / C.sum(0) * 1e6, samples=s)
    L = rng.normal(0, 1, size=(40, 10))
    af.add(st, tmp_path / "signed", X=L, samples=s)
    M = 2.0 ** rng.normal(10, 1, size=(40, 10))
    af.add(st, tmp_path / "scaled", X=M / np.median(M, axis=1, keepdims=True), samples=s)
    man = engine.run(st, factors=["A2", "A3"])
    got = {d: find(man, st, "A2", d)["measures"]["classification"] for d in ("D1", "D2", "D3", "D4", "D5")}
    assert got == {"D1": "count_like", "D2": "proportion_like", "D3": "compositional_like", "D4": "continuous_signed",
                   "D5": "continuous_linear"}
    assert "out_of_scope_type" in [i["code"] for i in find(man, st, "A2", "D1")["indicators"]]
    assert find(man, st, "A3", "D1")["status"] == "not_applicable"
    assert find(man, st, "A2", "D5")["measures"]["median_scaling"]["signature"] is True
    assert find(man, st, "A3", "D4")["method"]["transform"]["transform"] == "none"


# ------------------------------------------------------------------ stage 6: A1, A6, A9, A10, declared vs observed


def test_integrity(st, tmp_path):
    rng = np.random.default_rng(4)
    X = 2.0 ** rng.normal(8, 1, size=(30, 8))
    X[:, 7] = X[:, 6]                        # duplicate sample column
    X[:, 5] = X[:, 4] * 1.0000001            # near-duplicate on log scale
    X[3] = 5.0                               # constant feature
    X[4, :6] = np.nan                        # fewer than 3 observed
    X[5, 0] = np.inf
    X[10] = X[11]                            # duplicate features
    samples = [f"S{i}" for i in range(8)]
    out = af.make_output(tmp_path / "integ", X=X, samples=samples)
    doc = json.loads((out / "schema.json").read_text())
    doc["assays"][0]["value_blocks"][0]["n_features"] = 31          # schema disagrees with the file
    (out / "schema.json").write_text(json.dumps(doc))
    lines = (out / "sample_metadata.csv").read_text().splitlines()
    (out / "sample_metadata.csv").write_text("\n".join(lines[:-1]) + "\n")   # S7 missing from the table
    d = st.add_dataset("integ", "output_folder", "imported_awaiting_confirm")
    store.publish_output(st, d["dataset_id"], out, {"mode": "output_folder"})
    man = engine.run(st, factors=["A1"])
    f = find(man, st, "A1")
    m = f["measures"]
    codes = {i["code"] for i in f["indicators"]}
    assert {"dimension_mismatch", "infinite_values", "constant_features", "sparse_features", "duplicate_samples",
            "duplicate_features", "near_duplicate_samples", "samples_x_vs_m"} <= codes
    assert m["dimensions"][0]["schema"] == [31, 8] and m["dimensions"][0]["file"] == [30, 8]
    assert m["duplicate_samples"] == [["S6", "S7"]] and m["ids"]["in_x_not_in_m"] == ["S7"]
    assert m["cells"]["infinite"] == 1 and "f3" in m["constant_features"]["examples"]
    assert m["top_sample_correlations"][0]["r"] >= 0.9999


def test_noise_qc_roles_and_drift(st, tmp_path):
    rng = np.random.default_rng(5)
    n, p = 16, 200
    samples = [f"S{i:02d}" for i in range(n)]
    base = rng.normal(10, 2, size=(p, 1))
    L = base + rng.normal(0, 1, size=(p, n))
    qc = [0, 5, 10, 15]
    L[:, qc] = base + rng.normal(0, 0.1, size=(p, len(qc)))          # QC: ~7% RSD
    order = np.arange(n)
    L += 0.08 * order[None, :]                                        # drift with run order
    is_study = [i not in qc for i in range(n)]
    suspect = {"Type": {"values": ["ctrl" if i < 10 else "protein" for i in range(p)], "marks_rows_as_suspect": True,
                        "flagged_values": ["ctrl"]},
               "QC_CV": {"values": [str(round(float(x), 3)) for x in rng.random(p)]}}
    af.add(st, tmp_path / "qc", X=2.0 ** L, samples=samples, is_study=is_study,
           columns={"run": [str(i + 1) for i in order]}, kinds={"run": "run_order"}, annotations=suspect)
    man = engine.run(st, factors=["A6"])
    f = find(man, st, "A6")
    assert f["measures"]["qc_rsd"]["status"] == "insufficient_metadata" and "sample roles" in f["needs"]
    assert "roles_needed" in [i["code"] for i in f["indicators"]]
    for i in qc:
        overrides.add(st, {"kind": "sample_role", "sample": samples[i], "role": "qc"})
    man = engine.run(st, factors=["A6"])
    f = find(man, st, "A6")
    m = f["measures"]
    assert m["roles"]["counts"] == {"qc": 4, "study": 12}
    # QC noise alone is ~7% RSD; the planted drift spreads the 4 QC runs over 1.2 log2 units
    assert 20 < m["qc_rsd"]["summary"]["median"] < 50 and m["qc_rsd"]["share_above"]["20"] > 0.9
    assert m["run_order"]["spearman_median_y"] > 0.9 and "run_order_drift" in [i["code"] for i in f["indicators"]]
    assert m["run_order"]["qc_feature_trend"]["summary"]["median"] > 0.5
    assert m["suspect_rows"]["flagged"]["n_features"] == 10 and m["suspect_rows"]["columns"] == ["Type"]
    assert [v["column"] for v in m["vendor_qc_columns"]] == ["QC_CV"]
    assert len(m["per_sample"]) == 12


def test_source_heterogeneity(st, tmp_path):
    rng = np.random.default_rng(6)
    n = 20
    tissue = ["plasma" if i % 2 else "serum" for i in range(n)]
    L = rng.normal(8, 1, size=(100, n)) + np.array([1.5 if t == "serum" else 0 for t in tissue])[None, :] * \
        (np.arange(100) < 50)[:, None]
    samples = [f"S{i}" for i in range(n)]
    af.add(st, tmp_path / "src", X=2.0 ** L, samples=samples, columns={"matrix": tissue, "site": ["A"] * n},
           kinds={"matrix": "sample_type", "site": "sample_type"})
    af.add(st, tmp_path / "none", X=2.0 ** L, samples=[f"T{i}" for i in range(n)])
    man = engine.run(st, factors=["A9"])
    f = find(man, st, "A9", "D1")
    v = {x["variable"]: x for x in f["measures"]["variables"]}
    assert v["site"]["homogeneous"] and not v["matrix"]["homogeneous"]
    perm = next(r for r in f["measures"]["associations"]["rows"] if r["test"] == "permanova")
    assert perm["q"] < 0.05 and perm["r2"] > 0.2
    assert {"single_source", "source_association"} <= {i["code"] for i in f["indicators"]}
    g = find(man, st, "A9", "D2")
    assert g["status"] == "insufficient_metadata" and g["needs"] == ["sample source"]


def test_multiomics_layers(st, tmp_path):
    rng = np.random.default_rng(7)
    n = 12
    shared = rng.normal(0, 1, size=(1, n))
    samples = [f"S{i}" for i in range(n)]
    A = 2.0 ** (8 + shared * rng.normal(1, 0.2, size=(80, 1)) + rng.normal(0, 0.3, size=(80, n)))
    B = 2.0 ** (5 + shared * rng.normal(1, 0.2, size=(40, 1)) + rng.normal(0, 0.3, size=(40, n)))
    af.add(st, tmp_path / "prot", X=A, samples=samples)
    af.add(st, tmp_path / "metab", X=B[:, 2:], samples=samples[2:] + [], family="metabolomics")
    man = engine.run(st, factors=["A10"])
    f = engine.finding(st, man["run_id"], "A10__session.json")
    m = f["measures"]
    assert m["overlap"]["pairs"][0]["n_shared"] == 10 and m["overlap"]["pairs"][0]["n_only_a"] == 2
    (r,) = m["rv"]
    assert r["n_shared"] == 10 and r["rv"] > 0.5 and r["p"] < 0.05
    assert [x["n_features"] for x in m["layers"]] == [80, 40]
    st2 = store.create("one")
    af.add(st2, tmp_path / "x" / "only", X=A, samples=samples)
    man = engine.run(st2, factors=["A10"])
    assert engine.finding(st2, man["run_id"], "A10__session.json")["status"] == "not_applicable"


def test_declared_vs_observed(st, tmp_path):
    rng = np.random.default_rng(8)
    s = [f"S{i}" for i in range(10)]
    M_ = 2.0 ** rng.normal(10, 1, size=(60, 10))
    M_[:20, :3] = M_[:20].min(1, keepdims=True)                       # floors
    scaled = M_ / np.median(M_, axis=1, keepdims=True)
    no = {"imputed": {"answer": "no"}, "normalized": {"answer": "no"}, "log_transformed": {"answer": "no"}}
    af.add(st, tmp_path / "a", X=scaled, samples=s, history=no)
    af.add(st, tmp_path / "b", X=np.log2(M_) - 10, samples=s, history=no)          # signed, log-like
    yes = {"imputed": {"answer": "yes"}, "normalized": {"answer": "yes"}, "log_transformed": {"answer": "not_sure"}}
    af.add(st, tmp_path / "c", X=scaled, samples=s, history=yes)
    man = engine.run(st, factors=["A4"])
    rows = {(r["dataset"], r["item"]): r["status"] for r in man["declared_vs_observed"]}
    assert rows[("D1", "imputed")] == "mismatch" and rows[("D1", "normalized")] == "mismatch"
    assert rows[("D1", "log_transformed")] == "consistent"
    assert rows[("D2", "log_transformed")] == "mismatch"
    assert rows[("D3", "imputed")] == "consistent" and rows[("D3", "log_transformed")] == "not_declared"
    codes = {i["code"] for i in find(man, st, "A4", "D1")["indicators"]}
    assert {"declared_not_imputed_floor_ties", "declared_not_normalized_signature"} <= codes
    assert "declared_not_logged_log_like" in {i["code"] for i in find(man, st, "A4", "D2")["indicators"]}
