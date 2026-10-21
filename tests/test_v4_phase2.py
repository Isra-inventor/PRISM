"""v4 stage 13: the Phase 2 hook (experimental imputation, kept out of the plan engine)."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from prism import registry
from prism.phase2 import imputation as imp

ROOT = Path(__file__).resolve().parent.parent


def nan_matrix(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(20, 1, size=(30, 1)) + rng.normal(0, 0.3, size=(30, 12))
    X[rng.random(X.shape) < 0.15] = np.nan
    return X


def test_min_prob_fills_only_missing_cells_seeded():
    X = nan_matrix()
    a = imp.impute_matrix(X, "min_prob", {"shift": 1.8, "scale": 0.3}, seed=3)
    b = imp.impute_matrix(X, "min_prob", {"shift": 1.8, "scale": 0.3}, seed=3)
    assert np.array_equal(a, b) and np.isfinite(a).all()
    obs = np.isfinite(X)
    assert np.array_equal(a[obs], X[obs])
    for j in range(X.shape[1]):
        m = ~obs[:, j]
        if m.any():
            assert a[m, j].mean() < np.nanmean(X[:, j])


def test_knn_and_within_batch():
    pytest.importorskip("sklearn")
    X = nan_matrix(1)
    out = imp.impute_matrix(X, "knn", {"k": 3})
    assert np.isfinite(out).all() and np.array_equal(out[np.isfinite(X)], X[np.isfinite(X)])
    batch = ["a"] * 6 + ["b"] * 6
    X2 = X.copy()
    X2[:, 6:] += 5
    out2 = imp.impute_matrix(X2, "knn", {"k": 3}, batch=batch, within_batch=True)
    filled_b = out2[:, 6:][~np.isfinite(X2[:, 6:])]
    assert filled_b.min() > 20                     # batch b values come from batch b only


def test_state_write_needs_approval(isolated, tmp_path):
    from prism import store
    import audit_fixtures as af
    st = store.create("p2")
    X, samples, cols, kinds = af.planted()
    af.add(st, tmp_path / "p", X=X, samples=samples, columns=cols, kinds=kinds)
    from prism.executor import state as S
    with pytest.raises(imp.ImputationError, match="approval"):
        imp.impute(st, "D1", S.root_id(st, "D1"), "min_prob")
    m, rec = imp.impute(st, "D1", S.root_id(st, "D1"), "min_prob", approval={"user": "u", "at": "t", "plan_entry": "none"})
    assert rec["experimental"] and rec["nan_after"] == 0 and m["counts"]["n_nan"] == 0
    assert S.head(st, "D1") == S.root_id(st, "D1")             # an experimental state never becomes HEAD


def test_unreachable_from_plan_executor_cli_api():
    from prism.executor import OPS
    assert not [o for o in OPS if "imput" in o]
    code = ("import sys\n"
            "import prism.plan.engine, prism.plan.facts, prism.executor, prism.cli, backend.study_api, backend.main\n"
            "assert 'prism.phase2' not in sys.modules and 'prism.phase2.imputation' not in sys.modules, "
            "[m for m in sys.modules if 'phase2' in m]\nprint('ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr
    for d in ("prism/plan", "prism/executor", "prism/audit", "backend"):
        for p in (ROOT / d).rglob("*.py"):
            assert "phase2" not in p.read_text(encoding="utf-8"), p
    assert "phase2" not in (ROOT / "prism" / "cli.py").read_text()


def test_methods_registry_seed():
    import yaml
    doc = yaml.safe_load((ROOT / "prism" / "phase2" / "methods_registry.yaml").read_text())
    by = {m["id"]: m for m in doc["methods"]}
    assert by["detection_or_hurdle_models"]["tolerates_missing"] == "native"
    assert by["pca_plsda_standard"]["tolerates_missing"] == "no"
    assert by["msstats_mofa"]["tolerates_missing"] is None
    assert len(doc["methods"]) == 3
    assert registry.check()["counted"]["phase2/methods_registry.yaml"] == 3
    assert registry.step("S6")["implemented"] is False
