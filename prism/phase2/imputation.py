"""EXPERIMENTAL imputation interface (Build Request v4, section 5.9). Not part of any Phase 1 plan:
the plan engine, the executor's op registry, the CLI and the API do not import this module.

    impute(st, did, state_id, method, params, within_batch=False, approval=...) -> (new_state_meta, op_record)

Two reference implementations:
- min_prob: per sample, draws from a Gaussian centred `shift` SDs below the sample's observed mean
  with SD `scale` x the sample's observed SD (seeded; parameters in the op record);
- knn: scikit-learn KNNImputer, feature-wise (a feature's missing values come from the most similar
  features), optionally within batch.
Only missing cells are written; observed values never change.
"""

from __future__ import annotations

import numpy as np

EXPERIMENTAL = True
METHODS = ("min_prob", "knn")


class ImputationError(Exception):
    pass


def _min_prob(X, params, rng):
    out = X.copy()
    shift, scale = float(params.get("shift", 1.8)), float(params.get("scale", 0.3))
    for j in range(X.shape[1]):
        obs = X[np.isfinite(X[:, j]), j]
        miss = ~np.isfinite(X[:, j])
        if not miss.any() or obs.size < 2:
            continue
        mu, sd = obs.mean(), obs.std(ddof=1)
        out[miss, j] = rng.normal(mu - shift * sd, scale * sd, size=int(miss.sum()))
    return out


def _knn(X, params):
    try:
        from sklearn.impute import KNNImputer
    except Exception:
        raise ImputationError("knn needs scikit-learn (not installed).")
    k = int(params.get("k", 5))
    imp = KNNImputer(n_neighbors=k, keep_empty_features=True) if _keeps_empty() else KNNImputer(n_neighbors=k)
    out = imp.fit_transform(X)                  # rows = features: neighbours are features
    if out.shape != X.shape:
        raise ImputationError("knn dropped all-missing columns; nothing written.")
    return out


def _keeps_empty():
    import inspect
    from sklearn.impute import KNNImputer
    return "keep_empty_features" in inspect.signature(KNNImputer.__init__).parameters


def impute_matrix(X, method, params=None, batch=None, within_batch=False, seed=0):
    """-> imputed copy of X (features x samples). Observed cells are unchanged."""
    if method not in METHODS:
        raise ImputationError(f"Unknown method '{method}'. Experimental methods: {', '.join(METHODS)}.")
    params = dict(params or {})
    rng = np.random.default_rng([int(seed), 5, 9])
    X = np.asarray(X, float)
    groups = [np.arange(X.shape[1])]
    if within_batch:
        if batch is None:
            raise ImputationError("within_batch needs a batch label per sample.")
        b = np.asarray([str(x) for x in batch])
        groups = [np.where(b == lv)[0] for lv in sorted(set(b.tolist()))]
    out = X.copy()
    for g in groups:
        sub = X[:, g]
        out[:, g] = _min_prob(sub, params, rng) if method == "min_prob" else _knn(sub, params)
    obs = np.isfinite(X)
    if not np.array_equal(out[obs], X[obs]):
        raise ImputationError("an observed value changed; nothing written.")
    return out


def impute(st, did, state_id, method, params=None, within_batch=False, batch=None, approval=None, seed=0):
    """Write an imputed child state (experimental). -> (new state meta, op_record)."""
    from ..executor import state as state_mod
    from ..util import now_iso
    if not approval or not approval.get("user"):
        raise ImputationError("Nothing runs without approval.")
    data, pm = state_mod.load_data(st, did, state_id)
    child = data.copy()
    before = int(sum(np.isnan(b.X).sum() for b in child.blocks))
    for b in child.blocks:
        bl = [batch.get(s, "") for s in b.sample_ids] if isinstance(batch, dict) else batch
        b.X = impute_matrix(b.X, method, params, bl, within_batch, seed)
    rec = {"op": f"impute_experimental:{method}", "step": "S6", "params": dict(params or {}, within_batch=within_batch),
           "seed": seed, "approval": dict(approval), "experimental": True, "nan_before": before,
           "nan_after": int(sum(np.isnan(b.X).sum() for b in child.blocks)), "executed_at": now_iso()}
    if pm["state_id"] == state_mod.root_id(st, did):
        state_mod.ensure_root(st, did)
    m = state_mod.write_state(st, did, pm, child, rec)
    return m, rec
