"""A1. Integrity and schema validation (v3 §6.3). On every sample of the unit, study or not:
it checks the files, and gates the other audits."""

from __future__ import annotations

import hashlib

import numpy as np

from . import stats
from .finding import Finding

AUDIT_ID, CODE = "A1", "integrity"


def _dupes(vectors, names):
    """Groups of names whose value vectors are exactly equal (NaN equal to NaN)."""
    groups = {}
    for v, n in zip(vectors, names):
        key = hashlib.sha256(np.where(np.isfinite(v), v, np.inf).astype("<f8").tobytes()).hexdigest()
        groups.setdefault(key, []).append(n)
    return [g for g in groups.values() if len(g) > 1]


def run(unit, ctx, params, results):
    f = Finding(AUDIT_ID, CODE, unit).feeds("gates_other_audits")
    X = unit.X_all
    samples = unit.all_samples
    keys = unit.feature_keys
    M = f.measures
    # 1. dimensions, unique keys and IDs, X vs M
    dims = []
    for b in unit.blocks:
        exp_f, exp_s = b.meta.get("n_features"), b.meta.get("n_samples")
        got_f, got_s = b.X.shape
        dims.append({"block": b.block_id, "schema": [exp_f, exp_s], "file": [int(got_f), int(got_s)],
                     "match": (exp_f in (None, got_f)) and (exp_s in (None, got_s))})
    M["dimensions"] = dims
    dup_keys = sorted({k for k in keys if keys.count(k) > 1}) if len(set(keys)) != len(keys) else []
    dup_samples = sorted({s for s in samples if samples.count(s) > 1}) if len(set(samples)) != len(samples) else []
    in_m = set(ctx.M)
    M["ids"] = {"duplicate_feature_keys": dup_keys[:100], "n_duplicate_feature_keys": len(dup_keys),
                "duplicate_sample_ids": dup_samples, "in_x_not_in_m": sorted(set(samples) - in_m),
                "in_m_not_in_x": sorted(in_m - set(samples)), "n_unparsable_cells": unit.n_unparsable}
    for d in dims:
        if not d["match"]:
            f.indicate("dimension_mismatch", d, block=d["block"], file=f"{d['file'][0]} x {d['file'][1]}",
                       schema=f"{d['schema'][0]} x {d['schema'][1]}")
    if dup_keys:
        f.indicate("duplicate_feature_keys", {"n": len(dup_keys), "examples": dup_keys[:5]}, n=len(dup_keys))
    if dup_samples:
        f.indicate("duplicate_sample_ids", {"samples": dup_samples}, n=len(dup_samples))
    if M["ids"]["in_x_not_in_m"] or M["ids"]["in_m_not_in_x"]:
        f.indicate("samples_x_vs_m", {k: M["ids"][k] for k in ("in_x_not_in_m", "in_m_not_in_x")},
                   a=len(M["ids"]["in_x_not_in_m"]), b=len(M["ids"]["in_m_not_in_x"]))
    if unit.n_unparsable:
        f.indicate("unparsable_cells", {"n": unit.n_unparsable}, n=unit.n_unparsable)

    # 2. cell counts
    fin = np.isfinite(X)
    M["cells"] = {"n": int(X.size), "missing": int(np.isnan(X).sum()), "infinite": int(np.isinf(X).sum()),
                  "negative": int((fin & (X < 0)).sum()), "zero": int((X == 0).sum())}
    if M["cells"]["infinite"]:
        f.indicate("infinite_values", {"n": M["cells"]["infinite"]}, n=M["cells"]["infinite"])

    # 3. constant features and samples, sparse features
    n_obs = fin.sum(1)
    with np.errstate(invalid="ignore"):
        sd_f = np.nanstd(np.where(fin, X, np.nan), axis=1)
        sd_s = np.nanstd(np.where(fin, X, np.nan), axis=0)
    const_f = [keys[i] for i in np.where((n_obs >= 2) & (sd_f == 0))[0]]
    const_s = [samples[j] for j in np.where((fin.sum(0) >= 2) & (sd_s == 0))[0]]
    few = [keys[i] for i in np.where(n_obs < 3)[0]]
    M["constant_features"] = {"n": len(const_f), "examples": const_f[:20]}
    M["constant_samples"] = const_s
    M["features_fewer_than_3_observed"] = {"n": len(few), "examples": few[:20]}
    if const_f:
        f.indicate("constant_features", {"n": len(const_f)}, n=len(const_f))
    if few:
        f.indicate("sparse_features", {"n": len(few)}, n=len(few))

    # 4. exact duplicates; near-duplicate samples on Y
    ds = _dupes(X.T, samples)
    dfeat = _dupes(X, keys)
    M["duplicate_samples"] = ds
    M["duplicate_features"] = {"n_groups": len(dfeat), "examples": dfeat[:20]}
    for g in ds:
        f.indicate("duplicate_samples", {"samples": g}, samples=", ".join(g))
    if dfeat:
        f.indicate("duplicate_features", {"n_groups": len(dfeat)}, n=len(dfeat))
    Y = unit.diagnostic(X)
    Yc = stats.complete_features(Y)
    near = []
    if Yc.shape[0] >= 3 and Yc.shape[1] >= 2:
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.corrcoef(Yc.T)
        iu = np.triu_indices(len(samples), 1)
        vals = C[iu]
        order = np.lexsort((iu[1], iu[0], -np.nan_to_num(vals, nan=-2)))[:5]
        near = [{"a": samples[iu[0][o]], "b": samples[iu[1][o]], "r": float(vals[o])} for o in order]
        for x in near:
            if x["r"] >= params["near_duplicate_r"] and not any(x["a"] in g and x["b"] in g for g in ds):
                f.indicate("near_duplicate_samples", x, a=x["a"], b=x["b"], r=round(x["r"], 6))
    M["top_sample_correlations"] = near

    # 5. metadata completeness (the sample table for this unit's samples)
    meta = []
    for c in ctx.M_columns:
        if c in ("unified_id", "present_in") or c.startswith(("sample_id@", "in_")):
            continue
        v = [(ctx.M.get(s) or {}).get(c, "") for s in samples]
        filled = [x for x in v if x]
        if not filled and not any(c == x or c.startswith(x + "@") for x in ctx.view.columns):
            continue                                   # another dataset's column, empty for these samples
        meta.append({"column": c, "audit_kind": ctx.col_kind.get(c), "share_missing": 1 - len(filled) / len(v) if v else None,
                     "n_levels": len(set(filled)), "constant": len(set(filled)) == 1 and len(filled) == len(v),
                     "all_unique": len(set(filled)) == len(filled) == len(v) and len(v) > 1})
    M["metadata"] = meta
    f.method("dimensions, keys, cell counts, constant and duplicate vectors (sha256), top sample correlations, "
             "metadata completeness", n_used=len(samples), transform=unit.diagnostic_rule(X))
    return f
