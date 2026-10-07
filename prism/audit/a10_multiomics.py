"""A10. Multi-omics sample overlap and layer completeness (v3 §6.3): one session-level finding."""

from __future__ import annotations

import numpy as np

from . import stats
from .a02_scale import median_scaling
from .finding import Finding

AUDIT_ID, CODE = "A10", "multiomics_overlap"


def rv(Ka, Kb):
    """RV coefficient from two centered sample Gram matrices."""
    num = float(np.sum(Ka * Kb))
    den = float(np.sqrt(np.sum(Ka * Ka) * np.sum(Kb * Kb)))
    return num / den if den > 0 else float("nan")


def _centered_gram(Y):
    K = stats.gram(Y)
    return K - K.mean(0) - K.mean(1)[:, None] + K.mean()


def run_session(merged, ctxs, params):
    f = Finding(AUDIT_ID, CODE, dataset="session").feeds("multiomics_harmonization")
    f.d["assay"] = "session"
    M = f.measures
    units = [(c, u) for c in ctxs for u in c.units]
    if len(ctxs) < 2:
        return f.status("not_applicable", "one dataset in the session: nothing to compare across layers")
    ov = merged["overlap"]
    M["overlap"] = {"pairs": [{k: p[k] for k in ("a", "b", "n_shared", "n_only_a", "n_only_b", "jaccard")} for p in ov["pairs"]],
                    "n_unified": ov["n_unified"], "n_in_all": ov["n_in_all"], "per_dataset": ov["per_dataset"],
                    "open_id_suggestions": sum(1 for s in merged["suggestions"] if s["status"] == "open")}
    M["design_agreement"] = merged["design_agreement"]
    for p in ov["pairs"]:
        f.indicate("layer_overlap", p, a=p["a"], b=p["b"], n=p["n_shared"], only_a=p["n_only_a"], only_b=p["n_only_b"],
                   j=round(p["jaccard"], 3) if p["jaccard"] is not None else "n/a")
    layers = []
    for c, u in units:
        Y = u.Y
        with np.errstate(invalid="ignore"):
            tv = float(np.nansum(np.nanvar(Y, axis=1, ddof=1)))
        ms = median_scaling(u.X, params["median_scaling_mad"])
        layers.append({"dataset": c.dataset_id, "unit": u.unit_id, "label": u.label, "n_features": u.p, "n_samples": u.n,
                       "median_spread_mad_log2": ms["mad_log2_feature_medians"] if ms else None,
                       "total_variance_y": tv, "transform": u.transform})
    M["layers"] = layers
    # RV between layers on shared samples (unified IDs), free permutation of one layer's samples
    rvs = []
    members = merged["members"]
    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            (ca, ua), (cb, ub) = units[i], units[j]
            if ca.dataset_id == cb.dataset_id:
                continue
            ua_ids = {members_key(members, ca.dataset_id, s): k for k, s in enumerate(ua.samples)}
            ub_ids = {members_key(members, cb.dataset_id, s): k for k, s in enumerate(ub.samples)}
            shared = [x for x in ua_ids if x in ub_ids and x is not None]
            row = {"a": f"{ca.dataset_id}/{ua.unit_id}", "b": f"{cb.dataset_id}/{ub.unit_id}", "n_shared": len(shared)}
            Ya = stats.complete_features(ua.Y[:, [ua_ids[x] for x in shared]])
            Yb = stats.complete_features(ub.Y[:, [ub_ids[x] for x in shared]])
            if len(shared) < 4 or Ya.shape[0] < 2 or Yb.shape[0] < 2:
                rvs.append(dict(row, status="insufficient_data", reason="fewer than 4 shared samples or 2 complete features"))
                continue
            Ka, Kb = _centered_gram(Ya), _centered_gram(Yb)
            obs = rv(Ka, Kb)
            perm = stats.Permuter(np.arange(len(shared)), None, params["permutations"], params["exhaustive_max"],
                                  params["seed"], f"A10|{row['a']}|{row['b']}")
            null = np.array([rv(Ka, Kb[np.ix_(p_, p_)]) for p_ in perm.matrix()])
            p, pmin = perm.p_value(null, obs)
            rvs.append(dict(row, status="computed", rv=obs, p=p, p_min_attainable=pmin, **perm.describe()))
    M["rv"] = rvs
    f.method("overlap and agreement from the merge; per-layer facts; RV coefficient with permutation p",
             n_used=ov["n_unified"], permutations=params["permutations"], seed=params["seed"])
    return f


def members_key(members, did, sid):
    for u, m in members.items():
        if m.get(did) == sid:
            return u
    return None
