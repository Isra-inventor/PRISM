"""A8. Repeated-measures, paired and nested structure (v3 §6.3)."""

from __future__ import annotations

import numpy as np

from . import stats
from .context import is_num
from .finding import Finding

AUDIT_ID, CODE = "A8", "repeated_measures"


def icc1(Y, subj_codes):
    """Feature-level ICC(1), unbalanced one-way random effects, over subjects with >= 2 observed
    samples for that feature. -> (icc[p], a[p], N[p]) with NaN where a < 3 or N - a < 3."""
    p = Y.shape[0]
    k = int(subj_codes.max()) + 1 if len(subj_codes) else 0
    ok = np.isfinite(Y)
    Z = np.where(ok, Y, 0.0)
    onehot = np.zeros((len(subj_codes), k))
    onehot[np.arange(len(subj_codes)), subj_codes] = 1
    C = ok.astype(float) @ onehot             # p x k observed counts
    Ssum = Z @ onehot
    use = C >= 2
    Cu = np.where(use, C, 0)
    Su = np.where(use, Ssum, 0)
    a = use.sum(1)
    N = Cu.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(use, Su / np.maximum(Cu, 1), 0)
        grand = Su.sum(1) / np.maximum(N, 1)
        msb = (Cu * (means - grand[:, None]) ** 2).sum(1) / np.maximum(a - 1, 1)
        # within: sum over used samples of (y - subject mean)^2
        sample_use = use @ onehot.T > 0        # p x n: sample's subject is used for this feature
        mean_of_sample = means @ onehot.T
        dev = np.where(ok & sample_use, Y - mean_of_sample, 0)
        msw = (dev ** 2).sum(1) / np.maximum(N - a, 1)
        k0 = (N - (Cu ** 2).sum(1) / np.maximum(N, 1)) / np.maximum(a - 1, 1)
        icc = (msb - msw) / (msb + (k0 - 1) * msw)
    valid = (a >= 3) & (N - a >= 3) & np.isfinite(icc)
    icc = np.where(valid, icc, np.nan)
    return icc, a, N


def run(unit, ctx, params, results):
    f = Finding(AUDIT_ID, CODE, unit).feeds("within_between_subject_handling", "timepoint_spacing", "A5")
    if unit.out_of_scope:
        return f.status("not_applicable", f"Tier 1 does not audit {unit.out_of_scope} ({unit.scale}).")
    if not ctx.subject:
        return f.status("insufficient_metadata", "no subject is defined (Step 0 design: subject source 'none')",
                        needs=["subject"])
    M = f.measures
    samples = unit.samples
    subj = [ctx.subject["values"].get(s, "") for s in samples]
    have = [i for i, s in enumerate(subj) if s]
    M["n_samples"], M["n_samples_without_subject"] = len(samples), len(samples) - len(have)
    by = {}
    for i in have:
        by.setdefault(subj[i], []).append(i)
    sizes = {s: len(ix) for s, ix in by.items()}
    dist = {}
    for v in sizes.values():
        dist[v] = dist.get(v, 0) + 1
    M["design"] = {"subject_column": ctx.subject["column"], "n_subjects": len(by),
                   "subjects_per_cluster_size": {str(k): dist[k] for k in sorted(dist)},
                   "samples_per_subject": {s: sizes[s] for s in sorted(sizes)},
                   "n_singletons": dist.get(1, 0), "balanced": len(dist) == 1,
                   "paired": set(dist) == {2}, "repeated": any(v > 1 for v in sizes.values())}
    if M["design"]["repeated"]:
        f.indicate("repeated_measures", {"n_subjects": len(by), "cluster_sizes": M["design"]["subjects_per_cluster_size"]},
                   n_subjects=len(by), max_size=max(sizes.values()), n_singletons=dist.get(1, 0))
    if M["design"]["paired"]:
        f.indicate("paired_design", {"n_subjects": len(by)})

    # 2. time grid
    if ctx.time:
        tv = [ctx.time["values"].get(s, "") for s in samples]
        per_time = {}
        subj_time = {}
        for i, t in enumerate(tv):
            if not t:
                continue
            per_time.setdefault(t, set()).add(subj[i] or f"_s{i}")
            subj_time.setdefault(t, 0)
            subj_time[t] += 1
        order = sorted(per_time, key=lambda x: (not is_num(x), float(x) if is_num(x) else 0, x))
        grid = {"time_column": ctx.time["column"], "unit": ctx.time_unit,
                "samples_per_time": {t: subj_time[t] for t in order},
                "subjects_per_time": {t: len(per_time[t]) for t in order}}
        numeric = all(is_num(t) for t in order)
        if numeric:
            gaps, spans = [], {}
            for s, ix in sorted(by.items()):
                ts = sorted(float(tv[i]) for i in ix if tv[i])
                if len(ts) >= 2:
                    spans[s] = ts[-1] - ts[0]
                    gaps += [b - a for a, b in zip(ts, ts[1:])]
            gaps.sort()
            grid.update(within_subject_gaps=[_n(g) for g in gaps], span_per_subject={s: _n(v) for s, v in spans.items()},
                        gap_min=_n(min(gaps)) if gaps else None, gap_median=_n(float(np.median(gaps))) if gaps else None,
                        gap_max=_n(max(gaps)) if gaps else None, regular=len(set(gaps)) <= 1 if gaps else None)
            if gaps and len(set(gaps)) > 1:
                f.indicate("irregular_time_grid", {"gaps": grid["within_subject_gaps"]}, min_gap=_n(min(gaps)),
                           max_gap=_n(max(gaps)), unit=ctx.time_unit or "time units")
        M["time_grid"] = grid
    else:
        M["time_grid"] = None
        f.status("computed", needs=["time"])

    # 3. constant within subject, nesting
    cols = [c for c in ctx.M_columns if c not in ("unified_id", "present_in") and not c.startswith(("sample_id@", "in_"))
            and c not in ("sample_label",)]
    multi = [ix for ix in by.values() if len(ix) >= 2]
    const = {}
    catvals = {}
    for c in cols:
        vals = ctx.column(c)
        v = [vals.get(s, "") for s in samples]
        if not any(v):
            continue
        cw = []
        for ix in multi:
            seen = {v[i] for i in ix if v[i]}
            if seen:
                cw.append(len(seen) == 1)
        const[c] = all(cw) if cw else None
        levels = {x for x in v if x}
        if 2 <= len(levels) <= params["categorical_max_levels"] or c == ctx.subject["column"]:
            catvals[c] = v
    M["constant_within_subject"] = const
    nested = []
    names = sorted(catvals)
    for a_ in names:
        for b_ in names:
            if a_ == b_:
                continue
            va, vb = catvals[a_], catvals[b_]
            mp = {}
            okk = True
            for x, y in zip(va, vb):
                if x and y:
                    if mp.setdefault(x, y) != y:
                        okk = False
                        break
            if okk and len(set(mp.values())) >= 2 and len(set(mp)) > len(set(mp.values())):
                nested.append({"inner": a_, "outer": b_})
    M["nested"] = nested
    batch_cols = {c for c in const if ctx.col_kind.get(c) in ("batch", "run_order")} | \
        {o["column"] for o in ctx.overrides if o["kind"] == "batch_variable"}
    for c, v in sorted(const.items()):
        if v and c in batch_cols and c in catvals and len({x for x in catvals[c] if x}) >= 2:
            f.indicate("variable_constant_within_subject", {"variable": c}, variable=c)

    # 4. ICC(1) on Y
    Y = unit.Y
    codes = {s: j for j, s in enumerate(sorted(by))}
    idx = np.array(have, dtype=int)
    sc = np.array([codes[subj[i]] for i in have], dtype=int)
    n_multi = sum(1 for ix in by.values() if len(ix) >= 2)
    N_all = sum(len(ix) for ix in by.values() if len(ix) >= 2)
    if n_multi < 3 or N_all - n_multi < 3:
        M["icc"] = {"status": "insufficient_data", "reason": f"{n_multi} subject(s) with 2 or more samples "
                                                             "(needs a >= 3 and N - a >= 3)"}
    else:
        icc, a, N = icc1(Y[:, idx], sc)
        valid = np.isfinite(icc)
        tr = np.maximum(icc[valid], 0)
        M["icc"] = {"status": "computed", "n_features": int(valid.sum()), "of": int(len(icc)),
                    "raw": stats.summary(icc), "truncated_at_0": stats.summary(tr),
                    "iqr": [float(np.quantile(icc[valid], 0.25)), float(np.quantile(icc[valid], 0.75))] if valid.any() else None,
                    "share_above": {"cut": params["icc_indicator"],
                                    "share": float((icc[valid] > params["icc_indicator"]).mean()) if valid.any() else None},
                    "n_subjects_used": n_multi, "n_samples_used": N_all}
        f.plot["icc_histogram"] = stats.histogram(icc[valid], 30)
        share = M["icc"]["share_above"]["share"]
        if share:
            f.indicate("high_icc", {"share": share, "n_features": int(valid.sum())}, pct=round(100 * share, 1),
                       cut=params["icc_indicator"])
        results.setdefault(unit.dataset_id, {}).setdefault(unit.unit_id, {})["median_icc"] = M["icc"]["raw"]["median"]

    # 5. mean Pearson r, same vs different subject (features observed in every used sample)
    Ys = Y[:, idx]
    full = np.isfinite(Ys).all(1)
    if full.sum() >= 3 and len(idx) >= 3:
        C = np.corrcoef(Ys[full].T)
        same, diff = [], []
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                (same if sc[i] == sc[j] else diff).append(C[i, j])
        M["sample_correlation"] = {"n_features": int(full.sum()), "same_subject_mean_r": float(np.mean(same)) if same else None,
                                   "different_subject_mean_r": float(np.mean(diff)) if diff else None,
                                   "n_same_pairs": len(same), "n_different_pairs": len(diff)}
        if same and diff:
            d = float(np.mean(same) - np.mean(diff))
            M["sample_correlation"]["difference"] = d
            if d > 0:
                f.indicate("within_subject_correlation", {"same": float(np.mean(same)), "different": float(np.mean(diff))},
                           same=round(float(np.mean(same)), 3), diff=round(float(np.mean(diff)), 3))
    else:
        M["sample_correlation"] = None
    f.plot["timeline"] = [{"sample": samples[i], "subject": subj[i],
                           "time": ctx.time["values"].get(samples[i], "") if ctx.time else None} for i in range(len(samples))]
    f.method("cluster sizes, time grid, nesting, ICC(1) one-way random effects (unbalanced), within-subject r",
             n_used=len(have), transform=unit.transform)
    return f


def _n(x):
    return int(x) if float(x).is_integer() else float(x)
