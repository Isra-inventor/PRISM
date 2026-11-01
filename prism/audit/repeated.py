"""S5 + S10 / A8: repeated-measures, paired and nested structure. Ported from the v4 build.

Needs a subject (step 8). On the audit set:
 1. cluster sizes: subjects, samples per subject, singletons, balanced, paired (every subject has 2)
 2. the time grid (when the design has a time): samples and subjects per time point; for numeric
    times, the gaps within subjects (irregular spacing is reported)
 3. which sample variables are constant within every subject, and which variables are nested in
    which (e.g. subjects nested in plates: a batch correction that ignores this can remove subject
    effects); a batch constant within subject is reported
 4. ICC(1) per protein (one-way random effects, unbalanced): the share of variation explained by the
    subject; the median ICC feeds the effective sample size in A5
 5. mean Pearson r between samples of the same subject against different subjects (pairwise-complete)
"""

from __future__ import annotations

import numpy as np

from . import bridge, stats
from . import diagnostic as diagnostic_matrix
from .bridge import is_num
from .finding import Finding


def icc1(Y, subj_codes):
    """Protein-level ICC(1), unbalanced one-way random effects, over subjects with >= 2 observed
    samples for that protein. -> (icc[p], a[p], N[p]) with NaN where a < 3 or N - a < 3."""
    k = int(subj_codes.max()) + 1 if len(subj_codes) else 0
    ok = np.isfinite(Y)
    Z = np.where(ok, Y, 0.0)
    onehot = np.zeros((len(subj_codes), k))
    onehot[np.arange(len(subj_codes)), subj_codes] = 1
    C = ok.astype(float) @ onehot
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
        sample_use = use @ onehot.T > 0
        mean_of_sample = means @ onehot.T
        dev = np.where(ok & sample_use, Y - mean_of_sample, 0)
        msw = (dev ** 2).sum(1) / np.maximum(N - a, 1)
        k0 = (N - (Cu ** 2).sum(1) / np.maximum(N, 1)) / np.maximum(a - 1, 1)
        icc = (msb - msw) / (msb + (k0 - 1) * msw)
    valid = (a >= 3) & (N - a >= 3) & np.isfinite(icc)
    return np.where(valid, icc, np.nan), a, N


def _n(x):
    return int(x) if float(x).is_integer() else float(x)


def run(audit) -> Finding:
    f = Finding("A8", "Repeated measures", step="S5", feeds=["S5 design constraints", "A5 effective sample size",
                                                             "S10 confounding and design"])
    params = audit.params
    d = audit.schema.design()
    time_unit = d.get("time_unit")

    def block(unit, f):
        cols = bridge.column_values(audit, unit.samples)
        if "subject" not in cols:
            f.set_status("insufficient_metadata", "no subject is defined (step 8 design: subject source 'none').",
                         needs=["subject"])
            return
        M = f.measures
        samples = unit.samples
        subj = cols["subject"][1]
        have = [i for i, s in enumerate(subj) if s]
        M["n_samples"], M["n_samples_without_subject"] = len(samples), len(samples) - len(have)
        by = {}
        for i in have:
            by.setdefault(subj[i], []).append(i)
        sizes = {s: len(ix) for s, ix in by.items()}
        dist = {}
        for v in sizes.values():
            dist[v] = dist.get(v, 0) + 1
        M["design"] = {"n_subjects": len(by), "subjects_per_cluster_size": {str(k): dist[k] for k in sorted(dist)},
                       "samples_per_subject": {s: sizes[s] for s in sorted(sizes)}, "n_singletons": dist.get(1, 0),
                       "balanced": len(dist) == 1, "paired": set(dist) == {2},
                       "repeated": any(v > 1 for v in sizes.values())}
        if M["design"]["repeated"]:
            f.indicate("repeated_measures", {"n_subjects": len(by)}, n_subjects=len(by), max_size=max(sizes.values()),
                       n_singletons=dist.get(1, 0))
        if M["design"]["paired"]:
            f.indicate("paired_design", {"n_subjects": len(by)})

        if "time" in cols:
            tv = cols["time"][1]
            per_time, subj_time = {}, {}
            for i, t in enumerate(tv):
                if not t:
                    continue
                per_time.setdefault(t, set()).add(subj[i] or f"_s{i}")
                subj_time[t] = subj_time.get(t, 0) + 1
            order = sorted(per_time, key=lambda x: (not is_num(x), float(x) if is_num(x) else 0, x))
            grid = {"unit": time_unit, "samples_per_time": {t: subj_time[t] for t in order},
                    "subjects_per_time": {t: len(per_time[t]) for t in order}}
            if order and all(is_num(t) for t in order):
                gaps, spans = [], {}
                for s, ix in sorted(by.items()):
                    ts = sorted(float(tv[i]) for i in ix if tv[i])
                    if len(ts) >= 2:
                        spans[s] = ts[-1] - ts[0]
                        gaps += [b - a for a, b in zip(ts, ts[1:])]
                gaps.sort()
                grid.update(within_subject_gaps=[_n(g) for g in gaps], gap_min=_n(min(gaps)) if gaps else None,
                            gap_median=_n(float(np.median(gaps))) if gaps else None,
                            gap_max=_n(max(gaps)) if gaps else None, regular=len(set(gaps)) <= 1 if gaps else None)
                if gaps and len(set(gaps)) > 1:
                    f.indicate("irregular_time_grid", {"gaps": grid["within_subject_gaps"]}, min_gap=_n(min(gaps)),
                               max_gap=_n(max(gaps)), unit=time_unit or "time units")
            M["time_grid"] = grid
        else:
            M["time_grid"] = None

        multi = [ix for ix in by.values() if len(ix) >= 2]
        const, catvals = {}, {}
        for c, (kind, v) in cols.items():
            if not any(v):
                continue
            cw = [len({v[i] for i in ix if v[i]}) == 1 for ix in multi if any(v[i] for i in ix)]
            const[c] = all(cw) if cw else None
            levels = {x for x in v if x}
            if 2 <= len(levels) <= params["categorical_max_levels"] or c == "subject":
                catvals[c] = v
        M["constant_within_subject"] = const
        nested = []
        names = sorted(catvals)
        for a_ in names:
            for b_ in names:
                if a_ == b_:
                    continue
                mp, okk = {}, True
                for x, y in zip(catvals[a_], catvals[b_]):
                    if x and y and mp.setdefault(x, y) != y:
                        okk = False
                        break
                if okk and len(set(mp.values())) >= 2 and len(set(mp)) > len(set(mp.values())):
                    nested.append({"inner": a_, "outer": b_})
        M["nested"] = nested
        for c, v in sorted(const.items()):
            if v and cols[c][0] in ("batch", "run_order") and c in catvals and len({x for x in catvals[c] if x}) >= 2:
                f.indicate("variable_constant_within_subject", {"variable": c}, variable=c)

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
            M["icc"] = {"status": "computed", "n_proteins": int(valid.sum()), "of": int(len(icc)),
                        "raw": stats.summary(icc), "truncated_at_0": stats.summary(np.maximum(icc[valid], 0)),
                        "share_above": {"cut": params["icc_indicator"],
                                        "share": float((icc[valid] > params["icc_indicator"]).mean())
                                        if valid.any() else None},
                        "n_subjects_used": n_multi, "n_samples_used": N_all}
            share = M["icc"]["share_above"]["share"]
            if share:
                f.indicate("high_icc", {"share": share}, pct=round(100 * share, 1), cut=params["icc_indicator"])
            audit.results.setdefault("median_icc", {})[unit.block] = M["icc"]["raw"]["median"]

        Ys = Y[:, idx]
        if len(idx) >= 3:
            C, Nsh = diagnostic_matrix.pairwise_corr(Ys)
            same, diff = [], []
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    if np.isfinite(C[i, j]):
                        (same if sc[i] == sc[j] else diff).append(C[i, j])
            M["sample_correlation"] = dict(diagnostic_matrix.pairwise_report(Nsh),
                                           same_subject_mean_r=float(np.mean(same)) if same else None,
                                           different_subject_mean_r=float(np.mean(diff)) if diff else None,
                                           n_same_pairs=len(same), n_different_pairs=len(diff))
            if same and diff and np.mean(same) > np.mean(diff):
                M["sample_correlation"]["difference"] = float(np.mean(same) - np.mean(diff))
                f.indicate("within_subject_correlation", None, same=round(float(np.mean(same)), 3),
                           diff=round(float(np.mean(diff)), 3))
        f.set_method("cluster sizes, time grid, nesting, ICC(1) one-way random effects (unbalanced), within-subject r",
                     n_used=len(have), transform=unit.transform)

    return bridge.per_block(audit, f, block)
