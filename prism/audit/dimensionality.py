"""A5: dimensionality, sparsity, effective sample size. Ported from the v4 build. Numbers only.

 1. n samples, p proteins, n / p, subjects / p
 2. the smallest design cell (level of a batch or design variable with fewest samples)
 3. sparsity: shares of cells missing, zero, at a floor; detection rate per protein
 4. Kish design effect with the median ICC from A8: deff = 1 + (m - 1) * ICC, m = samples per
    subject; n_eff = n / deff: how many independent samples the repeated measures are worth
"""

from __future__ import annotations

import numpy as np

from . import bridge, stats
from .bridge import is_num
from .finding import Finding
from .missingness import floor_cells


def _key(v):
    return (0, float(v), "") if is_num(v) else (1, 0.0, v)


def run(audit) -> Finding:
    f = Finding("A5", "Dimensionality and effective sample size", step="S10",
                feeds=["S3 feature filtering", "S5 design constraints"])
    params = audit.params

    def block(unit, f):
        X = unit.X
        p, n = X.shape
        M = f.measures
        subj = bridge.subject_codes(audit, unit.samples)
        n_subj = len({s for s in subj if s}) if subj else None
        M.update(n=n, p=p, n_over_p=n / p if p else None, n_subjects=n_subj)
        if p and n / p < 1:
            f.indicate("few_samples_per_feature", {"n": n, "p": p}, n=n, p=p, ratio=round(n / p, 4))
        B, G, _ = bridge.variables(audit, unit.samples)
        cells = []
        for v in B + G:
            if v.numeric or v.role == "subject":
                continue
            counts = {}
            for x in v.values:
                if x:
                    counts[x] = counts.get(x, 0) + 1
            if counts:
                counts = {k: counts[k] for k in sorted(counts, key=_key)}
                lev = min(counts, key=lambda k: (counts[k], k))
                cells.append({"variable": v.name, "role": v.role, "counts": counts,
                              "smallest": {"level": lev, "n": counts[lev]}})
        M["cells"] = cells
        if cells:
            sm = min(cells, key=lambda c: (c["smallest"]["n"], c["variable"]))
            M["smallest_cell"] = {"variable": sm["variable"], "level": sm["smallest"]["level"], "n": sm["smallest"]["n"]}
            f.indicate("small_cell", M["smallest_cell"], size=sm["smallest"]["n"], variable=sm["variable"],
                       level=sm["smallest"]["level"])
        R = ~np.isfinite(X)
        S = audit.results.get("floor", {}).get(unit.block)
        if S is None:
            S = floor_cells(X, params["floor_min_observed"])[1]
        zero = (X == 0) & ~S
        M["sparsity"] = {"share_missing_zero_or_floor": float((R | zero | S).mean()), "share_missing": float(R.mean()),
                         "share_zero": float(zero.mean()), "share_floor": float(S.mean())}
        det = 1 - (R.mean(1) + S.mean(1))
        M["detection_rate"] = dict(stats.summary(det), **stats.quantiles(det, (0.1, 0.9)))
        icc = audit.results.get("median_icc", {}).get(unit.block)
        if n_subj and icc is not None:
            m = n / n_subj
            rho = max(0.0, icc)
            deff = 1 + (m - 1) * rho
            M["design_effect"] = {"m": m, "median_icc": icc, "rho_used": rho, "deff": deff, "n_eff": n / deff,
                                  "note": "median ICC truncated at 0"}
            audit.results.setdefault("n_eff", {})[unit.block] = n / deff
            f.indicate("effective_sample_size", M["design_effect"], n_subjects=n_subj, icc=round(icc, 3),
                       n_eff=round(n / deff, 1), n=n, deff=round(deff, 3))
        else:
            M["design_effect"] = {"status": "insufficient_data",
                                  "reason": "needs a subject and the A8 ICC (at least 3 subjects with 2 or more samples)"}
        if not f.sentences:
            f.say("ok", f"{n} samples, {p} proteins (n / p = {n / p:.3g}).")
        f.set_method("n, p, cell counts, sparsity, Kish design effect", n_used=n, transform=unit.transform)

    return bridge.per_block(audit, f, block)
