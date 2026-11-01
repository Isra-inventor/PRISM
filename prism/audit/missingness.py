"""G7 missingness, part 1: describe what is missing and where. Never fills anything.

Hard rule (Build Request v4, section 5): nothing here writes a filled value anywhere. Missing (NaN)
and zero are counted apart; zeros are never turned into missing here (if you said in step 7 that
zeros mean "not detected", the plan will later propose the op recode_zero_as_missing, applied
only after approval). Floor-tied values are numbers in the file and count as observed.

A4 missingness (audit set: study samples, unflagged proteins), per block:
  1. how much is missing, per protein and per sample; zeros apart
  2. floor ties: proteins with >= 2 samples exactly at their minimum value (>= 5 observed values);
     how many cells sit at that floor
  3. abundance dependence: Spearman rho between a protein's missing rate (and floor rate) and its
     mean log2 value over cells that are neither missing nor at the floor. Strongly negative
     (rho <= -0.1, heuristic): low-abundance proteins miss more, consistent with a detection limit.

A12 missingness profile, part 1:
  5.1 cell encoding: every cell is exactly one of missing, zero, floor-tied, observed (counts)
  5.2 rates by batch, by group, by time point; share of proteins with no / <= 10% / <= 25% / <= 50%
      missing (reported, none chosen)
  5.3 batch-wide absence: proteins with no observed value in at least one batch level that has
      >= 3 samples. This pattern breaks both imputation and ComBat, so it is counted per batch.
      Also batch coverage: share of proteins with >= 1 (and >= 2) observed values in every level.
Part 2 (next step) adds the detection curve, design tests, feature classes, the pattern summary
and the imputation hand-off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .finding import Finding
from .processing_state import floor_ties
from .scale import diagnostic_log


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float | None, int]:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return None, int(ok.sum())
    rx, ry = pd.Series(x[ok]).rank(), pd.Series(y[ok]).rank()
    return float(np.corrcoef(rx, ry)[0, 1]), int(ok.sum())


def _summary(v: np.ndarray) -> dict:
    return {"min": float(v.min()), "median": float(np.median(v)), "p90": float(np.quantile(v, 0.9)),
            "max": float(v.max())} if v.size else {}


def floor_cells(X: np.ndarray, min_obs: int) -> tuple[np.ndarray, np.ndarray]:
    """(tied per protein, S: cells sitting at the floor of a tied protein)."""
    tied, _ = floor_ties(X, min_obs)
    with np.errstate(invalid="ignore"):
        m = np.nanmin(np.where(np.isfinite(X), X, np.inf), axis=1)
    S = np.isfinite(X) & (X == m[:, None]) & tied[:, None]
    return tied, S


def run_a4(audit) -> Finding:
    p = audit.params
    f = Finding("A4", "Missingness", step="G7", feeds=["G7 missingness profile", "S2 batch correction",
                                                       "G5 log transform", "S3 feature filtering"],
                method="exact counts on the audit set; floor ties on proteins with >= "
                       f"{p['floor_min_observed']} observed values; Spearman rho on log2 means")
    for bid, g in audit.schema.block_map().items():
        X = audit.audit_matrix(bid).to_numpy(dtype=float)
        n_p, n_s = X.shape
        R = np.isnan(X)
        n_cells, n_miss, n_zero = X.size, int(R.sum()), int((X == 0).sum())
        prate, srate = R.mean(1), R.mean(0)
        m = {"n_proteins": n_p, "n_samples": n_s, "n_cells": n_cells,
             "missing": {"n": n_miss, "rate": n_miss / n_cells if n_cells else 0, "per_protein": _summary(prate),
                         "per_sample": _summary(srate), "proteins_with_any": int((prate > 0).sum()),
                         "proteins_all_missing": int((prate == 1).sum()), "samples_with_any": int((srate > 0).sum())},
             "zeros": {"n": n_zero, "rate": n_zero / n_cells if n_cells else 0, "zero_means": g.zero_means}}
        if n_miss:
            f.say("note", f"{bid}: {100 * n_miss / n_cells:.2f}% of cells are missing ({n_miss} of {n_cells}); "
                          f"{m['missing']['proteins_with_any']} of {n_p} proteins miss at least one value.")
        else:
            f.say("ok", f"{bid}: no value is missing.")
        if n_zero:
            f.say("note", f"{bid}: {100 * n_zero / n_cells:.2f}% of cells are exactly zero ({n_zero}); zeros are counted "
                          f"apart from missing values. You said zeros mean '{g.zero_means}'"
                          + (": the plan will propose recoding them as missing (after your approval)."
                             if g.zero_means == "not_detected" else "."))

        tied, S = floor_cells(X, p["floor_min_observed"])
        n_tied = int(tied.sum())
        eligible = int((np.isfinite(X).sum(1) >= p["floor_min_observed"]).sum())
        m["floor_ties"] = {"n_proteins": n_tied, "n_eligible": eligible, "share": n_tied / eligible if eligible else 0,
                           "n_cells": int(S.sum())}
        if n_tied and n_tied / max(eligible, 1) >= p["floor_tie_share_indicator"]:
            f.say("note", f"{bid}: {100 * n_tied / eligible:.1f}% of proteins ({n_tied} of {eligible}) have two or more "
                          f"samples at their minimum value ({int(S.sum())} cells): consistent with values set to a floor "
                          "(for example a detection limit or an imputed minimum).")
        elif n_tied:
            f.say("ok", f"{bid}: {n_tied} of {eligible} proteins have floor ties (below the "
                        f"{100 * p['floor_tie_share_indicator']:.0f}% indicator).")

        Y = diagnostic_log(X)[0]
        keep = ~R & ~S
        with np.errstate(invalid="ignore"):
            mean_y = np.where(keep.sum(1) > 0, np.where(keep, Y, 0).sum(1) / np.maximum(keep.sum(1), 1), np.nan)
        dep = {}
        for what, rate, active in (("missing", prate, n_miss > 0), ("floor", S.mean(1), n_tied > 0)):
            if not active:
                continue
            rho, n_used = spearman(rate, mean_y)
            dep[what] = {"rho": rho, "n_proteins": n_used}
            if rho is not None and rho <= p["abundance_rho_indicator"]:
                f.say("note", f"{bid}: lower-abundance proteins have more {what} values (Spearman rho {rho:.3f} over "
                              f"{n_used} proteins), consistent with a detection limit.")
        m["abundance_dependence"] = dep
        f.measures[bid] = m
        audit.results.setdefault("floor", {})[bid] = S
    return f


def _levels(values: pd.Series, min_n: int) -> dict:
    out = {}
    for lev, idx in values.groupby(values).groups.items():
        out[lev] = list(idx)
    return {k: v for k, v in out.items() if k not in ("", None)}


def run_a12_part1(audit) -> Finding:
    p = audit.params
    f = Finding("A12", "Missingness profile", step="G7", feeds=["S2 batch correction", "S3 feature filtering",
                                                               "the later imputation decision"],
                method="exact counts on the audit set, per block; batch / group / time from your sample "
                       f"information; batch levels need >= {p['a12_min_level_n']} samples for batch-wide absence")
    variables = audit.variables()
    for bid in audit.schema.block_map():
        Xd = audit.audit_matrix(bid)
        X = Xd.to_numpy(dtype=float)
        R = np.isnan(X)
        S = audit.results.get("floor", {}).get(bid)
        if S is None:
            _, S = floor_cells(X, p["floor_min_observed"])
        n_cells = X.size
        Z = X == 0
        # every cell gets exactly one class, in this order: missing, zero, floor-tied, observed
        enc = {"observed": int((~R & ~Z & ~S).sum()), "missing": int(R.sum()), "zero": int(Z.sum()),
               "floor_tied": int((S & ~Z).sum()), "n_cells": n_cells}
        m = {"cell_encoding": enc}
        f.say("ok", f"{bid}: cells: {enc['observed']} observed, {enc['missing']} missing, {enc['zero']} zero, "
                    f"{enc['floor_tied']} at a protein's floor (of {n_cells}).")

        prate = R.mean(1)
        m["protein_missing_shares"] = {k: round(float(v), 4) for k, v in (
            ("no_missing", (prate == 0).mean()), ("at_most_10pct", (prate <= 0.10).mean()),
            ("at_most_25pct", (prate <= 0.25).mean()), ("at_most_50pct", (prate <= 0.50).mean()))}

        by = {}
        for name, (kind, values) in variables.items():
            vals = values.reindex(Xd.columns).fillna("").astype(str).str.strip()
            rows = []
            for lev, cols in _levels(vals, 1).items():
                j = [Xd.columns.get_loc(c) for c in cols]
                rows.append({"level": lev, "n_samples": len(j), "missing_rate": round(float(R[:, j].mean()), 4)})
            if rows and kind in ("batch", "group", "timepoint"):
                by[name] = {"audit_kind": kind, "levels": rows}
        m["rates_by_variable"] = by

        absence = {}
        for name, (kind, values) in variables.items():
            if kind != "batch":
                continue
            vals = values.reindex(Xd.columns).fillna("").astype(str).str.strip()
            levels = {lev: [Xd.columns.get_loc(c) for c in cols] for lev, cols in _levels(vals, 1).items()}
            if len(levels) < 2:
                continue
            big = {lev: j for lev, j in levels.items() if len(j) >= p["a12_min_level_n"]}
            obs_per_level = {lev: (~R[:, j]).sum(1) for lev, j in levels.items()}
            absent = np.zeros(X.shape[0], bool)
            per_level = {}
            for lev, j in big.items():
                a = obs_per_level[lev] == 0
                per_level[lev] = int(a.sum())
                absent |= a
            cov1 = np.all([obs_per_level[lev] >= 1 for lev in levels], axis=0)
            cov2 = np.all([obs_per_level[lev] >= 2 for lev in levels], axis=0)
            absence[name] = {"n_proteins_batch_wide_absent": int(absent.sum()), "share": round(float(absent.mean()), 4),
                             "per_level": per_level, "levels_too_small": sorted(set(levels) - set(big)),
                             "coverage_at_least_1_every_level": round(float(cov1.mean()), 4),
                             "coverage_at_least_2_every_level": round(float(cov2.mean()), 4)}
            if absent.any():
                f.say("note", f"{bid}: {int(absent.sum())} protein(s) ({100 * absent.mean():.1f}%) have no observed value "
                              f"in at least one level of '{name}' (per level: {per_level}). This pattern breaks both "
                              "imputation across batches and ComBat.")
            f.say("ok", f"{bid}: {100 * cov1.mean():.1f}% of proteins have at least one observed value in every level of "
                        f"'{name}' ({100 * cov2.mean():.1f}% at least two).")
        if not any(k == "batch" for k, _ in variables.values()):
            f.say("not_applicable", f"{bid}: no batch variable in the sample information, so batch-wide absence "
                                    "cannot be assessed.")
        m["batch_wide_absence"] = absence
        f.measures[bid] = m
    return f
