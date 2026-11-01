"""G1 / A1: integrity. Is the data sound enough to compute anything on?

Runs on ALL samples and ALL proteins (QC and flagged ones included), block by block:

 1. size of every block, and whether the blocks share their samples
 2. cells in a value column that are not numbers (e.g. "<LOD", "n.d.", "1,234"): the matrix turns
    them into missing values, so they must be seen here               -> problem
 3. infinite values ("inf")                                              -> problem
 4. repeated protein IDs or sample names                                -> problem
 5. samples with no value at all                                        -> problem
 6. proteins with no value, fewer than 3 values, or one constant value  -> note
 7. two samples (or two proteins) with exactly the same values: often a copy-paste or
    export error                                                          -> note
 8. one non-zero value that covers a large share of all cells (e.g. 1, -9, 999, or a detection
    limit written in): a plausible sign of a placeholder                 -> note
 9. study samples without a subject (when the design has one)          -> note

It also states the AUDIT SET used by the next checks: study samples only (QC, blank, pool,
calibrator and reference samples are used only where a check says so) and unflagged proteins.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .finding import Finding



def _identical(frame: pd.DataFrame) -> list[list[str]]:
    """Groups of columns whose values are exactly equal (missing equal to missing)."""
    groups: dict[str, list[str]] = {}
    for name in frame.columns:
        v = frame[name].to_numpy(dtype=float)
        if np.isnan(v).all():
            continue
        key = hashlib.sha256(np.where(np.isnan(v), np.inf, v).astype("<f8").tobytes()).hexdigest()
        groups.setdefault(key, []).append(str(name))
    return [g for g in groups.values() if len(g) > 1]


def _examples(items, n=5) -> str:
    items = list(items)
    return ", ".join(map(str, items[:n])) + (f" ... (+{len(items) - n})" if len(items) > n else "")


def run(audit) -> Finding:
    s = audit.schema
    PARAMS = audit.params
    f = Finding("A1", "Integrity", step="G1", feeds=["every later check", "feature filtering"],
                method="exact counts on the values as written in the file, all samples and all proteins")
    blocks = {}
    for bid in s.block_map():
        X = audit.full_matrix(bid)
        text = s.matrix(bid) if audit.state is None else X.notna().astype(str).where(X.notna(), "")
        m = {"n_proteins": X.shape[0], "n_samples": X.shape[1]}

        bad = (text != "") & X.isna()
        m["n_not_a_number"] = int(bad.values.sum())
        if m["n_not_a_number"]:
            ex = text.where(bad).stack().head(5)
            f.say("problem", f"{bid}: {m['n_not_a_number']} cell(s) are not numbers and would be read as missing, "
                             f"e.g. {', '.join(f'{v!r} ({p}, {q})' for (p, q), v in ex.items())}. Fix them in the "
                             "file or decide what they mean (e.g. below detection) before auditing.")
        inf = np.isinf(X.to_numpy(dtype=float))
        m["n_infinite"] = int(inf.sum())
        if m["n_infinite"]:
            f.say("problem", f"{bid}: {m['n_infinite']} infinite value(s).")
        X = X.mask(np.isinf(X))

        dup_p = X.index[X.index.duplicated()].unique()
        dup_s = X.columns[X.columns.duplicated()].unique()
        if len(dup_p):
            f.say("problem", f"{bid}: {len(dup_p)} protein ID(s) repeat: {_examples(dup_p)}.")
        if len(dup_s):
            f.say("problem", f"{bid}: {len(dup_s)} sample name(s) repeat: {_examples(dup_s)}.")

        n_obs_s = X.notna().sum(axis=0)
        empty_s = list(n_obs_s[n_obs_s == 0].index)
        m["samples_without_values"] = empty_s
        if empty_s:
            f.say("problem", f"{bid}: {len(empty_s)} sample(s) have no value at all: {_examples(empty_s)}.")

        n_obs_p = X.notna().sum(axis=1)
        m["n_proteins_without_values"] = int((n_obs_p == 0).sum())
        m["n_proteins_sparse"] = int(((n_obs_p > 0) & (n_obs_p < PARAMS["min_observed"])).sum())
        sd = X.std(axis=1, ddof=0)
        const = list(X.index[(n_obs_p >= 2) & (sd == 0)])
        m["n_proteins_constant"] = len(const)
        if m["n_proteins_without_values"]:
            f.say("note", f"{bid}: {m['n_proteins_without_values']} protein(s) have no value in any sample.")
        if m["n_proteins_sparse"]:
            f.say("note", f"{bid}: {m['n_proteins_sparse']} protein(s) have fewer than {PARAMS['min_observed']} "
                          "values: too few to estimate anything for them.")
        if const:
            f.say("note", f"{bid}: {len(const)} protein(s) have the same value in every sample where measured: "
                          f"{_examples(const)}.")

        same_s, same_p = _identical(X), _identical(X.T)
        m["identical_samples"], m["n_identical_protein_groups"] = same_s, len(same_p)
        for g in same_s:
            f.say("note", f"{bid}: samples {', '.join(g)} have exactly the same values: a plausible sign of a "
                          "duplicated column or a copy-paste error.")
        if same_p:
            f.say("note", f"{bid}: {len(same_p)} group(s) of proteins have exactly the same values, e.g. "
                          f"{', '.join(same_p[0][:3])}.")

        v = X.stack()
        v = v[v != 0]
        if len(v):
            top, n_top = v.value_counts().index[0], int(v.value_counts().iloc[0])
            share = n_top / X.size
            m["most_common_nonzero"] = {"value": float(top), "n": n_top, "share": round(share, 4)}
            if share >= PARAMS["sentinel_share"] and n_top >= PARAMS["sentinel_min_cells"]:
                f.say("note", f"{bid}: the value {top:g} fills {share:.1%} of all cells ({n_top} cells): a plausible sign "
                              "of a placeholder or a value written in for 'not measured' (e.g. a detection limit).")
        blocks[bid] = m

    names = [tuple(sorted(s.sample_names(b))) for b in s.block_map()]
    if len(set(names)) > 1:
        f.say("note", "The blocks do not all have the same samples.")

    d = s.design()
    if d["subject"]["source"] != "none":
        n_missing = d["facts"].get("n_study_samples_without_subject", 0)
        if n_missing:
            f.say("note", f"{n_missing} study sample(s) have no subject: they will be treated as separate individuals.")

    st = s.samples()
    flagged = int(s.flag_mask()[0].sum())
    types = st["sample_type"].value_counts().to_dict()
    n_study = int(st["is_study_sample"].sum())
    f.measures = {"blocks": blocks, "sample_types": types, "n_flagged_proteins": flagged,
                  "audit_set": {"n_study_samples": n_study, "n_proteins_unflagged": len(s.feature_ids()) - flagged}}
    other = ", ".join(f"{n} {t}" for t, n in types.items() if t != "study") or "none"
    f.say("ok", f"Audit set for the next checks: {n_study} study samples (other samples: {other}) and "
                f"{len(s.feature_ids()) - flagged} proteins ({flagged} flagged proteins left out).")
    if n_study < 3:
        f.say("problem", f"Only {n_study} study sample(s): too few for the statistical checks.")
    if not any(st_ != "ok" for st_, _ in f.sentences[:-1]):
        f.sentences.insert(0, ("ok", "No integrity problem found: every value cell is a number, IDs and sample "
                                     "names are unique, every sample has values."))
    return f
