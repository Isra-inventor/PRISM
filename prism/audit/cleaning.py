"""G3 technical-record cleaning candidates (Build Request v4, section 4.3). Ported from the v4 build.
Flags only: each class is a separate flag with counts and protein IDs; nothing is removed by the
audit. The plan proposes remove_features per class and you approve per class.

Classes (on all proteins, study samples):
    flagged:<column>        proteins flagged in step 6 (controls, other species, decoys ...)
    all_missing             no value in any study sample
    sd_zero                 the same value in every sample where measured
    fewer_than_3_observed   1 or 2 values
    blank_dominated         mean blank signal >= 0.5 x mean study signal (heuristic; needs blanks)
"""

from __future__ import annotations

import numpy as np

MAX_IDS = 5000


def candidates(audit, block: str) -> dict:
    s = audit.schema
    full = audit.full_matrix(block)
    types = s.samples().set_index("sample")["sample_type"]
    study = [c for c in full.columns if types.get(c) == "study"]
    blanks = [c for c in full.columns if types.get(c) == "blank"]
    X = full[study].to_numpy(dtype=float)
    keys = [str(k) for k in full.index]
    fin = np.isfinite(X)
    n_obs = fin.sum(1)
    import warnings
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = np.nanstd(np.where(fin, X, np.nan), axis=1)
    classes = []

    def add(cls, mask, why, source, **extra):
        ids = [keys[i] for i in np.where(mask)[0]]
        classes.append(dict({"class": cls, "n": len(ids), "ids": ids[:MAX_IDS], "truncated": len(ids) > MAX_IDS,
                             "why": why, "source": source, "block": block}, **extra))

    _, reason = s.flag_mask()
    fm = s.feature_metadata()
    by_feature = dict(zip(fm["feature"].astype(str), fm["flag_reason"]))
    for col in s.flag_decisions:
        m = np.array([col + "=" in (by_feature.get(k) or "") for k in keys])
        if m.any():
            add(f"flagged:{col}", m, f"flagged in step 6 by '{col}'", "step 6 flags", column=col)
    add("all_missing", n_obs == 0, "every value is missing", "A1")
    add("sd_zero", (n_obs >= 2) & (sd == 0), "SD 0 over the observed values", "A1")
    add("fewer_than_3_observed", (n_obs >= 1) & (n_obs < 3), "fewer than 3 observed values", "A1")
    if blanks and study:
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.nanmean(full[blanks].to_numpy(float), axis=1) / np.nanmean(X, axis=1)
        cut = audit.params["blank_dominated_ratio"]
        add("blank_dominated", np.isfinite(ratio) & (ratio >= cut),
            f"mean blank signal at least {cut} x the mean study signal (heuristic)", "A6 blank ratio")
    return {"classes": [c for c in classes if c["n"]], "n_classes": sum(1 for c in classes if c["n"]),
            "note": "flags only; removal is a plan entry approved per class and always runs before any other op"}
