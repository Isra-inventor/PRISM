"""G3 technical-record cleaning candidates (Build Request v4, section 4.3). Flags only: each class is
a separate flag with counts and ids; nothing is removed by the audit. The plan proposes
remove_features per class and the user approves per class."""

from __future__ import annotations

import numpy as np

MAX_IDS = 5000


def candidates(unit, ctx, params):
    X = unit.X
    keys = unit.feature_keys
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
                             "why": why, "source": source, "assay": unit.assay["assay_id"]}, **extra))

    for a in ctx.schema.get("feature_annotations", []):
        if not (a.get("marks_rows_as_suspect") and a.get("keep", True)):
            continue
        vals = set(a.get("flagged_values") or [])
        m = np.array([(row.get(a["column"]) or "") in vals for row in unit.F])
        if m.any():
            add(f"suspect:{a['column']}", m, f"rows marked suspect in Step 0 ('{a['column']}' in "
                                             f"{sorted(vals)})", "Step 0 marks_rows_as_suspect", column=a["column"])
    add("all_missing", n_obs == 0, "every value is missing", "A1")
    add("sd_zero", (n_obs >= 2) & (sd == 0), "SD 0 over the observed values", "A1")
    add("fewer_than_3_observed", (n_obs >= 1) & (n_obs < 3), "fewer than 3 observed values", "A1")
    roles = {s: ctx.role(s) for s in unit.all_samples if s not in ctx.excluded_by_override}
    col = {s: j for j, s in enumerate(unit.all_samples)}
    blanks = [col[s] for s, r in roles.items() if r == "blank"]
    study = [col[s] for s, r in roles.items() if r == "study"]
    if blanks and study:
        Xa = unit.X_all
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.nanmean(Xa[:, blanks], axis=1) / np.nanmean(Xa[:, study], axis=1)
        cut = params["blank_dominated_ratio"]
        add("blank_dominated", np.isfinite(ratio) & (ratio >= cut),
            f"mean blank signal at least {cut} x the mean study signal (heuristic)", "A6 blank ratio")
    return {"classes": [c for c in classes if c["n"]], "n_classes": sum(1 for c in classes if c["n"]),
            "note": "flags only; removal is a plan entry approved per class and always runs before any other op"}
