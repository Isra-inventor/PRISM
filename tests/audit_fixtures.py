"""Synthetic Step 0 output folders with planted ground truth for the audit tests (seeded)."""

import csv
import json

import numpy as np

from prism import store


def make_output(path, X, samples, columns=None, kinds=None, design=None, keys=None, history=None,
                classes=None, family="proteomics"):
    """X: features x samples (NaN = empty cell). classes: per-feature feature_class (derived)."""
    columns, kinds = columns or {}, kinds or {}
    path.mkdir(parents=True)
    keys = keys or [f"f{i}" for i in range(X.shape[0])]
    with open(path / "value_matrix_A1_B1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature_key"] + list(samples))
        for k, row in zip(keys, X):
            w.writerow([k] + ["" if not np.isfinite(v) else repr(float(v)) for v in row])
    with open(path / "feature_metadata.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature_key", "assay_id"] + (["feature_class"] if classes else []))
        for i, k in enumerate(keys):
            w.writerow([k, "A1"] + ([classes[i]] if classes else []))
    names = list(columns)
    with open(path / "sample_metadata.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "sample_label", "is_study_sample"] + names)
        for j, s in enumerate(samples):
            w.writerow([s, "study sample", "true"] + [columns[c][j] for c in names])
    ann = [{"column": "feature_class", "derived_from": "feature_names", "keep": True, "display_labels": {"": "unclassified"},
            "provenance": "ai_proposed_confirmed"}] if classes else []
    schema = {"schema_version": "0.4.1", "source_file": path.name + ".csv", "file_sha256": "0" * 64,
              "omics_family": {"value": family},
              "assays": [{"assay_id": "A1", "assay_label": family, "omics_type": family, "n_features": X.shape[0],
                          "n_samples": len(samples),
                          "value_blocks": [{"block_id": "B1", "group_id": "g1", "label": "values",
                                            "file": "value_matrix_A1_B1.csv"}]}],
              "feature_annotations": ann,
              "sample_metadata": [{"column": c, "audit_kind": kinds.get(c, "covariate")} for c in names],
              "design": design or {"subject": {"source": "none"}, "time": {"source": "none"}},
              "processing_history": history or {}}
    (path / "schema.json").write_text(json.dumps(schema))
    return path


def add(st, path, **kw):
    out = make_output(path, **kw)
    d = st.add_dataset(path.name, "output_folder", "imported_awaiting_confirm")
    store.publish_output(st, d["dataset_id"], out, {"mode": "output_folder"})
    return d["dataset_id"]


SUBJ_DESIGN = {"subject": {"source": "metadata_column", "column": "animal"},
               "time": {"source": "metadata_column", "column": "week", "unit": {"value": "weeks"}}}


def planted(seed=1, n_subj=12, per=3, p=400, icc=0.8):
    """12 subjects x 3 weeks. Planted: subject effects giving a known ICC on log2 scale; plate
    constant within subject (4 plates x 3 animals); a batch (run day) that doubles the missing
    rate of low-abundance features; floors in features 0..19; one outlier cell."""
    rng = np.random.default_rng(seed)
    samples, animal, week, plate, day = [], [], [], [], []
    for a in range(n_subj):
        for t in range(per):
            samples.append(f"A{a:02d}_{t * 4}")
            animal.append(f"A{a:02d}")
            week.append(str(t * 4))
            plate.append(f"P{a // 3 + 1}")
            day.append("d1" if (a + t) % 2 == 0 else "d2")
    n = len(samples)
    sb, sw = np.sqrt(icc), np.sqrt(1 - icc)
    base = rng.normal(10, 2, size=(p, 1))
    subj_eff = rng.normal(0, sb, size=(p, n_subj))
    L = base + subj_eff[:, [int(a[1:]) for a in animal]] + rng.normal(0, sw, size=(p, n))
    X = 2.0 ** L
    # floors: features 0..19 have 4 samples at the feature's minimum
    for i in range(20):
        fl = X[i].min() * 0.5
        X[i, rng.choice(n, 4, replace=False)] = fl
    # missingness: low-abundance features lose values, twice as often on day 2
    low = [i for i in np.argsort(base[:, 0]) if i >= 20][: p // 4]
    for i in low:
        for j in range(n):
            if rng.random() < (0.25 if day[j] == "d2" else 0.04):
                X[i, j] = np.nan
    cols = {"animal": animal, "week": week, "plate": plate, "run_day": day}
    kinds = {"animal": "subject_id", "week": "timepoint", "plate": "batch", "run_day": "batch"}
    return X, samples, cols, kinds
