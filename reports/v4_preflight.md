# v4 preflight (Build Request v4, section 0)

Date: 2026-10-21. Branch `claude/prism-data-input-recognition-9adu1r`, head `0df3277` (clean, equal to origin).

**Outcome: no blocker.** The golden tests pass, so the build continues. The one gap: v3.1 does not exist in this
repository. Item 5 lists where v4 depends on it and how each point was resolved.

## 1. State of v3 and v3.1

| Part | State | Tests |
|---|---|---|
| Step 0 wizard (v2.x) | present | pass |
| Sessions, output-folder loader, manifest, ledger, CLI (v3 stage 1) | present | pass |
| Schema import (exact / template / seeded wizard) | present | pass (`test_v3_import.py`) |
| Merge M1–M7 | present; the Merge step appears only with 2+ datasets | pass (`test_v3_merge.py`) |
| Audits A1–A11 | all 11 present (`prism/audit/a01…a11`) | pass (`test_v3_audit.py`) |
| Report, final report, HTML export | present | pass (`test_v3_report.py`) |
| **v3.1** (metadata `scope`, report fixes, `partial` status, `not_assessed` history items, `sample_median_spread_mad`) | **not present.** There are no commits, files or spec text for v3.1 in this repository or conversation | — |

Full suite on Python 3.11: **168 passed, 2 skipped** (the 2 skips are the real-Gemini tests, which need an API key).

Golden tests on the two real files (`tests/data/*_common.csv`, git-ignored, present):
`test_golden_real_files` and `test_golden_real_files_share_all_27_samples` **pass**:
- metabolomics: 397/1,174 floor-tied features;
- SomaScan: 20/11,083 floor-tied features;
- 8 subjects;
- the 12,301.6 cell flagged;
- the median-scaling signature present for metabolomics and absent for SomaScan.

## 2. NaN and floor-tied values in A7, A8 and A11 today

The diagnostic copy is `Y = log2(X)`, or `log2(X + c)` with `c = 0.5 × smallest positive value`
(`context.py`). Floor-tied values are numbers and stay in everywhere, counted as observed.

- **A7 PCA and PERMANOVA** (`a07_batch.py:89`). `Y = stats.complete_features(unit.Y)` keeps only features finite in
  every sample (`stats.py:264`, `Y[np.isfinite(Y).all(1)]`). The PCA (SVD of feature-centred Y) and the PERMANOVA
  Gram matrix use that subset. The number kept is reported as `pca.n_features_complete` out of `of`. There is no
  fallback when too few features survive: below 2 it returns `insufficient_data`. There is no bias warning and no
  sensitivity check.
- **A8 ICC** (`a08_repeated.py:14-44`). Each feature uses its own observed mask: subjects with ≥2 observed samples
  for that feature, and ICC is NaN where a < 3 subjects or N − a < 3. Same- versus different-subject correlation
  (`a08_repeated.py:184`) uses features observed in every used sample (complete features).
- **A11 outlier distance** (`a11_outliers.py:25`): `stats.complete_features(Y)`, as in A7. Per-feature extreme
  values use the observed values only.

Section 3 of v4 replaces this with `diagnostic_matrix(Y, policy)`, reported policy, the q fallback, the bias warning
and the sensitivity block.

## 3. R, packages and which oracle tests can run

- **Bioconductor and CRAN are blocked** by the network policy (`bioconductor.org`, `git.bioconductor.org`,
  `cloud.r-project.org` all return 403). `proBatch` and `pvca` therefore cannot be installed.
- **Ubuntu's apt mirror works.** Installed R 4.3.3, `sva` 3.50.0, `preprocessCore` 1.64.0, `limma` 3.58.1,
  `lme4` 1.1-35.1 and `vegan` 2.6-4 from the Ubuntu packages.
- **PyPI works.** Installed on both interpreters:

  | Package | Python 3.11 | Python 3.7 (numpy 1.21.6) |
  |---|---|---|
  | scipy | 1.17.1 | 1.7.3 |
  | statsmodels | 0.15.0 | 0.13.5 |
  | scikit-learn | 1.9.1 | 1.0.2 |
  | scikit-misc | 0.5.3 | 0.1.4 |

  scikit-misc wraps the original LOESS code.

| Oracle test (section 11) | Can run? | How |
|---|---|---|
| `log2(x + offset)` | yes | base R |
| median centering, per-batch median centering | yes | base R |
| quantile normalization | yes | `preprocessCore::normalize.quantiles` |
| LOESS drift | yes | R `stats::loess` with the same span and degree, plus the proBatch fitting function sourced from the downloaded repo (base R only) |
| ComBat | yes | `sva::ComBat` |
| PVCA | **no** | `pvca` is Bioconductor-only and blocked. Python PVCA is compared with an `lme4` re-implementation of the PVCA steps instead, and the test says so |
| tutorial datasets (optional test) | **no** | the datasets come from the proBatch / PRIDE servers, not reachable; skipped |

The R oracle tests skip themselves when `Rscript` or a package is missing, so the suite also runs without R.

## 4. proBatch source

- Bioconductor's git URL (`git.bioconductor.org/packages/proBatch`) is blocked. The source came from GitHub:
  `https://github.com/symbioticMe/proBatch`, the URL listed in `DESCRIPTION` as `URL` and `BugReports`.
- Commit `524249ee9db539ca0e93442ca698bb14cf827d90`, 25 Apr 2025 ("Merge pull request #16 from Freddsle/master").
  Branches: `master` and `RELEASE_3_10`.
- `DESCRIPTION` **Version: 1.3.1**. This is the GitHub development line. It differs from the Bioconductor release
  numbering, and it has no `ProBatchFeatures` container, so it is **not proBatch 2.0**. Treat it as "the GitHub
  master as of April 2025", not as the current Bioconductor release.
- `DESCRIPTION` **License: GPL-3**. The repo holds **no separate LICENSE text file**, only that field.
- `DESCRIPTION` authors:
  - Jelena Cuklina (aut)
  - Chloe H. Lee (aut)
  - Patrick Pedrioli (aut)
  - Yuliya Burankova (cre)

  The paper's other authors are not in the package's author list.
- The suspected bugs in 7.4 are visible at these locations. Stage 8 checks each one, ports the intended behaviour
  and adds a failing-on-bug test.

  | Suspected bug | Location |
  |---|---|
  | CV filter | `R/CV_calculation.R:55` `if (any(df_long$n) > 2)` |
  | `rle_func` | `R/fit_non_linear.R:198` uses `df$Intensity` |
  | `unlog_dm` / `unlog_df` | `R/transform_raw.R:48-85` |
  | `keep_all` | `R/correct_batch_effects.R` |
  | median centering | `R/normalize.R` |

- **protDP (Li and Smyth, detection probability curve)**: `mengbo-li.github.io` is blocked, but
  `git clone https://github.com/mengbo-li/protDP` worked. Version 1.1.0, commit `133da7e` (26 Oct 2023), author
  Mengbo Li, **License: Apache-2.0** (full text in the repo). It is an R package (`dpc()`); it needs `limma`, which is
  installed. An A12.4 oracle comparison is therefore possible.

## 5. Where v3, v3.1 and v4 disagree, and how each was resolved

| # | Point | Resolution |
|---|---|---|
| 1 | v4 builds on v3.1, which is not in the repo | Proceeded on v3. Each v3.1 feature v4 refers to is inferred from v4's own wording and listed below. Nothing from v3.1 beyond what v4 states is invented. |
| 2 | `partial` status ("three statuses plus `partial`"); v3 has `computed`, `insufficient_data`, `insufficient_metadata`, `not_applicable` | `partial` is added as a valid status: some measures computed, others not, with the missing parts named in `status_reason`. v3's four statuses keep their meaning. |
| 3 | "v3.1 scope rules" for batch candidates | The v3 rule is used: sample-metadata columns with `audit_kind` `batch` or `run_order`, plus the `batch_variable` override (`context.variables`). |
| 4 | `batch_corrected` and `removed_before_upload` "stay `not_assessed` as in v3.1" | A13 marks those two `not_assessed` and never asks about them. |
| 5 | Parameter `sample_median_spread_mad` (0.1 log2, section 14.4); v3 has `narrow_median_spread` (0.05) for "reads as normalized" | The two are kept separate. `narrow_median_spread` keeps its v3 meaning; `sample_median_spread_mad: 0.1` is added (heuristic) and used only by the S1 decision node "medians differ". Both are listed for calibration. |
| 6 | Report "v3.1 conventions: static SVG in print, no collapsed cards" | The final report already prints with every card expanded (the `@media print` rule). The new cards follow the same pattern: inline SVG, no collapse in print. |
| 7 | Metabolomics A7 "still returns `insufficient_metadata` for the batch variable (v3.1)" | v3 already does this: no batch candidate gives `insufficient_metadata`, `needs: ["batch variable"]`. Kept. |
| 8 | Metadata `scope` change (v3.1) | Unknown content. Nothing in v4 depends on its details beyond item 3, so nothing is built for it. |
| 9 | Dependencies: v3's audit is numpy-only; v4 asks for `scipy.optimize` (A12.4), `statsmodels` (PVCA), `skmisc` (LOESS), scikit-learn (Phase 2 kNN) | `scipy` becomes a core dependency (A12.4). `statsmodels`, `scikit-misc` and `scikit-learn` are optional: PVCA falls back to eta-squared partitioning and says so; LOESS falls back to PRISM's own tricube local regression with `source: fallback` in the op record; Phase 2 kNN refuses without scikit-learn. `pyyaml` is added for the nested registry, rules and evidence files; the v3 flat `audit_params.yaml` reader is unchanged. All versions install on Python 3.7. |
| 10 | Python 3.7 (ROADMAP) | Kept. Every new module runs on 3.7: no walrus, no `math.prod`, no 3.8+ typing. Both suites run at every stage. |
| 11 | A6 LOESS: v3 A6 uses PRISM's own lowess for QC trends; v4 7.3 prefers `skmisc.loess` | The audit A6 is left as is (an audit measure). The executor ops (`drift_qc_loess`, `drift_sample_loess`) use `skmisc` when present. |
| 12 | v4 says Step 0 `processing_history` has five items; Step 0 stores `log_transformed`, `normalized`, `imputed`, `batch_corrected`, `removed_before_upload` | Matches. No schema change: confirmations go to `overrides.json` (`processing_state` kind), as v4 asks. |
| 13 | The v3 diagnostic transform is `log2(X + 0.5 × min positive)`; proBatch defaults to offset 1 | The G5 default is `half_min`. `proBatch_compat` (c = 1) exists for the oracle tests only. |
| 14 | `pvca` cannot be installed (item 3) | The PVCA oracle uses an `lme4` re-implementation of the same steps. Recorded as a deviation in `docs/probatch_port.md`. |
