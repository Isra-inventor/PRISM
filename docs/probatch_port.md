# The proBatch port (Build Request v4, section 7)

`source_verified: true`. The port was adapted from the proBatch source (GitHub master, `DESCRIPTION`
version 1.3.1, commit `524249e`, 25 April 2025), not only from the paper. The Bioconductor git
server was blocked, so the current Bioconductor release was not compared.

License: GPL-3 (see `prism/ports/probatch/NOTICE.md` and `LICENSING.md`). The core reaches the port
only through `prism/batch/backend.py`. With `PRISM_DISABLE_PORTS=1`, the port is never imported and
the ops that need it refuse.

## What was ported, and where

| proBatch | PRISM | Notes |
|---|---|---|
| `log_transform_dm`, `unlog_dm` | `ports/probatch/transform.py` | The offset has no default. proBatch's offset 1 is only the `proBatch_compat` mode of G5, used in oracle tests |
| `normalize_sample_medians_dm` | `batch/normalize.median_center` (core, v4 6.1) | Identical result (oracle) |
| `quantile_normalize_dm` (`preprocessCore`) | `batch/normalize.quantile` (core, v4 6.1) | Identical with no NaN, ties included (oracle). The NA-aware form is PRISM's own and differs from preprocessCore on data with NaN |
| `adjust_batch_trend_df`, `fit_nonlinear`, `loess_regression` | `ports/probatch/drift.py` | Per feature, per batch: LOESS on run order with at least 8 consecutive measured values; value + (mean of the fit − fit); a feature whose fit fails is left unchanged and counted |
| `center_feature_batch_medians_dm`, `..._means_dm` | `ports/probatch/correct.center_feature_batch` | Identical (oracle) |
| `correct_with_ComBat_dm` (`sva::ComBat`, `~1`, parametric) | `batch/combat.py` (core) + `correct.combat_intercept` | ComBat implemented from Johnson, Li and Rabinovic (2007), following sva's documented numerical conventions; no sva code. Identical to sva (oracle) |
| `correct_batch_effects_dm` | `correct.correct_batch_effects` | Continuous (LOESS) then discrete correction |
| `plot_sample_mean`, boxplots, `plot_sample_corr_heatmap`, `plot_hierarchical_clustering`, `plot_PCA`, `calculate_PVCA` | `ports/probatch/diagnostics.py` | Run on PRISM's diagnostic matrix (v4 §3), never on filled values |
| `calculate_sample_corr_distr`, `calculate_peptide_corr_distr`, `calculate_feature_CV` | `ports/probatch/qc.py` | Pearson, pairwise-complete, with the number of shared features. Peptide correlation is `not_applicable` without a peptide-to-protein map |

Not ported: `loess_regression_opt` (the optimised span). Its code fits `y ~ x_all` with a `y` of a
different length and predicts at `x_to_fit`, so the intended behaviour is unclear.

## Python equivalents (7.3)

- **LOESS:** `skmisc.loess` (scikit-misc wraps the original loess C/Fortran code) with
  `surface="direct"`, `family="gaussian"`, degree 2, span 0.75 (R's default, which proBatch passes
  through `...`). Against R `loess` with the same settings, the largest difference is 1.7e-15, within
  the 1e-6 relative tolerance. Without scikit-misc, PRISM's own direct local regression is used and
  `loess_engine: prism_fallback` is recorded. On the oracle data it differed from R by 4.4e-14.
- **ComBat:** a PRISM implementation, validated against `sva::ComBat` 3.50.0.
- **PVCA:** principal components of the sample correlation matrix (at least 3, up to the 0.6
  cumulative share), then a random-effects variance decomposition per component. It uses
  statsmodels `MixedLM` (REML, one variance component per factor and per pairwise interaction).
  If a model fails, eta-squared partitioning is used instead, and `method_per_pc` records which one
  ran. **The Bioconductor `pvca` package could not be installed (network policy).** The oracle is an
  `lme4` re-implementation of `pvcaBatchAssess`'s steps, written for the test.
- **Quantile:** see 6.1.

## Oracle results (R 4.3.3; Python 3.11 and 3.7 gave the same numbers)

| Comparison | Largest absolute difference | Tolerance |
|---|---|---|
| `log2(x + 1)` vs R | 5.3e-15 | 1e-12 |
| median centering vs proBatch `normalize_sample_medians_dm` | 5.0e-14 | 1e-10 |
| quantile vs `preprocessCore::normalize.quantiles` (with ties) | 3.6e-14 | 1e-8 |
| LOESS (skmisc) vs R `loess`, surface direct | 1.7e-15 | 1e-6 relative |
| LOESS (PRISM fallback) vs R `loess` | 4.4e-14 | documented |
| LOESS drift vs proBatch `adjust_batch_trend_df` | 7.5e-14 | 1e-6 |
| per-batch median centering vs proBatch `center_feature_batch_medians_dm` | 5.0e-14 | 1e-10 |
| ComBat vs `sva::ComBat` (`~1`, parametric, 3 batches) | 5.3e-14 | 1e-6 |
| PVCA (statsmodels) vs the lme4 re-implementation, crossed 3 x 2 design | 0.0018 (0.011 at most over seeds 0–2) | 0.02 |
| PVCA, same comparison, on a design with singular fits | 0.043 | not tested: lme4 puts boundary components at exactly 0, statsmodels at small positive values |
| A12.4 detection curve vs protDP `dpc` (datasetA, protein, log2) | b1 0.896 (PRISM) vs 0.878 (protDP), difference 0.018 | recorded only |

The proBatch oracles source the R files of the repository (`PROBATCH_SRC`). The tests skip
without R, a package or `PROBATCH_SRC`.

## Bugs checked (7.4)

All six suspected bugs are present in the source used:

| Suspected bug | Confirmed? | Where | Ported behaviour | Test |
|---|---|---|---|---|
| median centering returns the string `measure_col` | yes | `normalize.R`, `normalize_sample_medians_df`, `no_fit_imputed` branch: `ifelse(flag, NA, measure_col)` | flagged cells are excluded from the statistic and keep numbers | `test_bug_median_centering_never_returns_a_column_name` |
| CV filter `any(df_long$n) > 2` never filters | yes | `CV_calculation.R:55` | features need at least 3 measured values | `test_bug_cv_filter_filters` |
| default log offset 1 everywhere | yes | `transform_raw.R` (`offset = 1` defaults) | no default; PRISM's G5 uses `half_min` | `test_bug_no_default_log_offset` |
| `unlog_dm` / `unlog_df` inconsistent | yes | `unlog_df`: `base^(x − offset)`; `unlog_dm`: `base^x − offset` | `base^x − offset` (the inverse of `log(x + offset)`) | `test_bug_unlog_is_the_inverse_of_log` |
| `correct_batch_effects_dm` hard-codes `keep_all = FALSE` | yes: in R the function fails (shown by `test_bug_keep_all_false_fails_in_r`) | `correct_batch_effects.R` | returns the corrected matrix | `test_bug_correct_batch_effects_returns_a_matrix` |
| `rle_func` hard-codes `df$Intensity` | yes | `fit_non_linear.R:198` | the run length is counted on the fitted values | `test_bug_rle_uses_the_fitted_values` |

## Missing values

Diagnostics use the policies of v4 section 3. LOESS and centering tolerate NaN, and NaN stays NaN.
ComBat refuses any NaN, and the plan returns `Do not automate` when coverage is incomplete (6.2).

## Operation log

Every executor op that calls the port writes PRISM's `op_record`. It records the backend name, the
proBatch source and `source_verified`, the LOESS engine, and the fits that were done or left
unchanged. proBatch 2.0's operation log and its `ProBatchFeatures` container are not used.
