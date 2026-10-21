# v4 build report

Branch `claude/prism-data-input-recognition-9adu1r`. All 13 stages of the build order are done. Each
stage was committed and pushed after the full test suite passed on Python 3.11 and Python 3.7.

## Tests

| Run | Result |
|---|---|
| Python 3.11 | 246 passed, 6 skipped |
| Python 3.7 (numpy 1.21.6) | 246 passed, 6 skipped |
| Python 3.11, `PRISM_DISABLE_PORTS=1` | 224 passed, 8 skipped |
| `prism evidence check` | ok |

The skips:
- 2 real-Gemini tests (they need an API key);
- 4 oracle tests that need the proBatch repository (`PROBATCH_SRC`). They all pass when it is set.

With the port disabled, the port tests and the S2 approval tests skip as well.

## What was built (by section)

**Registry and evidence (§1, §8.5)**
- `prism/registry/steps.yaml` (G1–G7, S1–S10), validated by `steps.schema.json`.
- `evidence.yaml` holds only the references listed in §8.5, plus two §5.9 sources:
  - the msqrob2 documentation, stated as verified;
  - the 2023 bioRxiv study, `verified: false`.
- `parameters.yaml` gives every audit parameter an evidence level.
- `prism registry show` and `prism evidence check`.

**States, executor, checkpoints (§2)**
- States: `datasets/<did>/states/<state_id>/` with `state.json`. The S0 id is computed without writing anything.
- Executor: 11 Phase 1 ops plus the `recode_zero_as_missing` NaN exception. There is no imputation op; registering one is refused.
- Invariants are checked before anything is written. Approval records are required. Replay is exact.
- `audit run --state` runs the checkpoint battery for that state. Findings carry `state_id` and `checkpoint`.

**Diagnostic matrix (§3)**
- All five policies, with the q = 100 → 75 → 50 fallback and the bias warning.
- A7 and A11 run a sensitivity block that emits `diagnostic_conclusion_depends_on_policy` when the conclusion changes.
- A8's correlations use pairwise-complete values. A1, A6, A9 and A10 report their policy.

**G2/A13 (§4.2):** declared vs observed vs confirmed; `confirm_processing_state` questions; SomaScan platform note (shown, never applied); the G5/S1 gate.

**G3, G4, G5, G6 (§4.3–4.6)**
- G3 candidates per class, with ids.
- G4 measures added to A7: samples per batch, batch coverage, relationship-stratified correlation, control features.
- `log_transform` offset rules; `arcsinh_transform` (the cofactor must be justified).
- `drift_qc_loess`, evidence level `unverified`.

**A12 (§5)**
- Cell encoding and the zeros question.
- Missing rates by batch, group and time.
- Batch-wide absence.
- Zero-truncated binomial detection curve with a seeded bootstrap CI and the stated limitation.
- Per-sample and per-feature association with batch (permutations, BH).
- Feature classes.
- `pattern_class` with three-valued indicators and the wording rule.
- `imputation_candidates`, unranked.
- No "MCAR/MAR/MNAR" appears as a diagnosis (tested).

**S1–S5 (§6)**
- `normalize_median`, NA-aware `normalize_quantile`, `drift_sample_loess`, `batch_center`, `batch_combat` (which refuses any NaN).
- The `median_shift_technical` decision node.
- The S3 completeness table: overall, within a group, and by subject cluster.
- The S4 list, which needs per-sample approval.
- `design_constraints.json`.

**proBatch port (§7)**
- `prism/ports/probatch/`, license-isolated behind `prism/batch/backend.py`.
- All six suspected bugs are confirmed in the source, and each has a test.
- Details and oracle numbers: `docs/probatch_port.md`.

**Plan engine (§8)**
- `intake.json`, the gate, `rules_v0.yaml` (29 rules, typed), `plan_order.yaml` and the D2 joint node as data.
- Approval runs the executor and then re-runs the checkpoint battery. Rejections are recorded.
- `prism plan replay`.

**Verification (§9)**
- Before and after, comparing C1 vs C2 and S0 vs C2: negative and positive controls.
- A test shows that no function returns a method name or a ranking.

**API and report (§10)**
- All endpoints listed in §10.
- Report sections: the 2×2 steps panel, Processing state, Missingness profile, diagnostic-policy notes and sensitivity tables, Plan, checkpoints.
- The live page has an intake form and the plan shown one entry at a time.
- Checked in Chromium: no console errors.

**Phase 2 hook (§5.9):** `methods_registry.yaml` with three seeded entries, and the experimental `min_prob` and `knn`. Neither is reachable from the plan, the executor, the CLI or the API (tested).

## Oracle results (R 4.3.3)

| Comparison | Largest difference |
|---|---|
| ComBat vs `sva::ComBat` | 5.3e-14 |
| quantile vs `preprocessCore` | 3.6e-14 |
| LOESS (skmisc) vs R `loess` | 1.7e-15 |
| LOESS drift vs proBatch | 7.5e-14 |
| median centering vs proBatch | 5.0e-14 |
| per-batch median centering vs proBatch | 5.0e-14 |
| `log2(x + 1)` | 5.3e-15 |
| PVCA vs an lme4 re-implementation | 0.0018 (crossed design) |
| A12.4 b1 vs `protDP::dpc` (datasetA) | PRISM 0.896, protDP 0.878 |

`pvca` itself could not be installed.

## Golden tests on the two real files

The v3 numbers still hold:
- floor ties: 397/1,174 and 20/11,083;
- 8 subjects;
- the 12,301.6 cell flagged.

v4 expectations, checked on the first run:

| Expectation | Result |
|---|---|
| Metabolomics: 0 NaN, 397 floor-tied features in the cell encoding | as expected |
| Metabolomics: A12.4 `not_applicable`, A4 item 4 as the substitute | as expected |
| `confirm_processing_state` appears for both files | as expected, but **for different reasons** (see below) |
| G5 and S1 `Insufficient metadata` until answered | as expected |
| Metabolomics A7 `insufficient_metadata`; S2 "needs batch variable" | as expected |
| SomaScan A7 uses its plate variables; nesting reported | **different** (see below) |

The two differences:
- **Why the SomaScan question appears.** Metabolomics gets its question because its declared `imputed: no` and `normalized: no` do not match the evidence. SomaScan's only declared item (`imputed: no`) does match: 20 of 11,083 features are floor-tied, below the 1% indicator. Its questions appear because `log_transformed` and `normalized` were declared `not_sure`. The SomaScan platform note is shown on `normalized`.
- **SomaScan has no plate variables in this session.** The golden wizard run skips the sample-metadata step and the expression file has no plate column. So A7 has no batch candidate and the plate–animal nesting cannot appear. S2 is `Insufficient metadata: needs batch variable`, and no batch correction is proposed. The nesting rule itself (`S5-nesting`: S2 becomes `Do not automate` when subjects are nested in the batch) is tested on the planted data.

## Changes to v3 behaviour

- Two v3 tests were updated:
  - "A12" now exists (the unknown-audit test uses "A42");
  - `partial` is now a valid status.
- A4's declared-vs-observed and A13 apply to S0 only. After PRISM's own ops, the declaration would be compared with the wrong data.
- A8's same- vs different-subject correlation now uses pairwise-complete features instead of complete features.

## Disagreements between v3, v3.1 and v4

See `reports/v4_preflight.md` §5. In short:
- v3.1 is not in the repository; each feature v4 references from it was inferred from v4's own wording.
- scipy and pyyaml are now dependencies.
- statsmodels, scikit-misc and scikit-learn are optional.

## Things you need to decide or check (§14, not decided here)

1. **The package license.** The port is GPL-3 derived: `DESCRIPTION` says GPL-3, and the repository has no separate license text. Check with the university licensing office before publishing anything. `LICENSING.md` says the license is undecided and that nothing is to be published.
2. **The processing-history answers for the two real files.**
   - Metabolomics: the floor ties and the median-scaling signature conflict with the declared `imputed: no` and `normalized: no`.
   - SomaScan: data are vendor-normalized, but the note is unverified.
   - Both files had `log_transformed` and `normalized` declared `not_sure`.
3. **Verify or drop:**
   - Dunn et al. 2011 (QC drift; G-R5 stays `unverified`);
   - the 25-sample batch-size suggestion;
   - the RSD levels;
   - the SomaScan vendor steps;
   - Kong et al. 2022's decision chart.

   Also check the citation details in `evidence.yaml` marked `citation_checked: false`: titles and pages were filled from memory.
4. **Threshold calibration:**
   - `sample_median_spread_mad` (0.1 log2);
   - `median_scaling_mad` (0.5);
   - the `pattern_class` rules (§5.7, listed in `a12_missingness_profile.PATTERN_RULES`);
   - the feature-class cutoffs (§5.6, `a12_low_abundance_fraction`).

   Also new heuristic parameters introduced in this build:
   - `diag_min_features`, `diag_min_share`, `diag_sample_missing_max`;
   - `a12_b1_negligible`, `a12_batch_absence_share`;
   - `blank_dominated_ratio`, `qc_drift_share`, `neff_ratio_note`.
5. **Whether to replace PRISM's detection-curve estimator with protDP's.** On datasetA, b1 was 0.896 (PRISM) vs 0.878 (protDP). protDP is Apache-2.0.

## Other open points

- **PVCA.** The Bioconductor `pvca` package was not available, so the oracle is an lme4 re-implementation. On designs with singular fits, statsmodels and lme4 differ by up to about 0.04.
- **proBatch version.** The port follows the GitHub master (1.3.1, April 2025). Its relation to the current Bioconductor release is not verified.
- **Rule R3.** It is implemented as "flagged samples present": quantile normalization and sample exclusion become `Do not automate`. The intake has no field for "many features expected to differ".
