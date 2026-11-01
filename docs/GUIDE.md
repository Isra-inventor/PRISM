# PRISM: the complete guide

PRISM helps you choose preprocessing for a **proteomics abundance table** (SomaScan, Olink, MaxQuant,
DIA-NN, Spectronaut, FragPipe protein-level exports) that fits your study design. It does this in three
stages, each one built from the steps below:

1. **Recognize the file** (steps 1-10): what every column is, which values are measurements, which
   samples are study samples, what the design is. You confirm everything; the AI only proposes.
2. **Audit the data** (step 11): measure the data with a fixed battery of checks. The audit never
   changes the data and never recommends.
3. **Plan, act, verify** (steps 12-13): a rule base turns the audit into a plan of preprocessing steps,
   each with a status. Nothing changes the data until you approve a step; each approved step makes a new
   *data state*, the audit is re-run on it, and at the end the same measurements are compared before and
   after.

Other omics files are recognized but marked out of scope. This guide follows the order you use PRISM in.

---

## The idea behind it: two axes

PRISM sorts every preprocessing step on two independent axes (Build Request v4):

| | **Global** (needs only the data and its technical metadata) | **Study-dependent** (needs the research question or the design) |
|---|---|---|
| **Audit** (measures, never changes data) | G1 integrity, G2 processing history and scale, G4 initial assessment, G7 missingness profile | S10 confounding and design (and the inputs to S4, S5) |
| **Action** (changes data, only after your approval) | G3 technical-record cleaning, G5 log transform, G6 drift correction on pooled QC | S1 normalization, S2 batch correction, S3 protein filtering, S4 sample exclusion, S5 repeated-measures constraints, S6 imputation (later stage) |

"Global" does not mean "always run": a log transform still needs the scale check, drift correction
still needs QC samples and run order. "Study-dependent" means the step needs at least one answer only
you can give: the contrast variable, the variables to protect, the question type, or the downstream
analysis family (the *intake*, step 12).

The whole battery of audits is the measuring instrument; actions are interventions; verification is
re-measurement with the same instrument. The steps and their properties live in one file,
`prism/registry/steps.yaml`, which the code, the plan and this guide read.

---

## Before you start (Colab)

1. **GitHub token (once).** GitHub → Settings → Developer settings → Personal access tokens →
   Fine-grained tokens → Generate new token; repository access *Only select repositories* →
   `Isra-inventor/PRISM`; permission *Contents: Read-only*. In Colab: key icon (Secrets) → Add new secret →
   name `GITHUB_TOKEN` → Notebook access on.
2. **Gemini key.** From https://aistudio.google.com/apikey. Either save it in Colab Secrets as
   `GEMINI_API_KEY`, or let PRISM ask you once per session (hidden input; kept in memory only, never
   written to a file).
3. **Data** in Google Drive, e.g. `MyDrive/PRISM_data/`. The repository holds no data.

The whole flow in code:

```python
import prism
table = prism.read(f"{DATA}/SomaExpr_common.csv")      # step 1
s = prism.start(table)                                 # steps 2-4 (the AI proposes)
s.todo()                                               # steps 5-9: what is still open, with the command for each
...                                                    # confirm / correct until s.todo() is empty
s.review(); s.save(f"{OUT}/soma")                      # step 10

st = prism.Study(s, folder=f"{OUT}/soma/work")         # steps 11-13
st.audit()                                             # the full audit (C0)
st.questions()                                         # what only you can answer
st.set_intake(contrast_variable="Group", downstream_family="differential_abundance")
plan = st.build_plan(); plan                           # every step with its status and candidates
st.approve("A1B1:G5")                                  # one step at a time; rebuild the plan after each
st.verify(); st.replay(); st.save()
```

Notebooks `01` to `10` in `notebooks/` run each part on your files.

---

# Part 1. Recognizing the file (steps 1-10)

## Step 1. Read the table exactly as written (`prism.read`, `prism/table.py`)

**What it does.** Reads a CSV or TSV file with every cell as text, so nothing is changed on reading: an
ID like `007` stays `007`, a gene name like `MARCH1` or `1E5` is never turned into a date or a number.
The delimiter (comma, tab, semicolon) is detected from the first 20 lines.

**What it records per column** (facts only):
- *kind*: `numeric` if every non-missing cell reads as a number, `text` otherwise, `empty` if every cell
  is missing;
- missing cells, distinct values, the first value.

**Missing values** are exactly these tokens: empty, `NA`, `N/A`, `NaN`, `nan`, `null`, `NULL`, `None`.
Anything else (for example `<LOD`) is a value, and is caught later by integrity (G1).

**You:** look at `table` (a summary table per column).

## Step 2. A first layout proposal by code (`prism.propose_layout`, `prism/layout.py`)

**What it does.** A quick, rule-based proposal used as a *hint* for the AI and as a fallback:
- *orientation*: if there are more than twice as many numeric columns as rows, one row per sample
  (`samples_in_rows`), else one row per protein (`samples_in_columns`);
- *the ID column*: the first text column with a value in every row and none repeated (a numeric column
  only if it is the first column and holds whole numbers); otherwise rows are numbered;
- *measurements*: every numeric column (with samples in rows, whole-number columns with at most 5
  values are treated as codes, e.g. a 0/1 sex code).

Numeric annotation columns (a CV, a limit of detection, a Kd) look like samples by type alone; the
proposal says so, and the AI (step 4) and you (step 5) sort them out.

## Step 3. What the AI may see: the digest (`prism.digest`, `prism/digest.py`) and the connection (`prism/ai.py`)

**The AI never receives data rows.** It receives, per column: name, position, kind, missing share,
distinct values and a few statistics (min, median, max, share of zeros and negatives, whole numbers or
not, how many powers of ten the values span). Three rules keep it small and private:
1. **Name templates.** Numeric columns whose names differ only in digits (`S01 ... S27`,
   `Amino Acid_1 ... _400`) become one entry with statistics over all of them.
2. **Example values only from repetitive text columns.** A text column where more than half of the
   values are distinct (protein names, sequence IDs, patient codes) sends only the *shape* of its values
   (`SL000001` → `AA999999`). A column with few values (Type, Organism, Group) sends up to 12 values with
   their counts, so the AI can tell what it is. `examples=False` turns example values off completely.
   For text columns PRISM also counts IDs starting with known decoy / contaminant prefixes
   (`REV__`, `CON__`, `contam_` ...), so the AI can propose a flag without seeing the IDs.
3. **No file name**, only its extension.

`prism.show(d)` prints exactly what would be sent.

**The connection.** Gemini over HTTPS, temperature 0, answers must be JSON in a fixed shape (a response
schema). Models are tried in order (gemini-3.5-flash-lite first); an overloaded or missing model falls
through to the next, a rate limit is waited out when Google says how long (up to 35 s), a bad key stops
at once. Every call records which model answered.

## Step 4. The AI groups and labels the columns (`prism.propose_groups`, `prism/grouping.py`, `prism/prompts.py`)

**What it does.** The AI reads the digest and the briefing in `prism/prompts.py` (what PRISM is, the
current scope, recognition hints for SomaScan, Olink, MaxQuant, DIA-NN, Spectronaut, FragPipe) and
proposes:
- the layout (`samples_in_columns`, `samples_in_rows` or `long`) with confidence and evidence;
- *groups*: which columns are one kind of thing (all sample columns of one measurement; each annotation
  on its own), each with a role (`feature_id`, `feature_annotation`, `value`, `sample_id`,
  `sample_metadata`, `ignore`, `unresolved`), a label in its own words, for sample information an
  `audit_kind` (subject, time point, batch, run order, replicate, sample type, group, covariate, other),
  and for flag columns the values or prefixes that mark a row as suspect;
- *assays*: what assay each value group belongs to, its omics family and whether it is in scope.

**Code checks the answer, it never decides the grouping itself:** a template is expanded to its
columns; every column named must exist and is claimed once (first claim wins); every column the AI
did not mention becomes its own `unresolved` group (nothing is silently dropped); merges apply only when
both groups exist; closed fields must hold allowed values. Everything refused is listed in
`g.rejected` with the reason.

## Step 5. Confirm or correct the groups (`prism/schema.py`)

`prism.start(table)` runs steps 2-4 and returns a **Schema**. `s.todo()` always lists what is still
open, with the exact command to close it.

- `s.groups()`: one row per group; `s.show("g3")`: every column, the AI's evidence and the facts.
- `s.confirm("g1", "g2")`, `s.confirm_all()` (confirms the layout and every AI group you did not change;
  unresolved groups stay open).
- `s.set("g7", role="feature_annotation", label="dissociation constant")`,
  `s.split("g9", ["Normal Plasma RFU"], role=..., label=...)`, `s.merge("g10", "g11")`,
  `s.set_layout(...)`.

**Provenance.** Every decision records who made it: `computed` (worked out by code from facts),
`ai_proposed` (proposed, not yet looked at: still open), `ai_proposed_confirmed`,
`ai_proposed_corrected`, `user_set`. Every action is appended to `s.log` with a time stamp.

## Step 6. Protein ID and flagged rows (`prism/features.py`)

- **Protein ID.** Proposed from the AI's assay information; code checks that every row has an ID and
  none repeats (several columns make a composite ID joined with `|`). `s.feature_id()`,
  `s.confirm_feature_id()` or `s.set_feature_id([...])`. With samples in rows the proteins are the
  value column names.
- **Flagged rows.** Rows that are not ordinary measured proteins: SomaScan controls (Type other than
  "Protein"), other species, decoys (`REV__`), contaminants (`CON__`, MaxQuant "Potential contaminant" =
  `+`). The AI proposes the column and values; `s.values_of("Type")` shows a column's values;
  `s.flag_rows("Type", values=[...])`, `s.flag_rows("feature_id", prefixes=["CON__"])`,
  `s.unflag(...)`, `s.confirm_flags()`. **Flagged rows are kept and labelled**, never dropped here;
  removing them is an action (G3) you approve later.

## Step 7. Values and samples (`prism/values.py`)

- **Blocks.** Every kept value group is a block with an id `A1B1`, `A1B2`, `A2B1` ... (assay, block).
  `s.blocks()` gives proteins, samples, missing share, zeros, negatives, median. Nothing ranks blocks.
- **Zeros.** Whenever a block holds zeros PRISM asks what they mean: `s.set_block("A1B1",
  zero_means="not_detected")` or `"zero"`. The values are not changed here; recoding is an approved
  action later.
- **Assays.** `s.assays_table()`, `s.set_assay(...)`, `s.confirm_values()`.
- **Samples.** Sample names come from the value columns (shared text such as `LFQ intensity ` is
  removed when there are several blocks), from the sample ID column, or from a long table.
  `s.propose_sample_types()` sends the **sample names only** to the AI, which proposes patterns for QC,
  blank, pool, calibrator and reference samples; everything else is a study sample.
  `s.set_sample_type("QC_*", "qc")`, `s.confirm_samples()`.
- **Long tables (Olink).** The matrix is built by pivoting the value column on the protein ID and the
  sample ID; a protein-sample pair that occurs twice is reported (often several plates or panels).

## Step 8. Sample information and study design (`prism/design.py`)

- **A sample-information file** (patient, visit, plate, group ...): `s.add_sample_info(path)`. The AI
  sees its digest and proposes the sample ID column and each column's audit kind; code reports how many
  sample names match. `s.sample_info()`, `s.set_info_column(...)`, `s.confirm_sample_info()`.
- **Design.** Where the subject and the time point come from: a column, the sample names
  (`s.name_patterns()` shows how the names split, e.g. `P01_T2` at the last `_`), or nowhere.
  `s.set_design(subject="Patient", time="Visit", time_unit="weeks")` or
  `s.set_design(derive={...}, time_unit=...)`. Code computes the facts (subjects, samples per subject,
  time points) and a label such as "longitudinal, balanced (2 samples per subject)". **The time unit is
  always asked, never guessed.** `s.confirm_design()`.

## Step 9. Processing history (`prism/history.py`)

Five questions per assay: normalized? log-transformed? imputed? batch-corrected? proteins or samples
removed before upload? You answer yes / no / not_sure. Code adds hints from the numbers (negative values
usually mean log scale; sample medians within 2% of each other suggest normalization; no missing value
and no zero may mean filling). A hint raises the question; it never answers it. The audit compares your
answers with the evidence (A13).

## Step 10. Review and save (`prism/outputs.py`)

`s.review()` prints every decision with its provenance; `s.save(folder)` refuses while anything is open
and writes:
- `schema.json`: every decision, its provenance, the log, the AI models used, and the fact that no data
  rows were sent;
- `value_matrix_A1_B1.csv` (one per block): proteins x samples, **values exactly as in the file**
  (missing cells empty; nothing filled, scaled or removed);
- `feature_metadata.csv` (ID, annotations, flagged, flag_reason), `sample_metadata.csv` (sample type,
  study or not, sample information, subject, time);
- `import_manifest.json`: source file name, size, SHA-256 fingerprint, counts, models.

`prism.load(schema_path, table)` rebuilds the schema on the same file without asking again (it checks
the columns match and warns if the contents changed).

---

# Part 2. The audit (step 11)

`prism.Audit(s)` (or `st.audit()`) runs on a schema with nothing open. Every check returns a
**finding**:
- *status*: `ok`; `note` (worth knowing, nothing wrong by itself); `problem` (would make later results
  wrong: fix the file or the schema); `not_applicable` / `insufficient_data` / `insufficient_metadata`
  (could not be assessed, and why);
- *sentences* in careful wording ("consistent with", "a plausible sign of"; never "diagnosed");
- *measures*: every number, per block; *method*: which data and how; *feeds*: which later steps use it.

**The audit set.** From A2 on, checks use **study samples only** and **unflagged proteins**. QC, blank,
pool samples come back only where a check needs them (A6, verification).

**The diagnostic copy.** Statistics that need a symmetric scale use log2 of the values *only when A2
found a linear scale*; values already on a log scale (e.g. Olink NPX) are used as they are. This copy
is never saved.

**Missing values in diagnostics (v4 section 3).** PCA, PERMANOVA, distances and correlations need
complete data, and **PRISM never fills values to get it**. Default: only proteins observed in every
sample; if fewer than 20 proteins (or 10%) remain, 75% and then 50% are tried, dropping samples missing
more than half their values. Every finding that uses this reports the policy, how many proteins were
used, and the warning that *these proteins are biased toward higher-abundance, consistently detected
ones*. Correlations use pairwise-complete proteins. The **sensitivity block** (A7, A11) recomputes the
main conclusion under every policy (base, without floor-tied proteins, floor values treated as
censored, 100 / 75 / 50%) and says when it changes; no variant is preferred.

**Permutation tests.** Associations with sample variables use restricted permutations that respect the
design: whole subjects are shuffled when the variable is constant within subject, values are shuffled
within subject when it varies within subject, samples freely otherwise. 999 permutations (seed 0), or
all of them when there are at most 10,000. p-values are corrected across tests (Benjamini-Hochberg).

`a.run_all()` runs the full battery in this order:

### G1 / A1 Integrity (`prism/audit/integrity.py`)
On all samples and all proteins: value cells that are not numbers (they would silently become missing)
→ problem; infinite values → problem; repeated protein IDs or sample names → problem; samples with no
value → problem; proteins with no value, fewer than 3 values, or one constant value → note; two samples
or two proteins with exactly the same values (copy-paste or export error) → note; one non-zero value
filling at least 1% of all cells (a placeholder or a written-in detection limit) → note; study samples
without a subject → note. It also states the audit set.

### G2 / A2 Measurement scale (`prism/audit/scale.py`)
Ordered tests; the first that applies wins: count-like (every value a non-negative whole number),
proportion-like (all between 0 and 1), compositional-like (every sample sums to the same total) — all
three out of scope —, signed (negative values: already centred or log, e.g. NPX), linear (right-skewed
per protein and log2 makes it more symmetric: raw intensities, RFU), log-like (log2 does not make it
more symmetric). Also the shared-median signature: every protein's median nearly the same (MAD of log2
medians below 0.5) suggests proteins were scaled one by one.

### G2 / A13 Processing state (`prism/audit/processing_state.py`)
Your step-9 answers against the evidence, per block: *declared*, *observed* (sentences with numbers),
*match* (matches / does_not_match / not_assessed), *confirmed* (your answer after seeing a mismatch).
- log-transformed ↔ the A2 class and the A3 slope;
- normalized ↔ the shared-median signature and how close the sample medians are (MAD below 0.05);
- imputed ↔ floor ties (proteins with two or more samples exactly at their minimum value) and whether
  anything is missing;
- batch-corrected, removed before upload: cannot be seen in the values.

Absence of a signature is not proof ("declared yes, no signature" is not_assessed). Platform notes
(`prism/audit/platform_notes.yaml`: SomaScan vendor normalization, Olink NPX is log2, MaxLFQ is
normalized) are shown next to a mismatch and never decide; each is marked "verify with vendor
documentation". **Gate:** while a mismatch is unconfirmed, G5 and S1 are "Insufficient metadata:
confirm processing state". Answer with `a.confirm_processing_state(item, value, assay=...)` (or
`st.confirm_processing_state`); confirmations are kept in `overrides.json`, not in `schema.json`.

### G4 / A3 Distribution and variance-mean (`prism/audit/distribution.py`)
Slope of log SD on log mean per protein (95% CI from 999 bootstrap resamples over proteins): near 1 =
multiplicative noise (evidence for a log transform), near 0 = spread independent of level; the same
relation on the diagnostic copy (Spearman); skewness; relative log expression (each value minus its
protein's median, summarized per sample); and the **S1 decision node**: if sample medians differ (MAD
above 0.1 on the diagnostic scale), does that follow a technical variable (batch, run order) or a design
variable (group, time)? A shift that follows the design could be biology.

### G4 / A6 Noise, QC and sample quality (`prism/audit/quality.py`)
QC repeatability (per-protein RSD over QC and pool samples, shares above 20% and 30%; the SD on the log
scale for log data); blank signal against study signal; drift over run order (sample medians against run
order: Spearman, slope, LOWESS range, PC1; per-protein trends in the QC samples); each study sample's
correlation with the median profile; vendor per-protein QC columns (CV, LoDB, ColCheck ...); flagged
against unflagged proteins.

### G7 / A4 Missingness (`prism/audit/missingness.py`)
Missing values per protein and per sample; zeros counted apart (never recoded here); floor ties (≥ 2
samples exactly at a protein's minimum, proteins with ≥ 5 values; reported from 1% of proteins);
abundance dependence (Spearman rho between a protein's missing rate and its mean value; rho ≤ −0.1 is
consistent with a detection limit).

### G7 / A12 Missingness profile (`prism/audit/missingness.py`) — nothing is ever filled
- **5.1** every cell is exactly one of missing, zero, floor-tied, observed;
- **5.2** missing rates by plate, group and time; share of proteins with none / ≤ 10% / ≤ 25% / ≤ 50%
  missing (reported, none chosen);
- **5.3** batch-wide absence: proteins with no value in a whole batch level (≥ 3 samples), and batch
  coverage. This pattern breaks imputation across batches and ComBat;
- **5.4** the detection curve: `logit P(detected) = b0 + b1 × mean log2 value`, a zero-truncated
  binomial fitted by maximum likelihood, 95% CI of b1 from 200 bootstrap resamples. It is a simplified
  reference method (the mean is over detected values only, so it is biased upward for censored proteins;
  Li and Smyth 2023 is the careful version); read b1 as descriptive. Skipped when nothing is missing or
  fewer than 50 proteins are partly detected;
- **5.5** the missing pattern against the design (per-sample rates against every variable; per-protein
  missing indicator against each batch, chi-square with permutation p, BH over proteins);
- **5.6** protein classes, in order: absent from a whole batch, lowest third of abundance, scattered;
  complete otherwise (descriptive labels, never mechanisms);
- **5.7** the overall pattern (none, mostly intensity-dependent, batch-associated, mixed, insufficient
  data) with three yes / no / indeterminate indicators. MCAR, MAR and MNAR are never used as a
  diagnosis: *models in which values are missing at random and models in which they are not can fit the
  observed data equally well (Molenberghs et al. 2008)*;
- **5.8** the hand-off: imputation families that fit the profile (`prism/audit/imputation_catalog.yaml`),
  with references and caveats, **not ranked and not applied**.

### S10 / A7 Batch structure (`prism/audit/batch.py`)
Batch candidates are sample variables with audit kind batch or run order; design variables are subject,
time, group, sample type, covariate.
- Structure of every batch against every design variable: confounded (cannot be separated), nested,
  crossed and balanced, or partially confounded; batches against each other (rule R9).
- Samples per batch (fewer than 25 is reported; a heuristic quoted in Cuklina et al. 2021 from other
  work, unverified) and coverage.
- **Relationship-stratified correlation** (this is where sample correlation is used as analysis): Pearson
  r between all sample pairs (pairwise-complete), as same / different subject and same / different batch;
  `batch_effect_on_correlation` = median r same batch − median r different batch.
- PCA (first 5 PCs) against every variable, and PERMANOVA, with permutations and BH; then the
  sensitivity block.
- Control proteins you mark (`a.mark_control_protein(...)`) get a trend against run order.

### S5 / A8 Repeated measures (`prism/audit/repeated.py`)
Subjects, samples per subject, singletons, balanced, paired; the time grid and irregular spacing;
variables constant within subject and nesting (e.g. subjects nested in plates: a per-plate correction
would absorb subject differences); ICC(1) per protein (the share of variation due to the subject);
same-subject against different-subject correlation.

### S10 / A5 Dimensionality and effective sample size (`prism/audit/dimensionality.py`)
n, p, n / p; the smallest design cell; sparsity; the Kish design effect with the median ICC:
`deff = 1 + (m − 1) × ICC`, `n_eff = n / deff` — how many independent samples the repeated measures are
worth.

### S4 / A11 Outliers (`prism/audit/outliers.py`) — flags only
Samples: distance to the median profile and PC1 / PC2 scores as modified z (median and MAD); |z| > 3.5
flags a sample (Iglewicz and Hoaglin; heuristic), with the sensitivity block; cells: the modified z within
each protein; whether flagged samples cluster in the run.

### G3 cleaning candidates (`prism/audit/cleaning.py`)
Classes of technical records, each a separate flag with IDs: proteins flagged in step 6 (per column),
all missing, constant, fewer than 3 values, blank-dominated. Nothing is removed by the audit.

---

# Part 3. Actions and data states (`prism/executor/`)

**Data states.** S0 is the confirmed data of step 10 (all proteins, all samples, as numbers) on the
confirmed scale (linear when you confirmed "not log-transformed", log2 when you confirmed it was). Every
approved action makes a **new** state; parents are never changed. The state id is a SHA-256 of the
parent id, the operation (op, step, parameters, seed, plan entry) and every matrix, so the same actions
on the same data always give the same id. With a folder, each state is written once to
`work/states/<id>/` (matrices, protein table, sample table, `state.json`).

**The operations** (Phase 1; there is no imputation operation, and one cannot be registered):

| Op | Step | What it does | Refuses when |
|---|---|---|---|
| `remove_features` | G3, S3 | drops proteins (rows); changes no value | unknown IDs, no reason |
| `remove_samples` | S4 | drops samples; each needs its own approval | fewer than 3 would remain |
| `recode_zero_as_missing` | G7 | exact zeros become missing (the one documented exception to "no op creates missing values") | you did not say zeros mean missing |
| `log_transform` | G5 | log2(x + c), c = half the smallest positive value (or none / fixed c) | not confirmed linear, already log, negative values |
| `arcsinh_transform` | G5 | arcsinh(x / cofactor), cofactor and justification required | same as above |
| `drift_qc_loess` | G6 | per protein, LOESS of QC values against run order, removed from every sample | fewer than 5 QC samples, QC not spanning the run, not on a log scale |
| `normalize_median` | S1 | x − median(sample) + median(all) | not on a log scale |
| `normalize_quantile` | S1 | quantile normalization (missing-aware form with missing values) | not on a log scale |
| `batch_center` | S2 | per protein, per batch: x − median(batch) + median(all) (or mean) | not on a log scale, fewer than 2 batches |
| `batch_combat` | S2 | ComBat, parametric, intercept-only | not on a log scale, any missing value, a batch with fewer than 2 samples |
| `drift_sample_loess` | S2 | per protein, per batch, LOESS on run order over study samples (proBatch); a protein whose fit fails is left unchanged | not on a log scale, fewer than 2 batches, run order for fewer than 8 samples |

**Invariants**, checked after every operation before anything is kept: the missing-value pattern is
identical (except recode_zero), no operation turns a missing value into a number or a number into a
missing value, only removals change the shape (and they change no value), values change only in the
proteins an operation says it touched. **Nothing runs without an approval record** (who, when, which
plan entry).

**The proBatch port** (`prism/ports/probatch/`, `prism/batch/`): the batch-correction methods of the
tutorial we anchor on (Cuklina et al. 2021) in Python, adapted from proBatch's source (GPL-3). It is kept
apart, reached only through `prism/batch/backend.py`, and `PRISM_DISABLE_PORTS=1` switches it off; see
`prism/ports/probatch/NOTICE.md`, `docs/probatch_port.md` and `LICENSING.md`. Six bugs found in the
proBatch source are not reproduced, each with a test. LOESS uses scikit-misc (the original loess code)
when installed, otherwise PRISM's own local regression; the engine used is in every op record.

---

# Part 4. The plan (step 12; `prism/plan/`, `prism/rules/`)

**The intake** (`st.set_intake(...)`) — four answers that unlock the study-dependent steps:
`contrast_variable` (the sample variable your question compares, or `none`), `protected_variables`
(default: the contrast) and `nuisance_variables` (default: the batch candidates), `question_type`
(`global_untargeted` / `targeted_features`), `downstream_family` (`differential_abundance`,
`clustering_exploration`, `prediction`, `integration`, `unsure`). An empty field is not an error; it
makes the dependent steps "Insufficient metadata" with the field named.

**Facts.** For each block, the findings are turned into one flat set of facts (`prism/plan/facts.py`):
the confirmed log state, the scale class, the A3 slope, the G3 classes, zeros and floors, the
missingness pattern, QC and run order, batch association and structure, confounding with the contrast,
median shifts, outliers, n and n_eff. A fact that could not be computed is *indeterminate*, and **a
condition on an indeterminate fact never holds, so it can never lead to "Recommended"**.

**Statuses.** Exactly four, plus one for scope: `Recommended`, `Alternative`, `Do not automate`,
`Insufficient metadata`, and `Not applicable` (counts, compositional data). Only Recommended and
Alternative candidates can be approved.

**The gate.** Each step lists what it needs (`steps.yaml`: data, QC samples, run order, batch variable,
contrast, ...). An action whose needs are not met is "Insufficient metadata: needs ...".

**The rules** (`prism/rules/rules_v0.yaml`, data, not code; all listed in Appendix B). Per step, the
first matching *decide* rule sets the status and candidates; *annotate* rules add warnings or change a
candidate's status; *scope* rules are checked before the gate. Rules R2 to R13 come from one expert
tutorial (Cuklina et al. 2021), so their evidence level is `expert_guidance` until paired with an
independent source.

**The joint node D2** (`joint_nodes.yaml`): batch correction is decided together with missingness. No
missing values: decided on its own. Missing values and a method that tolerates them (median centering,
LOESS): correct first, and if the later statistical method needs complete data, impute after correction
and within batch (hand-off). Missing values and ComBat: Do not automate (alternatives: batch_center, or
remove the incomplete proteins).

**Order** (`plan_order.yaml`): G3 → G7 (zeros, floors) → G5 → G6 → S1 → **C1** → S2 → S3 → S4 → **C2**.

**Also in the plan:** S3 reports, for thresholds 100 / 90 / 75 / 50 / 25%, how many proteins are
observed overall, within at least one contrast group, and per subject cluster (convention; you choose);
S4 lists flagged samples for your decision (Do not automate, one approval per sample); S5 writes the
design constraints (subject variable, within-subject variables, nesting, smallest cell, n_eff) that
batch correction and filtering must respect; the hand-off lists the imputation candidates.

**Approving.** `st.approve(entry_id, candidate=0)` checks that the entry and candidate are approvable
and that the plan is not stale, runs the operation on the current state, and re-runs the **C1 battery**
(A1, A2, A3, A4, A6, A7, A8, A12) on the new state. Rebuild the plan after each approval:
`st.build_plan()`. `st.reject(entry_id, reason=...)` is recorded too.

---

# Part 5. Verification, replay, saving (step 13)

**Checkpoints.** C0 = S0 with the full battery; C1 = after G3, G5, G6, S1 actions; C2 = after S2, S3, S4
actions.

**Verification** (`st.verify()`, at C2; `prism/audit/verification.py`): the same descriptive measures at
S0, at the last C1 state and now — batch association (PERMANOVA; PVCA through the port), the
relationship-stratified correlation, control proteins, QC variability, completeness — as before / after
tables, with negative controls (no batch clustering; same-batch r not above different-batch r) and
positive controls (same-subject or replicate r above unrelated r). **These measures check the chosen
plan; they are never used to rank or select methods** (the module has no such function, and a test
checks it).

**Replay** (`st.replay()`): re-applies every approved operation from S0 in memory and must reproduce the
final state id exactly.

**Saving** (`st.save()`): `plan.json`, `decisions.jsonl` (every approval and rejection),
`intake.json` (with its history), `overrides.json` (your confirmations), `design_constraints.json`;
the states are already in `work/states/`.

**Phase 2 hook** (`prism/phase2/`): `methods_registry.yaml` (whether later statistical methods tolerate
missing values; three seed entries only) and an experimental imputation module (`min_prob`, `knn`,
optionally within batch) that writes only missing cells. It is not imported by the plan: whether to
impute is decided later, with the statistical method.

---

# Part 6. What is not built, and what you need to decide or check

**Not built, or different from Build Request v4:**
1. Scope is **proteomics only** (v4 also covers metabolomics and lipidomics); other omics are recognized
   and marked out of scope.
2. **No command line, web API, sessions or HTML report yet.** PRISM is a Colab package; those come with
   the website rebuild.
3. **The comparisons against R** (proBatch, sva, preprocessCore, R loess, pvca) are not run: R is not
   available here. The port's own tests run (one per confirmed proBatch bug, behaviour, license
   isolation, the core without the port).
4. **S7, S8, S9 are not built** (scaling for multivariate methods; multi-layer harmonization A10;
   tissue / cell composition A9): single-assay proteomics files for now.
5. Answering **"unsure"** to a processing-state question opens the gate (a human confirmation), but the
   effective answer stays undecided, so G5 / S1 cannot become Recommended on it.
6. Control proteins are marked with `a.mark_control_protein(...)` (v4's override `control_feature`);
   overrides `feature_level` and `sample_role` are not needed (sample types come from step 7).
7. The golden tests on your two real files from v3 (e.g. 20 of 11,083 SomaScan proteins with floor
   ties) are not run here: the files are not in the repository. Compare in notebook 08.
8. Nothing has been run on your real files or against the real Gemini yet from this package.

**For you to decide or check (v4 section 14):**
1. **The package license** (the proBatch port is GPL-3 derived). Check with the university licensing
   office before publishing anything. Nothing is published until then.
2. **Your processing-history answers** for the real files (SomaScan data are usually vendor-normalized).
3. **Verify or drop:** Dunn et al. 2011 (QC drift, G-R5), the 25-sample batch-size suggestion, the RSD
   levels (Broadhurst et al. 2018), the SomaScan / Olink / MaxQuant platform notes, Kong et al. 2022's
   decision chart.
4. **Calibrate the thresholds** marked heuristic (Appendix A), especially `sample_median_spread_mad`,
   `median_scaling_mad`, the missingness pattern rules and the protein-class cut-offs.
5. Whether PRISM's detection-curve estimator should be replaced by Li and Smyth's implementation after a
   comparison.

---

## Appendix A. Every audit parameter

From `prism/audit/params.py` and `prism/registry/parameters.yaml` (generated).

| Parameter | Value | What it does | Evidence level |
|---|---|---|---|
| `sentinel_share` | 0.01 | one non-zero value covering this share of cells reads as a placeholder | heuristic |
| `sentinel_min_cells` | 3 | ... and at least this many cells | convention |
| `min_observed` | 3 | proteins with fewer observed values are sparse | convention |
| `median_scaling_mad` | 0.5 | MAD of log2 per-protein medians below this: proteins share a common median | heuristic |
| `narrow_median_spread` | 0.05 | MAD (log2) of sample medians below this reads as normalized | heuristic |
| `floor_min_observed` | 5 | proteins need this many observed values for floor ties | convention |
| `floor_tie_share_indicator` | 0.01 | floor ties count as "present" from this share of proteins | heuristic |
| `compositional_cv` | 0.001 | sample sums this constant read as compositional | heuristic |
| `abundance_rho_indicator` | -0.1 | Spearman rho at or below this gets the detection-limit sentence | heuristic |
| `a12_min_level_n` | 3 | a batch level needs this many samples for batch-wide absence | convention |
| `a12_min_partial_features` | 50 | proteins with 0 < detected < n needed for the detection curve | heuristic |
| `a12_bootstrap` | 200 | bootstrap resamples over proteins for the CI of b1 | convention |
| `a12_b1_negligible` | 0.1 | /b1/ CI inside +-this (per log2 unit) reads as "no" intensity dependence | heuristic |
| `a12_batch_absence_share` | 0.01 | batch-wide absence above this share reads as batch-associated | heuristic |
| `a12_low_abundance_fraction` | 0.3333 | "lowest third" of proteins by mean observed log2 value | heuristic |
| `a12_feature_permutations` | 999 | per-protein missing-indicator tests: Monte Carlo permutations | convention |
| `bootstrap` | 999 | A3 bootstrap resamples | convention |
| `sample_median_spread_mad` | 0.1 | MAD (log2) of sample medians above this reads as "medians differ" | heuristic |
| `rsd_levels` | [20, 30] | QC RSD levels, percent | heuristic (broadhurst2018) |
| `run_order_rho_indicator` | 0.3 | /Spearman rho/ of sample medians with run order reported from here | heuristic |
| `qc_drift_share` | 0.2 | share of proteins with /rho/ > 0.5 against run order in QC samples | heuristic |
| `diag_min_features` | 20 | complete_features(100) needs this many proteins, else q = 75, 50 | heuristic |
| `diag_min_share` | 0.1 | ... and this share of all proteins | heuristic |
| `diag_sample_missing_max` | 0.5 | for q < 100, samples missing more than this share are dropped | heuristic |
| `categorical_max_levels` | 20 | categorical variables need 2 to this many levels | convention |
| `time_ordered_max_levels` | 12 | time is an ordered factor up to this many values, else numeric | convention |
| `max_pcs` | 5 | k = min(max_pcs, n - 1) | convention |
| `modified_z_cutoff` | 3.5 | outlier cut-off (Iglewicz and Hoaglin) | heuristic (iglewicz1993) |
| `icc_indicator` | 0.5 | share of proteins with ICC above this is reported | heuristic |
| `batch_min_samples` | 25 | quoted in Cuklina et al. 2021 as a suggestion from other work; unverified | heuristic (cuklina2021) |
| `blank_dominated_ratio` | 0.5 | G3 flags proteins whose mean blank signal is this share of the study mean | heuristic |
| `near_duplicate_r` | 0.9999 | samples correlating above this read as near-duplicates | heuristic |
| `neff_ratio_note` | 0.5 | the plan notes when n_eff / n is below this (S5) | heuristic |
| `seed` | 0 |  | convention |
| `permutations` | 999 |  | convention |
| `exhaustive_max` | 10000 | enumerate every distinct permutation up to this many | convention |


## Appendix B. Every rule

From `prism/rules/rules_v0.yaml` (generated). *decide*: the first match of a step sets its status; *annotate*: adds text or changes a candidate; *scope*: checked before the gate.

| Rule | Step | Kind | When | Result | Evidence level |
|---|---|---|---|---|---|
| G-R6 | G3 | decide | `g3.n_candidates gt 0` | Recommended: remove the flagged features ($item.class) | convention |
| G7-zeros-yes | G7-zeros | decide | `zeros.present is_true AND zeros.answer eq yes` | Recommended: recode zeros as missing values | convention |
| G7-zeros-ask | G7-zeros | decide | `zeros.question is_true` | Insufficient metadata | convention |
| M-R2 | G7-floor | decide | `floor.present is_true AND floor.answer exists False` | Insufficient metadata | convention |
| G-R3 | G5 | scope | `scale.out_of_scope is_true OR scale.class eq continuous_signed` | Not applicable | convention |
| G-R1 | G5 | scope | `processing_state.any_unconfirmed is_true` | Insufficient metadata | convention |
| G-R4 | G5 | decide | `processing_state.log_transformed eq yes OR state_scale in ['log2', 'arcsinh']` | Do not automate | convention |
| G-R2 | G5 | decide | `scale.class eq continuous_linear AND a3.slope_ci_excludes_zero is_true AND processing_state.log_transformed eq no` | Recommended: log2(x + c), c = 0.5 x smallest positive value; arcsinh(x / cofactor): needs a justified cofactor | expert_guidance |
| G-R2b | G5 | decide | `scale.class eq continuous_linear AND processing_state.log_transformed eq no` | Alternative: log2(x + c), c = 0.5 x smallest positive value | convention |
| G-R5 | G6 | decide | `qc.n ge 5 AND run_order.present is_true AND qc.drift is_true` | Recommended: LOESS on pooled QC against run order | unverified |
| S1-node-contrast | S1 | decide | `median.differ is_true AND median.contrast is_true` | Do not automate | expert_guidance |
| R2 | S1 | decide | `median.differ is_true AND median.technical is_true AND median.contrast is_false` | Recommended: median centering; quantile normalization | expert_guidance |
| S1-untracked | S1 | decide | `median.differ is_true` | Alternative: median centering; quantile normalization | expert_guidance |
| S1-similar | S1 | decide | `median.differ is_false` | Alternative: median centering | heuristic |
| R3 | S1 | annotate | `outliers.flagged nonempty` | (adds) — sets normalize_quantile to Do not automate | expert_guidance |
| R13 | S2 | scope | `batch.present is_false AND run_order.present is_false` | Insufficient metadata | expert_guidance |
| R10 | S2 | decide | `confound.contrast is_true` | Do not automate | expert_guidance |
| S5-nesting | S2 | decide | `nested.subject_in_batch is_true` | Do not automate | expert_guidance |
| R9 | S2 | decide | `batch.two_confounded is_true` | Insufficient metadata | expert_guidance |
| R4 | S2 | decide | `checkpoint in ['C1', 'C2'] AND batch.association is_false AND batch.same_r_above_diff is_false` | Recommended: do not correct: no batch clustering and same-batch r not above different-batch r | expert_guidance |
| R5 | S2 | decide | `run_order.drift is_true AND batch.present is_true` | Recommended: sample-based LOESS drift (proBatch), span 0.75: check the span; then per-batch median centering | expert_guidance |
| R6 | S2 | decide | `batch.association is_true` | Recommended: per-batch median centering; ComBat (parametric, ~1) | expert_guidance |
| S2-none | S2 | decide | `batch.association is_false` | Alternative: no correction: the batch candidates show no association | expert_guidance |
| R11 | S2 | annotate | `batch.small is_true` | (adds) | heuristic |
| R12 | S2 | annotate | `intake.downstream_family eq differential_abundance AND batch.present is_true` | (adds): batch as a covariate in the final model (no change to the data) | expert_guidance |
| M-R3 | S2 | annotate | `missing.batch_wide_absent_any is_true` | (adds) — sets batch_combat to Do not automate | literature_independent |
| M-R4 | S2 | annotate | `missing.batch_association eq yes` | (adds) | literature_independent |
| S3-table | S3 | decide | `missing.any is_true` | Alternative | convention |
| R3-S4 | S4 | decide | `outliers.flagged nonempty` | Do not automate | heuristic |
