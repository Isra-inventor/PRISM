# PRISM — Step 0: Schema recognition and guided confirmation

**P**reprocessing and **R**easoning for **I**nformed **S**tatistical **M**ethodology.

PRISM helps you preprocess omics data (proteomics and metabolomics for now). It doesn't push every
dataset through one fixed pipeline. It first works out what the data is, then reasons about what to do with it.
Step 0: you upload a quantified table, PRISM recognizes its structure, you confirm it step by step, and the
result is written out as a schema plus canonical tables. On top of Step 0 (v3, package `prism/`): **sessions** of
one or more datasets, **schema import** (skip the wizard with a `schema.json` you confirmed before), a
**multi-dataset merge** of sample IDs, sample tables and design, and the **Tier 1 audit**: eleven deterministic
audits that report numbers and templated indicator sentences, never verdicts. Normalization, imputation, filtering
and recommendations (Tier 2) are later phases.

## Run it

Works with **Python 3.7 or newer**. `requirements.txt` picks versions that match your Python.
Run these commands from the project's top folder (the one that contains `backend/`).

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload
```

Open <http://127.0.0.1:8000/tool.html>. The home page is at `/`.
`python main.py` does **not** work: start the server through uvicorn as shown above.

To try it without your own data, upload any file from `tests/fixtures/`.

## AI provider and key

The AI only labels column groups. It never sees raw rows. You can switch it off in the page ("AI suggestions" toggle),
and the wizard still works in manual mode with hints computed from the data.

Put the key in a file named **`.env`** next to this README (copy `.env.example`). `.env` is git-ignored.

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | – | Google Gemini key (default provider). Get one at <https://aistudio.google.com/apikey> |
| `LLM_PROVIDER` | auto | `gemini`, `anthropic`, `openai` or `mock`. Auto picks whichever key is set |
| `PRISM_LLM_MODEL` | provider default | Model name(s), comma-separated. For Gemini, several models are tried in order, because individual models are often briefly overloaded |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | – | For those providers. Needs `pip install anthropic` / `pip install openai` |
| `AI_SEND_EXAMPLE_VALUES` | `true` | When `false`, only column names and statistics are sent |
| `PRISM_MAX_UPLOAD_MB` | `200` | Upload size limit |

`LLM_PROVIDER=mock` gives an offline demo with canned, rule-based proposals. The tests use the same mock.

To check that the AI works on your machine, run `python -m backend.check_ai`.
The server terminal prints every AI step as `[PRISM AI] …` lines, including the exact error from the provider.

## How it works

```
upload ─► parse (strings only, report oddities) ─► facts per column: statistics + shared name parts
      ─► known-format signature (MaxQuant, DIA-NN, Spectronaut, FragPipe, mz/rt) → a one-line hint
      ─► the AI GROUPS the columns and labels each group, briefed by backend/briefing.md
         (wide files: chunks of 150 columns + one consolidation call)
      ─► code applies the grouping only where columns / ids exist; every column ends up in exactly
         one group; structural checks; contradicted claims become "unresolved"
      ─► 8-step wizard: every proposal is editable in place; "Disagree?" re-asks the AI with your note
      ─► consistency checks by code (near-identical blocks named by codes, sample-ID collisions):
         flags you answer, never silent fixes
      ─► "Ask the AI to do it": an instruction becomes a previewed list of changes you apply or discard
      ─► schema.json + one value matrix per kept block + feature / sample metadata
         (finishing is refused if the sample counts do not reconcile)
```

**Principles, enforced in code:**
- **Facts are code, judgment is proposed.** Parsing, statistics, the shared-name-part facts, signatures and every
  structural check are plain code. Which columns belong together and what they are is judgment: the AI proposes
  it (or you decide it), and code only checks and applies it.
- **The AI proposes, code checks, you confirm.** Nothing counts until you confirm it. Each item carries provenance:
  `computed`, `ai_proposed_confirmed`, `ai_proposed_corrected` or `user_set`. The wizard shows this as a badge.
  A known-format signature is only a hint: it is written into the AI digest (and recorded as `signature_hint`
  in the schema). When the AI is off or unavailable, it gives the manual-mode starting point instead.
- **Closed fields vs open labels.** Code branches only on closed fields, served by `/api/vocabulary`: `layout`,
  `role`, `audit_kind` (for sample information: subject_id, timepoint, batch, run_order,
  technical_replicate, sample_type, group, covariate, other), yes / no / not sure, and the booleans `keep`,
  `marks_rows_as_suspect`, `is_study_sample`. Everything descriptive is free text in plain words: omics type,
  source software, assay label, and the label of every column, value block and sample. The AI says what it
  actually sees instead of squeezing it into a list; you can overwrite any of it, with suggestions from labels
  already used in the session.
- **Scope is a notice, not a block.** `SCOPE_DESCRIPTION` (in `backend/config.py`, or env `PRISM_SCOPE_DESCRIPTION`)
  says what PRISM's audit supports today. Each assay gets `in_supported_scope` yes / no / unsure with a reason;
  a 16S OTU table or a methylation matrix is still described, flagged, and can be confirmed and exported.
- **No value is modified.** The outputs are re-oriented (transposed or pivoted) and split into tables. Values are copied
  exactly; only missing-value tokens become empty cells (the tokens seen are listed in the parse report).
  No sample or feature is ever removed. QC, blank and pool samples, and decoy or contaminant rows, are labelled and kept.
- **Privacy.** The AI gets a digest: layout hints, and per column its name, position, computed statistics and shared name parts.
  It never gets raw rows. Example values are only sent for low-cardinality text columns (≤ 20 distinct values),
  and never for identifier-like ones (≥ 50 % distinct). For text columns the digest also gives `value_shapes`, the format of the
  values with digits masked (`cg########`, `P#####`, `OTU_##`): a shared prefix names the ID system, never an individual entry.
  Setting `AI_SEND_EXAMPLE_VALUES=false` withholds both. "What the AI saw" in the page shows the exact digest.

### The AI briefing (`backend/briefing.md`)
Loaded into the system prompt on every call: what PRISM is, the AI's job (group the columns and label the groups, quote the
digest's computed facts as evidence, say "unresolved" or ask rather than guess), what the later audit steps
need (subject IDs, time points, batch / run order / plate, technical replicates, QC / blank / pool samples,
group-like variables, columns that mark rows as suspect), and situations to watch for (clinical columns next to
features, several measurement families side by side, technical scale factors, tables outside the current scope).
Edit it to brief the model differently; `prompt_version` in `backend/schema.py` goes into the cache key and the log.

### Column grouping: an AI proposal (`ai.py`, `grouping.py`)
Grouping used to be done by engineered rules (name families of at least 3 columns, deviants a decade away, profile
clusters of at least 5). Those were judgment dressed up as code, and they produced only singletons on messy sample
names (`Pt003_visit1`, `004-w1`), which is exactly where grouping matters most. Now:
1. **Facts only, in code** (`profiling.py`). Per column: its statistics, and the name parts it shares with other
   columns: for every other column, the longest common prefix and suffix, summarised as levels with counts, e.g.
   `QC_01 Peak area` shares the suffix `_01 Peak area` with 1 other column, `1 Peak area` with 4 and ` Peak area`
   with 16. No threshold, no minimum group size; this describes the names and groups nothing.
2. **The AI groups and labels.** It gets the per-column digests and returns `groups[]`, each with its own `columns`
   list plus role, label and audit kind, and optionally `propose_merge` and `suggest_split`.
3. **Chunking is infrastructure.** `GROUPING_CHUNK_SIZE` (default 150, `PRISM_GROUPING_CHUNK_SIZE`) is a prompt size
   limit, not a claim about family size. Wider files are sent in chunks. Columns that share literal name parts are
   placed in the same chunk where possible: cut points are chosen by a small DP that prefers one more call over
   cutting through a long shared name part. Then one **consolidation** call gets a compact summary of every
   proposed group (label, size, first and last columns, shared name part, one line of statistics; no column data)
   and returns `cross_chunk_merges`. Files at or under the limit make a single call and skip consolidation.
4. **Code only checks and applies** (`grouping.py`):
   - a column named in `groups[].columns` that is not in that call's digest is rejected, like an invalid group id;
   - a column claimed twice stays in its first group, and the second claim is rejected;
   - **every column the AI did not mention becomes its own `unresolved` group**, so nothing is ever silently dropped;
   - `propose_merge`, `cross_chunk_merges` and `suggest_split` are applied only when every group id they name
     exists, and otherwise are rejected and logged.
5. **You decide in the wizard.** Every group card has **"Same thing as…"**: pick another group, optionally
   **ask the AI to check**, then **merge them**. Multi-column groups also let you tick columns and **take them out**.
   "Disagree? Tell the AI what's wrong" can now regroup as well as relabel. **Shared name parts**, under the table,
   lists the literal parts several columns share, with a "group these N" button: that is how you group by hand when
   the AI is off. Known formats (MaxQuant, DIA-NN…) give the manual-mode groups from their exact column names.
6. **If a call fails** (quota, timeouts), that chunk's columns stay as unresolved single columns and the page offers
   "Ask the AI again for these columns" and "Retry joining chunks".

**Cost and latency.** A file under 150 columns is one call, as before. A wide file costs one call per chunk plus
one consolidation call: e.g. SomaScan-scale 1,510 columns = 11 + 1 calls (about 2 minutes with Gemini flash-lite
in a test run); 2,045 columns = 15 + 1. On a free-tier key this can hit the per-minute or daily quota. Every call
is cached on disk, keyed by file, prompt version, models and the exact chunk contents, so re-uploading the same file
costs nothing. A request slower than `PRISM_AI_TIMEOUT_S` (default 120 s) moves on to the next model.

### Wizard steps
Every proposal is shown as editable fields straight away: closed fields are dropdowns, labels are text inputs. There
is no hidden edit mode. Under each step there is a **"Disagree? Tell the AI what's wrong"** box: write what is wrong
(e.g. "the iron column is a clinical value, not a metabolite") and the AI re-proposes that step's items with your
note. Items you already edited yourself are kept as you set them. Each item also has "Ask the AI about this one".
You still confirm the result. With the AI off, the box says so and you edit directly.

1. **Layout.** Samples in columns, samples in rows, or long. One card per assay: its label, omics type and source software
   (your words), and whether it is in PRISM's supported scope. Outside the scope you see "Recognized as <omics type> — not yet
   supported by PRISM's audit", a notice only.
2. **Feature ID.** One column or a composite key (e.g. m/z + RT). Duplicates are reported, never merged.
3. **Annotations.** Each column's label and role, and whether to keep it. For a column that marks rows as suspect
   (decoy, contaminant…), PRISM lists its distinct values with counts (≤ 5) and asks which one means "flagged", then shows
   "N rows flagged — nothing is removed now; recorded for the audit step."
4. **Values.** Nothing is ranked: there is no "primary" block. You keep or exclude each value block, and every kept block
   is exported as its own matrix. The block's label (the AI's words, editable) sits beside the profile computed from the
   data (histogram, min / median / max, zeros, whole numbers, span), so you can see whether they match. Below it,
   Numeric columns that don't look like features are listed separately. If several blocks of one assay are named by
   short codes and look statistically identical, you are asked whether they are one measurement (see Consistency checks).
5. **Samples.** Sample IDs, with an editable prefix/suffix strip and a live preview, and duplicates. Each sample has a label and
   a "study sample" tick: untick it for QC, blanks, pools, calibrators. Nothing is dropped.
6. **Sample info.** Samples-in-rows: role, audit kind, label and detail of each sample column. Samples-in-columns: an optional
   metadata file with a matching report; each of its columns gets an audit kind and label. Near-miss IDs are only suggested, never merged.
7. **Processing history.** Normalized? Log-transformed? Imputed? Batch-corrected? Anything removed? Always asked, never inferred;
   no answer is pre-selected.
8. **Review.** Finishing is blocked while anything is unresolved. Then you download the files.

Keyboard: **Enter** confirms, **Esc** leaves a field. Clicking a step in the stepper goes back to it.
Changing the layout re-runs the proposal with your layout fixed.

### "Ask the AI to do it" (instructions, not just questions)
At the top of every step there is a box for instructions such as *"don't include the score and count columns in
the output"*, *"the visit and week columns are time points"*, *"mark QC_* samples as non-study"* or *"merge the
two intensity blocks"*. The AI turns the instruction into actions from a closed list: `set_keep` (leave columns
out of the outputs), `set_role`, `set_label`, `set_audit_kind`, `set_assay`, `set_suspect`, `merge`, `split` and
`set_samples`. Code checks every action before you see it:
- unknown group ids are ignored;
- identifier columns cannot be excluded;
- a role the data contradicts (e.g. `value` on text) is refused;
- values outside the closed lists are refused.

The checks are shown next to each action, in plain words. **Nothing is applied until you tick the actions and click
"Apply selected"**; "Discard" drops them. Anything outside these actions (deleting rows, changing values,
normalizing) is answered in `not_possible` instead of being attempted: PRISM never deletes data, and excluded
columns are listed in the schema. Steps the changes touch are reopened for you to check. Every instruction, plan and
applied change is in the session log.

### Consistency checks — `backend/consistency.py` (code, not AI; flags, never fixes)
- **Near-identical blocks.** The check looks at the value blocks of one assay whose names are short identifier codes
  (`DEAB_0 … DEAB_136`, `GA43_4 …`: a code with a digit or in capitals, not a measurement term like
  `LFQ intensity`). If two or more of them have near-identical computed profiles, they are flagged, including
  single-column ones (`T623_0`). "Near-identical" means:
  - medians within 1 decade;
  - log10 span within 0.3;
  - zeros and missing within 0.1;
  - same whole-number pattern.

  The tolerances are settings in `config.py` (`PRISM_CONSISTENCY_*`). The flag reads: *"These N blocks look
  statistically identical — are they really different measurements, or the same measurement across different
  subjects/samples?"* You answer it one of two ways:
  - **"Same measurement: treat as one block".** The blocks are merged, sample IDs keep the full column names, and
    each name is split into `subject_code` (audit kind subject_id) and `name_suffix` (proposed as timepoint).
    These two appear in step 6 for you to check, and are written to `sample_metadata.csv`.
  - **"They really are different measurements".** The flag is dismissed.

  Finishing is blocked until you answer. This runs in addition to the AI's own `propose_merge`.
- **Sample IDs are checked across blocks, not per block.** Within an assay, blocks either measure the same samples
  (identical ID sets, descriptive names: MaxQuant LFQ / raw / iBAQ) or different samples (disjoint sets). Anything
  else is a collision, e.g. `DEAB_0` and `T623_0` both becoming `0`: *"Sample ID '0' appears in more than one block —
  these need to be distinguished."* It is fixed by using the full column names, or by a short label per block
  (`liver_S1`, `kidney_S1`), and is never merged silently.
- **Finalization invariant.** For each assay, the unique sample IDs must equal the entries in `samples[]` and be
  explainable from the raw value-column count: one block, or K blocks × N samples (same samples), or disjoint blocks
  adding up. Otherwise finishing stops and says which three numbers disagree. The schema reports `n_samples`,
  `n_value_columns` and `sample_structure` per assay.

### Literature (deferred)
`backend/literature.py` (Europe PMC retrieval with word-for-word quote verification) is **not used in Step 0** (v2.3):
Step 0 runs before the research-focus intake, so "should this block be analysed" cannot be answered yet. In one real
run, subject codes were also used as search terms and matched an unrelated compound. The module and its tests are
kept for a future Tier 2 feature.

### Outputs (per session, in `backend/sessions/<id>/outputs/`)
- `schema.json`: the confirmed schema: layout; assays (label, omics type, software, scope, feature identity, value blocks with
  label, file, computed profile); `feature_annotations` (label, `marks_rows_as_suspect`, `flag_values`, `flagged_value`,
  `n_flagged`, keep, provenance); `sample_metadata` (audit kind, label, detail); `samples` (sample, label, `is_study_sample`,
  provenance); excluded columns, processing history, parse report, integrity flags (incl. `outside_supported_scope`),
  `signature_hint`, and the AI provider / model / prompt version.
- `value_matrix_<assay>_<block>.csv` (e.g. `value_matrix_A1_B2.csv`): features × samples, one per kept block. For long tables, the pivot runs only if every
  (feature, sample) pair is unique; otherwise PRISM stops and explains that aggregation is not supported.
- `feature_metadata.csv`, `sample_metadata.csv` (`sample_id`, `sample_label`, `is_study_sample`, then the metadata columns).

Every upload, parse report, AI request (digest), raw AI response, validation result, confirmed step and
finalization is logged to `backend/logs/<session_id>.jsonl`, with timestamps, the provider, the model, `prompt_version` and temperature 0.

## v3: sessions, schema import, merge and the Tier 1 audit (`prism/`)

Principles: import, merge and audit are deterministic (no LLM; same inputs, parameters and seed give byte-identical
findings); the audit reads only the Step 0 output folders plus `overrides.json`; nothing is changed, filtered,
imputed or removed; nothing is imported or merged without an explicit confirmation.

```
sessions/<session_id>/
  session.json  session_schema.json  session_sample_table.csv  overrides.json
  datasets/<D1>/ upload/  schema.json  import_report.json  output/      (output/ = the audit's only input)
  audit/<run_id>/ manifest.json  findings/<audit>__<D>_<unit>.json  ledger.jsonl  report.html
  audit/ledger.jsonl (AI operator calls)   requests/NNN.md   logs/session.jsonl
```

- **Schema import** (`prism/io/`): `schema.json` is validated against the 0.4 contract (Pydantic models; the
  published JSON Schema is `schema/prism_schema_0.4.json`), then bound to the file: `exact` (same sha256: accepted
  as stored, every value-dependent number recomputed and diffed), `template` (same columns, new values: structure
  reused, statistics, processing history and the time unit asked again) or `seeded_wizard` (columns differ: the
  wizard opens with what matches). Review screen: Accept / Open in wizard / Reject, plus "Explain differences"
  (fixed sentences, no AI). Exact round trip: wizard → schema → import → byte-identical output folder.
  Confirmed schemas go to a saved-schemas library keyed by the file's sha256; uploading the same file again
  offers it ("use it?"), never applies it.
- **Merge** (`prism/session/merge.py`): exact sample-ID matching; near misses (case, whitespace, `-`/`_`/space,
  leading zeros) are suggestions only; unified IDs from confirmed suggestions or a mapping CSV (injective per
  dataset); overlap with Jaccard and a presence matrix; the unified sample table, where disagreeing values become
  conflicts (take / keep both / drop) and differing `audit_kind`s become questions; subject and time agreement on
  shared samples, time units, possible ID reuse; "ready for audit" when nothing is open. Original IDs are never
  rewritten.
- **Tier 1 audit** (`prism/audit/`): A1 integrity, A2 scale, A3 distribution and variance-mean, A4 missingness and
  floor values (with declared vs observed processing history), A5 dimensionality and effective n, A6 technical
  noise and QC, A7 batch structure (codes, PCA associations, PERMANOVA), A8 repeated measures (ICC(1), time grid,
  nesting), A9 sample source, A10 multi-omics overlap (session level, RV coefficient), A11 outlier flags.
  numpy only. Conventions: diagnostic copy Y = log2(X [+ c]) never written out; permutations restricted by
  design (whole subjects / within subject / free), 999 or exhaustive up to 10,000, `p_min_attainable` reported;
  guards give `insufficient_data`; BH q within each table; count, proportion and compositional data are
  `not_applicable`. Parameters in `prism/audit/audit_params.yaml` (heuristics labelled); every run writes a
  manifest (version, git hash, parameters and their hash, input sha256s, overrides hash, timings) and a ledger.
- **Workspace and theme**: the tool page is a 4-step journey (Data → Merge → Audit → Report) that shows
  where you are and what is still open. Every page has a light / night toggle (remembered per browser; the
  OS setting is used until you choose).
- **Final report**: step 4, or `python -m prism report --session SID`. One document for the whole session:
  a plain-language summary, key findings per dataset, what needs your input, declared vs observed, how each
  dataset was read, key figures, every check, and the methods. It downloads as a single self-contained
  HTML file (interactive, light / night) and prints cleanly to PDF.
- **Report**: the "Run audit" panel (run history, parameter drawer with re-run and diff, overrides) and a
  self-contained HTML export with the same renderer (`frontend/audit_report.js`): header with what was used,
  Declared vs observed first, a tab per dataset, a card per audit with plots from `plot_data`.
- **AI operator** (thin): "Ask the AI" in the audit panel. It sees the schemas and finding JSON only (never data
  rows) and calls a fixed set of tools that run the same engine: `run_audit`, `rerun`, `show_finding`,
  `set_override` (a proposal you apply), `plot`, `request_addition` (writes `requests/NNN.md`, nothing runs).
  A parameter it was not given in your message is refused; numbers in its reply that are not in the findings
  are flagged; every call is in the ledger.

CLI (the same engine as the page):
```bash
python -m prism session new --name study
python -m prism import --data X.csv --schema schema.json [--metadata M.csv] --session SID [--accept]
python -m prism session add --session SID --output backend/sessions/<step0 id>/outputs
python -m prism session merge-report --session SID
python -m prism session merge-decide --session SID ITEM_ID confirm|dismiss|take:D1|keep_both|drop|<option>
python -m prism session merge-mapping --session SID mapping.csv
python -m prism audit override --session SID --kind sample_role --sample S1 --role qc --reason "pooled QC"
python -m prism audit run --session SID [--factor A4] [--dataset D1] [--param permutations=199]
python -m prism audit show --session SID [--run RUN]
python -m prism audit report --session SID [--run RUN] --format html|json [--out FILE]
python -m prism audit compare --session SID RUN_A RUN_B
python -m prism report --session SID [--format html|json] [--out FILE]
```
Python 3.7 or later; numpy, FastAPI and Pydantic 2 (no pandas, scipy or scikit-learn).

## API

| Method | Path | |
|---|---|---|
| GET | `/api/vocabulary` | All vocabularies, definitions and history questions (the single source of truth) |
| GET | `/api/health` | AI provider status |
| POST | `/api/upload` | multipart `file` → session, parse report, preview (groups come with the proposal) |
| POST | `/api/propose` | `{session_id, ai}` → the draft (validated AI proposal, or the manual starting point) and the exact digest(s) sent |
| POST | `/api/confirm-step` | `{session_id, step_id, decision}` → updated draft; changing the layout triggers a re-proposal |
| POST | `/api/ai-command` | `{session_id, instruction}` → the AI's plan as checked actions (nothing applied) |
| POST | `/api/apply-command` | `{session_id, command_id, accept?}` → your confirmation: apply the chosen actions |
| POST | `/api/discard-command` | `{session_id}` → drop the pending plan |
| POST | `/api/consistency` | `{session_id, action: one_block / dismiss / full_names / labels, key?, group_ids?, labels?}` → your answer to a consistency flag |
| POST | `/api/reconsider` | `{session_id, group_ids, user_hint}` → re-asks the AI about these groups' columns with your note (it may relabel, regroup, split or merge) |
| POST | `/api/merge-check` | `{session_id, group_ids, user_hint}` → the AI's opinion on "these are the same thing"; nothing is applied |
| POST | `/api/merge` | `{session_id, group_ids}` → your decision: merge these groups |
| POST | `/api/split` | `{session_id, group_id, columns}` → your decision: take these columns out, one group each |
| POST | `/api/group-columns` | `{session_id, columns}` → your decision: these columns are one group (e.g. a shared name part) |
| POST | `/api/consolidate` | `{session_id}` → retry the cross-chunk consolidation call |
| POST | `/api/metadata-upload` | multipart `session_id`, `file` → matching report |
| POST | `/api/finalize` | → schema, artifact list, integrity flags |
| GET | `/api/export/{session_id}/{artifact}` | download an output file |
| GET | `/api/progress/{id}` | live progress of an upload / AI call |
| GET | `/api/sessions/{id}/log` | the session's event log |

## Tests

```bash
python -m pytest -q
```
The LLM is always mocked in the tests, including deliberately wrong proposals, and Europe PMC is replaced by a fake
(`tests/fake_europepmc.py`, same response format, invented test papers). The tests assert structure (role, keep, audit kind, `marks_rows_as_suspect`, `is_study_sample`), never label wording. They cover:
- parsing and the parse report;
- facts: shared name parts (checked against brute force), chunking that keeps literal families together;
- grouping as a proposal (`test_grouping.py`): every column in exactly one group for every fixture, an incomplete
  AI answer (unmentioned columns become unresolved single columns), double claims and unknown columns rejected,
  `propose_merge` / `suggest_split` applied only with valid ids, your merge-check / merge / split, a 2,040-column
  file with messy sample names (15 chunk calls + consolidation, full coverage, second upload fully cached), and
  recovery after a failed chunk and a failed consolidation;
- signatures as hint and manual starting point;
- the structural checks: hallucinated group id, value role on a text column, a closed field outside its set;
- full API flows for fixtures A–F (MaxQuant, DIA-NN, MZmine, samples-in-rows multi-omics, SomaScan-like, long unique/duplicate);
- AI off, invalid JSON, suspect-flag values, out-of-scope notice, disagree / reconsider (incl. a split), provenance and the outputs;
- consistency (`test_consistency.py`, fixture J: one measurement split into subject-code blocks such as `DEAB_0`):
  the flag fires and blocks finishing; "one block" parses subject / suffix into sample information; dismissing
  leaves a global sample-ID collision that blocks finishing until fixed; per-block labels; MaxQuant parallel blocks
  are not a collision; deliberately mismatched sample counts block finishing;
- instructions (`test_command.py`): one instruction excludes many columns, the preview changes nothing, invalid
  actions are checked and refused, samples, discard, AI off;
- the literature module only (kept for later): Step 0 has no literature anywhere, quote verification, the HTTP client;
- v3 (`test_v3_*.py`): sessions and the loader; schema import (byte-identical round trip, template and seeded
  modes, hand-edited profile, rejections); merge (near misses never applied, non-injective mapping rejected,
  conflicts, audit-kind questions, possible ID reuse, time-unit questions); the audit on planted ground truth
  (floors, batch shift, nesting, ICC, outlier sample and cell, QC drift, source effect, layer RV, declared vs
  observed), statuses, the contract (uploads deleted, identical findings for the same seed, parameter hash,
  `output/` untouched), the API and the self-contained export, the AI operator (mock), the saved-schemas library.

Golden tests run when the real files are in `tests/data/` (git-ignored; skipped otherwise): SomaScan floor ties
20 of 11,083, metabolomics 397 of 1,174 with every class count, the design (8 subjects, cluster sizes, samples per
time, within-subject gaps), 27 of 27 shared samples, the 12,301.6 outlier cell, declared vs observed, and the
median-scaling signature (present for metabolomics, absent for SomaScan). The whole audit of both files takes a
few seconds.


A manual acceptance test runs against the real AI with tables outside the current scope (fixtures H: 16S OTU
counts, I: methylation beta values). They must be described sensibly and flagged `no` / `unsure`, and the wizard
must complete. It is skipped unless you ask for it; keep the key in the environment, never in a file you commit:
```bash
PRISM_REAL_API=1 GEMINI_API_KEY=... python -m pytest -s tests/test_real_api.py
```

To regenerate the fixtures: `python tests/fixtures/make_fixtures.py`.

## Layout

```
backend/   main.py (API) · schema.py (vocabulary) · config.py (scope) · briefing.md (AI briefing)
           parsing.py · profiling.py (facts) · format_detect.py (signatures) · grouping.py (applies proposals)
           validation.py · consistency.py · ai.py · literature.py (deferred, not used in Step 0) · llm_providers.py · mock_llm.py
           workflow.py (draft, steps) · outputs.py
           session_log.py · envfile.py · check_ai.py
frontend/  index.html (home + 3D prism) · tool.html + app.js (wizard) · style.css · home.js · fonts/
prism/     store.py (sessions) · io/ (loader, schema_model, importer, rebuild, library) · session/merge.py
           audit/ (engine, context, stats, finding, params + audit_params.yaml, overrides, a01..a11, report, operator)
           cli.py · ledger.py · manifest.py
frontend/  theme.js (light / night) · session.js (journey, import review, merge) · audit.js + final.js +
           audit_report.js/.css (audit panel, final report, exports)
tests/     fixtures/ (A–F, H 16S, I methylation, J subject-code blocks, messy file) · fake_europepmc.py · test_deterministic.py · test_grouping.py
           test_flow.py · test_consistency.py · test_command.py · test_literature.py · test_real_api.py
```
