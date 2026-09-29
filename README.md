# PRISM — Step 0: Schema recognition and guided confirmation

**P**reprocessing and **R**easoning for **I**nformed **S**tatistical **M**ethodology.

PRISM helps you preprocess omics data (proteomics and metabolomics for now). It doesn't push every
dataset through one fixed pipeline. It first works out what the data is, then reasons about what to do with it.
This build is **Step 0 only**: you upload a quantified table, PRISM recognizes its structure, you confirm it
step by step, and the result is written out as a schema plus canonical tables. Audits (missingness,
PCA, batch, distributions), normalization, imputation, filtering and recommendations are later phases.

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
upload ─► parse (strings only, report oddities) ─► profile every column ─► group columns
      ─► known-format signature (MaxQuant, DIA-NN, Spectronaut, FragPipe, mz/rt) → a one-line hint
      ─► AI labels each group from a statistics digest, briefed by backend/briefing.md (optional)
      ─► structural checks; contradicted claims become "unresolved"
      ─► 8-step wizard: every proposal is editable in place; "Disagree?" re-asks the AI with your note
      ─► schema.json + value matrices + feature / sample metadata
```

**Principles, enforced in code:**
- **Deterministic first.** Parsing, profiling, grouping, signatures and every structural check are plain code.
  The AI only interprets meaning from names and computed statistics.
- **The AI proposes, code checks, you confirm.** Nothing counts until you confirm it. Each item carries provenance:
  `computed`, `ai_proposed_confirmed`, `ai_proposed_corrected` or `user_set`. The wizard shows this as a badge.
  A known-format signature is only a hint: it is written into the AI digest (and recorded as `signature_hint`
  in the schema). When the AI is off or unavailable, it gives the manual-mode starting point instead.
- **Closed fields vs open labels.** Code branches only on closed fields, served by `/api/vocabulary`: `layout`,
  `role`, `block_role`, `audit_kind` (for sample information: subject_id, timepoint, batch, run_order,
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
- **Privacy.** The AI gets a digest: layout hints, and per column group its name pattern, a few column names and aggregated statistics.
  It never gets raw rows. Example values are only sent for low-cardinality text columns (≤ 20 distinct values),
  and never for identifier-like ones (≥ 50 % distinct). For text columns the digest also gives `value_shapes`, the format of the
  values with digits masked (`cg########`, `P#####`, `OTU_##`): a shared prefix names the ID system, never an individual entry.
  Setting `AI_SEND_EXAMPLE_VALUES=false` withholds both. "What the AI saw" in the page shows the exact digest.

### The AI briefing (`backend/briefing.md`)
Loaded into the system prompt on every call: what PRISM is, the AI's job (label pre-built groups, quote the
digest's computed facts as evidence, say "unresolved" or ask rather than guess), what the later audit steps
need (subject IDs, time points, batch / run order / plate, technical replicates, QC / blank / pool samples,
group-like variables, columns that mark rows as suspect), and situations to watch for (clinical columns next to
features, several measurement families side by side, technical scale factors, tables outside the current scope).
Edit it to brief the model differently; `prompt_version` in `backend/schema.py` goes into the cache key and the log.

### Column grouping (`backend/profiling.py`)
1. Numeric columns sharing a name prefix/suffix (≥ 3 columns, ≥ 3 characters) form a family, e.g. `LFQ intensity S01…`.
   Two checks stop mixed groups:
   - Families that swallow a more specific family are rejected: `X Intensity` never absorbs `X MaxLFQ Intensity`.
   - Per-sample slices are rejected: suffix ` S01` across `Intensity S01`, `iBAQ S01`… is not a family.
2. Columns whose profile deviates strongly from their family are split out: the typical value is far off,
   or the whole-number / missing-value pattern differs. This keeps age, CD4 count or a scale factor out of a feature block.
   Columns named like QC / blank / pool samples stay in their family. A column must be at least one decade away
   from the family's typical value, so sparse count columns (e.g. OTU counts) are not split by chance.
   Sample names that carry a design code (`Stool.D0.A`, `Stool.D7.A`, `Stool.D14.A`) stay one block, not one family per day.
3. Remaining numeric columns are clustered by typical value, keeping clusters of ≥ 5 columns; the rest become singletons.
   Text columns are singletons.
4. The AI may suggest splitting a column out of a block (e.g. `iron` among metabolites), in the first proposal or
   when you disagree. The code checks the columns exist and applies the split. The split then shows in the wizard for you to confirm.

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
4. **Values.** Primary / auxiliary / excluded per assay. The block's label (the AI's words, editable) sits beside the profile
   computed from the data (histogram, min / median / max, zeros, whole numbers, span), so you can see whether they match.
   Numeric columns that don't look like features are listed separately.
5. **Samples.** Sample IDs, with an editable prefix/suffix strip and a live preview, and duplicates. Each sample has a label and
   a "study sample" tick: untick it for QC, blanks, pools, calibrators. Nothing is dropped.
6. **Sample info.** Samples-in-rows: role, audit kind, label and detail of each sample column. Samples-in-columns: an optional
   metadata file with a matching report; each of its columns gets an audit kind and label. Near-miss IDs are only suggested, never merged.
7. **Processing history.** Normalized? Log-transformed? Imputed? Batch-corrected? Anything removed? Always asked, never inferred;
   no answer is pre-selected.
8. **Review.** Finishing is blocked while anything is unresolved. Then you download the files.

Keyboard: **Enter** confirms, **Esc** leaves a field. Clicking a step in the stepper goes back to it.
Changing the layout re-runs the proposal with your layout fixed.

### Outputs (per session, in `backend/sessions/<id>/outputs/`)
- `schema.json`: the confirmed schema: layout; assays (label, omics type, software, scope, feature identity, value blocks with
  label, block role and computed profile); `feature_annotations` (label, `marks_rows_as_suspect`, `flag_values`, `flagged_value`,
  `n_flagged`, keep, provenance); `sample_metadata` (audit kind, label, detail); `samples` (sample, label, `is_study_sample`,
  provenance); excluded columns, processing history, parse report, integrity flags (incl. `outside_supported_scope`),
  `signature_hint`, and the AI provider / model / prompt version.
- `value_matrix_<assay>.csv`: features × samples, from the primary block only. For long tables, the pivot runs only if every
  (feature, sample) pair is unique; otherwise PRISM stops and explains that aggregation is not supported.
- `feature_metadata.csv`, `sample_metadata.csv` (`sample_id`, `sample_label`, `is_study_sample`, then the metadata columns).

Every upload, parse report, AI request (digest), raw AI response, validation result, confirmed step and
finalization is logged to `backend/logs/<session_id>.jsonl`, with timestamps, the provider, the model, `prompt_version` and temperature 0.

## API

| Method | Path | |
|---|---|---|
| GET | `/api/vocabulary` | All vocabularies, definitions and history questions (the single source of truth) |
| GET | `/api/health` | AI provider status |
| POST | `/api/upload` | multipart `file` → session, parse report, preview, groups with profiles and histograms |
| POST | `/api/propose` | `{session_id, ai}` → the draft (validated AI proposal, or the manual starting point) and the exact digest(s) sent |
| POST | `/api/confirm-step` | `{session_id, step_id, decision}` → updated draft; changing the layout triggers a re-proposal |
| POST | `/api/reconsider` | `{session_id, group_ids, user_hint}` → re-asks the AI about these groups with your note (splits it suggests are applied) |
| POST | `/api/metadata-upload` | multipart `session_id`, `file` → matching report |
| POST | `/api/finalize` | → schema, artifact list, integrity flags |
| GET | `/api/export/{session_id}/{artifact}` | download an output file |
| GET | `/api/progress/{id}` | live progress of an upload / AI call |
| GET | `/api/sessions/{id}/log` | the session's event log |

## Tests

```bash
python -m pytest -q
```
The LLM is always mocked in the tests, including deliberately wrong proposals. The tests assert structure (role,
block role, audit kind, `marks_rows_as_suspect`, `is_study_sample`), never label wording. They cover:
- parsing and the parse report;
- digests and grouping (incl. the 16S and methylation tables);
- signatures as hint and manual starting point;
- the structural checks: hallucinated group id, value role on a text column, a closed field outside its set;
- full API flows for fixtures A–F (MaxQuant, DIA-NN, MZmine, samples-in-rows multi-omics, SomaScan-like, long unique/duplicate);
- AI off, invalid JSON, suspect-flag values, out-of-scope notice, disagree / reconsider (incl. a split), provenance and the outputs.

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
           parsing.py · profiling.py · format_detect.py (signatures)
           validation.py · ai.py · llm_providers.py · mock_llm.py · workflow.py (draft, steps) · outputs.py
           session_log.py · envfile.py · check_ai.py
frontend/  index.html (home + 3D prism) · tool.html + app.js (wizard) · style.css · home.js · fonts/
tests/     fixtures/ (A–F, H 16S, I methylation, messy file) · test_deterministic.py · test_flow.py · test_real_api.py
```
