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
      ─► signature match (MaxQuant, DIA-NN, Spectronaut, FragPipe, mz/rt) pre-fills what is certain
      ─► AI labels each group from a statistics digest (optional)
      ─► every AI claim is checked against the data; contradictions become "unresolved"
      ─► 8-step wizard: you confirm or correct each part
      ─► schema.json + value matrices + feature / sample metadata
```

**Principles, enforced in code:**
- **Deterministic first.** Parsing, profiling, grouping, signatures and every consistency check are plain code.
  The AI only interprets meaning from names and computed statistics.
- **The AI proposes, code checks, you confirm.** Nothing counts until you confirm it. Each item carries provenance:
  `signature`, `computed`, `ai_proposed_confirmed`, `ai_proposed_corrected` or `user_set`.
  The wizard shows this as a badge.
- **No value is modified.** The outputs are re-oriented (transposed or pivoted) and split into tables. Values are copied
  exactly; only missing-value tokens become empty cells (the tokens seen are listed in the parse report).
  No sample or feature is ever removed. QC, blank and pool samples, and decoy or contaminant rows, are labelled and kept.
- **Privacy.** The AI gets a digest: layout hints, and per column group its name pattern, a few column names and aggregated statistics.
  It never gets raw rows. Example values are only sent for low-cardinality text columns (≤ 20 distinct values),
  and never for identifier-like ones (≥ 50 % distinct). "What the AI saw" in the page shows the exact digest.

### Column grouping (`backend/profiling.py`)
1. Numeric columns sharing a name prefix/suffix (≥ 3 columns, ≥ 3 characters) form a family, e.g. `LFQ intensity S01…`.
   Two checks stop mixed groups:
   - Families that swallow a more specific family are rejected: `X Intensity` never absorbs `X MaxLFQ Intensity`.
   - Per-sample slices are rejected: suffix ` S01` across `Intensity S01`, `iBAQ S01`… is not a family.
2. Columns whose profile deviates strongly from their family are split out: the typical value is far off,
   or the whole-number / missing-value pattern differs. This keeps age, CD4 count or a scale factor out of a feature block.
   Columns named like QC / blank / pool samples stay in their family.
3. Remaining numeric columns are clustered by typical value, keeping clusters of ≥ 5 columns; the rest become singletons.
   Text columns are singletons.
4. The AI may suggest splitting a column out of a block (e.g. `iron` among metabolites). The code checks the columns exist and applies the split.
   The split then shows in the wizard for you to confirm.

### Wizard steps
1. **Layout.** Samples in columns, samples in rows, or long. Also the omics type and the source software.
2. **Feature ID.** One column or a composite key (e.g. m/z + RT). Duplicates are reported, never merged.
3. **Annotations.** The kind of each feature annotation column, and whether to keep it. Flag columns show how many rows are flagged.
4. **Values.** The primary / auxiliary / excluded block, measurement type, scale, histogram and statistics. Numeric columns
   that don't look like features are listed separately and sent to step 6.
5. **Samples.** Sample IDs, with an editable prefix/suffix strip and a live preview, duplicates, and sample types (QC, blank, pool…).
6. **Sample info.** Samples-in-rows: roles and kinds of the sample columns. Samples-in-columns: an optional metadata
   file with a matching report. Near-miss IDs are only suggested, never merged automatically.
7. **Processing history.** Normalized? Log-transformed? Imputed? Batch-corrected? Anything removed? Always asked, never inferred;
   no answer is pre-selected.
8. **Review.** Finishing is blocked while anything is unresolved. Then you download the files.

Keyboard: **Enter** confirms, **Esc** opens or closes the editor. Clicking a step in the stepper goes back to it.
Changing the layout re-runs the proposal with your layout fixed.

### Outputs (per session, in `backend/sessions/<id>/outputs/`)
- `schema.json`: the confirmed schema (layout, assays and value blocks with profiles, annotations, sample
  metadata, sample types, excluded columns, processing history, parse report, integrity flags, AI provider/model/prompt).
- `value_matrix_<assay>.csv`: features × samples, from the primary block only. For long tables, the pivot runs only if every
  (feature, sample) pair is unique; otherwise PRISM stops and explains that aggregation is not supported.
- `feature_metadata.csv`, `sample_metadata.csv`.

Every upload, parse report, AI request (digest), raw AI response, validation result, confirmed step and
finalization is logged to `backend/logs/<session_id>.jsonl`, with timestamps, the provider, the model, `prompt_version` and temperature 0.

## API

| Method | Path | |
|---|---|---|
| GET | `/api/vocabulary` | All vocabularies, definitions and history questions (the single source of truth) |
| GET | `/api/health` | AI provider status |
| POST | `/api/upload` | multipart `file` → session, parse report, preview, groups with profiles and histograms |
| POST | `/api/propose` | `{session_id, ai}` → the draft (signature + validated AI proposal) and the exact digest(s) sent |
| POST | `/api/confirm-step` | `{session_id, step_id, decision}` → updated draft; changing the layout triggers a re-proposal |
| POST | `/api/reconsider` | `{session_id, group_id, user_hint}` → re-asks the AI about one group |
| POST | `/api/metadata-upload` | multipart `session_id`, `file` → matching report |
| POST | `/api/finalize` | → schema, artifact list, integrity flags |
| GET | `/api/export/{session_id}/{artifact}` | download an output file |
| GET | `/api/progress/{id}` | live progress of an upload / AI call |
| GET | `/api/sessions/{id}/log` | the session's event log |

## Tests

```bash
python -m pytest -q
```
The LLM is always mocked in the tests, including deliberately wrong proposals. The tests cover:
- parsing and the parse report;
- digests and grouping;
- signatures;
- every consistency rule;
- full API flows for fixtures A–F (MaxQuant, DIA-NN, MZmine, samples-in-rows multi-omics, SomaScan-like, long unique/duplicate);
- AI off, invalid JSON, hallucinated group ids, provenance and the canonical outputs.

To regenerate the fixtures: `python tests/fixtures/make_fixtures.py`.

## Layout

```
backend/   main.py (API) · schema.py (vocabulary) · parsing.py · profiling.py · format_detect.py (signatures)
           validation.py · ai.py · llm_providers.py · mock_llm.py · workflow.py (draft, steps) · outputs.py
           session_log.py · envfile.py · check_ai.py
frontend/  index.html (home + 3D prism) · tool.html + app.js (wizard) · style.css · home.js · fonts/
tests/     fixtures/ (A–F + messy file) · test_deterministic.py · test_flow.py
```
