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
upload ─► parse (strings only, report oddities) ─► facts per column: statistics + shared name parts
      ─► known-format signature (MaxQuant, DIA-NN, Spectronaut, FragPipe, mz/rt) → a one-line hint
      ─► the AI GROUPS the columns and labels each group, briefed by backend/briefing.md
         (wide files: chunks of 150 columns + one consolidation call)
      ─► code applies the grouping only where columns / ids exist; every column ends up in exactly
         one group; structural checks; contradicted claims become "unresolved"
      ─► 8-step wizard: every proposal is editable in place; "Disagree?" re-asks the AI with your note
      ─► step 4: Europe PMC literature search; the AI describes each value block from the papers,
         with citations checked word for word
      ─► schema.json + one value matrix per kept block + feature / sample metadata + the literature
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
   **what the literature says** (see next section). Numeric columns that don't look like features are listed separately.
5. **Samples.** Sample IDs, with an editable prefix/suffix strip and a live preview, and duplicates. Each sample has a label and
   a "study sample" tick: untick it for QC, blanks, pools, calibrators. Nothing is dropped.
6. **Sample info.** Samples-in-rows: role, audit kind, label and detail of each sample column. Samples-in-columns: an optional
   metadata file with a matching report; each of its columns gets an audit kind and label. Near-miss IDs are only suggested, never merged.
7. **Processing history.** Normalized? Log-transformed? Imputed? Batch-corrected? Anything removed? Always asked, never inferred;
   no answer is pre-selected.
8. **Review.** Finishing is blocked while anything is unresolved. Then you download the files.

Keyboard: **Enter** confirms, **Esc** leaves a field. Clicking a step in the stepper goes back to it.
Changing the layout re-runs the proposal with your layout fixed.

### Literature (RAG) — `backend/literature.py`
When you open step 4, PRISM searches **Europe PMC** (PubMed abstracts plus open-access full text) for papers with similar
data, and the AI describes each value block from what it finds:
1. **Queries**: 1–3 written by the AI in the first proposal, plus ones built from your confirmed description (software,
   omics type, block names such as `"LFQ intensity" AND "iBAQ"`). They are shown and editable; "Search again" re-runs them.
   Only these names and terms are sent to Europe PMC, never data.
2. **Retrieval**: the top papers per query; for the best open-access ones, the methods / data-processing / results
   sections of the full text (that is where papers say what they actually analysed). Everything is cut into passages
   and ranked (BM25, plus a bonus for the block names), at most 3 passages per paper. Responses are cached on disk
   (`backend/cache/literature/`).
3. **Suggestions**: the AI gets the passages and, per block, writes what the measurement is, how similar studies used it,
   and `suggested_for_analysis` yes / no / unsure — several blocks can be "yes"; nothing is ranked. Every claim must cite a
   passage with an **exact quote**. Code checks each quote word for word: quotes that are not in the passage are
   discarded, and a "yes" left without a verified citation becomes "unsure". The page shows the badge, the text,
   `[n]` links to the papers (hover shows the quote), and the papers and passages themselves.
4. **Kept for later steps**: all papers and passages, plus the passages the AI points to for normalization / missing
   values / batch effects (`for_later_steps`, pointers only, no advice), are stored in `schema.json`, so the audit and
   preprocessing steps can reuse them without searching again.

With the AI off, the search still runs and you get the ranked passages to read yourself. If Europe PMC cannot be
reached you get a clear message and can continue without it. `PRISM_LITERATURE=off` turns it off;
`PRISM_EUROPEPMC_URL` points at a mirror. RAG does not train the model: it puts relevant, citable text in front of it for each file.

### Outputs (per session, in `backend/sessions/<id>/outputs/`)
- `schema.json`: the confirmed schema: layout; assays (label, omics type, software, scope, feature identity, value blocks with
  label, file, computed profile and the literature suggestion); `feature_annotations` (label, `marks_rows_as_suspect`, `flag_values`, `flagged_value`,
  `n_flagged`, keep, provenance); `sample_metadata` (audit kind, label, detail); `samples` (sample, label, `is_study_sample`,
  provenance); excluded columns, processing history, parse report, integrity flags (incl. `outside_supported_scope`),
  `signature_hint`, `literature` (queries, papers, passages, per-block suggestions with verified citations, pointers for
  later steps), and the AI provider / model / prompt version.
- `value_matrix_<assay>_<block>.csv` (e.g. `value_matrix_A1_B2.csv`): features × samples, one per kept block. For long tables, the pivot runs only if every
  (feature, sample) pair is unique; otherwise PRISM stops and explains that aggregation is not supported.
- `feature_metadata.csv`, `sample_metadata.csv` (`sample_id`, `sample_label`, `is_study_sample`, then the metadata columns).

Every upload, parse report, AI request (digest), raw AI response, validation result, confirmed step and
finalization is logged to `backend/logs/<session_id>.jsonl`, with timestamps, the provider, the model, `prompt_version` and temperature 0.

## API

| Method | Path | |
|---|---|---|
| GET | `/api/vocabulary` | All vocabularies, definitions and history questions (the single source of truth) |
| GET | `/api/health` | AI provider status |
| POST | `/api/upload` | multipart `file` → session, parse report, preview (groups come with the proposal) |
| POST | `/api/propose` | `{session_id, ai}` → the draft (validated AI proposal, or the manual starting point) and the exact digest(s) sent |
| POST | `/api/confirm-step` | `{session_id, step_id, decision}` → updated draft; changing the layout triggers a re-proposal |
| POST | `/api/literature` | `{session_id, queries?}` → searches Europe PMC and adds cited suggestions per value block |
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
- literature: queries, full-text parsing, caching, citations verified word for word, invented quotes discarded and
  "yes" downgraded, AI off, no results, service unreachable, and the real HTTP client against the fake server.

To try the literature step offline, run the fake service and point PRISM at it:
```bash
python tests/fake_europepmc.py 8090
PRISM_EUROPEPMC_URL=http://127.0.0.1:8090 python -m uvicorn backend.main:app --reload
```

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
           validation.py · ai.py · literature.py (Europe PMC RAG) · llm_providers.py · mock_llm.py
           workflow.py (draft, steps) · outputs.py
           session_log.py · envfile.py · check_ai.py
frontend/  index.html (home + 3D prism) · tool.html + app.js (wizard) · style.css · home.js · fonts/
tests/     fixtures/ (A–F, H 16S, I methylation, messy file) · fake_europepmc.py · test_deterministic.py · test_grouping.py
           test_flow.py · test_literature.py · test_real_api.py
```
