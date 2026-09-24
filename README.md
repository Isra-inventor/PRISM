# PRISM — Step 0: Data Input & Recognition

**P**reprocessing and **R**easoning for **I**nformed **S**tatistical **M**ethodology.

PRISM helps you preprocess omics data (proteomics and metabolomics for now). It doesn't push every
dataset through one fixed pipeline. It first works out what the data is, then reasons about what to do with it.
This build covers **Step 0 only**: you upload a quantified table, PRISM identifies its structure,
you confirm it, and the tool shows the confirmed structure. Missingness, PCA, batch detection,
normalization, imputation and recommendations are out of scope for this build.

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

Open <http://127.0.0.1:8000>. The home page is at `/`, and the tool is at `/tool.html`.
FastAPI serves both the API and the static frontend, so you only run one command.
`python main.py` does **not** work, because the backend is a package and has to be started through uvicorn as shown.

The 3D prism on the home page loads three.js from `cdn.jsdelivr.net`. The page needs internet access for
the prism. Without it, the typographic hero still works. The fonts (Quicksand, Inter) are self-hosted in `frontend/fonts/`.

### Where to put the API key

The AI fallback reads **`ANTHROPIC_API_KEY`** from the server's environment. Set it in the same
terminal before you start the server: `$env:ANTHROPIC_API_KEY="sk-ant-..."` in PowerShell, or
`export ANTHROPIC_API_KEY=sk-ant-...` on macOS/Linux. The key is never hardcoded
and never sent to the browser. You can export it in your shell or put it in your process manager's environment.
Optional settings:

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | Turns on AI-assisted column proposals |
| `PRISM_LLM_MODEL` | `claude-opus-5` | Model used for proposals |
| `PRISM_LOG_DIR` | `backend/logs` | Where session logs are written |
| `PRISM_MAX_UPLOAD_MB` | `250` | Upload size limit |

If no key is set, uploads that match a known signature still work. For every other file, all
columns come back `unresolved` and you assign each role by hand.

## How it works

1. **Upload.** Only `.csv` and `.tsv` are accepted, plus `.txt` if it is tab-separated, like MaxQuant's `proteinGroups.txt`.
   PRISM rejects anything else, including binary files, with a message asking you to export a quantified table first.
   Every cell is read as an unmodified string.
2. **Deterministic detection** (`backend/format_detect.py`). The header is checked against the
   `SIGNATURES` table (MaxQuant, DIA-NN, Spectronaut, FragPipe, and generic mz/rt feature tables).
   A signature matches only when **all** of its required columns are present, with no fuzzy matching.
   On a match, value and feature-ID columns are resolved by fixed rules and **no AI is called**.
3. **AI fallback** (`backend/llm_fallback.py`). This step runs only when nothing matched. The model gets the header and the first 15
   rows (cells cut at 40 characters). It must return `{column, proposed_role, confidence, evidence}` for each column,
   using the fixed role vocabulary (`sample_id`, `subject_id`, `timepoint`, `batch`,
   `group_or_outcome`, `feature_value`, `feature_annotation`, `ignore`) or `unresolved`.
   If an entry is missing, uses an unknown role, or the call fails, that column becomes `unresolved`.
   PRISM never defaults it to a real role.
4. **Confirmation** (`POST /api/sessions/{id}/confirm`). The UI requires an explicit confirm click,
   including for signature matches. It can't be confirmed while any column is `unresolved`. The endpoint returns the recognized
   structure: platform, omics type, sample and feature counts, layout, and the final role of every column.

**No values are transformed** at any point in this step.

### Logs

Each session writes a JSON Lines file to `backend/logs/<session_id>.jsonl` with these events:
`upload` → `signature_match` → `ai_proposal` (with the prompts and raw model output) →
`confirmation`. For each column, the confirmation event records the original proposal (role, confidence,
evidence), the final role, and whether it was *accepted*, *corrected* or *assigned*. It also has a UTC timestamp.
You can see the log in the UI (“View session log”) or at `GET /api/sessions/{id}/log`.

### How samples and features are counted

- No `sample_id` column: one row per feature, and each `feature_value` column is one sample (typical proteomics/metabolomics export).
- `sample_id` unique on every row: one row per sample, and each `feature_value` column is one feature.
- `sample_id` repeats: long format. Samples are the distinct sample IDs, and features are the distinct feature IDs.

## API

| Method | Path | |
|---|---|---|
| GET | `/api/health` | Whether AI is available, the model, roles and signatures |
| POST | `/api/upload` | multipart `file` → preview, detection result or AI proposals |
| POST | `/api/sessions/{id}/confirm` | `{"columns": [{"index", "column", "role"}]}` → structure summary |
| GET | `/api/sessions/{id}/log` | Full event log for the session |

Sessions are held in memory, so a restart clears them. The logs on disk remain.

## Examples & tests

`examples/` contains small synthetic files:

- `maxquant_proteinGroups.txt` matches a signature (MaxQuant).
- `xcms_feature_table.csv` matches a signature (generic mz/rt metabolomics).
- `cohort_metabolites_wide.csv` matches no signature, so it goes to the AI fallback (or manual assignment).

```bash
python -m pytest -q
```

## Layout

```
backend/   main.py (API) · format_detect.py · llm_fallback.py · models.py · session_log.py · logs/
frontend/  index.html (home + 3D prism) · tool.html · style.css · home.js · app.js · fonts/
examples/  sample input tables
tests/     pytest suite (the AI is mocked, so no key is needed)
```
