# PRISM

Recognize, audit and preprocess omics abundance tables. Rebuilt step by step as a Python package
that runs in Google Colab; the website will use it later.

## Steps

| Step | What | Status |
|---|---|---|
| 1 | Read a table exactly as written and describe its columns (`prism.read`) | done |
| 2 | Which columns are row IDs, measurements, annotations; the measurements as numbers (`prism.propose_layout`) | done |
| 3 | The AI connection (Gemini) and the digest: what the AI may see about a table, never data rows (`prism.digest`, `prism.connect`) | done |
| 4 | The AI proposes column groups and labels; code checks every column is claimed once, unmentioned ones become unresolved (`prism.propose_groups`) | done, for review |
| 5+ | Confirmation steps (layout, feature ID, annotations, values, samples, sample info, design, history, review), provenance, `schema.json` and output files | later |

The previous version (website, v1 to v4) is in the Git history at commit `c0e8a83`.

## Use it in Google Colab

1. **Make a GitHub token (once).** The repository is private, so Colab needs a token to download it.
   - Go to GitHub → Settings → Developer settings → Personal access tokens → **Fine-grained tokens** →
     **Generate new token**.
   - Repository access: **Only select repositories** → `Isra-inventor/PRISM`.
   - Permissions → Repository permissions → **Contents: Read-only**. Nothing else.
   - Generate it and copy it (it starts with `github_pat_`).
2. **Store it in Colab (once).** In a Colab notebook, click the key icon (**Secrets**) in the left
   bar → **Add new secret** → name `GITHUB_TOKEN`, value the token → switch on **Notebook access**.
3. **Open a notebook from `notebooks/`** (`01_read_a_table`, `02_columns_and_layout`, `03_ai_connection_and_digest`, `04_ai_groups_the_columns`) in Colab (File → Upload notebook, or open it from GitHub)
   and run the cells. It installs PRISM from GitHub and reads a file from your Google Drive.

From step 3 on, PRISM needs a Gemini key (https://aistudio.google.com/apikey). If it is not in Colab
Secrets as `GEMINI_API_KEY`, PRISM asks you to paste it once per session; save it in Secrets to
never be asked.

Put your data files in Google Drive (for example `MyDrive/PRISM_data/`), not in the repository.

## Run the tests (on any computer with Python 3.10+)

```bash
pip install -e ".[test]"
pytest
```
