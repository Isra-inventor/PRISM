# PRISM

Recognize, audit and preprocess **proteomics** abundance tables (SomaScan, Olink, MaxQuant, DIA-NN,
Spectronaut, FragPipe protein-level outputs). Other omics are recognized but marked out of scope for now.
A Python package that runs in Google Colab; the website will use it later.

**Read [`docs/GUIDE.md`](docs/GUIDE.md)**: every step explained in order, with every threshold and rule.

## The steps

| Steps | What | Where |
|---|---|---|
| 1-4 | Read the file as written; a first layout hint; the digest the AI may see (no data rows); the AI groups and labels the columns, code checks the answer | `prism.read`, `propose_layout`, `digest`, `propose_groups` |
| 5-10 | The schema wizard: confirm or correct groups, protein ID and flagged rows, value blocks and samples, sample information and design, processing history; save `schema.json` and the output folder | `prism.start`, `prism/schema.py` ... `outputs.py` |
| 11 | The audit (Build Request v4): G1 integrity, G2 scale and processing state, G4 distributions, QC and batch structure, G7 missingness profile, S4/S5/S10 inputs (outliers, repeated measures, design) | `prism.Audit`, `prism/audit/` |
| 12 | Intake, the metadata gate, the rule base and the plan; approval runs one action on a new data state and re-audits it | `prism.Study`, `prism/plan/`, `prism/rules/`, `prism/executor/` |
| 13 | Verification at C2 (before / after, never used to rank methods), replay, saving | `Study.verify`, `replay`, `save` |
| later | Method selection (imputation is decided there; `prism/phase2/` is the hook), the website | |

The steps and their properties are in `prism/registry/steps.yaml`; references in `prism/registry/evidence.yaml`.
The previous website (v1 to v4) is in the Git history at commit `c0e8a83` (v4 package code at `aee0912`).

## Use it in Google Colab

1. **Make a GitHub token (once).** GitHub → Settings → Developer settings → Personal access tokens →
   **Fine-grained tokens** → **Generate new token**. Repository access: **Only select repositories** →
   `Isra-inventor/PRISM`. Permissions → Repository permissions → **Contents: Read-only**.
2. **Store it in Colab (once).** Key icon (**Secrets**) in the left bar → **Add new secret** → name
   `GITHUB_TOKEN` → switch on **Notebook access**.
3. **Gemini key.** From https://aistudio.google.com/apikey. Save it in Secrets as `GEMINI_API_KEY`, or
   PRISM asks you once per session.
4. **Open the notebooks in `notebooks/`** in order (`01` ... `10`). Put your data in Google Drive (for
   example `MyDrive/PRISM_data/`), not in the repository.

## Licensing

Undecided: see `LICENSING.md`. `prism/ports/probatch/` is derived from proBatch (GPL-3) and kept apart;
`PRISM_DISABLE_PORTS=1` runs PRISM without it. Nothing is to be published until the license is decided.

## Run the tests (on any computer with Python 3.10+)

```bash
pip install -e ".[test]"
pytest
```
