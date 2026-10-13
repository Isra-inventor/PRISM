# Is the LMM the right method? Review of `HIV_Anemia.ipynb` + all-elements pipeline

## Verdict
**A random-intercept mixed model is the right family** (repeated samples per animal, unbalanced, 1–7 per animal, 3 animals have one sample).
But the notebook's *implementation* has problems that I measured on your data (27 samples, 8 animals, 11,011 proteins):

| Issue | Evidence | Fix in `lmm_engine.py` |
|---|---|---|
| **Fe→Ca find/replace broke the code** | `reCarence`, `efCacts`, duplicated `Ca` column (`Ca`,`Ca`; second is Fe), so `Treatment(reCarence="Pre-ART")` raises | New code is element-generic |
| **~50% of proteins silently dropped** | `except Exception: return None`; statsmodels raises `Singular matrix` when animal variance → 0 (76/150 proteins tested) | Profile-REML over variance ratio; ratio 0 = plain OLS, never crashes (≈12% of proteins sit on that boundary) |
| **ML + z-tests anti-conservative** | Permutation null (element shuffled): 11.2% false positives at p<0.05, 4.3% at p<0.01 | REML + t with df = n−G−(p−1) = 14: 5.5% / 1.4% |
| **Viral load ≈ ART stage** | R² of VL on stage = 0.85; Mid/Late-ART VL is *all* at the 50-copy floor (log10 1.70); VIF 6–10 with VL in model | Run both models (`--vl both`), see below |
| Hard-coded element table | 25 rows, 10× the scale of `Run2_common.csv`, has Ni; file has 27 rows, no Ni | Reads `Run2_common.csv` (single source) |
| SomaScan controls kept | 12 hybridisation-control + 3 non-human rows in the 11,083 | Filtered with `Type_meta=="Protein"` (11,011 left) |

## Biggest scientific point: VL vs stage
With VL in the model, stage effects are near-zero (0–2 proteins per element at FDR<0.05); without VL, **24–49 proteins per element** are
stage-associated (omnibus F test). Stage and VL carry the same information here, so adjusting for VL removes the stage signal.
Recommended primary model: `omics ~ element + stage + (1|animal)`; VL-adjusted model as sensitivity.

## Results (proteomics, FDR<0.05 within element)
Element association: only **Cu (101 proteins, 72 with VL)** and Mg (1). Not significant after FDR across all 8 elements together (`q_E_global`).
Fe: 0. Cu hits are sign-stable after removing the 2 highest-Cu samples, but Cu is driven by few samples (4 animals with high Cu). Treat as hypothesis-generating.

## Limits you should know
* 27 samples / 8 animals → low power; the element term is partly *between-animal* (ICC of Fe 0.63, Mg 0.39, Cu 0.52), not within-animal.
* Mid-ART has only 4 samples from 3 animals; element×stage interaction is not estimable.
* 8 elements × 11k tests: use `q_E_global` if you want to claim a hit across elements.
* Better still (if you can use R): `limma::duplicateCorrelation` / `variancePartition::dream` add variance moderation across proteins, which gains power at this n.
* Metabolomics: same pipeline, but I had no metabolomics file, so that path is untested.
* Ni is not in `Run2_common.csv`; add it as a column to include it.

## Run
`python run_all_elements.py --data-dir <dir> --out-dir results --vl both`, then `python make_plots.py results/proteomics/vl_none`.
