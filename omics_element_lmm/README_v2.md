# Elements x proteome in HIV/ART macaques - full re-analysis (all samples kept)

Inputs: `Cleaned_data.xlsx` (37 element samples incl. Ni), `SomaExpr_common.csv`, `sample_metadata.tsv`.
Proteomics exists for 27 of the 37 element samples (8 infected animals); the other 10 (7 naive animals, 3 infected samples with no proteomics)
are used in the element-level analysis. **No sample is excluded anywhere.** (Leave-one-animal-out columns are diagnostics only.)

## Methods (what and why)
* Random-intercept LMM per protein (animal), REML, t-tests. Vectorised custom engine: never crashes on zero animal variance (the notebook silently lost ~50% of proteins).
* **Degrees of freedom were calibrated with permutations that keep the animal clustering of the element** (8 animals): element terms df=8, between-animal term df=4.
  The first version of this pipeline used df=14 and gave too many false positives for elements that mostly differ between animals (Cu, Fe, Mg). Earlier Cu "101 hits" disappear.
* Models M1-M8 (see COLUMNS.txt): stage, viral load, week, within/between animal, element x week, stage-specific slopes, hemolysis-adjusted, rank-normalised element.
* Ni used as negative control, plus "Ni_resFe" (Ni with the Fe-explained part removed, because Ni ~ 0.46% of Fe, r=0.62).
* Multi-protein: protein modules; exhaustive screen of all 60.6M protein pairs with a permutation FWER; element-modulated coupling among the top-60 proteins.

## Key results
1. **No single protein-element association passes FDR<0.05 for any element, in any model (M1-M8).** (Cu M4 within-animal: 25 proteins, see 4.)
2. Calibration check: Ni and Fe-corrected Ni give lambda 1.04 / 0.77 (QQ in fig2) -> the p-values are well calibrated. Mg, P, Cu have lambda ~2.0-2.5: many proteins shift a little together with these elements.
3. Fe: top hits (CDK2, PFDN4 ...) are all pulled by one sample (T545_0, Fe 41.7, high hemolysis score). Fe correlates with the hemolysis score (r=0.50, p=0.009). Rank-normalised Fe gives the same non-significant result.
4. Cu: association is mostly between-animal (DEAB and T544 baseline), the within-animal Cu effect (M4) has 25 proteins at q<0.05 (CLPSL1, TESK1, MND1, CSDC2 ...); driven by 2-3 animals whose Cu falls on ART. Treat as a lead.
5. Along treatment time: element x week interaction (M5) - no protein passes FDR; stage-specific slopes (M6) heterogeneous only for P (61 proteins, but Mid-ART has 4 samples).
6. **Proteins change strongly with ART independent of elements**: 48 proteins FDR<0.05 for stage, 545 for week (CXCL13, SEMA4A, CRTAM, CD28, ISG15, KLHDC2 down). Module M05 (interferon-stimulated: MX1, ISG15) falls with ART (q=0.0015) = built-in positive control.
7. Elements themselves (fig1, all 37 samples): Na, K, Ca lower in infected than naive animals; P and Mg rise on ART (q 0.003 / 0.012 with week); Zn up early; Cu higher pre-ART than naive. Naive animals are a different group (not matched), so naive-vs-infected differences could also be batch/cohort.
8. **Protein pairs ("tri-relationships")**: for Mg (5 pairs, FWER p=0.01), P (1 pair, 0.03) and Cu (18 pairs, 0.01) the best pairs beat chance. Pattern: two proteins correlated with each other (r~0.8) that individually barely relate to the element, but their difference (or sum if r<0) does (partial R2 0.5-0.8; stable when any single animal is left out). Not significant for Na, K, Ca, Fe, Zn, Fe-corrected Ni; Ni itself is borderline (p=0.05) -> treat as exploratory and needing replication.
9. Element-modulated coupling (A->B depends on element): 7 pairs for Cu, 25 for Zn at q<0.05 among the top-60 proteins - lowest-powered analysis; exploratory.

## Caveats
8 animals / 27 samples: only large effects detectable; the element is largely an animal-level trait (ICC 0.5-0.6 for Fe, Mg, Cu), so effective sample size for element effects is close to 8.
Mid-ART has 4 samples. Hemolysis score is a proxy from SomaScan red-cell proteins. Viral load for blanks in metadata set to 50 (undetectable); GA43_4 is 50 in metadata vs 67 in the old Run2 file.

Run: `python full_analysis.py --data-dir D --out-dir results_v2; python tri_analysis.py ...; python pair_followup.py ...; python figures_v2.py ...`

## Addendum - Cu analysed animal by animal (`per_animal_assoc.py`, results_v2/per_animal)
Per animal (>=4 samples: DEAB, DEL5, GA43, T544, T623), each protein's within-animal Spearman rho and slope vs Cu; then compared across animals
(random-effects meta-analysis, exact within-animal permutation test of the mean rho, leave-one-animal-out replication, animal-vs-animal profile similarity).
Result: 1 protein at FDR<0.05 (FAM170B, same sign in all 5 animals), GSN q=0.08; spread of mean rho = chance (ratio 1.03); top-100 proteins chosen on 4 animals do not replicate in the 5th (p 0.14-1.0);
animal profiles uncorrelated (mean off-diagonal 0.007). The pooled-model Cu signal (25 proteins) is therefore not supported as a consistent per-animal pattern.

## Addendum 2 - best-practice within-animal re-analysis for every element (`within_animal_models.py`, `predict_loao.py`, `adjust_cellular.py`, `time_spec_check.py`; results_v2/within)
* Common-slope within-animal model (animal fixed intercepts, shared quadratic time trend; = repeated-measures correlation) + slope-heterogeneity F test; p-values from 1000 within-animal permutations of the element (the parametric heterogeneity F test was too liberal, 11-12% FPR at 0.05).
  Variance shrinkage across proteins (limma-style) changed nothing (d0~2.3).
* Held-out-animal prediction (ridge on proteome, element shuffled within animals as null).
* Results: no protein FDR<0.05 for any element. Excess of small p (p<0.001; chance = 11): Mg 166, P 162, Cu 37; Fe, Na, K, Ca, Zn and both Ni controls at chance.
  Held-out prediction: Mg rho=0.56 (p=0.023), Cu 0.49 (p=0.033), P 0.31 (p=0.13); others and Ni controls none.
* Mg and P within-animal slopes correlate r=0.97 across proteins; ~90% positive; mostly intracellular/blood-cell proteins (MYL9, MYL12A, PTK2B, PRKCB, G6PD, S100A9, STAT1/3). Partly explained by global protein level (Mg p<0.001: 166 -> 55) and by red-cell/platelet scores (P q<0.10: 304 -> 0).
* Min detectable within-animal slope (80% power, p=1e-4): 0.33-0.51 log2 per SD of element.
