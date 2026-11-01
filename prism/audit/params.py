"""Every number the audit uses as a threshold, in one place. Each one has an evidence level in
prism/registry/parameters.yaml (literature_independent, expert_guidance, convention, heuristic,
unverified); `prism.registry.check()` fails if one is missing. A "heuristic" is a rule of thumb:
the report says so, and the value is open to discussion."""

DEFAULTS = {
    # G1 / A1 integrity
    "sentinel_share": 0.01,            # one non-zero value covering this share of cells reads as a placeholder
    "sentinel_min_cells": 3,           # ... and at least this many cells
    "min_observed": 3,                 # proteins with fewer observed values are sparse
    # G2 / A2 scale, A13 processing state
    "median_scaling_mad": 0.5,         # MAD of log2 per-protein medians below this: proteins share a common median
    "narrow_median_spread": 0.05,      # MAD (log2) of sample medians below this reads as normalized
    "floor_min_observed": 5,           # proteins need this many observed values for floor ties
    "floor_tie_share_indicator": 0.01, # floor ties count as "present" from this share of proteins
    "compositional_cv": 0.001,         # sample sums this constant read as compositional
    # G7 / A4, A12 missingness
    "abundance_rho_indicator": -0.1,   # Spearman rho at or below this gets the detection-limit sentence
    "a12_min_level_n": 3,              # a batch level needs this many samples for batch-wide absence
    "a12_min_partial_features": 50,    # proteins with 0 < detected < n needed for the detection curve
    "a12_bootstrap": 200,              # bootstrap resamples over proteins for the CI of b1
    "a12_b1_negligible": 0.1,          # |b1| CI inside +-this (per log2 unit) reads as "no" intensity dependence
    "a12_batch_absence_share": 0.01,   # batch-wide absence above this share reads as batch-associated
    "a12_low_abundance_fraction": 0.3333,  # "lowest third" of proteins by mean observed log2 value
    "a12_feature_permutations": 999,   # per-protein missing-indicator tests: Monte Carlo permutations
    # G4 / A3, A6 and the diagnostic matrix (Build Request v4, section 3)
    "bootstrap": 999,                  # A3 bootstrap resamples
    "sample_median_spread_mad": 0.1,   # MAD (log2) of sample medians above this reads as "medians differ"
    "rsd_levels": [20, 30],            # QC RSD levels, percent
    "run_order_rho_indicator": 0.3,    # |Spearman rho| of sample medians with run order reported from here
    "qc_drift_share": 0.2,             # share of proteins with |rho| > 0.5 against run order in QC samples
    "diag_min_features": 20,           # complete_features(100) needs this many proteins, else q = 75, 50
    "diag_min_share": 0.1,             # ... and this share of all proteins
    "diag_sample_missing_max": 0.5,    # for q < 100, samples missing more than this share are dropped
    # A5, A7, A8, A11, G3 (ported from the v4 build)
    "categorical_max_levels": 20,      # categorical variables need 2 to this many levels
    "time_ordered_max_levels": 12,     # time is an ordered factor up to this many values, else numeric
    "max_pcs": 5,                      # k = min(max_pcs, n - 1)
    "modified_z_cutoff": 3.5,          # outlier cut-off (Iglewicz and Hoaglin)
    "icc_indicator": 0.5,              # share of proteins with ICC above this is reported
    "batch_min_samples": 25,           # quoted in Cuklina et al. 2021 as a suggestion from other work; unverified
    "blank_dominated_ratio": 0.5,      # G3 flags proteins whose mean blank signal is this share of the study mean
    "near_duplicate_r": 0.9999,        # samples correlating above this read as near-duplicates
    # shared by every permutation test (v3 section 6.1)
    "seed": 0,
    "permutations": 999,
    "exhaustive_max": 10000,           # enumerate every distinct permutation up to this many
}


def defaults() -> dict:
    return dict(DEFAULTS)
