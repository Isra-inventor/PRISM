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
}


def defaults() -> dict:
    return dict(DEFAULTS)
