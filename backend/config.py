"""Editable settings.

SCOPE_DESCRIPTION tells the AI what PRISM's later steps currently support.
Files outside it are still recognized and described, but flagged
in_supported_scope "no" / "unsure". Widening scope later = editing this text
(or setting PRISM_SCOPE_DESCRIPTION in .env).
"""

import os

SCOPE_DESCRIPTION = os.environ.get("PRISM_SCOPE_DESCRIPTION") or (
    "Quantified proteomics and metabolomics tables (MS-based tool outputs such as MaxQuant, DIA-NN, "
    "XCMS/MZmine-style tables, and platform exports such as SomaScan), including paired and "
    "longitudinal designs."
)

# Prompt size limit for the grouping call: files with more columns are sent to the
# AI in chunks of this many columns (plus one consolidation call). It is an
# infrastructure constraint, NOT a claim about how big a column family can be.
GROUPING_CHUNK_SIZE = int(os.environ.get("PRISM_GROUPING_CHUNK_SIZE") or 150)

# v2.4 fragmentation check (§16): kept value blocks are "near-identical" when their medians are
# within FRAGMENT_MEDIAN_FACTOR of each other and their p1 and p99 each within FRAGMENT_TAIL_FACTOR.
# Tuned on the two real files (SomaScan, metabolomics); the result is a question, never a merge.
FRAGMENT_MEDIAN_FACTOR = float(os.environ.get("PRISM_FRAGMENT_MEDIAN_FACTOR") or 2.0)
FRAGMENT_TAIL_FACTOR = float(os.environ.get("PRISM_FRAGMENT_TAIL_FACTOR") or 3.0)
