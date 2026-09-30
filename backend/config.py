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
