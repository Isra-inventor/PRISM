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
