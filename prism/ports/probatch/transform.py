# Derived from proBatch (Bioconductor; https://github.com/symbioticMe/proBatch, GitHub master,
# DESCRIPTION version 1.3.1, commit 524249e, April 2025).
# Original authors (DESCRIPTION): Jelena Cuklina, Chloe H. Lee, Patrick Pedrioli (aut);
# Yuliya Burankova (cre). Original license: GPL-3 (DESCRIPTION field; the repository has no
# separate license text file).
# Adapted to Python by PRISM, 2026-10-21.
# What changed: one log / unlog pair with an explicit offset (proBatch defaults to offset 1
# everywhere, and its unlog_df computes base^(x - offset) while unlog_dm computes base^x - offset;
# the second is the inverse of log(x + offset) and is the one ported).
"""log_transform_dm / unlog_dm."""

import numpy as np

_LOG = {2: np.log2, 10: np.log10, "e": np.log}
_EXP = {2: lambda x: np.power(2.0, x), 10: lambda x: np.power(10.0, x), "e": np.exp}


def log_transform(X, log_base=2, offset=None):
    """log_base(X + offset). The offset has no default (proBatch's 1 is used only in compatibility tests)."""
    if offset is None:
        raise ValueError("log_transform needs an explicit offset (no default).")
    return _LOG[log_base](np.asarray(X, float) + offset)


def unlog(X, log_base=2, offset=None):
    """The inverse of log_transform: base^X - offset."""
    if offset is None:
        raise ValueError("unlog needs the offset used by log_transform.")
    return _EXP[log_base](np.asarray(X, float)) - offset
