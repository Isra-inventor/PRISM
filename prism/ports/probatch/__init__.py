# Derived from proBatch (Bioconductor; https://github.com/symbioticMe/proBatch, GitHub master,
# DESCRIPTION version 1.3.1, commit 524249e, April 2025).
# Original authors (DESCRIPTION): Jelena Cuklina, Chloe H. Lee, Patrick Pedrioli (aut);
# Yuliya Burankova (cre). Original license: GPL-3 (DESCRIPTION field; the repository has no
# separate license text file).
# Adapted to Python by PRISM, 2026-10-21.
# What changed: matrix (features x samples) interface instead of long data frames; missing values
# are never filled; every call reports what it did for the PRISM op record; the bugs listed in
# docs/probatch_port.md are not reproduced.
"""The proBatch port (Build Request v4, section 7). GPL-3 derived: see NOTICE.md and LICENSING.md.
The core reaches it only through prism.batch.backend."""

SOURCE = {"project": "proBatch", "url": "https://github.com/symbioticMe/proBatch", "version": "1.3.1 (GitHub master)",
          "commit": "524249ee9db539ca0e93442ca698bb14cf827d90", "license": "GPL-3", "source_verified": True}
