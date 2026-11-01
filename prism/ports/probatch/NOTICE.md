# NOTICE: the proBatch port

The modules in `prism/ports/probatch/` are adapted to Python from **proBatch**.

| | |
|---|---|
| Project | proBatch: tools for diagnostics and corrections of batch effects in proteomics (Bioconductor) |
| Source used | `https://github.com/symbioticMe/proBatch`, branch `master`, commit `524249ee9db539ca0e93442ca698bb14cf827d90` (25 April 2025), `DESCRIPTION` version 1.3.1 |
| Authors (`DESCRIPTION`) | Jelena Cuklina (aut), Chloe H. Lee (aut), Patrick Pedrioli (aut), Yuliya Burankova (cre) |
| License | GPL-3 (the `License:` field of `DESCRIPTION`; the repository has no separate license text file) |
| Adapted by | PRISM, 2026-10-21 |

Paper: Cuklina J, Lee CH, Williams EG, et al. Diagnostics and correction of batch effects in large-scale
proteomic studies: a tutorial. Mol Syst Biol. 2021;17:e10240.

What changed in the adaptation:
- matrix interface (features x samples) instead of long data frames;
- missing values are never filled (proBatch fills 0 or -1 in some diagnostics);
- the confirmed bugs listed in `docs/probatch_port.md` are not reproduced; each has a test;
- every call reports its parameters and counts for PRISM's op record (proBatch 2.0's operation log
  and its `ProBatchFeatures` container are not used).

The Bioconductor release could not be downloaded in the build environment (network policy), so the
GitHub master was used. Its relation to the current Bioconductor release is not verified.

PRISM's package license is undecided (see `LICENSING.md`). Nothing is to be published until that
decision is made.
