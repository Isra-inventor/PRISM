# Licensing

**The license of the PRISM package is undecided.** No license has been chosen for this repository.

- `prism/ports/probatch/` is adapted from proBatch (Bioconductor), whose `DESCRIPTION` states
  **GPL-3**. Code in that folder is GPL-3 derived. See `prism/ports/probatch/NOTICE.md` for the
  source, the authors and what changed.
- The rest of PRISM reaches that code only through `prism/batch/backend.py`, and runs its tests with
  the port disabled (`PRISM_DISABLE_PORTS=1`) for every part that does not need it. This keeps the
  licensing decision reversible: the port can be kept (with a GPL-compatible package license),
  replaced, or shipped separately.
- protDP (Li and Smyth, Apache-2.0) was downloaded only to compare results; none of its code is in
  this repository.

**Nothing is to be published until the decision is made**, including to PyPI or a public
repository. The decision belongs to the project owner, together with the university licensing
office.
