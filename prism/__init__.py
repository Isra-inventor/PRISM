"""PRISM beyond Step 0: sessions of several datasets, schema import, merging and the Tier 1 audit.

Everything here is deterministic: no LLM is called in import, merge or audit. The Step 0
wizard (backend/) produces each dataset's output folder; the audit reads only those folders
plus overrides.json. Python 3.7+, numpy only.
"""

__version__ = "0.1.0"
