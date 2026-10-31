"""PRISM: recognize, audit and preprocess proteomics abundance tables (other omics: out of scope for now).

Built one step at a time. What exists so far:

    import prism
    table = prism.read("my_file.csv")       # step 1: read a table exactly as written, and describe it
    layout = prism.propose_layout(table)    # step 2: which columns are IDs, measurements, annotations
    layout = layout.set(annotations=[...])  #         correct anything
    X = layout.matrix()                     #         the measurements as numbers, features x samples
    d = prism.digest(table, layout)         # step 3: what the AI is allowed to see (no data rows)
    prism.show(d)                           #         print it
    ai = prism.connect()                    #         Gemini, key from Colab Secrets (GEMINI_API_KEY)
    ai.check()                              #         is the key working
    g = prism.propose_groups(table)         # step 4: the AI groups and labels the columns; code checks it
    g.summary(), g.columns(), g.rejected    #         look at the proposal
"""

from .ai import AIError, MockAI, connect
from .digest import digest, show
from .grouping import Group, Grouping, propose_groups
from .layout import Layout, propose_layout
from .table import Table, read

__version__ = "0.4.0"
__all__ = ["read", "Table", "propose_layout", "Layout", "digest", "show", "connect", "MockAI", "AIError",
           "propose_groups", "Grouping", "Group", "__version__"]
