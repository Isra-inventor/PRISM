"""PRISM: recognize, audit and preprocess omics abundance tables.

Built one step at a time. What exists so far:

    import prism
    table = prism.read("my_file.csv")       # step 1: read a table exactly as written, and describe it
    layout = prism.propose_layout(table)    # step 2: which columns are IDs, measurements, annotations
    layout = layout.set(annotations=[...])  #         correct anything
    X = layout.matrix()                     #         the measurements as numbers, features x samples
"""

from .layout import Layout, propose_layout
from .table import Table, read

__version__ = "0.2.0"
__all__ = ["read", "Table", "propose_layout", "Layout", "__version__"]
