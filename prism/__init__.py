"""PRISM: recognize, audit and preprocess omics abundance tables.

Built one step at a time. What exists so far:

    import prism
    table = prism.read("my_file.csv")   # step 1: read a table exactly as written, and describe it
    table                               # in a notebook, shows the summary
"""

from .table import Table, read

__version__ = "0.1.0"
__all__ = ["read", "Table", "__version__"]
