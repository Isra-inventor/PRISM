"""Batch-effect and normalization methods used by the executor (Build Request v4, sections 6-7).

The core reaches the proBatch port only through `backend.get()`; with PRISM_DISABLE_PORTS=1 the
port is not loaded and the ops that need it refuse with a clear message.
"""
