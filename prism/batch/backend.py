"""BatchBackend (v4 section 7.1): the only door from the core to batch-correction code derived
from other projects. The proBatch port is one implementation. With PRISM_DISABLE_PORTS=1 the port
is not imported and `get()` returns a backend whose methods refuse with a clear message."""

from __future__ import annotations

from .. import ports


class BackendUnavailable(Exception):
    pass


class BatchBackend:
    name = "none"
    source = None

    def drift_loess(self, X, batch, order, span=0.75, degree=2, min_measurements=8):
        raise BackendUnavailable("sample-based LOESS drift correction needs the proBatch port, which is disabled "
                                 "(PRISM_DISABLE_PORTS=1).")

    def center(self, X, batch, stat="median"):
        raise BackendUnavailable("per-batch centering needs the proBatch port, which is disabled "
                                 "(PRISM_DISABLE_PORTS=1).")

    def combat(self, X, batch):
        raise BackendUnavailable("ComBat via the batch backend needs the proBatch port, which is disabled "
                                 "(PRISM_DISABLE_PORTS=1).")

    def diagnostics(self):
        raise BackendUnavailable("proBatch diagnostics need the proBatch port, which is disabled "
                                 "(PRISM_DISABLE_PORTS=1).")


class ProBatchBackend(BatchBackend):
    name = "probatch_port"

    def __init__(self):
        from ..ports import probatch
        from ..ports.probatch import correct, diagnostics, drift
        self.source = probatch.SOURCE
        self._correct, self._drift, self._diag = correct, drift, diagnostics

    def drift_loess(self, X, batch, order, span=0.75, degree=2, min_measurements=8):
        return self._drift.adjust_batch_trend(X, batch, order, span, degree, min_measurements)

    def center(self, X, batch, stat="median"):
        return self._correct.center_feature_batch(X, batch, stat)

    def combat(self, X, batch):
        return self._correct.combat_intercept(X, batch)

    def diagnostics(self):
        return self._diag


def get():
    if not ports.enabled():
        return BatchBackend()
    return ProBatchBackend()
