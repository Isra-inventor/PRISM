"""Code derived from third-party projects, kept apart so the licensing decision stays reversible
(LICENSING.md). Set PRISM_DISABLE_PORTS=1 to run PRISM without any of it."""

import os


def enabled():
    return os.environ.get("PRISM_DISABLE_PORTS", "").strip() not in ("1", "true", "yes")
