"""Auto-loaded shim: installs the ZAYA router probe in EVERY python process.

Put this directory FIRST on PYTHONPATH. CPython imports ``sitecustomize`` at
interpreter startup, so this reaches the tokenizer manager, the scheduler
processes and all 8 TP ranks without touching a single source file.

It is a no-op unless ``ZAYA_PROBE_DIR`` is set, so leaving the directory on
PYTHONPATH costs nothing when you are not probing.
"""

import os

if os.environ.get("ZAYA_PROBE_DIR"):
    try:
        import zaya_router_probe  # noqa: F401

        zaya_router_probe.install()
    except Exception as exc:  # pragma: no cover - never break the server
        import sys

        print(f"[zaya-probe] install failed: {exc!r}", file=sys.stderr)
