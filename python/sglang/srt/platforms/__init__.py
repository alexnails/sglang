"""SRT platform discovery, registry, and ``current_platform`` global.

Three sources of platform plugins, checked in order:

1. **Built-in** -- always available (CUDA, CPU).
2. **``entry_points``** -- pip-installed packages that declare
   ``[project.entry-points."sglang.platforms"]``.
3. **``SGLANG_PLATFORM_PLUGIN``** env-var -- ``"name:qualname"`` override
   for development / testing.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import os
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.platforms.interface import SRTPlatform

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

_discovered: dict[str, type[SRTPlatform]] | None = None


def discover_platforms() -> dict[str, type[SRTPlatform]]:
    """Return a mapping ``device_name -> SRTPlatform subclass``."""
    global _discovered
    if _discovered is not None:
        return _discovered

    platforms: dict[str, type[SRTPlatform]] = {}

    # 1. Built-in (always available)
    from sglang.srt.platforms.cpu import CPUSRTPlatform
    from sglang.srt.platforms.cuda import CUDASRTPlatform

    platforms["cuda"] = CUDASRTPlatform
    platforms["cpu"] = CPUSRTPlatform

    # Stubs for platforms that don't have a full class yet.
    # These will be replaced by real classes in Phase 2-3.
    from sglang.srt.platforms._stubs import (
        HPUSRTPlatformStub,
        MUSASRTPlatformStub,
        NPUSRTPlatformStub,
        XPUSRTPlatformStub,
    )

    platforms["npu"] = NPUSRTPlatformStub
    platforms["xpu"] = XPUSRTPlatformStub
    platforms["musa"] = MUSASRTPlatformStub
    platforms["hpu"] = HPUSRTPlatformStub

    # 2. entry_points from pip-installed packages
    try:
        for ep in importlib.metadata.entry_points(group="sglang.platforms"):
            try:
                cls = ep.load()
                platforms[ep.name] = cls
            except Exception as exc:
                logger.warning("Failed to load platform plugin %s: %s", ep.name, exc)
    except Exception:
        pass

    # 3. SGLANG_PLATFORM_PLUGIN env-var override  (``name:qualname``)
    plugin_spec = os.environ.get("SGLANG_PLATFORM_PLUGIN")
    if plugin_spec:
        if ":" in plugin_spec:
            name, qualname = plugin_spec.split(":", 1)
        else:
            name = plugin_spec.rsplit(".", 1)[-1]
            qualname = plugin_spec
        try:
            module_name, obj_name = qualname.rsplit(".", 1)
            mod = importlib.import_module(module_name)
            cls = getattr(mod, obj_name)
            platforms[name] = cls
        except Exception as exc:
            logger.warning(
                "Failed to load SGLANG_PLATFORM_PLUGIN=%s: %s", plugin_spec, exc
            )

    _discovered = platforms
    return platforms


# ---------------------------------------------------------------------------
# Instantiation cache
# ---------------------------------------------------------------------------

_instances: dict[str, SRTPlatform] = {}


def get_platform(device: str) -> SRTPlatform:
    """Return a (cached) ``SRTPlatform`` instance for *device*."""
    # Normalise ``"cuda:0"`` → ``"cuda"``
    base_device = device.split(":")[0] if ":" in device else device

    if base_device in _instances:
        return _instances[base_device]

    platforms = discover_platforms()
    cls = platforms.get(base_device)
    if cls is None:
        logger.warning(
            "No SRTPlatform registered for device '%s'; falling back to CUDA default.",
            base_device,
        )
        from sglang.srt.platforms.cuda import CUDASRTPlatform

        cls = CUDASRTPlatform

    instance = cls()
    _instances[base_device] = instance
    return instance


# ---------------------------------------------------------------------------
# Lazy ``current_platform`` global
# ---------------------------------------------------------------------------

_current_platform: SRTPlatform | None = None
_init_trace: str = ""


def _resolve_current_platform() -> SRTPlatform:
    """Auto-detect the current device and return its platform instance.

    Detection order mirrors ``sglang.srt.utils.common.get_device()``:
    CPU (opt-in) → CUDA → XPU → NPU → HPU → MUSA → CPU (fallback).
    """
    from sglang.srt.utils.common import get_device

    device = get_device()
    return get_platform(device)


def __getattr__(name: str):
    if name == "current_platform":
        global _current_platform
        if _current_platform is None:
            _current_platform = _resolve_current_platform()
            global _init_trace
            _init_trace = "".join(traceback.format_stack())
        return _current_platform
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "discover_platforms",
    "get_platform",
    "current_platform",
]
