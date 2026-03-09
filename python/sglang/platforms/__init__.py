"""Shared platform types for SGLang.

These are re-exported here so both ``sglang.srt`` and
``sglang.multimodal_gen`` can depend on them without cross-importing.
"""

from sglang.platforms.interface import (  # noqa: F401
    BasePlatform,
    CpuArchEnum,
    DeviceCapability,
    PlatformEnum,
)

__all__ = [
    "BasePlatform",
    "CpuArchEnum",
    "DeviceCapability",
    "PlatformEnum",
]
