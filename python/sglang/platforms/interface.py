"""Shared platform interface for SGLang runtimes (SRT, multimodal_gen, etc.).

BasePlatform defines the common device abstraction that all platform plugins
must implement.  SRT and MM each extend it with runtime-specific methods
(see ``sglang.srt.platforms.interface.SRTPlatform``).
"""

from __future__ import annotations

import enum
import platform as _platform
import random
from abc import ABC
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Enums & value types
# ---------------------------------------------------------------------------


class PlatformEnum(enum.Enum):
    CUDA = enum.auto()
    ROCM = enum.auto()
    TPU = enum.auto()
    CPU = enum.auto()
    MPS = enum.auto()
    NPU = enum.auto()
    MUSA = enum.auto()
    HPU = enum.auto()
    XPU = enum.auto()
    OOT = enum.auto()
    UNSPECIFIED = enum.auto()


class CpuArchEnum(enum.Enum):
    X86 = enum.auto()
    ARM = enum.auto()
    UNSPECIFIED = enum.auto()


class DeviceCapability(NamedTuple):
    major: int
    minor: int

    def as_version_str(self) -> str:
        return f"{self.major}.{self.minor}"

    def to_int(self) -> int:
        """Express capability as ``<major><minor>`` (minor is single digit)."""
        assert 0 <= self.minor < 10
        return self.major * 10 + self.minor


# ---------------------------------------------------------------------------
# BasePlatform
# ---------------------------------------------------------------------------


class BasePlatform(ABC):
    """Hardware-agnostic base that both SRT and MM platform classes inherit."""

    name: str
    _enum: PlatformEnum
    device_type: str
    dispatch_key: str = "CPU"

    # ------------------------------------------------------------------
    # Detection helpers (derived from ``_enum``)
    # ------------------------------------------------------------------

    def is_cuda(self) -> bool:
        return self._enum == PlatformEnum.CUDA

    def is_hip(self) -> bool:
        return self._enum == PlatformEnum.ROCM

    def is_rocm(self) -> bool:
        return self._enum == PlatformEnum.ROCM

    def is_npu(self) -> bool:
        return self._enum == PlatformEnum.NPU

    def is_xpu(self) -> bool:
        return self._enum == PlatformEnum.XPU

    def is_musa(self) -> bool:
        return self._enum == PlatformEnum.MUSA

    def is_hpu(self) -> bool:
        return self._enum == PlatformEnum.HPU

    def is_mps(self) -> bool:
        return self._enum == PlatformEnum.MPS

    def is_cpu(self) -> bool:
        return self._enum == PlatformEnum.CPU

    def is_cuda_alike(self) -> bool:
        return self._enum in (PlatformEnum.CUDA, PlatformEnum.ROCM)

    def is_out_of_tree(self) -> bool:
        return self._enum == PlatformEnum.OOT

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def get_device(self, local_rank: int) -> torch.device:
        raise NotImplementedError

    def get_device_name(self, device_id: int = 0) -> str:
        raise NotImplementedError

    def get_device_uuid(self, device_id: int = 0) -> str:
        raise NotImplementedError

    def get_device_capability(self, device_id: int = 0) -> DeviceCapability | None:
        return None

    def has_device_capability(
        self,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        current = self.get_device_capability(device_id=device_id)
        if current is None:
            return False
        if isinstance(capability, tuple):
            return current >= capability
        return current.to_int() >= capability

    def empty_cache(self) -> None:
        pass

    def synchronize(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def get_device_total_memory(self, device_id: int = 0) -> int:
        """Total device memory in bytes."""
        raise NotImplementedError

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        """Return ``(free_bytes, total_bytes)``."""
        raise NotImplementedError

    def get_current_memory_usage(self, device: torch.device | None = None) -> float:
        """Return current memory usage in bytes."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Distributed
    # ------------------------------------------------------------------

    def get_distributed_backend(self) -> str:
        raise NotImplementedError

    def get_communicator_class(self) -> type | None:
        return None

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    @classmethod
    def inference_mode(cls):
        return torch.inference_mode(mode=True)

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def verify_quantization(self, quant: str) -> None:
        pass

    @classmethod
    def get_cpu_architecture(cls) -> CpuArchEnum:
        machine = _platform.machine().lower()
        if machine in ("x86_64", "amd64", "i386", "i686"):
            return CpuArchEnum.X86
        elif machine in ("arm64", "aarch64"):
            return CpuArchEnum.ARM
        return CpuArchEnum.UNSPECIFIED
