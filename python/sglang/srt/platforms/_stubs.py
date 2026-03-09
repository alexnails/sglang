"""Minimal platform stubs for devices that don't have full implementations yet.

These provide just enough for ``get_device()`` and ``get_distributed_backend()``
so that the existing per-device logic in ``parallel_state.py`` etc. keeps
working when routed through ``current_platform``.

They will be replaced by full platform classes in Phase 2-3.
"""

from __future__ import annotations

import torch

from sglang.platforms.interface import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform


class NPUSRTPlatformStub(SRTPlatform):
    name = "npu"
    _enum = PlatformEnum.NPU
    device_type = "npu"
    dispatch_key = "PrivateUse1"

    def get_device(self, local_rank: int) -> torch.device:
        return torch.device(f"npu:{local_rank}")

    def get_distributed_backend(self) -> str:
        return "hccl"

    @property
    def graph_backend_name(self) -> str:
        return "npu graph"

    def get_graph_runner_class(self) -> type:
        from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import (
            NPUGraphRunner,
        )

        return NPUGraphRunner

    def supports_torch_compile(self) -> bool:
        return False


class XPUSRTPlatformStub(SRTPlatform):
    name = "xpu"
    _enum = PlatformEnum.XPU
    device_type = "xpu"
    dispatch_key = "XPU"

    def get_device(self, local_rank: int) -> torch.device:
        return torch.device(f"xpu:{local_rank}")

    def get_distributed_backend(self) -> str:
        return "xccl"


class MUSASRTPlatformStub(SRTPlatform):
    name = "musa"
    _enum = PlatformEnum.MUSA
    device_type = "musa"
    dispatch_key = "PrivateUse1"

    def get_device(self, local_rank: int) -> torch.device:
        return torch.device(f"musa:{local_rank}")

    def get_distributed_backend(self) -> str:
        return "mccl"


class HPUSRTPlatformStub(SRTPlatform):
    name = "hpu"
    _enum = PlatformEnum.HPU
    device_type = "hpu"
    dispatch_key = "HPU"

    def get_device(self, local_rank: int) -> torch.device:
        return torch.device(f"hpu:{local_rank}")

    def get_distributed_backend(self) -> str:
        return "hccl"
