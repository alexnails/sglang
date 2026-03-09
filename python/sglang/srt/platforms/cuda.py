"""CUDA platform implementation for SRT."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.platforms.interface import DeviceCapability, PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform

if TYPE_CHECKING:
    pass


class CUDASRTPlatform(SRTPlatform):
    name = "cuda"
    _enum = PlatformEnum.CUDA
    device_type = "cuda"
    dispatch_key = "CUDA"

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def get_device(self, local_rank: int) -> torch.device:
        from sglang.srt import environ as envs

        device_id = (
            0 if envs.SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS.get() else local_rank
        )
        return torch.device(f"cuda:{device_id}")

    def get_device_name(self, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)

    def get_device_uuid(self, device_id: int = 0) -> str:
        return str(torch.cuda.get_device_properties(device_id).uuid)

    def get_device_capability(self, device_id: int = 0) -> DeviceCapability | None:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def get_device_total_memory(self, device_id: int = 0) -> int:
        return torch.cuda.get_device_properties(device_id).total_memory

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        return torch.cuda.mem_get_info(device_id)

    def get_current_memory_usage(self, device: torch.device | None = None) -> float:
        torch.cuda.reset_peak_memory_stats()
        return torch.cuda.max_memory_allocated(device)

    # ------------------------------------------------------------------
    # Distributed
    # ------------------------------------------------------------------

    def get_distributed_backend(self) -> str:
        return "nccl"
