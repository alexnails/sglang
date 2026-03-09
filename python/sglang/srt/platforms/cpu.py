"""CPU platform implementation for SRT."""

from __future__ import annotations

from typing import TYPE_CHECKING

import psutil
import torch

from sglang.platforms.interface import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


class CPUSRTPlatform(SRTPlatform):
    name = "cpu"
    _enum = PlatformEnum.CPU
    device_type = "cpu"
    dispatch_key = "CPU"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def configure_server_args(self, server_args: ServerArgs) -> None:
        if server_args.attention_backend is None:
            from sglang.srt.utils.common import cpu_has_amx_support

            if cpu_has_amx_support():
                server_args.attention_backend = "intel_amx"
            else:
                server_args.attention_backend = "torch_native"
        server_args.sampling_backend = "pytorch"

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def get_device(self, local_rank: int) -> torch.device:
        return torch.device("cpu")

    def get_device_name(self, device_id: int = 0) -> str:
        import platform as _platform

        return _platform.processor() or _platform.machine()

    def get_device_uuid(self, device_id: int = 0) -> str:
        import platform as _platform

        return _platform.machine()

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def get_device_total_memory(self, device_id: int = 0) -> int:
        return psutil.virtual_memory().total

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        mem = psutil.virtual_memory()
        n_numa = self._numa_node_count()
        free = mem.available // max(n_numa, 1)
        total = mem.total // max(n_numa, 1)
        return (free, total)

    def get_current_memory_usage(self, device: torch.device | None = None) -> float:
        return 0.0

    # ------------------------------------------------------------------
    # Distributed
    # ------------------------------------------------------------------

    def get_distributed_backend(self) -> str:
        return "gloo"

    # ------------------------------------------------------------------
    # Graph runners
    # ------------------------------------------------------------------

    def get_graph_runner_class(self) -> type:
        from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner

        return CPUGraphRunner

    @property
    def graph_backend_name(self) -> str:
        return "cpu graph"

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def supports_torch_compile(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Profiler
    # ------------------------------------------------------------------

    def get_profiler_activities(self) -> list:
        import torch.profiler

        return [torch.profiler.ProfilerActivity.CPU]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _numa_node_count() -> int:
        try:
            from sglang.srt.utils.common import get_cpu_ids_by_node

            return len(get_cpu_ids_by_node())
        except Exception:
            return 1
