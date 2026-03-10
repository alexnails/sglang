# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Apple MPS (Metal) platform implementation for SRT."""

from __future__ import annotations

import logging
import platform as _platform
from typing import TYPE_CHECKING

import psutil
import torch

from sglang.platforms.interface import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class MPSSRTPlatform(SRTPlatform):
    name = "mps"
    _enum = PlatformEnum.MPS
    device_type = "mps"
    dispatch_key = "MPS"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def configure_server_args(self, server_args: ServerArgs) -> None:
        server_args.attention_backend = "torch_native"
        server_args.sampling_backend = "pytorch"
        server_args.disable_cuda_graph = True
        server_args.disable_overlap_schedule = True

        if server_args.mem_fraction_static is None:
            server_args.mem_fraction_static = 0.50

        if server_args.tp_size > 1:
            logger.warning(
                "MPS supports only tp_size=1 (single device). Overriding tp_size to 1."
            )
            server_args.tp_size = 1
        if server_args.dp_size > 1:
            logger.warning(
                "MPS supports only dp_size=1 (single device). Overriding dp_size to 1."
            )
            server_args.dp_size = 1

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def get_device(self, local_rank: int) -> torch.device:
        return torch.device("mps")

    def get_device_name(self, device_id: int = 0) -> str:
        processor = _platform.processor() or _platform.machine()
        return f"Apple {processor}"

    def get_device_uuid(self, device_id: int = 0) -> str:
        return _platform.machine()

    def empty_cache(self) -> None:
        torch.mps.empty_cache()

    def synchronize(self) -> None:
        torch.mps.synchronize()

    # ------------------------------------------------------------------
    # Memory (unified memory)
    # ------------------------------------------------------------------

    def get_device_total_memory(self, device_id: int = 0) -> int:
        return psutil.virtual_memory().total

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        recommended = torch.mps.recommended_max_memory()
        allocated = torch.mps.current_allocated_memory()
        free = max(recommended - allocated, 0)
        return (free, recommended)

    def get_current_memory_usage(self, device: torch.device | None = None) -> float:
        return float(torch.mps.current_allocated_memory())

    # ------------------------------------------------------------------
    # Distributed
    # ------------------------------------------------------------------

    def get_distributed_backend(self) -> str:
        return "gloo"

    # ------------------------------------------------------------------
    # Graph runners (torch.compile-based, reuses CPUGraphRunner)
    # ------------------------------------------------------------------

    def get_graph_runner_class(self) -> type:
        from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner

        return CPUGraphRunner

    @property
    def graph_backend_name(self) -> str:
        return "mps compile graph"

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
    # Quantization
    # ------------------------------------------------------------------

    def verify_quantization(self, quant: str) -> None:
        if quant is not None and quant.lower() not in ("none", ""):
            raise ValueError(
                f"MPS does not support {quant} quantization. "
                "Only unquantized models are supported."
            )

    # ------------------------------------------------------------------
    # Dtype guard (Metal does not support float64)
    # ------------------------------------------------------------------

    def guard_dtype(self, dtype: torch.dtype) -> torch.dtype:
        """Convert unsupported dtypes to safe alternatives for MPS."""
        if dtype == torch.float64:
            logger.warning("MPS does not support float64; converting to float32.")
            return torch.float32
        return dtype

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        if seed is not None:
            import random

            import numpy as np

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
