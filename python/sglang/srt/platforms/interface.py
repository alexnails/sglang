"""SRT-specific platform interface.

``SRTPlatform`` extends ``BasePlatform`` with methods needed by the SGLang
serving runtime: graph runners, KV pools, quantisation overrides, compilation
settings, and registry hooks.

Default implementations follow the current CUDA/PyTorch-on-accelerator path
so that ``CUDASRTPlatform`` can inherit most behaviour unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sglang.platforms.interface import BasePlatform

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


class SRTPlatform(BasePlatform):
    """Base class for SRT platform plugins.

    Every method has a default implementation that matches the current CUDA
    behaviour, so new platform classes only need to override what differs.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_backend(self) -> None:
        """One-time initialisation (e.g. import device-specific libs)."""

    def configure_server_args(self, server_args: ServerArgs) -> None:
        """Adjust ``ServerArgs`` defaults for this platform.

        Called during ``ServerArgs.__post_init__`` before memory/backend
        configuration.  The default is a no-op (CUDA defaults are fine).
        """

    # ------------------------------------------------------------------
    # Execution model
    # ------------------------------------------------------------------

    def get_model_runner_class(self) -> type:
        from sglang.srt.model_executor.model_runner import ModelRunner

        return ModelRunner

    def requires_kv_cache(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Graph runners
    # ------------------------------------------------------------------

    def get_graph_runner_class(self) -> type:
        from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner

        return CudaGraphRunner

    @property
    def graph_backend_name(self) -> str:
        """Human-readable name for logging, e.g. 'cuda graph'."""
        return "cuda graph"

    def get_eagle_draft_graph_runner_class(self) -> type | None:
        return None

    def get_eagle_draft_extend_graph_runner_class(self) -> type | None:
        return None

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def get_piecewise_backend_class(self) -> type:
        from sglang.srt.compilation.piecewise_cuda_graph_runner import (
            CUDAPiecewiseBackend,
        )

        return CUDAPiecewiseBackend

    def supports_torch_compile(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Memory pools
    # ------------------------------------------------------------------

    def get_kv_pool_class(self, use_mla: bool) -> type | None:
        """Return a custom KV pool class, or ``None`` for the default."""
        return None

    def get_kv_pool_allocator_class(self) -> type | None:
        return None

    # ------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------

    def get_quant_method_override(self, quant_name: str, layer: Any) -> Any | None:
        """Return a platform-specific quant method, or ``None`` for default."""
        return None

    def post_process_weights(self, layer: Any, weight_name: str) -> None:
        pass

    # ------------------------------------------------------------------
    # Profiler
    # ------------------------------------------------------------------

    def get_profiler_activities(self) -> list:
        import torch.profiler

        return [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]

    # ------------------------------------------------------------------
    # Registry hooks
    # ------------------------------------------------------------------

    def register_attention_backends(self) -> None:
        """Register platform-specific attention backends at startup."""

    def register_lora_backends(self) -> None:
        """Register platform-specific LoRA backends at startup."""
