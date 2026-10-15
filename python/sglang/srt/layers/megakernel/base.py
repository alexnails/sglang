"""Contract between SGLang and a pluggable megakernel provider.

A megakernel replaces a span of a model's forward (a block inside a layer, or
eventually the whole decode step) with one persistent launch. Unlike a
:class:`~sglang.kernels.fused_op.BaseFusedOp` backend it is not a drop-in for a
single op: it is bound to one model family's weights, KV layout and
parallelism, it needs collective state built before graph capture, and it only
serves some steps, so every call site keeps the normal path as the fallback.

A provider declares what it can serve in a :class:`MegaKernelCapability`
(checked generically at startup and per step) and implements the lifecycle
hooks of :class:`MegaKernelProvider`. The model owns the seams: it asks the
registry for a layer's launch at each step and falls back when it gets None.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, FrozenSet, Optional

import msgspec

from sglang.kernels.spec import CapabilityRequirement, PlatformInfo


class MegaKernelScope(str, enum.Enum):
    """The span of the forward one launch replaces; the model defines each seam."""

    # attention output reduce -> pre-FFN mix/norm -> FFN or MoE -> FFN reduce
    LAYER_FFN = "layer.ffn"
    # pre-attention mix/norm -> QKV projections, rope, KV-cache write
    LAYER_ATTN_FRONT = "layer.attn_front"


class StepMode(str, enum.Enum):
    """Forward modes a launch can serve (a subset of ``ForwardMode``)."""

    DECODE = "decode"
    TARGET_VERIFY = "target_verify"


class MegaKernelCapability(msgspec.Struct, frozen=True):
    """What a provider can serve. Empty sets mean unrestricted.

    Startup checks (``devices``, ``architectures``, parallelism, speculative
    algorithm, compute units) reject an opted-in deployment with a reason;
    per-step checks (``step_modes``, ``max_rows``) route a step to the normal
    path instead.
    """

    devices: FrozenSet[CapabilityRequirement] = frozenset()
    architectures: FrozenSet[str] = frozenset()
    tp_sizes: FrozenSet[int] = frozenset()
    allow_attn_dp: bool = False
    allow_moe_ep: bool = False
    # upper-case names as ``get_spec().speculative_algorithm``; "NONE" = off
    speculative_algorithms: FrozenSet[str] = frozenset({"NONE"})
    # a persistent launch keeps every CTA resident; fewer SMs/CUs deadlocks
    min_compute_units: int = 0
    step_modes: FrozenSet[StepMode] = frozenset({StepMode.DECODE})
    max_rows: int = 0  # 0 = unbounded

    def unsupported_platform(self, platform: PlatformInfo) -> Optional[str]:
        if self.devices and not any(c.is_satisfied_by(platform) for c in self.devices):
            return f"does not run on {platform.device.value}"
        return None


class MegaKernelProvider(ABC):
    """One megakernel integration, registered under a name.

    One instance serves the whole process. Hooks run in this order:
    :meth:`unsupported` (startup), :meth:`bind_layer` (model construction),
    :meth:`on_weight_loaded` / :meth:`post_load_weights` (loading),
    :meth:`prepare_before_capture` (warmup, eagerly, before graph capture),
    then the bound launches on eligible steps.
    """

    capability: ClassVar[MegaKernelCapability] = MegaKernelCapability()

    def __init__(self, name: str) -> None:
        self.name = name

    def unsupported(self, hf_config: Any) -> Optional[str]:
        """Deployment checks the capability cannot express, or None."""
        return None

    @abstractmethod
    def bind_layer(self, layer: Any, scope: MegaKernelScope) -> Optional[Callable]:
        """Return the launch serving ``scope`` of ``layer``, or None to keep it on the normal path."""

    def on_weight_loaded(self, layers: Any, name: str, tensor: Any) -> None:
        """See one checkpoint tensor before quantization post-processing rewrites it."""

    def post_load_weights(self, layers: Any) -> None:
        """Repack or validate the bound layers' weights after loading."""

    def prepare_before_capture(self, model_runner: Any) -> None:
        """Build collective state, compile every width, allocate scratch (no graph capture is active)."""

    def eligible(self, forward_batch: Any, rows: int) -> bool:
        """Per-step gate beyond ``capability.step_modes`` / ``max_rows``."""
        return True
