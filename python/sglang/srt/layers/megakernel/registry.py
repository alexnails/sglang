"""Process-wide megakernel registry and the seams models call.

``--megakernel-backend <name>`` selects one registered provider. Models bind a
:class:`LayerMegaKernel` per layer at construction and ask it for a launch at
each step; ``None`` means "take the normal path". Built-in providers are
registered by import path so nothing is imported until one is selected;
out-of-tree providers call :func:`register_megakernel` from their plugin.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Union

import msgspec

from sglang.kernels.spec import PlatformInfo
from sglang.srt.layers.megakernel.base import (
    MegaKernelCapability,
    MegaKernelProvider,
    MegaKernelScope,
    StepMode,
)
from sglang.srt.runtime_context import get_exec, get_parallel, get_spec

logger = logging.getLogger(__name__)

ProviderFactory = Callable[[str], MegaKernelProvider]

# name -> factory, or "module:attr" resolved on first selection
_DSV41_MONO = (
    "sglang.srt.models.deepseek_common.amd.dsv41_mono_decode:DSV41MonoProvider"
)
_FACTORIES: Dict[str, Union[ProviderFactory, str]] = {
    "dsv41_mono": _DSV41_MONO,
    "dsv41_mono_k1": _DSV41_MONO,
}
# backend name -> validated provider; one entry per process
_ACTIVE: Dict[str, MegaKernelProvider] = {}


def register_megakernel(name: str, factory: Union[ProviderFactory, str]) -> None:
    """Register a provider factory (or its ``"module:attr"`` path) under ``name``."""
    if name in _FACTORIES and _FACTORIES[name] != factory:
        raise ValueError(f"Megakernel {name!r} is already registered")
    _FACTORIES[name] = factory


def registered_megakernels() -> list:
    return sorted(_FACTORIES)


def get_megakernel(hf_config: Any) -> Optional[MegaKernelProvider]:
    """The selected provider, built and validated once; None when none is selected.

    Raises ``ValueError`` with the reason when the selected provider cannot
    serve this deployment, so an opt-in never degrades silently.
    """
    name = get_exec().kernel.megakernel_backend
    if name is None:
        return None
    provider = _ACTIVE.get(name)
    if provider is None:
        provider = _build(name)
        why = unsupported_deployment(provider.capability, hf_config=hf_config)
        if why is None:
            why = provider.unsupported(hf_config)
        if why is not None:
            raise ValueError(f"--megakernel-backend {name} {why}.")
        _ACTIVE.clear()
        _ACTIVE[name] = provider
        logger.info("Megakernel backend %r enabled", name)
    return provider


def _build(name: str) -> MegaKernelProvider:
    factory = _FACTORIES.get(name)
    if factory is None:
        raise ValueError(
            f"Unknown --megakernel-backend {name!r}; registered: {registered_megakernels()}"
        )
    if isinstance(factory, str):
        module_path, _, attr = factory.partition(":")
        factory = getattr(importlib.import_module(module_path), attr)
    return factory(name)


def unsupported_deployment(
    capability: MegaKernelCapability, *, hf_config: Any
) -> Optional[str]:
    """The generic startup checks of ``capability``: why it cannot serve, or None."""
    why = capability.unsupported_platform(PlatformInfo.detect())
    if why is not None:
        return why
    archs = set(hf_config.architectures or ())
    if capability.architectures and not archs & capability.architectures:
        return f"supports {sorted(capability.architectures)}, not {sorted(archs)}"
    p = get_parallel()
    if capability.tp_sizes and p.tp_size not in capability.tp_sizes:
        return f"needs tensor parallel size in {sorted(capability.tp_sizes)}"
    if not capability.allow_attn_dp and p.attn_tp_size != p.tp_size:
        return "does not support attention data parallelism"
    if not capability.allow_moe_ep and p.moe_ep_size != 1:
        return "does not support expert parallelism"
    algo = (get_spec().speculative_algorithm or "NONE").upper()
    if algo not in capability.speculative_algorithms:
        return f"supports speculative algorithms {sorted(capability.speculative_algorithms)}, not {algo}"
    if capability.min_compute_units:
        import torch

        cus = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        if cus < capability.min_compute_units:
            return (
                f"needs {capability.min_compute_units} compute units, the GPU has {cus}"
            )
    return None


_STEP_MODE_PREDICATES = {
    StepMode.DECODE: "is_decode",
    StepMode.TARGET_VERIFY: "is_target_verify",
}


def step_eligible(
    capability: MegaKernelCapability, *, forward_mode: Any, rows: int
) -> bool:
    """Whether a step of ``rows`` rows in ``forward_mode`` is inside ``capability``."""
    if rows < 1 or (capability.max_rows and rows > capability.max_rows):
        return False
    return any(
        getattr(forward_mode, _STEP_MODE_PREDICATES[m])() for m in capability.step_modes
    )


class LayerMegaKernel(msgspec.Struct, frozen=True):
    """One layer's bound launches; a layer with none takes the normal path everywhere."""

    provider: Optional[MegaKernelProvider] = None
    launches: Mapping[MegaKernelScope, Callable] = {}

    def has(self, scope: MegaKernelScope) -> bool:
        return scope in self.launches

    def launch(
        self, scope: MegaKernelScope, *, forward_batch: Any, rows: int
    ) -> Optional[Callable]:
        """The launch serving ``scope`` on this step, or None for the normal path."""
        launch = self.launches.get(scope)
        if launch is None:
            return None
        if not step_eligible(
            self.provider.capability, forward_mode=forward_batch.forward_mode, rows=rows
        ):
            return None
        if not self.provider.eligible(forward_batch, rows):
            return None
        return launch


NO_MEGAKERNEL = LayerMegaKernel()


def bind_layer(
    layer: Any, *, hf_config: Any, scopes: Iterable[MegaKernelScope]
) -> LayerMegaKernel:
    """Ask the selected provider for ``layer``'s launches at the seams the model offers."""
    provider = get_megakernel(hf_config)
    if provider is None:
        return NO_MEGAKERNEL
    launches = {}
    for scope in scopes:
        launch = provider.bind_layer(layer, scope)
        if launch is not None:
            launches[scope] = launch
    if not launches:
        return NO_MEGAKERNEL
    return LayerMegaKernel(provider=provider, launches=launches)


def _active() -> Optional[MegaKernelProvider]:
    return next(iter(_ACTIVE.values()), None)


def on_weight_loaded(*, layers: Any, name: str, tensor: Any) -> None:
    """Forward one checkpoint tensor to the active provider (models call this in load_weights)."""
    provider = _active()
    if provider is not None:
        provider.on_weight_loaded(layers, name, tensor)


def post_load_weights(*, layers: Any) -> None:
    provider = _active()
    if provider is not None:
        provider.post_load_weights(layers)


def prepare_before_capture(*, device: Any) -> None:
    """Run the active provider's eager setup; called once from runner warmup."""
    provider = _active()
    if provider is not None:
        provider.prepare_before_capture(device)
