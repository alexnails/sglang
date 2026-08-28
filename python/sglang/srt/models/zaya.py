# SPDX-License-Identifier: Apache-2.0
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
"""Inference-only Zyphra ZAYA1 (CCA attention + MoE) model implementation.

Architecture summary (see docs/supported_models/text_generation/zaya_design.md
for the full design notes):

- Even-indexed layers run :class:`ZayaAttention`, which feeds hidden states to
  the :class:`CCA` (Compressed Convolutional Attention) projection. CCA emits
  q/k/v via two small (``kernel_size=2``) depthwise + grouped 1D convolutions
  over the time axis plus a learnable per-K-head temperature. The conv needs a
  two-token left padding that is sourced from a per-request state cache owned
  by the CCA module itself. The q/k/v then go through partial rotary embedding
  (``partial_rotary_factor=0.5``) and SGLang's :class:`RadixAttention` for the
  softmax MHA. The implementation only uses ``torch`` / ``torch.nn`` ops, so the
  same code runs on NVIDIA and AMD GPUs.
- Odd-indexed layers run :class:`ZayaBlock`, an MoE mixer built around SGLang's
  :class:`FusedMoE`. Expert routing uses a 3-layer MLP with EDA (depth-wise
  averaging across MoE layers) and MOD (mixture-of-depths skip expert).
- Per-layer :class:`ResidualScaling` keeps the residual stream in fp32 with
  affine scale/bias both on the residual and on the post-mixer hidden states.
- Per-request CCA state (``conv_state`` + the ``val_proj2`` lag) lives in
  SGLang's centralized ``MambaPool`` inside ``HybridReqToTokenPool``. The
  state plumbing (slot indices, prefix mask, cuda-graph buffers) is owned by
  ``ShortConvAttnBackend`` and reached via
  ``get_attn_backend().conv_state_metadata()``, so the model holds no pool
  access; CCA runs its own conv (:func:`cca_extend` / :func:`cca_decode`)
  against the returned handle.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Callable, List, NamedTuple, Optional, Tuple

import msgspec
import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.zaya import ZayaConfig
from sglang.srt.distributed import (
    get_pp_group,
    get_tp_group,
    moe_expert_parallel_all_reduce,
    moe_tensor_model_parallel_all_reduce,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    attn_tp_all_reduce,
    dp_gather_partial_out,
    dp_gather_replicate,
    dp_scatter,
    get_attention_dp_rank,
    get_dp_global_num_tokens,
    get_global_dp_buffer,
    get_local_dp_buffer,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    should_skip_post_experts_all_reduce,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import RotaryEmbedding, get_rope
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.utils.cp_utils import is_prefill_context_parallel_enabled
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, MHATokenToKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_exec, get_parallel, get_server_args
from sglang.srt.utils import add_prefix, make_layers, set_weight_attrs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Residual scaling
# ---------------------------------------------------------------------------


class ResidualScaling(nn.Module):
    """Affine fp32 scaling applied to the residual / hidden_states streams.

    Layer 0 has no incoming residual stream, so its checkpoint omits
    ``residual_scale`` / ``residual_bias`` and ``has_residual`` stays False.
    """

    def __init__(self, config: ZayaConfig, layer_n: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.has_residual = layer_n != 0
        self.hidden_states_scale = nn.Parameter(torch.ones(self.hidden_size))
        self.hidden_states_bias = nn.Parameter(torch.zeros(self.hidden_size))
        if self.has_residual:
            self.residual_scale = nn.Parameter(torch.ones(self.hidden_size))
            self.residual_bias = nn.Parameter(torch.zeros(self.hidden_size))
        # Folded constants, recomputed after every weight load by
        # ``fold_scales``. Explicitly fp32 (not the ambient default dtype, which
        # model loading sets to the checkpoint dtype): the original formulation
        # cast scale/bias up to fp32 before the arithmetic, and these buffers
        # are what preserve that accumulation precision in the fused form.
        # Non-persistent -- derived from parameters, never part of a checkpoint.
        for name in (
            ("hidden_states", "residual") if self.has_residual else ("hidden_states",)
        ):
            self.register_buffer(
                f"{name}_bias_scaled",
                torch.zeros(self.hidden_size, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                f"{name}_scale_f32",
                torch.ones(self.hidden_size, dtype=torch.float32),
                persistent=False,
            )
        # Gate for the fused residual chain: the folded buffers above are only
        # valid once fold_scales has run against loaded weights.
        self._scales_folded = False

    @torch.no_grad()
    def fold_scales(self) -> None:
        """Recompute the folded fp32 constants from the loaded parameters.

        Called after weight loading (and after any weight reload) via
        ``ZayaForCausalLM.fold_decode_constants``.
        """
        self.hidden_states_scale_f32.copy_(self.hidden_states_scale.float())
        self.hidden_states_bias_scaled.copy_(
            self.hidden_states_bias.float() * self.hidden_states_scale_f32
        )
        if self.has_residual:
            self.residual_scale_f32.copy_(self.residual_scale.float())
            self.residual_bias_scaled.copy_(
                self.residual_bias.float() * self.residual_scale_f32
            )
        self._scales_folded = True

    def forward(
        self,
        residual: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        # ``(x + b) * s == x * s + (b * s)``. ``b`` and ``s`` are load-time
        # constants, so the ``b * s`` product is folded once (``fold_scales``)
        # and each stream costs a single fused multiply-add instead of a
        # cast + add + mul chain. ZAYA1 runs this twice per layer over 120
        # layers, so the saved launches are the single largest elementwise
        # contributor in a decode step (measured: 274-624 us/step on MI350X,
        # and bit-comparable at fp32 -- rel err ~1e-7).
        hidden_states = torch.addcmul(
            self.hidden_states_bias_scaled, hidden_states, self.hidden_states_scale_f32
        )

        if self.has_residual and residual is not None:
            residual = torch.addcmul(
                self.residual_bias_scaled, residual, self.residual_scale_f32
            )

        return residual, hidden_states


def _apply_norm_with_fp32_residual(
    norm: nn.Module,
    residual: torch.Tensor,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Normalize ``residual`` (typically fp32) and cast back to ``target_dtype``.

    The fp32 residual stream is preserved by the caller (the residual tensor
    is kept around for the next accumulation), so the norm itself can run at
    ``target_dtype`` -- this lets us hit the fused sgl_kernel rmsnorm path
    instead of the eager ``forward_native`` fallback (5+ kernel launches per
    call, ×120 norms per step).
    """
    return norm(residual.to(target_dtype))


# ---------------------------------------------------------------------------
# CCA conv-state kernels (v1 torch)
#
# ZAYA1-specific conv step: the CCA conv is a causal two-stage conv over
# ``qk = [W_q hs || W_k hs]`` plus a one-token lag for the ``val_proj2`` value
# stream. The per-request conv state lives in the centralized MambaPool; the
# backend (ShortConvAttnBackend) hands out the slot indices + prefix flags and
# CCA runs these functions against them. ``conv_qk`` is the module's two-stage
# conv; both functions mutate ``conv_state`` / ``lag_state`` in place and return
# ``(qk_out, lag_prev)`` -- the conv output ``[T, in_out_ch]`` and the
# right-shifted lag stream ``[T, lag_dim]``.
#
# The lag stream carries the *projected* value ``W_v2 . hs`` rather than the raw
# hidden state. ``val_proj2`` is linear and bias-free, so ``W_v2 . shift(hs) ==
# shift(W_v2 . hs)``: projecting before the state write is the same function and
# the cached quantity shrinks from ``hidden_size`` to ``latent_k_dim / 2`` (4096
# -> 128 on ZAYA1-74B). See ``CCA.__init__`` for the gate and
# ``ZayaConfig.cca_cache_projected_v2`` for the pool-shape side of it.
#
# ``lag_now`` / ``lag_state`` are ``None`` on a rank whose K heads all come from
# ``val_proj1``: it never reads the lag, so it neither computes nor stores it.
# ---------------------------------------------------------------------------


def _shift_lag_into(
    lag_prev: torch.Tensor,
    lag_now: torch.Tensor,
    start: int,
    end: int,
    prefix: bool,
    lag_state: torch.Tensor,
    slot: int,
) -> None:
    """Right-shift one request's lag segment and park its last row in the pool.

    ``lag_prev[t] = lag_now[t - 1]``, with the row before the segment taken from
    the request's cached slot when it resumes a prefix and from zero when it does
    not (a fresh request's first ``val_proj2`` input is defined to be zero, which
    is exactly what MambaPool's zeroed slot holds).

    Two slice copies rather than ``cat`` + one copy: the same launch count with
    half the bytes, since the concatenate's temporary is never materialized.
    """
    if end <= start:
        return
    if prefix:
        lag_prev[start : start + 1] = lag_state[slot].squeeze(-1).to(lag_now.dtype)
    else:
        lag_prev[start : start + 1].zero_()
    if end - start > 1:
        lag_prev[start + 1 : end] = lag_now[start : end - 1]
    lag_state[slot] = lag_now[end - 1].unsqueeze(-1).to(lag_state.dtype)


def cca_extend(
    qk: torch.Tensor,
    lag_now: Optional[torch.Tensor],
    conv_fn: Callable[[torch.Tensor], torch.Tensor],
    conv_state: torch.Tensor,
    lag_state: Optional[torch.Tensor],
    slot_ids: List[int],
    has_prefix: List[bool],
    extend_seq_lens_cpu: List[int],
    total_padding: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Prefill / extend conv-state step (v1, pure torch).

    Walks each request in the batch, applies ``conv_fn`` with the request's own
    initial state (zeros on a fresh first chunk, the cached ``conv_state`` slot
    otherwise), writes the updated ``conv_state`` / ``lag_state`` back, and
    returns ``(qk_out, lag_prev)`` in the original token layout.

    ``lag_now`` is this chunk's ``val_proj2`` value at each token and ``lag_prev``
    is that stream shifted right by one, sourced across a chunk boundary from the
    cached slot. Both are ``None`` when the rank does not use the lag.

    ``conv_fn`` maps ``[N, C, S + total_padding] -> [N, C, S]``; callers pass
    :meth:`CCA._conv_qk_run` so the folded single grouped conv is used when it is
    available.

    ``slot_ids`` is the host mirror of the per-request MambaPool slot indices and
    ``has_prefix[i]`` is ``True`` when request ``i`` resumes a cached prefix.

    The launch count here grows with the batch and the loop trip count comes from
    ``extend_seq_lens_cpu``, which is also what blocks the prefill CUDA graph;
    :func:`cca_conv1d_fn <sglang.kernels.ops.attention.cca_conv1d.cca_conv1d_fn>`
    is the device-driven replacement.
    """
    dtype = qk.dtype
    if total_padding is None:
        total_padding = conv_state.shape[-1]
    in_out_ch = qk.shape[-1]

    qk_out = torch.empty_like(qk)
    lag_prev = None if lag_now is None else torch.empty_like(lag_now)

    # Fresh-prefill fast path: when no request has a cached prefix the per-request
    # convs can be coalesced into a single packed convolution. Each request's
    # segment is laid out as ``[total_padding zeros, S_i tokens]``.
    all_fresh = bool(extend_seq_lens_cpu) and not any(has_prefix)

    if all_fresh:
        seq_lens = [int(s) for s in extend_seq_lens_cpu]
        pad = total_padding
        offsets_in = [0]
        for s in seq_lens:
            offsets_in.append(offsets_in[-1] + s + pad)
        packed = qk.new_zeros((1, in_out_ch, offsets_in[-1]))
        start = 0
        for i, s in enumerate(seq_lens):
            end = start + s
            packed[0, :, offsets_in[i] + pad : offsets_in[i + 1]] = qk[
                start:end
            ].transpose(0, 1)
            start = end

        packed_out = conv_fn(packed)  # [1, C, offsets_in[-1] - pad]

        start = 0
        for i, s in enumerate(seq_lens):
            end = start + s
            a_i = offsets_in[i]
            qk_out[start:end] = packed_out[0, :, a_i : a_i + s].transpose(0, 1)
            new_state = packed[0, :, a_i + s : a_i + s + pad]
            conv_state[slot_ids[i]] = new_state.to(conv_state.dtype)

            if lag_now is not None:
                # Fresh request: the first token's val_proj2 value is 0 by
                # definition, matching the zeroed slot MambaPool hands out.
                _shift_lag_into(
                    lag_prev, lag_now, start, end, False, lag_state, slot_ids[i]
                )
            start = end
    else:
        start = 0
        for i, seq_len in enumerate(extend_seq_lens_cpu):
            end = start + int(seq_len)
            slot = slot_ids[i]
            prefix = bool(has_prefix[i])

            qk_cur = qk[start:end].transpose(0, 1).unsqueeze(0)  # [1, C, S_cur]
            if prefix:
                left_pad = conv_state[slot].unsqueeze(0).to(dtype)
            else:
                left_pad = qk_cur.new_zeros((1, in_out_ch, total_padding))
            padded = torch.cat([left_pad, qk_cur], dim=-1)

            out = conv_fn(padded)  # [1, C, S_cur]
            qk_out[start:end] = out.squeeze(0).transpose(0, 1)

            new_state = padded[..., -total_padding:]
            conv_state[slot] = new_state.squeeze(0).to(conv_state.dtype)

            if lag_now is not None:
                # Chunked prefill: the boundary row carries the PROJECTED value
                # the previous chunk left behind, so a resumed prefix reads the
                # same v2 it would have seen in a single-chunk run. Writing the
                # raw hidden state here instead would silently degrade the
                # resumed tokens with no shape or dtype error to catch it.
                _shift_lag_into(lag_prev, lag_now, start, end, prefix, lag_state, slot)
            start = end

    return qk_out, lag_prev


def cca_decode(
    qk: torch.Tensor,
    lag_now: Optional[torch.Tensor],
    conv_qk: nn.Module,
    conv_state: torch.Tensor,
    lag_state: Optional[torch.Tensor],
    mamba_indices: torch.Tensor,
    total_padding: Optional[int] = None,
    decode_conv_weight: Optional[torch.Tensor] = None,
    decode_conv_bias: Optional[torch.Tensor] = None,
    decode_conv_groups: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Single-token decode conv-state step (v1, pure torch).

    Gathers each request's cached ``conv_state`` / ``lag_state`` via
    ``index_select``, applies the conv over the ``[T, C, total_padding + 1]``
    window, and scatters the updated state back with ``index_copy_``. All ops are
    on-device (``mamba_indices`` is a device ``long`` tensor), so this stays
    CUDA-graph capturable. Returns ``(qk_out, lag_prev)`` where ``lag_prev`` is
    the previous step's ``val_proj2`` value (``None`` when unused by this rank).

    When ``decode_conv_weight`` / ``_bias`` / ``_groups`` are supplied (see
    :meth:`CCA.fold_decode_conv`) the two conv stages are evaluated as a single
    grouped matmul; otherwise ``conv_qk`` is run as-is. The Triton swap is
    :func:`cca_conv1d_update`.
    """
    dtype = qk.dtype
    if total_padding is None:
        total_padding = conv_state.shape[-1]

    from sglang.kernels.ops.attention import cca_conv1d_update as _fused_update
    from sglang.kernels.ops.attention import cca_state_step as _state_step

    if envs.SGLANG_OPT_ZAYA_FUSED_CCA_DECODE.get() and _fused_update.covered(
        qk,
        lag_now,
        decode_conv_weight,
        decode_conv_bias,
        conv_state,
        lag_state,
        mamba_indices,
        total_padding,
        decode_conv_groups or 0,
    ):
        # One kernel for the window, the history shift and the grouped matmul.
        return _fused_update.cca_conv1d_update(
            qk,
            lag_now,
            decode_conv_weight,
            decode_conv_bias,
            conv_state,
            lag_state,
            mamba_indices,
            total_padding,
            decode_conv_groups,
        )

    if _state_step.covered(
        qk, lag_now, conv_state, lag_state, mamba_indices, total_padding
    ):
        # One kernel for the gathers, the concat and the scatters. When the
        # folded weight is available it carries the conv bias in a trailing
        # column, so ask for the matching constant-1.0 tap and let the bias land
        # in the matmul accumulator instead of a separate add (see
        # ``fold_decode_conv``).
        padded, lag_prev = _state_step.cca_state_step(
            qk,
            lag_now,
            conv_state,
            lag_state,
            mamba_indices,
            total_padding,
            ones_column=decode_conv_weight is not None,
        )
        qk_out = _cca_decode_conv(
            padded,
            conv_qk,
            decode_conv_weight,
            decode_conv_bias,
            decode_conv_groups,
        )
        return qk_out, lag_prev

    left_pad = conv_state.index_select(0, mamba_indices).to(dtype)
    cur = qk.unsqueeze(-1)  # [T, C, 1]
    padded = torch.cat([left_pad, cur], dim=-1)  # [T, C, total_padding + 1]
    qk_out = _cca_decode_conv(
        padded, conv_qk, decode_conv_weight, decode_conv_bias, decode_conv_groups
    )

    new_state = padded[..., -total_padding:]
    conv_state.index_copy_(0, mamba_indices, new_state.to(conv_state.dtype))

    if lag_now is None:
        return qk_out, None

    # Read the previous step's val_proj2 value BEFORE overwriting the slot with
    # this token's.
    lag_prev = lag_state.index_select(0, mamba_indices).squeeze(-1).to(lag_now.dtype)
    lag_state.index_copy_(0, mamba_indices, lag_now.unsqueeze(-1).to(lag_state.dtype))
    return qk_out, lag_prev


# Fused kernel seam (TODO) -- perf swap for the v1 torch paths above. These
# mirror the ``causal_conv1d_fn`` / ``causal_conv1d_update`` contract but for
# CCA's two-stage *grouped* conv (conv_qk[0] depthwise + conv_qk[1] grouped
# per-head), which the stock depthwise ``causal_conv1d`` cannot express. Once
# implemented they replace the per-request loop in ``cca_extend`` and the
# separate gather/conv/scatter launches in ``cca_decode`` with a single
# index-driven kernel. Same ``(qk_out, lag_prev)`` return contract.


def cca_conv1d_fn(*args, **kwargs):
    """Fused varlen prefill conv-with-state; see the kernel module for the contract."""
    from sglang.kernels.ops.attention.cca_conv1d import cca_conv1d_fn as _fused

    return _fused(*args, **kwargs)


def cca_conv1d_update(*args, **kwargs):
    """Fused decode conv-with-state (window + grouped matmul in one kernel)."""
    from sglang.kernels.ops.attention.cca_conv1d_update import (
        cca_conv1d_update as _fused,
    )

    return _fused(*args, **kwargs)


def _cca_decode_conv(
    padded: torch.Tensor,
    conv_qk: nn.Module,
    decode_conv_weight: Optional[torch.Tensor],
    decode_conv_bias: Optional[torch.Tensor],
    decode_conv_groups: Optional[int],
) -> torch.Tensor:
    """Apply the decode conv to a ``[T, C, taps]`` window, returning ``[T, C]``.

    Prefers the load-time-folded single grouped matmul (see
    :meth:`CCA.fold_decode_conv`) and falls back to running the real two-stage
    ``conv_qk``, which is what an unfolded module (e.g. a CPU unit test) gets.

    The folded weight spans ``taps + 1`` inputs per channel: the extra one is the
    bias, activated by a constant-1.0 column in the window. ``cca_state_step``
    writes that column for free; a window that arrives without it (the unfused
    gather/concat fallback) gets it appended here, which is one extra op on a
    path that is already off the fast one.
    """
    if decode_conv_weight is not None:
        num_tokens, num_channels = padded.shape[0], padded.shape[1]
        taps_ext = decode_conv_weight.shape[-1] // (num_channels // decode_conv_groups)
        if padded.shape[-1] != taps_ext:
            padded = F.pad(padded, (0, taps_ext - padded.shape[-1]), value=1.0)
        # [T, C, taps_ext] -> [T, G, Cg*taps_ext] (the trailing (Cg, taps_ext)
        # dims flatten in place, matching how fold_decode_conv laid out the
        # weight) -> one grouped matmul -> [T, C]. The bias is already inside the
        # accumulator, so there is nothing to add afterwards.
        grouped = padded.reshape(num_tokens, decode_conv_groups, -1)
        return torch.einsum("tgk,gok->tgo", grouped, decode_conv_weight).reshape(
            num_tokens, -1
        )
    return conv_qk(padded).squeeze(-1)


class CCARope(NamedTuple):
    """The layer's rotary inputs, handed to ``CCA.forward`` for in-kernel RoPE.

    ZAYA1-74B builds TWO rotary caches -- a full-attention one keyed on
    ``rope_theta`` (1e7) and a sliding-window one keyed on ``swa_rotary_base``
    (10000) -- and picks per layer, so the fused path must read *this* layer's
    buffer rather than a model-wide one. ``get_rope`` keys its process-wide cache
    on ``base`` among other things, so the two are distinct modules with distinct
    buffers (pinned by ``TestZayaSlidingWindowAttention``'s
    ``test_attention_selects_window_and_rope_base_per_layer``).

    CUDA-graph safety: ``cos_sin_cache`` is a plain ``register_buffer`` allocated
    once in ``RotaryEmbedding.__init__`` at ``max_position_embeddings`` rows. The
    only thing that can move it is ``_ensure_cos_sin_cache_length``, whose sole
    caller is ``reserve_rope_cache_for_long_sequences``, which ``ModelRunner``
    runs *before* ``init_cuda_graphs`` -- so no reallocation can happen after
    capture. ``forward_cuda``'s ``self.cos_sin_cache.to(device, dtype)`` is a
    no-op on ROCm (the cache is already stored in the model dtype) and returns
    the same tensor, so the baseline rotary launch relies on exactly the same
    guarantee. The pointer is read fresh from the module on every forward, so an
    eager run picks up any move regardless.
    """

    positions: torch.Tensor
    cos_sin_cache: torch.Tensor
    rotary_dim: int
    is_neox_style: bool

    @classmethod
    def of(cls, rotary_emb: nn.Module, positions: torch.Tensor) -> Optional[CCARope]:
        """Snapshot a ``RotaryEmbedding``, or ``None`` if it is not the plain kind.

        Exact-type, not isinstance: every subclass changes something the fused
        kernel does not model -- a per-scaling-factor cache offset
        (``LinearScalingRotaryEmbedding``), a second cache
        (``DualChunkRotaryEmbedding``), a 2-D position layout
        (``MRotaryEmbedding``). ZAYA1 passes no ``rope_scaling``, so ``get_rope``
        always hands it the base class; anything else falls back to a separate
        rotary launch rather than to a guess.

        ``_force_native`` (deterministic RL replay) also declines: that mode
        deliberately pins the eager torch rotary, and this kernel is not
        bit-identical to it.
        """
        if type(rotary_emb) is not RotaryEmbedding:
            return None
        if getattr(rotary_emb, "_force_native", False):
            return None
        cache = getattr(rotary_emb, "cos_sin_cache", None)
        if cache is None or hasattr(rotary_emb, "sin_cos_cache"):
            return None
        if positions is None or positions.ndim != 1:
            return None
        return cls(
            positions=positions,
            cos_sin_cache=cache,
            rotary_dim=int(rotary_emb.rotary_dim),
            is_neox_style=bool(rotary_emb.is_neox_style),
        )


class CCAKVStore(NamedTuple):
    """The layer's paged KV write target, handed to ``CCA.forward``.

    Lets the fused head-mix kernel scatter the post-rope ``k`` and the ``v`` it
    already has straight into the pool, so ``RadixAttention`` runs with
    ``save_kv_cache=False`` and the per-layer ``set_kv_buffer`` launch (60 per
    decode step) disappears.

    ``k_cache`` / ``v_cache`` are the plain 3-D NHD ``[size + page_size, H, D]``
    pool buffers, indexed flat by slot -- the same rows ``_set_kv_buffer_impl``
    writes, so no page arithmetic is involved and the write is correct for any
    page size. ``full_to_swa`` is the hybrid pool's slot indirection, present
    only on a sliding-window layer.

    CUDA-graph safety: all four tensors are allocated once and mutated in place
    (the pool buffers at startup, ``full_to_swa_index_mapping`` at allocator
    construction, ``out_cache_loc`` as the graph's own stable input buffer), so
    nothing here can be a stale address after capture. This is the same set of
    pointers ``create_fused_set_kv_buffer_arg`` bakes into the shared ROCm fused
    rope+store path for the other models that use it.
    """

    k_cache: torch.Tensor
    v_cache: torch.Tensor
    out_cache_loc: torch.Tensor
    full_to_swa: Optional[torch.Tensor]


# ---------------------------------------------------------------------------
# CCA: Compressed Convolutional Attention QKV projection
# ---------------------------------------------------------------------------


class CCA(nn.Module):
    """Compressed Convolutional Attention QKV projection.

    Given hidden states ``hs`` of shape ``[S, H]`` this layer produces
    ``(q, k, v)`` where:

        q = (W_q hs + Conv(W_q hs ‖ W_k hs)_q) / 2
            + mean_group(W_k hs) / 2                      (fp32, RMSNorm'd)
        k = (W_k hs + Conv(W_q hs ‖ W_k hs)_k) / 2
            + mean_group(W_q hs) / 2,  scaled by per-head temperature
        v = concat(W_{v1} hs, W_{v2} hs_prev_shifted)

    The two-stage conv on ``(W_q hs ‖ W_k hs)`` needs
    ``total_padding = (cca_time0 - 1) + (cca_time1 - 1)`` tokens of left padding.
    For the first prefill chunk of a request the padding is zero; for a resumed
    prefill or for decode it is read from a per-request cache that this module
    maintains internally.

    Parallelism: when ``tp_size > 1`` the CCA is head-parallel. Both the
    grouped-mean step and the second ``conv_qk`` stage with
    ``groups=num_q_heads+num_k_heads`` are head-local (each GQA group lives on
    a single rank), so the entire QKV projection runs without any cross-rank
    collective. The QKV projections become ``ColumnParallelLinear`` and the
    two ``nn.Conv1d`` layers are sized per-rank with custom weight loaders
    that slice the HF checkpoint rows into ``[rank's q heads, rank's k heads]``.
    """

    def __init__(
        self,
        config: ZayaConfig,
        cca_num_k_heads: int,
        cca_num_q_heads: int,
        hidden_size: int,
        head_dim: int,
        cca_time0: int,
        cca_time1: int,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = int(hidden_size)
        self.head_dim = int(head_dim)
        self.cca_time0 = int(cca_time0)
        self.cca_time1 = int(cca_time1)
        self.padding0 = self.cca_time0 - 1
        self.padding1 = self.cca_time1 - 1
        self.total_padding = self.padding0 + self.padding1

        # CCA is head-parallel over the *attention* TP group. That group equals
        # the global TP group unless DP attention is enabled, in which case it is
        # the per-DP-replica sub-group (tp_size / dp_size). Tests pass explicit
        # tp_rank/tp_size; production leaves them None and resolves here.
        if tp_rank is None:
            tp_rank = get_parallel().attn_tp_rank
        if tp_size is None:
            tp_size = get_parallel().attn_tp_size
        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)

        # Full (global) head counts retained for weight loading and shape asserts.
        self.num_q_heads_full = int(cca_num_q_heads)
        self.num_k_heads_full = int(cca_num_k_heads)
        assert (
            self.num_q_heads_full % self.num_k_heads_full == 0
        ), "num_q_heads must be a multiple of num_k_heads"
        self.gqa_groups = self.num_q_heads_full // self.num_k_heads_full

        # Head-parallel TP requires both head counts to be divisible by tp_size.
        # KV-replication-style TP (tp_size > num_k_heads) is not yet supported.
        assert self.num_q_heads_full % self.tp_size == 0, (
            f"num_q_heads ({self.num_q_heads_full}) must be divisible by "
            f"tp_size ({self.tp_size}) for ZAYA1 head-parallel CCA"
        )
        assert self.num_k_heads_full % self.tp_size == 0, (
            f"num_k_heads ({self.num_k_heads_full}) must be divisible by "
            f"tp_size ({self.tp_size}); KV-replication TP is not supported "
            "for ZAYA1 because both grouped-mean and conv_qk.1 are per-head"
        )

        # Per-rank head counts.
        self.num_q_heads = self.num_q_heads_full // self.tp_size
        self.num_k_heads = self.num_k_heads_full // self.tp_size

        # Per-rank channel layout.
        self.latent_q_dim_full = self.num_q_heads_full * self.head_dim
        self.latent_k_dim_full = self.num_k_heads_full * self.head_dim
        self.in_out_ch_full = self.latent_q_dim_full + self.latent_k_dim_full
        self.latent_q_dim = self.num_q_heads * self.head_dim
        self.latent_k_dim = self.num_k_heads * self.head_dim
        self.in_out_ch = self.latent_q_dim + self.latent_k_dim
        self.sqrt_head_dim = float(self.head_dim) ** 0.5
        self.clamp_temp = bool(getattr(config, "clamp_temp", False))

        bias = bool(getattr(config, "attention_bias", False))
        # ``linear_q`` / ``linear_k`` outputs are laid out as a contiguous head
        # sequence in the HF checkpoint, so the natural ColumnParallel shard
        # (``tp_rank * shard``) lands rank ``r`` on the head set
        # ``[r * heads_per_rank, (r+1) * heads_per_rank)``.
        #
        # At ``tp_size == 1`` there is nothing to shard, and on ROCm/aiter the
        # ColumnParallelLinear path selects a slower GEMM for the large-M prefill
        # (1.6-2.25x slower than ReplicatedLinear in bench_one_batch), so the
        # single-GPU case uses ReplicatedLinear. ``tp_size > 1`` keeps
        # ColumnParallelLinear for the per-rank head shard.
        # q and k read the same ``hidden_states`` and their outputs are
        # immediately concatenated, so they are one projection: a single wider
        # GEMM replaces two skinny ones and the ``cat`` disappears. Decode is
        # launch-bound on exactly these -- a profile of ZAYA1-base put the skinny
        # GEMV kernel at 280 launches per step (7 per attention layer) and the
        # concatenates at 80 -- and the k half is tiny (num_query_groups *
        # head_dim), so on its own it is nearly all overhead.
        if self.tp_size > 1:
            self.linear_qk = MergedColumnParallelLinear(
                self.hidden_size,
                [self.latent_q_dim_full, self.latent_k_dim_full],
                bias=bias,
                gather_output=False,
                quant_config=quant_config,
                prefix=add_prefix("linear_qk", prefix),
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
            )
        else:
            # ColumnParallelLinear measured 1.6-2.25x slower than
            # ReplicatedLinear at tp=1 (bench_one_batch), so the single-GPU case
            # keeps a plain replicated projection; ``_merged_qk_row_loader``
            # gives it the same shard-id loading contract as the merged
            # column-parallel one.
            self.linear_qk = ReplicatedLinear(
                self.hidden_size,
                self.latent_q_dim_full + self.latent_k_dim_full,
                bias=bias,
                quant_config=quant_config,
                prefix=add_prefix("linear_qk", prefix),
            )
            self._install_merged_qk_loader(bias=bias)
        # The HF V-projection layout maps val_proj1 to the FIRST half of K
        # heads and val_proj2 to the SECOND half (after ``cat([v1, v2]).view(
        # T, num_k_heads_full, head_dim)``). That doesn't align with a simple
        # output-dim ColumnParallel shard, so val_proj1 / val_proj2 are kept
        # Replicated and the per-rank K-head slice is taken in the forward
        # passes after ``cat + view``. The replicated weight memory is small
        # (~0.5 MB / layer) and the wasted compute is negligible compared to
        # linear_q / linear_k / o_proj.
        self.val_proj1 = ReplicatedLinear(
            self.hidden_size,
            self.latent_k_dim_full // 2,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("val_proj1", prefix),
        )
        self.val_proj2 = ReplicatedLinear(
            self.hidden_size,
            self.latent_k_dim_full // 2,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("val_proj2", prefix),
        )

        # Per-rank K head range, used for slicing the replicated v tensor.
        self.k_head_start = self.tp_rank * self.num_k_heads
        self.k_head_end = self.k_head_start + self.num_k_heads

        # ----- v2 lag stream (conv[1]) ------------------------------------
        # ``val_proj2`` consumes the PREVIOUS token's hidden state, which is why
        # CCA carries a second per-request state entry at all.
        #
        # Which of the two V projections this rank actually reads. ``val_proj1``
        # supplies K heads ``[0, v1_heads)`` and ``val_proj2`` the rest, so a
        # rank's contiguous K-head range can fall entirely inside one of them --
        # the common ZAYA1 case, where ``num_query_groups`` is 2 and attn_tp is
        # 2, giving rank 0 only val_proj1 and rank 1 only val_proj2. A rank that
        # does not read val_proj2 needs no lag stream at all: no projection, no
        # pool read, no pool write.
        v1_heads = self.num_k_heads_full // 2
        v_head_aligned = self.num_k_heads_full % 2 == 0
        self.v_uses_val1 = (not v_head_aligned) or self.k_head_start < v1_heads
        self.v_uses_val2 = (not v_head_aligned) or self.k_head_end > v1_heads

        # ``val_proj2`` is linear, so ``W_v2 . shift(hs) == shift(W_v2 . hs)``:
        # applying it before the state write caches ``latent_k_dim / 2`` channels
        # (128 on ZAYA1-74B) instead of ``hidden_size`` (4096) with no change to
        # the function computed. Two conditions gate it, mirrored by
        # ``ZayaConfig.cca_cache_projected_v2``, which sizes the pool entry:
        #
        #  * no bias -- a freshly allocated slot is zero and the first token's
        #    val_proj2 input is defined to be zero, and only ``W . 0 == 0``
        #    reproduces that. With a bias the cached zero would have to stand
        #    for ``b``, so a biased checkpoint keeps caching the raw hidden
        #    state and re-projecting after the read.
        #  * an even K-head count -- otherwise val_proj1 / val_proj2 split
        #    inside a head and the per-rank slice below is not head-aligned.
        #
        # Read the bias off the constructed module rather than the config flag:
        # it is the thing the forward pass would actually add.
        self.cache_projected_v2 = (
            getattr(self.val_proj2, "bias", None) is None and v_head_aligned
        )
        # Which slice of ``val_proj2``'s (replicated, ``latent_k_dim/2`` wide)
        # output this rank owns, in head units, when caching the projection.
        self.v2_head_lo = max(0, self.k_head_start - v1_heads)
        self.v2_head_hi = max(self.v2_head_lo, self.k_head_end - v1_heads)
        if not self.v_uses_val2:
            self.v2_lag_dim = 0
        elif self.cache_projected_v2:
            self.v2_lag_dim = (self.v2_head_hi - self.v2_head_lo) * self.head_dim
        else:
            self.v2_lag_dim = self.hidden_size

        # Two-stage depthwise + grouped conv along the time axis, sized for
        # this rank's head subset. Wrapping the two nn.Conv1d modules in
        # nn.Sequential makes the HF checkpoint keys ``conv_qk.{0,1}.weight``
        # / ``conv_qk.{0,1}.bias`` map onto submodules 1:1, with TP slicing
        # handled by the custom weight_loader attached below.
        self.conv_qk = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_out_ch,
                out_channels=self.in_out_ch,
                kernel_size=self.cca_time0,
                groups=self.in_out_ch,
                padding=0,
                stride=1,
            ),
            nn.Conv1d(
                in_channels=self.in_out_ch,
                out_channels=self.in_out_ch,
                kernel_size=self.cca_time1,
                groups=(self.num_k_heads + self.num_q_heads),
                padding=0,
                stride=1,
            ),
        )

        # Decode-time fold of the two conv stages into one grouped matmul.
        # Filled by ``fold_decode_conv`` after weight load; see that method and
        # ``cca_decode`` for the derivation. Non-persistent (derived from the
        # conv_qk parameters, and those are already TP-sliced per rank, so the
        # fold is automatically per-rank correct).
        self.decode_conv_groups = self.num_q_heads + self.num_k_heads
        self.decode_conv_taps = self.total_padding + 1
        # The matmul's window carries one extra constant-1.0 tap per channel so
        # the conv bias can ride in the weight instead of a separate add -- see
        # ``fold_decode_conv``. Only the last of those columns holds the bias;
        # the other ``ch_per_group - 1`` are zero, which costs a wider K on a
        # GEMM that is launch-bound anyway and buys back one launch per
        # attention layer (60 per decode step on ZAYA1-74B).
        self.decode_conv_taps_ext = self.decode_conv_taps + 1
        ch_per_group = self.in_out_ch // self.decode_conv_groups
        self.register_buffer(
            "decode_conv_weight",
            torch.zeros(
                self.decode_conv_groups,
                ch_per_group,
                ch_per_group * self.decode_conv_taps_ext,
            ),
            persistent=False,
        )
        # Kept alongside the folded weight even though the einsum path no longer
        # adds it: ``_conv_qk_run`` hands it to ``F.conv1d`` and the fused prefill
        # conv takes it as a kernel argument, both of which absorb a bias for
        # free.
        self.register_buffer(
            "decode_conv_bias",
            torch.zeros(self.decode_conv_groups, ch_per_group),
            persistent=False,
        )
        # Same folded coefficients, laid out as a grouped ``conv1d`` weight
        # ``[C_out, C_in/groups, kernel]``. The fold is a convolution -- the same
        # 3-tap kernel applies at every output position -- so it serves the
        # multi-timestep extend path as well as the single-step decode one,
        # replacing conv_qk's two MIOpen grouped convs with one.
        self.register_buffer(
            "fold_conv1d_weight",
            torch.zeros(self.in_out_ch, ch_per_group, self.decode_conv_taps),
            persistent=False,
        )
        # The folded buffers are only valid once ``fold_decode_conv`` has run
        # against loaded weights. Until then ``forward`` must keep using the real
        # ``conv_qk`` -- consuming the zero-initialized buffers would silently
        # emit bias-only output. Folding is done eagerly at load time rather than
        # lazily in ``forward``, because a lazy fold would execute inside CUDA
        # graph capture and bake stale constants into the replayed graph.
        self._decode_conv_folded = False

        # Tri-state cache for ``_fused_prefill_conv_allowed``; None == not yet
        # resolved (it reads server args, which do not exist this early).
        self._fused_prefill_conv_ok: Optional[bool] = None

        # Per-K-head learnable temperature scalar (per-rank slice).
        self.temp = nn.Parameter(torch.zeros(self.num_k_heads))

        # ``sqrt(head_dim) * temperature`` per k head, folded once after weight
        # load for the fused q/k head-mix kernel (see fold_qk_scales). fp32 to
        # match the accumulation precision of the torch path it replaces.
        self.register_buffer(
            "qk_k_scale",
            torch.zeros(self.num_k_heads, dtype=torch.float32),
            persistent=False,
        )
        self._qk_scales_folded = False

        # Set by every ``_mix_and_normalize_qk``: True when the fused kernel also
        # applied the rotary, so ``ZayaAttention`` skips its own rotary launch.
        # A python bool, like qwen3_moe's ``_used_fused_qk_norm_rope_last_call``:
        # it is decided from static properties (shapes, dtypes, devices) so it is
        # stable across a CUDA-graph capture and its replays.
        self.rope_fused = False
        # Likewise for the fused KV scatter: True when the kernel already wrote
        # k/v into the pool, so ``ZayaAttention`` passes save_kv_cache=False.
        self.kv_store_fused = False
        # Last logged (mix, rope, kv_store) decision -- see ``_note_fusion``.
        self._last_fusion_state: Optional[tuple] = None

        # Attach TP-aware weight loaders to conv_qk weights/biases and ``temp``
        # so the existing ``load_weights`` dispatch (``getattr(param,
        # "weight_loader", default_weight_loader)``) automatically slices the
        # HF checkpoint into rank-local rows.
        if self.tp_size > 1:
            self._install_tp_weight_loaders()

    # ----- TP weight loaders ----------------------------------------------

    def _install_tp_weight_loaders(self) -> None:
        """Attach TP-aware ``weight_loader`` attributes to parameters whose
        full-tensor → per-rank slicing cannot be expressed by a generic
        ColumnParallelLinear loader: the two ``conv_qk`` Conv1d weights and
        biases (where the per-rank "row" set is the discontiguous union of
        this rank's q heads and this rank's k heads) and the per-K-head
        ``temp`` parameter.
        """
        head_dim = self.head_dim
        latent_q_dim_full = self.latent_q_dim_full
        num_q_heads_per_rank = self.num_q_heads
        num_k_heads_per_rank = self.num_k_heads
        tp_rank = self.tp_rank

        q_start = tp_rank * num_q_heads_per_rank * head_dim
        q_end = q_start + num_q_heads_per_rank * head_dim
        k_start = latent_q_dim_full + tp_rank * num_k_heads_per_rank * head_dim
        k_end = k_start + num_k_heads_per_rank * head_dim
        k_temp_start = tp_rank * num_k_heads_per_rank
        k_temp_end = k_temp_start + num_k_heads_per_rank

        def conv_row_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
            # Both Conv1d.weight ([C_out, in_per_group, K]) and Conv1d.bias
            # ([C_out]) slice along the leading (output channel) dim. The
            # per-rank rows are the rank's q heads (contiguous) followed by
            # the rank's k heads (contiguous in the second half of the full
            # tensor).
            sliced = torch.cat(
                [loaded_weight[q_start:q_end], loaded_weight[k_start:k_end]],
                dim=0,
            )
            assert (
                sliced.shape == param.data.shape
            ), f"conv shard shape mismatch: {sliced.shape} vs {param.data.shape}"
            param.data.copy_(sliced)

        def temp_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
            sliced = loaded_weight[k_temp_start:k_temp_end]
            assert (
                sliced.shape == param.data.shape
            ), f"temp shard shape mismatch: {sliced.shape} vs {param.data.shape}"
            param.data.copy_(sliced)

        set_weight_attrs(self.conv_qk[0].weight, {"weight_loader": conv_row_loader})
        set_weight_attrs(self.conv_qk[0].bias, {"weight_loader": conv_row_loader})
        set_weight_attrs(self.conv_qk[1].weight, {"weight_loader": conv_row_loader})
        set_weight_attrs(self.conv_qk[1].bias, {"weight_loader": conv_row_loader})
        set_weight_attrs(self.temp, {"weight_loader": temp_loader})

    def _install_merged_qk_loader(self, *, bias: bool) -> None:
        """Give the tp=1 replicated q/k projection a shard-id weight loader.

        ``MergedColumnParallelLinear`` already accepts ``(param, weight,
        loaded_shard_id)``; ``ReplicatedLinear`` does not, so attach an
        equivalent that writes shard 0 (q) then shard 1 (k) into the merged rows.
        Keeps ``load_weights`` free of a tp==1 special case.
        """
        q_rows = self.latent_q_dim_full

        def merged_row_loader(
            param: torch.Tensor,
            loaded_weight: torch.Tensor,
            loaded_shard_id: int = 0,
        ) -> None:
            start = 0 if loaded_shard_id == 0 else q_rows
            end = q_rows if loaded_shard_id == 0 else param.data.shape[0]
            assert loaded_weight.shape[0] == end - start, (
                f"merged qk shard {loaded_shard_id} expects "
                f"{end - start} rows, got {loaded_weight.shape[0]}"
            )
            param.data[start:end].copy_(loaded_weight)

        # Assigned directly rather than through ``set_weight_attrs``:
        # ReplicatedLinear already installs its own single-shard weight_loader,
        # and set_weight_attrs asserts against overwriting an existing attribute.
        # Replacing it is the point here -- the merged parameter needs the
        # shard-aware loader.
        self.linear_qk.weight.weight_loader = merged_row_loader
        if bias:
            self.linear_qk.bias.weight_loader = merged_row_loader

    # ----- helpers ---------------------------------------------------------

    def _normalize_qk(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """RMSNorm (no learnable weight) + sqrt(head_dim) scaling on q and k,
        plus per-K-head temperature on k. Computed in fp32 for stability.
        """
        eps = 1e-12
        sqrt_head_dim = float(self.sqrt_head_dim)
        query_fp32 = query.to(torch.float32)
        inv_q = (
            torch.rsqrt(query_fp32.pow(2).sum(-1, keepdim=True) + eps) * sqrt_head_dim
        )
        query_fp32 = query_fp32 * inv_q

        key_fp32 = key.to(torch.float32)
        inv_k = torch.rsqrt(key_fp32.pow(2).sum(-1, keepdim=True) + eps) * sqrt_head_dim
        key_fp32 = key_fp32 * inv_k
        temp = self.temp.to(torch.float32).view(1, self.num_k_heads, 1)
        if self.clamp_temp:
            temp = torch.exp(torch.clamp(temp, 1e-7, 2.0))
        key_fp32 = key_fp32 * temp
        return query_fp32, key_fp32

    def _add_grouped_qk_means(
        self,
        query_conv: torch.Tensor,
        key_conv: torch.Tensor,
        query_pre: torch.Tensor,
        key_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Blend the post-conv q/k with the per-GQA-group mean of the
        pre-conv (raw projection) q/k, matching the ZAYA1 training formula.

        Shapes (T = num_tokens):
            query_conv : [T, num_q_heads, head_dim]      (fp32, post conv)
            key_conv   : [T, num_k_heads, head_dim]      (fp32, post conv)
            query_pre  : [T, num_q_heads, head_dim]      (raw W_q hs)
            key_base   : [T, num_k_heads, head_dim]      (raw W_k hs)
        """
        num_k_heads = key_base.shape[-2]
        key_base_fp32 = key_base.to(torch.float32)
        query_pre_grouped = query_pre.view(
            query_pre.shape[0], num_k_heads, self.gqa_groups, query_pre.shape[-1]
        )
        query_pre_grouped_fp32 = query_pre_grouped.to(torch.float32)
        query_out_grouped = (
            query_conv.view_as(query_pre_grouped).to(torch.float32)
            + 0.5 * query_pre_grouped_fp32
            + 0.5 * key_base_fp32.unsqueeze(-2)
        )
        query_out = query_out_grouped.reshape(
            query_pre.shape[0], -1, query_pre.shape[-1]
        )

        query_pre_mean = query_pre_grouped_fp32.mean(dim=-2, dtype=torch.float32)
        key_out = (
            key_conv.to(torch.float32) + 0.5 * query_pre_mean + 0.5 * key_base_fp32
        )
        return query_out, key_out

    @torch.no_grad()
    def fold_qk_scales(self) -> None:
        """Fold ``sqrt(head_dim) * temperature`` into one fp32 vector.

        ``_normalize_qk`` applies both factors to k (and the ``clamp_temp``
        exponential) on every forward; they depend only on loaded weights, so the
        fused kernel takes the product precomputed. Refreshed by
        ``ZayaForCausalLM.fold_decode_constants`` after every weight load.
        """
        temp = self.temp.detach().to(torch.float32)
        if self.clamp_temp:
            temp = torch.exp(torch.clamp(temp, 1e-12, 2.0))
        self.qk_k_scale.copy_(temp * float(self.sqrt_head_dim))
        self._qk_scales_folded = True

    def _mix_and_normalize_qk(
        self,
        qk_out: torch.Tensor,
        query_pre_flat: torch.Tensor,
        key_base_flat: torch.Tensor,
        query_conv: torch.Tensor,
        key_conv: torch.Tensor,
        query_pre: torch.Tensor,
        key_base: torch.Tensor,
        out_dtype: torch.dtype,
        rope: Optional[CCARope] = None,
        value: Optional[torch.Tensor] = None,
        kv_store: Optional[CCAKVStore] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Blend post-conv q/k with the grouped pre-conv means, then RMS-normalize.

        Prefers the fused Triton kernel and falls back to the two torch helpers
        when it cannot serve the shapes (see ``cca_qk_mix.covered``) -- notably
        before ``fold_qk_scales`` has run, so CPU unit tests keep the torch path.

        ``rope`` and ``kv_store`` are two further, independent gates on top of
        ``covered()``: an input the mix can serve but one of them cannot still
        gets the fused mix, and ``self.rope_fused`` / ``self.kv_store_fused``
        tell the caller which launches it still owes. ``rope`` must win for
        ``kv_store`` to be offered -- storing an un-rotated k would be silent KV
        corruption, and the rotation happens inside this same kernel.
        """
        from sglang.kernels.ops.attention import cca_qk_mix as _cca_qk_mix

        self.rope_fused = False
        self.kv_store_fused = False
        scale = self.qk_k_scale if self._qk_scales_folded else None
        mix_fused = False
        if _cca_qk_mix.covered(
            qk_out,
            query_pre_flat,
            key_base_flat,
            scale,
            num_q_heads=self.num_q_heads,
            num_k_heads=self.num_k_heads,
            head_dim=self.head_dim,
        ):
            mix_fused = True
            extra = {}
            if rope is not None and _cca_qk_mix.rope_covered(
                rope.positions,
                rope.cos_sin_cache,
                rope.rotary_dim,
                head_dim=self.head_dim,
                num_tokens=qk_out.shape[0],
                is_neox_style=rope.is_neox_style,
                device=qk_out.device,
            ):
                extra = {
                    "positions": rope.positions,
                    "cos_sin_cache": rope.cos_sin_cache,
                    "rotary_dim": rope.rotary_dim,
                }
                self.rope_fused = True
            if (
                self.rope_fused
                and kv_store is not None
                and _cca_qk_mix.store_covered(
                    value,
                    kv_store.k_cache,
                    kv_store.v_cache,
                    kv_store.out_cache_loc,
                    kv_store.full_to_swa,
                    num_k_heads=self.num_k_heads,
                    head_dim=self.head_dim,
                    num_tokens=qk_out.shape[0],
                    out_dtype=out_dtype,
                    device=qk_out.device,
                )
            ):
                extra["value"] = value
                extra["k_cache"] = kv_store.k_cache
                extra["v_cache"] = kv_store.v_cache
                extra["out_cache_loc"] = kv_store.out_cache_loc
                extra["full_to_swa"] = kv_store.full_to_swa
                self.kv_store_fused = True
            self._note_fusion(mix_fused)
            return _cca_qk_mix.cca_qk_mix(
                qk_out,
                query_pre_flat,
                key_base_flat,
                scale,
                num_q_heads=self.num_q_heads,
                num_k_heads=self.num_k_heads,
                head_dim=self.head_dim,
                q_scale=float(self.sqrt_head_dim),
                out_dtype=out_dtype,
                **extra,
            )

        self._note_fusion(mix_fused)
        query, key = self._add_grouped_qk_means(
            query_conv, key_conv, query_pre, key_base
        )
        query, key = self._normalize_qk(query, key)
        return query.to(out_dtype), key.to(out_dtype)

    def _note_fusion(self, mix_fused: bool) -> None:
        """Log which of the three fusions took, once per distinct outcome.

        Every gate here declines by *falling back*, so a precondition that stops
        matching costs launches and nothing else -- which is exactly how an
        earlier step of this campaign produced baseline-identical numbers that
        read as a clean null result. Saying so in the log turns that into an
        observation instead of a mystery.

        Cost is a three-bool tuple compare per forward; the message is built only
        when the outcome changes, and ``_log_dataflow_decision`` dedupes across
        layers so a uniform model logs one line, not sixty.
        """
        state = (mix_fused, self.rope_fused, self.kv_store_fused)
        if state == self._last_fusion_state:
            return
        self._last_fusion_state = state
        _log_dataflow_decision(
            f"zaya cca qk fusion: mix={mix_fused} rope={self.rope_fused} "
            f"kv_store={self.kv_store_fused}"
        )

    @torch.no_grad()
    def fold_decode_conv(self) -> None:
        """Collapse ``conv_qk`` into one grouped matmul for the decode step.

        Decode feeds a ``[T, C, total_padding + 1]`` window and needs a single
        output timestep, so the depthwise stage (``kernel_size = cca_time0``)
        followed by the grouped stage (``kernel_size = cca_time1``) is one affine
        map from ``t0 + t1 - 1 == total_padding + 1`` input taps::

            out[co] = sum_{ci in g} sum_{j<t1} w1[co,ci,j]
                                  * ( sum_{k<t0} w0[ci,k] * x[ci,j+k] ) + bias
                    = sum_{ci in g} sum_m A[co,ci,m] * x[ci,m] + bias
            A[co,ci,m] = sum_{j+k=m} w1[co,ci,j] * w0[ci,k]

        The depthwise bias passes through every tap of the grouped stage, hence
        ``b = b1 + sum_ci (sum_j w1[co,ci,j]) * b0[ci]``. Folding turns two
        MIOpen grouped convs into one einsum: measured 1.9x (T=1) to 4.6x (T=32)
        on MI350X under graph replay. Extend still uses the real two-stage conv,
        which produces many timesteps and cannot be folded this way.

        ``decode_conv_weight`` additionally carries that bias as a trailing
        column, matched by the constant-1.0 tap ``cca_state_step`` appends to the
        window. The affine map becomes a pure linear one over ``taps + 1`` inputs,
        so the bias is summed inside the matmul's fp32 accumulator instead of
        being added to an already-rounded bf16 output. That removes a launch per
        attention layer AND drops a rounding step, so the decode conv output is
        *closer* to the fp32 reference than before, not bit-identical to it. The
        column lands at ``ci == ch_per_group - 1`` and the other ``ci`` slots of
        that tap are zero, so each output gets the bias exactly once.
        """
        t0, t1 = self.cca_time0, self.cca_time1
        groups = self.decode_conv_groups
        cg = self.in_out_ch // groups
        taps = self.decode_conv_taps

        w0 = self.conv_qk[0].weight.float().view(groups, cg, t0)  # depthwise
        b0 = self.conv_qk[0].bias.float().view(groups, cg)
        w1 = self.conv_qk[1].weight.float().view(groups, cg, cg, t1)  # grouped
        b1 = self.conv_qk[1].bias.float().view(groups, cg)

        folded = torch.zeros(
            groups, cg, cg, taps, device=w0.device, dtype=torch.float32
        )
        for j in range(t1):
            for k in range(t0):
                # w1[..., j] weights the depthwise output at offset j, which
                # itself reads input tap j + k.
                folded[..., j + k] += w1[..., j] * w0[:, None, :, k]

        bias = b1 + (w1.sum(dim=3) * b0[:, None, :]).sum(dim=2)  # [G, Cg]

        # [G, Co_g, Ci_g, taps] -> [G, Co_g, Ci_g, taps + 1], the extra tap
        # holding the bias on the last input channel and zero on the rest. The
        # window's matching column is a constant 1.0, so the matmul contracts to
        # ``sum_ci sum_m A[co,ci,m] x[ci,m] + b[co]`` with the bias inside the
        # fp32 accumulator.
        folded_ext = torch.zeros(
            groups, cg, cg, taps + 1, device=w0.device, dtype=torch.float32
        )
        folded_ext[..., :taps] = folded
        folded_ext[:, :, cg - 1, taps] = bias
        self.decode_conv_weight.copy_(
            folded_ext.reshape(groups, cg, cg * (taps + 1)).to(
                self.decode_conv_weight.dtype
            )
        )
        self.decode_conv_bias.copy_(bias.to(self.decode_conv_bias.dtype))
        # [G, Co_g, Ci_g, taps] -> [G*Co_g, Ci_g, taps] == [C, C/groups, kernel]
        self.fold_conv1d_weight.copy_(
            folded.reshape(groups * cg, cg, taps).to(self.fold_conv1d_weight.dtype)
        )
        self._decode_conv_folded = True

    def _fused_prefill_conv_allowed(self) -> bool:
        """Whether the fused varlen prefill conv may run, resolved once.

        The fused conv and the prefill CUDA graph are each correct alone and
        corrupt together: under capture the fused path yields all-zero logits,
        meaning the conv state it writes is wrong, and the cause is not yet
        understood. The graph is the larger win and works with the reference host
        loop, so it takes precedence.

        Resolved lazily rather than in ``__init__`` because the config bag it reads
        is not published that early, and reached only once the env flag is on so a
        module built outside a server (a CPU unit test) never touches it.

        Reads the resolved leaf from the exec bag, NOT the published ServerArgs
        record: ``cuda_graph_config`` is filled in by resolution, so on the record
        it is whatever the operator typed -- ``None`` unless they passed
        ``--cuda-graph-backend-prefill`` explicitly, which crashed here.
        """
        if self._fused_prefill_conv_ok is None:
            from sglang.srt.model_executor.cuda_graph_config import Backend

            prefill_graph_on = (
                get_exec().graph.cuda_graph_config.prefill.backend != Backend.DISABLED
            )
            self._fused_prefill_conv_ok = not prefill_graph_on
            if prefill_graph_on:
                _log_dataflow_decision(
                    "cca prefill: fused kernel disabled because the prefill CUDA "
                    "graph is enabled; the two are not yet compatible (the fused "
                    "conv under capture produces zero logits)"
                )
        return self._fused_prefill_conv_ok

    def _run_extend_conv(
        self,
        qk: torch.Tensor,
        lag_now: Optional[torch.Tensor],
        meta,
        conv_state: torch.Tensor,
        lag_state: Optional[torch.Tensor],
        extend_seq_lens_cpu: List[int],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the prefill conv-with-state, preferring the fused varlen kernel.

        The fused path needs the folded single grouped weight and the backend's
        device-side request metadata; when either is missing it falls back to the
        reference host loop in :func:`cca_extend`.

        ``lag_now`` is the projected ``val_proj2`` value stream (or the raw hidden
        state on a checkpoint that cannot cache the projection), and ``None`` on a
        rank that never reads the lag. Both paths carry it at whatever width they
        are handed, so the 32x narrowing is entirely in what the caller passes.
        """
        from sglang.kernels.ops.attention import cca_conv1d as _conv1d

        if (
            envs.SGLANG_OPT_ZAYA_FUSED_CCA_PREFILL.get()
            and self._fused_prefill_conv_allowed()
        ):
            bias = self.decode_conv_bias.reshape(-1)
            weight = self.fold_conv1d_weight if self._decode_conv_folded else None
            if _conv1d.covered(
                qk,
                lag_now,
                weight,
                bias,
                conv_state,
                lag_state,
                meta.query_start_loc,
                meta.has_initial_state,
                meta.cache_indices,
                self.total_padding,
                self.decode_conv_groups,
            ):
                _log_dataflow_decision(
                    f"cca prefill: fused varlen kernel, {qk.shape[0]} tokens over "
                    f"{meta.query_start_loc.shape[0] - 1} requests"
                )
                return _conv1d.cca_conv1d_fn(
                    qk,
                    lag_now,
                    weight,
                    bias,
                    conv_state,
                    lag_state,
                    meta.query_start_loc,
                    meta.has_initial_state,
                    meta.cache_indices,
                    self.total_padding,
                    self.decode_conv_groups,
                )
            _log_dataflow_decision(
                "cca prefill: fused kernel declined by covered(), running the "
                f"per-request host loop (folded={self._decode_conv_folded})"
            )

        return cca_extend(
            qk,
            lag_now,
            self._conv_qk_run,
            conv_state,
            lag_state,
            meta.slot_ids_cpu,
            meta.has_prefix_cpu,
            extend_seq_lens_cpu,
            self.total_padding,
        )

    def _conv_qk_run(self, padded: torch.Tensor) -> torch.Tensor:
        """Run the conv on ``[N, C, S + total_padding]`` -> ``[N, C, S]``.

        Uses the single folded grouped conv when the weights have been folded,
        which is exactly equivalent to the two-stage ``conv_qk`` (see
        :meth:`fold_decode_conv`) and halves the number of grouped-conv launches
        in the extend path. Falls back to the real two stages otherwise, so an
        unfolded module -- a CPU unit test -- still exercises the reference.
        """
        if self._decode_conv_folded:
            return F.conv1d(
                padded,
                self.fold_conv1d_weight,
                self.decode_conv_bias.reshape(-1),
                groups=self.decode_conv_groups,
            )
        return self.conv_qk(padded)

    # ----- forward modes ---------------------------------------------------

    def _slice_v_per_rank(self, value_full: torch.Tensor) -> torch.Tensor:
        """Take this rank's K-head slice of the full ``value`` tensor.

        Returns a no-op view when ``tp_size == 1``. For ``tp_size > 1`` the
        full V tensor is computed on every rank (see the comment on
        ``val_proj1`` / ``val_proj2``) and the rank's contiguous K-head range
        is selected here, leaving the downstream RadixAttention call with a
        per-rank shape ``[T, num_k_heads_per_rank, head_dim]``.
        """
        if self.tp_size == 1:
            return value_full
        return value_full[:, self.k_head_start : self.k_head_end, :].contiguous()

    def _lag_now(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        """The quantity this rank parks in the conv[1] pool slot this step.

        ``None`` when the rank's K heads all come from ``val_proj1`` -- it never
        reads the lag, so it neither projects nor stores anything. Otherwise the
        rank's slice of ``val_proj2(hidden_states)`` when the projection is
        cached, and the raw hidden state when it is not (biased or
        non-head-aligned checkpoint).

        Note the argument: ``val_proj2`` runs on the CURRENT hidden state here,
        not the shifted one. The shift is what the pool slot provides.
        """
        if not self.v_uses_val2:
            return None
        if not self.cache_projected_v2:
            return hidden_states
        v2, _ = self.val_proj2(hidden_states)
        lo, hi = self.v2_head_lo * self.head_dim, self.v2_head_hi * self.head_dim
        if lo == 0 and hi == v2.shape[-1]:
            return v2
        return v2[:, lo:hi]

    def _lag_state(self, pool_entry: torch.Tensor) -> Optional[torch.Tensor]:
        """This rank's view of the conv[1] pool entry, or ``None`` if unused.

        The pool entry is rank-uniform (``val_proj2`` is replicated, and an
        asymmetric per-slot size would desync ``max_mamba_cache_size`` across the
        attention-TP group). A rank that owns only part of ``val_proj2``'s output
        uses a leading sub-slice; the offset convention only has to be
        self-consistent because nothing else reads the entry.
        """
        if self.v2_lag_dim == 0:
            return None
        if pool_entry.shape[-2] == self.v2_lag_dim:
            return pool_entry
        return pool_entry.narrow(-2, 0, self.v2_lag_dim)

    def _compute_value_per_rank(
        self, hidden_states: torch.Tensor, lag_prev: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """This rank's V heads, running only the projections that feed them.

        ``val_proj1`` supplies the first ``num_k_heads_full // 2`` K heads and
        ``val_proj2`` the rest (the HF layout, see their construction). When this
        rank's head range falls entirely inside one of those, the other
        projection is dead work: skipping it drops a GEMM, the ``cat`` and the
        ``contiguous`` copy that slicing the concatenated tensor needed. That is
        the common case for ZAYA1 -- ``num_query_groups`` is 2, so at
        ``attn_tp == 2`` each rank owns exactly one projection's output.

        ``lag_prev`` is what came back from the state step. Under
        ``cache_projected_v2`` it is already this rank's slice of the previous
        token's ``val_proj2`` output and goes straight through; otherwise it is
        the previous raw hidden state and still has to be projected here.

        Falls back to computing both and slicing when the range straddles the
        boundary (or the split is not head-aligned), which is also the tp=1 path.
        """
        head_dim = self.head_dim
        start, end = self.k_head_start, self.k_head_end
        v1_heads = self.num_k_heads_full // 2

        if not self.v_uses_val2:
            value, _ = self.val_proj1(hidden_states)
            value = value[:, start * head_dim : end * head_dim]
        elif not self.v_uses_val1:
            value = self._project_lag(lag_prev)
            if not self.cache_projected_v2:
                value = value[
                    :, (start - v1_heads) * head_dim : (end - v1_heads) * head_dim
                ]
        else:
            v1, _ = self.val_proj1(hidden_states)
            v2 = self._project_lag(lag_prev)
            if self.cache_projected_v2:
                # v2 already covers K heads [v1_heads, end); v1 supplies
                # [start, v1_heads). start < v1_heads holds on this branch.
                value = torch.cat(
                    [v1[:, start * head_dim : v1_heads * head_dim], v2], dim=-1
                )
            else:
                value = torch.cat([v1, v2], dim=-1)[
                    :, start * head_dim : end * head_dim
                ]

        return value.reshape(value.shape[0], self.num_k_heads, head_dim)

    def _project_lag(self, lag_prev: torch.Tensor) -> torch.Tensor:
        """``val_proj2`` applied to the lag, or the lag itself when precomputed."""
        if self.cache_projected_v2:
            return lag_prev
        v2, _ = self.val_proj2(lag_prev)
        return v2

    def _forward_no_state(
        self, hs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reference path: process the entire ``hs`` of shape ``[S, H]`` with
        a zero initial conv state and a zero ``val_proj2`` lag.

        Exercised by the CCA unit tests so the prefill / decode paths can be
        compared against a single-shot torch reference, and used as a fallback
        for profile / warmup runs where no state cache is meaningful.
        """
        S = hs.shape[0]
        hs_3d = hs.unsqueeze(1)  # [S, 1, H]

        qk, _ = self.linear_qk(hs_3d)  # [S, 1, in_out_ch_per_rank]

        query_pre = qk[..., : self.latent_q_dim].reshape(
            S, self.num_q_heads, self.head_dim
        )
        key_base = qk[..., self.latent_q_dim :].reshape(
            S, self.num_k_heads, self.head_dim
        )

        # [1, C, S+pad] -> [1, C, S]
        qk_perm = qk.permute(1, 2, 0)
        qk_pad = F.pad(qk_perm, (self.total_padding, 0))
        qk_out = self._conv_qk_run(qk_pad).permute(2, 0, 1).squeeze(1)  # [S, C]

        query_conv = qk_out[:, : self.latent_q_dim].view(
            S, self.num_q_heads, self.head_dim
        )
        key_conv = qk_out[:, self.latent_q_dim :].view(
            S, self.num_k_heads, self.head_dim
        )

        query, key = self._add_grouped_qk_means(
            query_conv, key_conv, query_pre, key_base
        )
        query, key = self._normalize_qk(query, key)
        query, key = query.to(hs.dtype), key.to(hs.dtype)

        # val_proj1 / val_proj2 are replicated; compute the full V tensor and
        # then take this rank's K-head slice.
        # val_proj2 uses a right-shifted hidden_state. First val_proj2 input is 0.
        hs_shifted = F.pad(hs_3d[:-1], (0, 0, 0, 0, 1, 0))  # [S, 1, H]
        v1, _ = self.val_proj1(hs_3d)
        v2, _ = self.val_proj2(hs_shifted)
        value_full = (
            torch.cat([v1, v2], dim=-1)
            .squeeze(1)
            .view(S, self.num_k_heads_full, self.head_dim)
        )
        value = self._slice_v_per_rank(value_full)
        return query, key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        rope: Optional[CCARope] = None,
        kv_store: Optional[CCAKVStore] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project ``hidden_states`` into ``(q, k, v)`` honoring per-request state.

        The per-request conv-state plumbing (slot gather/scatter, prefix mask,
        cuda-graph buffers) is owned by :class:`ShortConvAttnBackend
        <sglang.srt.layers.attention.linear.short_conv_backend.ShortConvAttnBackend>`,
        reached via ``get_attn_backend().conv_state_metadata``; CCA runs its own
        two-stage grouped conv (:func:`cca_extend` / :func:`cca_decode`) against
        that handle, so this module holds no pool access. Those functions return
        the conv output ``qk_out`` and the one-token-lagged ``val_proj2`` value
        stream ``lag_prev``, updating the ``conv_state`` / lag pool slots in
        place.

        ``q`` / ``k`` / ``v`` are all returned in ``hidden_states``' dtype. The
        blend and normalize still accumulate in fp32 internally, but the result is
        stored at model precision: the caller rounded it there immediately anyway,
        so materializing an fp32 copy first only cost a per-layer ``copy_``.

        ``rope`` (this layer's :class:`CCARope`) additionally folds the neox
        partial rotary into the same kernel, so ``q`` / ``k`` come back already
        rotated. ``self.rope_fused`` reports whether that happened; when it is
        False the caller still owes a rotary launch. Under the global-residual
        dataflow ``rope.positions`` is already the replica-local view, matching
        ``hidden_states``.

        ``kv_store`` (this layer's :class:`CCAKVStore`) further folds the paged
        KV scatter in, reported by ``self.kv_store_fused``. That is why ``v`` is
        computed *before* the mix here rather than after: the mix kernel is what
        writes it into the pool.

        Shapes::

            q : [T, num_q_heads, head_dim]
            k : [T, num_k_heads, head_dim]
            v : [T, num_k_heads, head_dim]
        """
        if hidden_states.shape[0] == 0:
            # Nothing was rotated or stored, so an idle replica reports neither --
            # its caller returns before the rotary and the attention anyway.
            self.rope_fused = False
            self.kv_store_fused = False
            zero = hidden_states.new_zeros((0,))
            return (
                zero.view(0, self.num_q_heads, self.head_dim),
                zero.view(0, self.num_k_heads, self.head_dim),
                zero.view(0, self.num_k_heads, self.head_dim),
            )

        T = hidden_states.shape[0]
        # One merged projection: ``qk`` is already the layout the conv wants, and
        # the q / k views are free slices of it (unit innermost stride preserved).
        qk, _ = self.linear_qk(hidden_states)  # [T, in_out_ch]
        q_raw = qk[:, : self.latent_q_dim]
        k_raw = qk[:, self.latent_q_dim :]

        query_pre = q_raw.view(T, self.num_q_heads, self.head_dim)
        key_base = k_raw.view(T, self.num_k_heads, self.head_dim)

        # The backend hands out the per-request conv-state handle (slot indices,
        # prefix mask, cuda-graph buffers); CCA runs its own two-stage grouped
        # conv against it and gets back the conv output + the lagged val_proj2
        # value, with the conv_state / lag pool slots updated in place.
        backend = get_attn_backend()
        meta = backend.conv_state_metadata(self.layer_id, forward_batch)
        conv_state = meta.layer_cache.conv[0]
        lag_state = self._lag_state(meta.layer_cache.conv[1])
        lag_now = self._lag_now(hidden_states)
        if forward_batch.forward_mode.is_decode_or_idle():
            qk_out, lag_prev = cca_decode(
                qk,
                lag_now,
                self.conv_qk,
                conv_state,
                lag_state,
                meta.cache_indices,
                self.total_padding,
                decode_conv_weight=(
                    self.decode_conv_weight if self._decode_conv_folded else None
                ),
                decode_conv_bias=self.decode_conv_bias,
                decode_conv_groups=self.decode_conv_groups,
            )
        else:
            qk_out, lag_prev = self._run_extend_conv(
                qk,
                lag_now,
                meta,
                conv_state,
                lag_state,
                forward_batch.extend_seq_lens_cpu,
            )
            # Radix mamba-cache checkpoint (extra_buffer strategy only): both
            # conv entries are windows of these two inputs -- conv[0] is the
            # last ``total_padding`` rows of ``qk``, conv[1] the last row of
            # ``hidden_states`` -- so the snapshot at the chunk-aligned track
            # position is a gather over them. No-op under ``no_buffer``.
            backend.track_conv_states_extend(meta.layer_cache, (qk, hidden_states))

        query_conv = qk_out[:, : self.latent_q_dim].view(
            T, self.num_q_heads, self.head_dim
        )
        key_conv = qk_out[:, self.latent_q_dim :].view(
            T, self.num_k_heads, self.head_dim
        )

        # Emit the model dtype straight away: the caller rounded the fp32 result
        # to it immediately, so this is the same single rounding minus two
        # per-layer copies (aten::copy_ was the largest single op in the profile).
        # V before the mix: the mix kernel is what scatters it into the KV pool
        # when the fused store is available. The two have no data dependency in
        # either direction, so the reorder is free.
        value = self._compute_value_per_rank(hidden_states, lag_prev)

        query, key = self._mix_and_normalize_qk(
            qk_out,
            q_raw,
            k_raw,
            query_conv,
            key_conv,
            query_pre,
            key_base,
            out_dtype=hidden_states.dtype,
            rope=rope,
            value=value,
            kv_store=kv_store,
        )
        return query, key, value


# ---------------------------------------------------------------------------
# Attention layer (CCA QKV + rotary + RadixAttention)
# ---------------------------------------------------------------------------


class ZayaAttention(nn.Module):
    def __init__(
        self,
        config: ZayaConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.num_q_heads_full = config.num_attention_heads
        self.num_k_heads_full = config.num_query_groups
        self.head_dim = config.head_dim

        # Head-parallel TP: split both Q and KV heads across ranks. Since the
        # grouped-mean and conv_qk.1 are head-local, no cross-rank collective
        # is required inside the QKV projection. Both head counts must be
        # divisible by tp_size; the KV-replicated GQA-TP variant (tp_size >
        # num_k_heads) is intentionally rejected with a clear error message
        # because both per-K-head paths assume each rank holds whole K heads.
        # Head-parallel attention runs on the *attention* TP group. With plain
        # tensor parallelism this is the global TP group; with DP attention
        # (``enable_dp_attention``) it is the per-DP-replica sub-group of size
        # ``tp_size / dp_size``. CCA, ``o_proj`` and ``ZayaConfig.
        # mamba2_cache_params`` are all organized on this same group, so they
        # stay consistent in either mode.
        self.tp_rank = get_parallel().attn_tp_rank
        self.tp_size = get_parallel().attn_tp_size
        assert self.num_q_heads_full % self.tp_size == 0, (
            f"num_attention_heads ({self.num_q_heads_full}) must be divisible "
            f"by attention tp_size ({self.tp_size}) for ZAYA1 head-parallel "
            "attention"
        )
        # ZAYA1's grouped-mean and ``conv_qk.1`` keep whole GQA groups on each
        # rank, so attention TP cannot exceed ``num_query_groups`` (KV-head
        # replication would need a cross-rank reduction inside CCA). To use more
        # GPUs than that, enable DP attention so the extra ranks form additional
        # data-parallel replicas while attention TP stays <= num_query_groups.
        assert self.num_k_heads_full % self.tp_size == 0, (
            f"num_query_groups ({self.num_k_heads_full}) must be divisible by "
            f"attention tp_size ({self.tp_size}); attention TP cannot exceed "
            "num_query_groups for ZAYA1. Enable DP attention "
            "(enable_dp_attention) to scale across the remaining GPUs."
        )
        self.num_q_heads = self.num_q_heads_full // self.tp_size
        self.num_k_heads = self.num_k_heads_full // self.tp_size
        self.q_dim_full = self.num_q_heads_full * self.head_dim
        self.scale = self.head_dim**-0.5

        # The HF checkpoint stores the CCA QKV projection under
        # ``self_attn.qkv.*``, so the CCA submodule is registered with that
        # exact name to keep weight loading a 1:1 key mapping.
        self.qkv = CCA(
            config=config,
            cca_num_k_heads=self.num_k_heads_full,
            cca_num_q_heads=self.num_q_heads_full,
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            cca_time0=config.cca_time0,
            cca_time1=config.cca_time1,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("qkv", prefix),
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )

        # RowParallel o_proj on the attention-TP group: per-rank input is the
        # rank's q heads. The cross-rank reduction is deferred to ``forward``
        # via ``attn_tp_all_reduce`` so it targets the attention-TP group rather
        # than the global TP group (the two coincide when DP attention is off).
        self.o_proj = RowParallelLinear(
            self.q_dim_full,
            self.hidden_size,
            bias=bool(getattr(config, "attention_bias", False)),
            input_is_parallel=True,
            reduce_results=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )

        # ZAYA1-74B interleaves sliding-window attention with full attention
        # (per-layer ``swa_layers``), and the sliding layers use their own RoPE
        # base (``swa_rotary_base``) instead of ``rope_theta``. Base checkpoints
        # have ``swa_layers = None`` and always take the full-attention path.
        swa_window = config.sliding_window_for_layer(layer_id)
        self.is_sliding = swa_window > 0
        rope_theta = float(getattr(config, "rope_theta", 1_000_000.0))
        if self.is_sliding:
            swa_rotary_base = getattr(config, "swa_rotary_base", None)
            rope_base = float(swa_rotary_base) if swa_rotary_base else rope_theta
        else:
            rope_base = rope_theta
        partial_rotary_factor = float(getattr(config, "partial_rotary_factor", 0.5))
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=int(config.max_position_embeddings),
            base=int(rope_base),
            is_neox_style=True,
            partial_rotary_factor=partial_rotary_factor,
        )

        # Store ``window - 1`` (exclusive boundary -- the convention the SGLang
        # attention backends expect via ``layer.sliding_window_size``); full
        # attention layers pass -1. The backends pick the window per layer from
        # this attribute, and ``ModelRunner`` learns the global window from
        # ``ZayaForCausalLM.get_attention_sliding_window_size``.
        self.sliding_window_size = (swa_window - 1) if self.is_sliding else -1
        self.attn = RadixAttention(
            num_heads=self.num_q_heads,
            head_dim=self.head_dim,
            scaling=self.scale,
            num_kv_heads=self.num_k_heads,
            layer_id=layer_id,
            sliding_window_size=self.sliding_window_size,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def _kv_store(self, forward_batch: ForwardBatch) -> Optional[CCAKVStore]:
        """This layer's paged KV write target, or ``None`` to keep the unfused store.

        Resolving this per forward rather than caching it is deliberate: the pool
        can be re-backed after capture (``post_capture_active``) and the SWA
        mapping is registered by the allocator, so the answer is read from the
        live objects. Under a captured CUDA graph the whole method runs once, at
        capture; every tensor it returns is allocated once and mutated in place,
        so the captured addresses stay valid on replay.

        Rejects, in order (all fall back to ``set_kv_buffer``):

        * a scaled (fp8) attention layer -- the kernel applies no k/v scale;
        * DCP or prefill context parallelism -- both rewrite the write loc;
        * any pool type other than a plain ``MHATokenToKVPool``, a ``SWAKVPool``
          of two of them, or a ``HybridLinearKVPool`` fronting one. Exact-type
          checks, so every subclass (the unified pools, FP4/MXFP8, page-major,
          the no-op embedding pool) declines rather than being assumed
          compatible;
        * a quantized pool, a ``store_dtype`` that differs from ``dtype``, or a
          non-NHD physical layout (HND, vectorized_5d);
        * a buffer whose row count is not ``size + page_size`` -- the invariant
          that makes a flat slot index correct for any page size, and the check
          that rejects a no-op pool's placeholder buffers.
        """
        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        if out_cache_loc is None:
            return None
        if self.attn.k_scale is not None or self.attn.v_scale is not None:
            return None
        if getattr(forward_batch, "dcp_kv_mask", None) is not None:
            return None
        if get_parallel().attn_dcp_size > 1 or is_prefill_context_parallel_enabled():
            return None

        pool = get_token_to_kv_pool()
        full_to_swa = None
        pool_type = type(pool)
        if pool_type is SWAKVPool:
            _, is_swa_layer = pool.layers_mapping[self.layer_id]
            # A disagreement here means the model and the pool split the layers
            # differently, which would put the write in the wrong sub-pool.
            if is_swa_layer != self.is_sliding:
                return None
            sub = pool.swa_kv_pool if is_swa_layer else pool.full_kv_pool
            if is_swa_layer:
                full_to_swa = pool.full_to_swa_index_mapping
                if full_to_swa is None:
                    return None
                # The mapping is indexed by FULL-pool slot id.
                full_rows = pool.full_kv_pool.size + pool.full_kv_pool.page_size
                if full_to_swa.numel() <= full_rows:
                    return None
        elif pool_type is HybridLinearKVPool:
            if pool.use_mla:
                return None
            sub = pool.full_kv_pool
        elif pool_type is MHATokenToKVPool:
            sub = pool
        else:
            return None

        if type(sub) is not MHATokenToKVPool:
            return None
        if sub.is_quantized_kv_cache or sub.store_dtype != sub.dtype:
            return None
        if sub.use_hnd or sub.kv_cache_layout != "nhd":
            return None

        # ``get_key_buffer`` carries the HiCache layer-transfer wait. The baseline
        # took that same wait a few kernels later, inside the attention backend's
        # own read of this layer, so this only moves it earlier within the layer.
        k_cache = pool.get_key_buffer(self.layer_id)
        v_cache = pool.get_value_buffer(self.layer_id)
        rows = sub.size + sub.page_size
        if k_cache.shape[0] != rows or v_cache.shape[0] != rows:
            return None
        return CCAKVStore(k_cache, v_cache, out_cache_loc, full_to_swa)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        reduce_output: bool = True,
    ) -> torch.Tensor:
        # Idle forward: under DP attention a replica with no requests this step
        # still runs a forward (to join the MoE layers' gather/scatter) with T=0.
        # Every op below is a no-op on an empty batch, but the ROCm rotary kernel
        # derives its launch grid from the token count and raises SIGFPE on zero,
        # so return the correctly-shaped empty output before touching any kernel.
        # This is safe for collectives: an idle replica is idle on *all* of its
        # attention-TP ranks, so they skip the o_proj all-reduce together, and the
        # cross-replica gather/scatter that idle replicas must participate in
        # lives in ``ZayaDecoderMLPLayer`` and still runs.
        if hidden_states.shape[0] == 0:
            return hidden_states.new_zeros((0, self.hidden_size))

        # CCA returns fp32 q/k and input-dtype v as ``[T, heads, head_dim]``
        # tensors; flatten the head dim and cast all to the model dtype before
        # rotary + RadixAttention.
        #
        # The rotary is handed *into* CCA rather than run after it: the fused
        # head-mix kernel already holds the whole head in registers, so the
        # rotation is free arithmetic there and the separate
        # ``sgl_kernel.rotary_embedding`` launch (one per attention layer, 60 per
        # decode step) disappears. ``qkv.rope_fused`` says whether it took the
        # offer -- everything the fused path cannot serve (see
        # ``cca_qk_mix.rope_covered``) falls back to the launch below.
        q, k, v = self.qkv(
            hidden_states,
            forward_batch,
            rope=CCARope.of(self.rotary_emb, positions),
            kv_store=self._kv_store(forward_batch),
        )
        target_dtype = hidden_states.dtype
        # ``flatten(1)`` rather than ``reshape(T, -1)``: under DP attention a
        # replica with no requests this step runs an idle forward with T=0, and
        # ``reshape(0, -1)`` raises (the ``-1`` is ambiguous for a 0-element
        # tensor). ``flatten`` multiplies the head dims explicitly, so it yields
        # ``[0, heads*head_dim]`` and is identical to the old reshape for T>0.
        q = q.flatten(1).to(target_dtype)
        k = k.flatten(1).to(target_dtype)
        v = v.flatten(1).to(target_dtype)

        if not self.qkv.rope_fused:
            q, k = self.rotary_emb(positions, q, k)
        # Some rotary backends (notably AITER on ROCm) hand back tensors with
        # a different stride than the input. RadixAttention's KV-store kernel
        # asserts contiguous layout, so normalize q/k/v before the attention.
        # The fused path already returns freshly-allocated contiguous tensors,
        # so these are no-op views there, not copies.
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        # ``save_kv_cache=False`` when the mix kernel already scattered k/v into
        # the pool -- the same handshake ``qwen3_moe`` uses with the shared fused
        # rope+store path. The attention still needs k/v as arguments: the extend
        # kernel reads the new tokens from them directly rather than from the
        # pool.
        attn_output = self.attn(
            q, k, v, forward_batch, save_kv_cache=not self.qkv.kv_store_fused
        )
        output, _ = self.o_proj(attn_output)
        # o_proj is RowParallel with ``reduce_results=False``; reduce the partial
        # sums across the attention-TP group (equals the global TP group unless
        # DP attention is enabled). A size-1 group makes this a no-op.
        #
        # ``reduce_output=False`` returns the per-rank partial instead. The
        # global-residual dataflow uses it to fold this reduction into the DP
        # gather: ``dp_gather_partial`` has every attention-TP rank memcpy its
        # partial into the *same* slot of the global buffer, so the all-reduce
        # that gathers across replicas sums the partials within each replica as a
        # side effect -- one collective doing both jobs.
        if reduce_output and self.tp_size > 1:
            output = attn_tp_all_reduce(output)
        return output


# ---------------------------------------------------------------------------
# Router (EDA + MOD) and MoE block
# ---------------------------------------------------------------------------


def fusable_router_mlp(router_mlp: nn.Sequential) -> bool:
    """Whether ``ZayaRouter.router_mlp`` has the shape the fused kernel assumes.

    Structural, not numerical: exactly Linear / GELU / Linear / GELU / Linear,
    no ``skip_bias_add``, no bias on the final projection, and -- the one thing
    the kernel's own ``covered()`` cannot see from its tensors -- both
    activations the *erf* GELU.

    That last check is the point of this function. ``nn.GELU()`` is
    ``approximate='none'``, i.e. ``0.5 * x * (1 + erf(x / sqrt(2)))``. The tanh
    approximation differs from it by only ~5e-4 absolute: small enough to pass
    for ordinary numerical noise in an end-to-end eval, large enough to flip
    routing on near-ties, in every one of the 74B's 60 MoE layers. Fusing the
    wrong flavour would be a silent accuracy regression, so nothing gets fused
    until the flavour is verified.

    Kept a free function -- like ``mod_premask_experts`` / ``mod_blend`` -- so
    the guard is unit-testable without building a router.
    """
    if len(router_mlp) != 5:
        return False
    for i in (0, 2, 4):
        stage = router_mlp[i]
        if not isinstance(stage, ReplicatedLinear) or stage.skip_bias_add:
            return False
    for i in (1, 3):
        activation = router_mlp[i]
        if not isinstance(activation, nn.GELU):
            return False
        if getattr(activation, "approximate", "none") != "none":
            return False
    # The kernel folds no bias into the final projection.
    return getattr(router_mlp[4], "bias", None) is None


class ZayaRouting(NamedTuple):
    """Everything :class:`ZayaBlock` needs out of one router forward.

    ``moe_weight`` / ``moe_ids`` are the FusedMoE-ready pair: fp32 weights --
    the dtype every other sglang top-k emits, so the MoE runner's opening
    ``topk_weights.to(torch.float32)`` is a no-op rather than a launch -- and
    int32 expert ids already clamped into ``[0, num_moe_experts - 1]``.

    ``route_prob`` is the same probability at the model dtype, which is what the
    MOD residual blend multiplies the hidden state by. It stays separate from
    ``moe_weight`` so fusing the tail does not quietly change the MOD
    arithmetic's precision.

    ``skip_ids`` is the *unclamped* choice. MOD needs it: the clamp folds the
    skip slot (``num_moe_experts``) onto real expert ``num_moe_experts - 1``, so
    ``moe_ids`` can no longer distinguish a skipped token from one genuinely
    routed to the last expert. Without MOD it aliases ``moe_ids``.

    ``hidden_states_next`` is the POST-EDA, PRE-NORM router state that the next
    MoE layer folds in. Publishing the normalized tensor instead would change
    the EDA recursion in every downstream MoE layer without crashing.
    """

    moe_weight: torch.Tensor
    moe_ids: torch.Tensor
    route_prob: torch.Tensor
    skip_ids: torch.Tensor
    hidden_states_next: torch.Tensor


class ZayaRouter(nn.Module):
    """ZAYA1 expert router: 3-layer MLP with optional EDA and MOD.

    EDA (Exponential Decay Averaging) adds a scaled copy of the previous MoE
    layer's router hidden_state to the current layer's input, threading state
    across MoE layers.

    MOD (Mixture of Depths) reserves the last expert slot as a "skip" expert
    whose contribution to the residual stream is just the routing probability
    times the unprocessed hidden_state, letting individual tokens bypass the
    MoE entirely when the router scores the skip expert highest.
    """

    def __init__(
        self,
        config: ZayaConfig,
        layer_id: int,
        num_moe_experts: int,
        moe_router_topk: int,
        mlp_expansion: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.router_softmax_fp32 = bool(getattr(config, "zaya_high_prec", False))

        self.use_mod = bool(getattr(config, "zaya_use_mod", False))
        self.num_moe_experts = int(num_moe_experts)
        self.num_experts = (num_moe_experts + 1) if self.use_mod else num_moe_experts
        # Whether the MOD skip slot can actually win the argmax on the LOADED
        # biases; resolved by fold_mod_reachability() after every weight load.
        # Assume it can until the weights say otherwise.
        self.mod_reachable = self.use_mod
        self.topk = int(moe_router_topk)
        self.mlp_expansion = int(mlp_expansion)

        # The router is left unquantized. Its final projection is
        # ``mlp_expansion -> num_experts (+1 for the MOD skip slot)``, which for
        # ZAYA1 is 25 -- not a multiple of 16, so an FP8 GEMM rejects it outright
        # ("mat2 shape (256x25) must be divisible by 16") and online
        # --quantization fp8 fails at the first router forward. Quantizing it buys
        # nothing anyway: the whole router is ~0.1% of the layer's weights, while
        # the experts it selects are ~99%. Routing precision also feeds an argmax,
        # where fp8 rounding could flip expert choice for near-ties.
        router_quant_config = None

        self.down_proj = ReplicatedLinear(
            self.hidden_size,
            self.mlp_expansion,
            bias=True,
            quant_config=router_quant_config,
            prefix=add_prefix("down_proj", prefix),
        )

        # EDA threads router state from the previous MoE layer through
        # ``router_states_scale``. The first MoE layer in the model has no
        # previous state; whether to fold it in is decided at call time based on
        # ``prev_router_hidden_states``.
        ln_eps = float(getattr(config, "norm_epsilon", 1e-5))
        self.use_eda = bool(getattr(config, "zaya_use_eda", False))
        self.rmsnorm_eda = RMSNorm(self.mlp_expansion, eps=ln_eps)
        if self.use_eda:
            self.router_states_scale = nn.Parameter(torch.ones(self.mlp_expansion))

        self.non_linearity = nn.GELU()
        self.router_mlp = nn.Sequential(
            ReplicatedLinear(
                self.mlp_expansion,
                self.mlp_expansion,
                bias=True,
                quant_config=router_quant_config,
                prefix=add_prefix("router_mlp.0", prefix),
            ),
            self.non_linearity,
            ReplicatedLinear(
                self.mlp_expansion,
                self.mlp_expansion,
                bias=True,
                quant_config=router_quant_config,
                prefix=add_prefix("router_mlp.2", prefix),
            ),
            self.non_linearity,
            ReplicatedLinear(
                self.mlp_expansion,
                self.num_experts,
                bias=False,
                quant_config=router_quant_config,
                prefix=add_prefix("router_mlp.4", prefix),
            ),
        )

        # Checked once, here, because it is a property of the module graph
        # rather than of the inputs.
        self.fused_router_mlp_ok = fusable_router_mlp(self.router_mlp)

        self.register_buffer(
            "balancing_biases",
            torch.zeros(self.num_experts, dtype=torch.float32),
            persistent=True,
        )
        if self.use_mod:
            with torch.no_grad():
                self.balancing_biases[-1] = -1.0

    def fold_mod_reachability(self) -> None:
        """Decide whether the MOD skip slot can ever win, from the loaded biases.

        ``balancing_biases`` is added to a *softmax probability*, not to a logit, so
        the skip slot's score is bounded by ``1 + b_skip`` while every real expert's
        is at least ``b_j``. When

            1 + b_skip  <  max_j b_j        (j over the real experts)

        the skip score is strictly below the best real score for every possible
        input, so no tie-breaking rule can select it and the whole MOD path is dead
        work. ZAYA1-74B ships ``b_skip = -1.0`` on all 60 MoE layers against real
        biases peaking around +0.03, so it is dead there -- which is worth two
        kernel launches per MoE layer.

        Deliberately conservative: any bias layout that does not prove
        unreachability keeps the MOD path.
        """
        if not self.use_mod:
            self.mod_reachable = False
            return
        biases = self.balancing_biases.detach().float()
        skip_bias = float(biases[-1])
        max_real_bias = float(biases[:-1].max())
        self.mod_reachable = not (1.0 + skip_bias < max_real_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_router_hidden_states: Optional[torch.Tensor] = None,
    ) -> ZayaRouting:
        # ``hidden_states`` is ``[T, H]``.
        hs, _ = self.down_proj(hidden_states)
        if (
            self.use_eda
            and prev_router_hidden_states is not None
            and hasattr(self, "router_states_scale")
        ):
            # One fused multiply-add instead of a separate mul and add: 2
            # launches become 1 on all but the first MoE layer (59 of 60 on the
            # 74B, since the first has no previous router state). In-place is
            # safe because ``hs`` is the freshly-allocated ``down_proj`` output
            # that nothing else aliases, and it drops the temporary as well.
            #
            # NOT bit-identical: ``a + b * c`` rounds the product before the
            # add, ``addcmul`` fuses them and rounds once. At bf16 that is up to
            # one ULP on the EDA term, which then threads through the whole
            # 60-layer recursion. Verified with assert_close, not assert_equal.
            hs.addcmul_(prev_router_hidden_states, self.router_states_scale)

        # ``hs`` is a freshly-allocated tensor (output of ``down_proj``, updated
        # in place by the EDA add above) and ``rmsnorm_eda`` is non-residual /
        # out-of-place, so we can hand the same buffer to the next layer without
        # cloning. This is the POST-EDA, PRE-NORM tensor: the EDA recursion is
        # defined on the un-normalized state, so publishing ``hs_norm`` here
        # instead would silently change routing in every downstream MoE layer.
        router_hidden_states_next = hs

        hs_norm = self.rmsnorm_eda(hs)
        logits = self._router_logits(hs_norm)

        # One kernel for the whole tail: softmax, balancing bias, top-1, the
        # probability gather, the model-dtype rounding, and the int32/clamp/int32
        # expert-id munging that ZayaBlock used to do on the way into FusedMoE.
        # Nine launches per MoE layer become one, and on the 74B there are 60 of
        # them in a decode step that is launch-bound rather than bandwidth-bound.
        #
        # ``.is_cuda`` is tested before the import rather than left to
        # ``covered()`` alone because the kernel module imports ``triton`` at
        # module scope, which a CPU-only environment need not have.
        if logits.is_cuda and envs.SGLANG_OPT_ZAYA_FUSED_ROUTER.get():
            from sglang.kernels.ops.moe import zaya_router_tail as _tail

            # The clamp ceiling: ids above this are the MOD skip slot.
            max_expert_id = self.num_moe_experts - 1
            if _tail.covered(
                logits,
                self.balancing_biases,
                num_experts=self.num_experts,
                max_expert_id=max_expert_id,
                topk=self.topk,
                out_dtype=hidden_states.dtype,
            ):
                moe_weight, moe_ids, route_prob, skip_ids = _tail.router_tail(
                    logits,
                    self.balancing_biases,
                    num_experts=self.num_experts,
                    max_expert_id=max_expert_id,
                    softmax_fp32=self.router_softmax_fp32,
                    out_dtype=hidden_states.dtype,
                )
                return ZayaRouting(
                    moe_weight=moe_weight,
                    moe_ids=moe_ids,
                    route_prob=route_prob,
                    skip_ids=skip_ids,
                    hidden_states_next=router_hidden_states_next,
                )

        return self._routing_reference(
            logits, hidden_states.dtype, router_hidden_states_next
        )

    def _router_logits(self, hs_norm: torch.Tensor) -> torch.Tensor:
        """Expert logits from the normalized router state, ``[T, num_experts]``.

        The 3-stage MLP with its two GELUs is five launches as an
        ``nn.Sequential`` and one as a fused kernel -- 240 per decode step across
        the 74B's 60 MoE layers. The weights are only ~270 KB per layer, so this
        is launch overhead, not bandwidth.
        """
        # ``.is_cuda`` gates the import as well as the dispatch: the kernel
        # module imports ``triton`` at module scope.
        if (
            self.fused_router_mlp_ok
            and hs_norm.is_cuda
            and envs.SGLANG_OPT_ZAYA_FUSED_ROUTER.get()
        ):
            from sglang.kernels.ops.moe import zaya_router_mlp as _mlp

            first, second, last = (self.router_mlp[i] for i in (0, 2, 4))
            operands = (
                hs_norm,
                first.weight,
                first.bias,
                second.weight,
                second.bias,
                last.weight,
            )
            if _mlp.covered(*operands, num_experts=self.num_experts):
                return _mlp.router_mlp(*operands, num_experts=self.num_experts)

        return self._router_logits_reference(hs_norm)

    def _router_logits_reference(self, hs_norm: torch.Tensor) -> torch.Tensor:
        """The unfused ``nn.Sequential``: reference semantics and fallback."""
        # Step through the Sequential manually so the ``(tensor, bias)`` tuple
        # returned by each ReplicatedLinear is unpacked correctly.
        out = hs_norm
        for stage in self.router_mlp:
            if isinstance(stage, ReplicatedLinear):
                out, _ = stage(out)
            else:
                out = stage(out)
        return out

    def _routing_reference(
        self,
        logits: torch.Tensor,
        model_dtype: torch.dtype,
        router_hidden_states_next: torch.Tensor,
    ) -> ZayaRouting:
        """The unfused torch chain: reference semantics and fallback in one place.

        This is the single definition of what the fused tail must reproduce, and
        the path taken whenever ``zaya_router_tail.covered`` says no -- CPU
        tensors, top-k > 1, or an expert axis wider than one Triton block.
        """
        if self.router_softmax_fp32:
            expert_prob = torch.softmax(logits, dim=-1, dtype=torch.float32)
        else:
            expert_prob = torch.softmax(logits, dim=-1)

        biased = expert_prob.detach().to(torch.float32) + self.balancing_biases
        if self.topk == 1:
            # ZAYA1 ships moe_router_topk=1. argmax is the same selection without
            # the sort/heap machinery a general top-k pays for, and it keeps the
            # trailing dim that ``torch.gather`` below expects. Measured 1.9% of
            # decode GPU time in aten::topk before this.
            expert_choice = biased.argmax(dim=-1, keepdim=True)
        else:
            _, expert_choice = torch.topk(biased, self.topk, dim=-1)

        if self.topk > 1 and self.use_mod:
            skip_idx = self.num_experts - 1
            n_mask = expert_choice == skip_idx
            cumsum_mask = torch.cumsum(n_mask, dim=-1)
            expert_choice = expert_choice.masked_fill(cumsum_mask > 0, skip_idx)

        route_prob = torch.gather(expert_prob, dim=1, index=expert_choice)
        # fp32 for the MoE runner -- a no-op cast whenever the softmax already
        # ran in fp32, which ``zaya_high_prec`` makes the default -- and the
        # model dtype for the MOD blend.
        moe_weight = route_prob.to(torch.float32)
        if route_prob.dtype != model_dtype:
            route_prob = route_prob.to(model_dtype)

        if self.use_mod:
            # Clamp the skip slot into the valid expert range so FusedMoE never
            # indexes out of bounds. ``skip_ids`` keeps the unclamped choice,
            # which is the only thing that can still identify a skipped token.
            moe_ids = torch.clamp(
                expert_choice, min=0, max=self.num_moe_experts - 1
            ).to(torch.int32)
        else:
            moe_ids = expert_choice.to(torch.int32)

        return ZayaRouting(
            moe_weight=moe_weight,
            moe_ids=moe_ids,
            route_prob=route_prob,
            skip_ids=expert_choice,
            hidden_states_next=router_hidden_states_next,
        )


def mod_premask_experts(
    experts_out: torch.Tensor,
    indices: torch.Tensor,
    num_moe_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask the (per-rank, pre-all-reduce) expert output for the MOD skip path.

    Returns ``(mod_mask, masked_experts)`` where ``mod_mask`` is ``1`` for
    tokens routed to a real expert and ``0`` for tokens routed to the skip
    slot (``indices == num_moe_experts``), and
    ``masked_experts = mod_mask * experts_out``.

    The masking is applied *before* the cross-rank all-reduce so the single
    reduction yields ``mask · sum_r(partial_r) = mask · experts_out_full``
    without the replicated ``mod_out`` term being summed ``tp_size`` times.
    Pairs with :func:`mod_blend`, which adds the skip-path term back after the
    reduce. Kept as a free function so the MOD math is unit-testable without a
    live ``torch.distributed`` group.
    """
    mod_mask = (indices != num_moe_experts).to(experts_out.dtype)
    return mod_mask, mod_mask * experts_out


def mod_blend(
    masked_experts_reduced: torch.Tensor,
    mod_mask: torch.Tensor,
    mod_out: torch.Tensor,
) -> torch.Tensor:
    """Combine the already-all-reduced masked expert output with the skip path.

    ``mod_out`` (the skip-expert residual, ``hidden_states * prob``) is
    replicated on every rank, so it is folded in here -- after the reduce of
    ``masked_experts`` -- weighted by ``(1 - mod_mask)``. See
    :func:`mod_premask_experts`.

    The weighted add is one ``addcmul`` rather than a separate multiply and add,
    which is exact (identical operations, just fused) and drops one launch per
    MoE layer. Note the ``1.0 - mod_mask`` complement is NOT folded away by
    rewriting this as ``masked + mod_out - mask*mod_out``: that form is not
    exact, since ``(a + b) - b != a`` in floating point, and it must be for the
    masked tokens where the skip term is supposed to vanish entirely.
    """
    return torch.addcmul(masked_experts_reduced, 1.0 - mod_mask, mod_out)


class ZayaBlock(nn.Module):
    """ZAYA1 MoE mixer: ZayaRouter feeding FusedMoE, with optional MOD residual blend."""

    def __init__(
        self,
        config: ZayaConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.num_moe_experts = int(config.num_experts)
        self.mlp_expansion = int(config.zaya_mlp_expansion)
        self.topk = int(getattr(config, "moe_router_topk", 1))

        # Reduce over the *MoE* parallel groups (not the global TP group) so the
        # block is correct under expert parallelism (EP) and DP attention, where
        # the experts are sharded across the EP group and/or each DP replica owns
        # a different token slice. Under plain TP, ``moe_tp == global_tp`` and
        # ``ep == 1``, so this is behaviour-preserving.
        self.tp_size = get_parallel().moe_tp_size
        self.ep_size = get_parallel().moe_ep_size
        if self.tp_size > self.num_moe_experts:
            raise ValueError(
                f"MoE tensor parallel size {self.tp_size} is greater than the "
                f"number of experts {self.num_moe_experts}"
            )

        assert (
            config.activation_func == "swiglu"
        ), "ZayaBlock only supports SwiGLU activation"
        assert config.gated_linear_unit, "ZayaBlock requires gated_linear_unit=True"

        self.router = ZayaRouter(
            config=config,
            layer_id=layer_id,
            num_moe_experts=self.num_moe_experts,
            moe_router_topk=self.topk,
            mlp_expansion=self.mlp_expansion,
            quant_config=quant_config,
            prefix=add_prefix("router", prefix),
        )

        # ffn_hidden_size is the merged (gate+up) hidden dim; the per-side
        # intermediate is half.
        intermediate = int(config.ffn_hidden_size) // 2
        self.experts = get_moe_impl_class(quant_config)(
            num_experts=self.num_moe_experts,
            top_k=self.topk,
            hidden_size=config.hidden_size,
            intermediate_size=intermediate,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            activation="silu",
            reduce_results=False,
            # FusedMoE defaults to inplace=True, which on the triton runner aliases
            # the expert output onto ``hidden_states``. The MOD blend below reads
            # ``hidden_states`` *after* the experts run, so aliasing would feed it
            # the unreduced per-rank partial -- a different value on every TP rank,
            # silently diverging the replicated residual stream. The aiter runner
            # allocates a fresh output so this never fired in the measured config;
            # pinned here so the triton runner is not a correctness trap.
            inplace=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_router_hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.shape[0] == 0:
            return hidden_states, hidden_states.new_zeros((0, self.mlp_expansion))

        routing = self.router(hidden_states, prev_router_hidden_states)
        probs = routing.route_prob
        # The *unclamped* choice. MOD's skip predicate is
        # ``id == num_moe_experts``, which ``routing.moe_ids`` has clamped away.
        indices = routing.skip_ids
        router_hs_next = routing.hidden_states_next

        # ``moe_ids`` arrives int32 and already clamped into the real-expert
        # range, and ``moe_weight`` arrives fp32 -- the dtype the MoE runners
        # cast to anyway -- so nothing is left to do here. See ``ZayaRouting``.
        topk_out = StandardTopKOutput(
            topk_weights=routing.moe_weight,
            topk_ids=routing.moe_ids,
            router_logits=routing.moe_weight,
        )

        # ``mod_reachable``, not ``zaya_use_mod``: a checkpoint whose skip bias puts
        # the slot permanently out of reach turns this whole branch into two
        # elementwise launches per MoE layer computing an always-false predicate
        # (see ZayaRouter.fold_mod_reachability). The clamp that used to live here
        # is now done inside the fused router tail, which emits both the clamped
        # ``moe_ids`` and the unclamped ``skip_ids`` the predicate below needs.
        if self.router.mod_reachable:
            experts_out = self.experts(hidden_states, topk_out)
            # ``mod_out`` is computed identically on every rank that owns this
            # token (both ``hidden_states`` and ``probs`` are replicated across
            # the MoE-TP / MoE-EP groups). Fold the skip mask into the per-rank
            # partial experts output *before* the reduce so the reduction yields:
            #   sum_r(mask · partial_r) + (1 - mask) · mod_out
            # = mask · experts_out_full + (1 - mask) · mod_out
            # without double-counting ``mod_out``. The two steps are
            # ``mod_premask_experts`` / ``mod_blend`` so the math is testable
            # without a live distributed group.
            from sglang.kernels.ops.moe import zaya_mod as _mod

            if _mod.covered(experts_out, indices, hidden_states, probs):
                # Two kernels instead of six elementwise launches; each
                # recomputes the skip predicate from ``indices``, so no mask
                # tensor is materialized or threaded across the reduce.
                masked_experts = _mod.mod_premask(
                    experts_out, indices, self.num_moe_experts
                )
                masked_experts = self._reduce_experts(masked_experts)
                hidden_out = _mod.mod_blend(
                    masked_experts,
                    indices,
                    hidden_states,
                    probs,
                    self.num_moe_experts,
                )
            else:
                mod_out = hidden_states * probs
                mod_mask, masked_experts = mod_premask_experts(
                    experts_out, indices, self.num_moe_experts
                )
                masked_experts = self._reduce_experts(masked_experts)
                hidden_out = mod_blend(masked_experts, mod_mask, mod_out)
        else:
            hidden_out = self._reduce_experts(self.experts(hidden_states, topk_out))

        return hidden_out, router_hs_next

    def _reduce_experts(self, experts_out: torch.Tensor) -> torch.Tensor:
        """Combine partial expert outputs over the MoE parallel groups.

        Mirrors the canonical SGLang MoE reduce (cf. ``qwen3_moe``): first an
        all-reduce over the expert-parallel (EP) group, then over the
        MoE-tensor-parallel (TP) group. Under plain TP this is a single reduce
        over the global TP group; under EP / DP attention it stays scoped to the
        MoE groups and never spans the DP-attention replicas.

        Both legs go through ``should_skip_post_experts_all_reduce``, which is what
        makes an A2A backend safe here: an a2a combine already reduces partial
        expert outputs back to the source rank, so reducing again double-counts
        (and overflows bf16). ``dp_gather_required()`` separately drops the DP
        gather once a2a is on, so without this guard the two changes would not be
        consistent with each other.
        """
        if self.ep_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=False
        ):
            experts_out = moe_expert_parallel_all_reduce(experts_out)
        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=True
        ):
            experts_out = moe_tensor_model_parallel_all_reduce(experts_out)
        return experts_out


# ---------------------------------------------------------------------------
# Decoder layers
# ---------------------------------------------------------------------------


def dp_gather_required() -> bool:
    """Whether the MoE layers need to see the *global* token set.

    A token must be visible to every rank the expert reduce spans, so the gather
    is needed exactly when that reduce is wider than the attention-TP group, i.e.
    when it crosses DP replicas. When ``moe_tp == attn_tp`` (e.g. --moe-dp-size
    equal to the attention DP size) each replica owns a self-contained MoE over
    its own ranks, the token is already replicated across them by
    ``attn_tp_all_reduce``, and the gather/scatter pair is pure overhead.

    Compare against the width of the group ``ZayaBlock._reduce_experts`` actually
    reduces over -- expert-parallel AND MoE-tensor-parallel -- not moe_tp alone.
    Under EP (``--ep-size 8``) moe_tp collapses to 1 while the reduce still spans
    all 8 ranks, so keying off moe_tp alone would skip a gather that is required
    and silently drop every token the rank does not own.
    """
    parallel = get_parallel()
    moe_reduce_width = parallel.moe_ep_size * parallel.moe_tp_size
    return (
        parallel.attn_dp_size > 1
        and moe_reduce_width > parallel.attn_tp_size
        and get_moe_a2a_backend().is_none()
    )


class GlobalResidualLayout(msgspec.Struct, frozen=True):
    """Where this DP replica's rows sit inside the global DP buffer.

    Present only on the global-residual dataflow (see
    ``SGLANG_OPT_ZAYA_GLOBAL_RESIDUAL``), where the fp32 residual stream and the
    normed hidden states are held in the global layout -- every rank holds every
    replica's rows -- rather than the DP-local one. That lets a single
    ``dp_gather_partial`` of the o_proj partials do the work of both the
    attention-TP all-reduce and the MoE layer's separate gather, and removes the
    MoE scatter: three collectives per attention+MoE layer pair become two, and
    the norms pay for it by running over every replica's rows.
    """

    local_start: int
    local_len: int

    def local_view(self, global_rows: torch.Tensor) -> torch.Tensor:
        """This replica's rows of a global-layout tensor, as a view.

        A row slice of a contiguous 2D tensor is itself contiguous, which the
        attention kernels and the gather's ``local_tokens`` assert require.
        """
        return global_rows[self.local_start : self.local_start + self.local_len]


_logged_dataflow_decisions: set[str] = set()


def _log_dataflow_decision(message: str) -> None:
    """Log a dataflow decision once per distinct message.

    Deduping on the *whole* message, shapes included, is deliberate: an earlier
    A/B in this campaign was declined by a layout precondition and produced
    baseline-identical numbers that read as a clean null result. A per-reason
    dedupe would not have caught it either, because the decline happened first at
    prefill and would have masked the decode decision behind it.
    """
    if message not in _logged_dataflow_decisions:
        _logged_dataflow_decisions.add(message)
        logger.info(message)


def global_residual_layout() -> Optional[GlobalResidualLayout]:
    """Layout for this forward, or None to run the DP-local dataflow."""
    if not envs.SGLANG_OPT_ZAYA_GLOBAL_RESIDUAL.get():
        return None
    if not dp_gather_required():
        _log_dataflow_decision(
            "zaya global residual: declined, the expert reduce does not span DP "
            "replicas so there is no gather to fold into"
        )
        return None
    # The padded per-rank token counts: the same list that sized the global
    # buffer (``set_dp_buffer_len``) and that ``get_dp_local_info`` cumsums on
    # the device to place the gather's memcpy. Deriving the CPU row arithmetic
    # from that one source is what keeps it from drifting out of agreement with
    # where the gather actually writes -- including inside a captured CUDA graph,
    # where every rank is padded to the bucket's token count.
    sizes = get_dp_global_num_tokens()
    if sizes is None:
        _log_dataflow_decision(
            "zaya global residual: declined, DP buffer metadata is unset"
        )
        return None
    rank = get_attention_dp_rank()
    layout = GlobalResidualLayout(local_start=sum(sizes[:rank]), local_len=sizes[rank])
    _log_dataflow_decision(
        f"zaya global residual: active, rows per rank {sizes}, this rank "
        f"[{layout.local_start}, {layout.local_start + layout.local_len})"
    )
    return layout


def _residual_scale_norm(
    res_scale: Optional[ResidualScaling],
    norm: nn.Module,
    residual: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    target_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a layer's opening ``res_scale -> accumulate -> norm`` chain.

    Prefers the fused kernel and falls back to the torch chain when it cannot
    serve the shapes -- notably before ``fold_scales`` has run, so CPU unit tests
    keep exercising the reference path. Returns ``(normed_hidden, new_residual)``.
    """
    from sglang.kernels.ops.elementwise import zaya_residual_norm as _rn

    folded = res_scale is not None and res_scale._scales_folded
    norm_weight = norm.weight if isinstance(norm, RMSNorm) else None
    if folded and _rn.covered(hidden_states, residual, norm_weight, folded):
        return _rn.residual_scale_accumulate_norm(
            hidden_states,
            residual,
            hs_scale=res_scale.hidden_states_scale_f32,
            hs_bias_scaled=res_scale.hidden_states_bias_scaled,
            res_scale=(
                res_scale.residual_scale_f32
                if (res_scale.has_residual and residual is not None)
                else None
            ),
            res_bias_scaled=(
                res_scale.residual_bias_scaled
                if (res_scale.has_residual and residual is not None)
                else None
            ),
            norm_weight=norm_weight,
            eps=norm.variance_epsilon,
            out_dtype=target_dtype,
        )

    if res_scale is not None:
        residual, hidden_states = res_scale(residual, hidden_states)
    if residual is not None:
        residual = residual + hidden_states
    else:
        residual = hidden_states.float()
    return _apply_norm_with_fp32_residual(norm, residual, target_dtype), residual


class ZayaDecoderATTLayer(nn.Module):
    """Attention decoder layer: ``res_scale → input_norm → ZayaAttention``."""

    def __init__(
        self,
        config: ZayaConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.self_attn = ZayaAttention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.input_norm = self._build_norm(config)
        if config.scale_residual_merge:
            self.res_scale = ResidualScaling(config, layer_id)
        else:
            self.res_scale = None

    @staticmethod
    def _build_norm(config: ZayaConfig) -> nn.Module:
        if config.normalization == "RMSNorm":
            return RMSNorm(config.hidden_size, eps=config.norm_epsilon)
        if config.normalization == "LayerNorm":
            return nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)
        raise ValueError(f"Unsupported normalization: {config.normalization}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        prev_router_hidden_states: Optional[torch.Tensor] = None,
        global_residual: Optional[GlobalResidualLayout] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        target_dtype = (
            self.input_norm.weight.dtype
            if isinstance(self.input_norm, RMSNorm)
            else hidden_states.dtype
        )
        hidden_states, residual = _residual_scale_norm(
            self.res_scale, self.input_norm, residual, hidden_states, target_dtype
        )
        if global_residual is None:
            hidden_states = self.self_attn(hidden_states, positions, forward_batch)
            return hidden_states, residual, prev_router_hidden_states

        # Global-residual dataflow: the residual stream -- and so the normed
        # hidden states just computed -- cover every replica's rows, but attention
        # is DP-local, so it runs on this replica's slice against its own
        # positions and conv state. The partial gather then both sums the
        # attention-TP partials and lifts the result back to the global layout,
        # which is what the next (MoE) layer needs, so it needs no gather of its
        # own and no scatter afterwards.
        #
        # The gather sits here rather than inside ``ZayaAttention.forward``
        # deliberately: a replica idle this step returns early from that function
        # before o_proj, and it must still take part in the collective or the
        # ranks that are busy hang.
        partial = self.self_attn(
            global_residual.local_view(hidden_states),
            positions,
            forward_batch,
            reduce_output=False,
        )
        #
        # ``dp_gather_partial_out`` rather than ``dp_gather_partial``: under
        # sum_len padding the all-reduce is out-of-place, so its output buffer
        # already holds the gathered rows and copying them back into the staging
        # buffer is a wasted launch per attention layer (60 of them per decode
        # step here, where every launch is ~1.5-2.5us of a launch-bound step).
        # The staging buffer is a fresh per-layer allocation with no other
        # reader, so nothing needs the result to land there.
        staging = get_global_dp_buffer(get_tp_group())
        hidden_states = dp_gather_partial_out(staging, partial, forward_batch)
        return hidden_states, residual, prev_router_hidden_states


class ZayaDecoderMLPLayer(nn.Module):
    """MoE decoder layer: ``res_scale → input_norm → ZayaBlock``."""

    def __init__(
        self,
        config: ZayaConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.zaya_block = ZayaBlock(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("zaya_block", prefix),
        )
        self.input_norm = ZayaDecoderATTLayer._build_norm(config)
        if config.scale_residual_merge:
            self.res_scale = ResidualScaling(config, layer_id)
        else:
            self.res_scale = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        prev_router_hidden_states: Optional[torch.Tensor] = None,
        global_residual: Optional[GlobalResidualLayout] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        target_dtype = (
            self.input_norm.weight.dtype
            if isinstance(self.input_norm, RMSNorm)
            else hidden_states.dtype
        )
        hidden_states, residual = _residual_scale_norm(
            self.res_scale, self.input_norm, residual, hidden_states, target_dtype
        )
        if global_residual is not None:
            # Global-residual dataflow: the preceding attention layer's partial
            # gather already left the hidden states in the global layout, so the
            # experts can run on them directly. This is where that dataflow pays
            # off -- no gather here, and no scatter after, because the residual
            # this feeds back into is global too.
            hidden_states, prev_router_hidden_states = self.zaya_block(
                hidden_states, prev_router_hidden_states
            )
            return hidden_states, residual, prev_router_hidden_states

        # DP attention: the attention layers kept each DP replica's tokens
        # local, but the experts (and their EP / MoE-TP all-reduce) must run over
        # the *full* token set. Gather the DP-local normed hidden states into the
        # global sequence -- tokens then become replicated across the whole TP
        # group, which is exactly the layout ``ZayaBlock``'s reduce expects --
        # run the experts, then scatter the per-token result back to this
        # replica's slice. The fp32 ``residual`` stays DP-local;
        # ``prev_router_hidden_states`` stays in the gathered (global) layout and
        # is threaded through the MoE layers (every gather uses the same global
        # token order, so router state and hidden states stay aligned).
        #
        # Use the *replicate* gather (not ``dp_gather_partial``): ``self_attn``
        # already ran ``attn_tp_all_reduce``, so within each DP replica the normed
        # hidden states are identical across the attention-TP ranks. The replicate
        # gather takes each replica's slice from its attn-TP rank 0 only; the
        # partial gather instead sums every attn-TP rank into the same slot, which
        # multiplies the tokens by ``attn_tp_size`` -- a no-op at attn_tp=1 (so
        # tp=2/dp=2 was correct) but corrupting every value once attn_tp>1 (e.g.
        # tp=4/dp=2 on 74B doubled them, producing garbage output).
        # Whether the gather is needed at all is ``dp_gather_required``; skipping
        # it when the MoE and attention groups coincide matters a lot, because
        # profiling tp=8/dp=4 on the 74B put ~83% of decode GPU time in
        # collectives, of which the per-MoE-layer all-gather alone was 29%.
        use_dp_gather = dp_gather_required()
        if use_dp_gather:
            hidden_states, local_hidden_states = (
                get_global_dp_buffer(get_tp_group()),
                hidden_states,
            )
            dp_gather_replicate(hidden_states, local_hidden_states, forward_batch)
        hidden_states, prev_router_hidden_states = self.zaya_block(
            hidden_states, prev_router_hidden_states
        )
        if use_dp_gather:
            hidden_states, global_hidden_states = (
                get_local_dp_buffer(get_tp_group()),
                hidden_states,
            )
            dp_scatter(hidden_states, global_hidden_states, forward_batch)
        return hidden_states, residual, prev_router_hidden_states


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


def _build_layer(
    layer_id: int,
    config: ZayaConfig,
    quant_config: Optional[QuantizationConfig],
    prefix: str,
) -> nn.Module:
    # Even layer ids are attention, odd layer ids are MoE. This matches the HF
    # checkpoint keys: ``model.layers.<2k>.self_attn.*`` (CCA) versus
    # ``model.layers.<2k+1>.zaya_block.*`` (MoE).
    if layer_id % 2 == 0:
        return ZayaDecoderATTLayer(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
        )
    return ZayaDecoderMLPLayer(
        config=config,
        layer_id=layer_id,
        quant_config=quant_config,
        prefix=prefix,
    )


class ZayaModel(nn.Module):
    def __init__(
        self,
        config: ZayaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            # Under DP attention each replica embeds its own DP-local token slice,
            # so the vocab is sharded over the *attention* TP sub-group (and
            # replicated across DP replicas) via ``use_attn_tp_group``. Sharding
            # over the global TP group instead would make the embedding reduce
            # span DP ranks and sum embeddings of unrelated tokens. With DP
            # attention off, ``use_attn_tp_group`` is False and this is the plain
            # global-TP vocab-parallel path.
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                use_attn_tp_group=is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: _build_layer(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )

        if self.pp_group.is_last_rank:
            self.final_norm = ZayaDecoderATTLayer._build_norm(config)
            if config.scale_residual_merge:
                self.res_scale = ResidualScaling(config, config.num_hidden_layers)
            else:
                self.res_scale = None
        else:
            self.final_norm = PPMissingLayer()
            self.res_scale = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        global_residual = global_residual_layout()

        if self.pp_group.is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
            if global_residual is not None:
                # Seed the stream in the global layout. Layer 0 has no incoming
                # residual, so its residual *is* these embeddings -- gathering
                # them once here is what makes the residual global, and from then
                # on each attention layer's partial gather keeps it that way at no
                # extra cost. This is the one collective the dataflow adds, against
                # the 60 attention-TP all-reduces it removes.
                global_hidden = get_global_dp_buffer(get_tp_group())
                dp_gather_replicate(global_hidden, hidden_states, forward_batch)
                hidden_states = global_hidden
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        prev_router_hidden_states: Optional[torch.Tensor] = None
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual, prev_router_hidden_states = layer(
                hidden_states=hidden_states,
                residual=residual,
                positions=positions,
                forward_batch=forward_batch,
                prev_router_hidden_states=prev_router_hidden_states,
                global_residual=global_residual,
            )

        # Radix mamba-cache checkpoint, decode side (extra_buffer strategy
        # only). Snapshots every CCA layer's conv_state + prev_hs into the
        # per-request track slots in ONE launch, and must run after the last
        # layer has updated its state. It stays inside the captured decode
        # graph -- see ShortConvAttnBackend.track_conv_states_decode. No-op
        # under ``no_buffer``.
        get_attn_backend().track_conv_states_decode(forward_batch)

        if not self.pp_group.is_last_rank:
            # Both streams stay global across the PP boundary; the next stage
            # derives the same layout and carries on.
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )

        if global_residual is not None:
            # Back to DP-local for the final norm and the logits: this replica
            # only produces logits for its own tokens, so narrowing here also
            # keeps the last norm off the other replicas' rows.
            hidden_states = global_residual.local_view(hidden_states)
            if residual is not None:
                residual = global_residual.local_view(residual)

        if self.res_scale is not None:
            residual, hidden_states = self.res_scale(residual, hidden_states)
        target_dtype = (
            self.final_norm.weight.dtype
            if isinstance(self.final_norm, RMSNorm)
            else hidden_states.dtype
        )
        if residual is not None:
            merged = hidden_states.float() + residual.float()
        else:
            merged = hidden_states.float()
        hidden_states = _apply_norm_with_fp32_residual(
            self.final_norm, merged, target_dtype
        )
        return hidden_states


class ZayaForCausalLM(nn.Module):
    def __init__(
        self,
        config: ZayaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        self.model = ZayaModel(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        if self.pp_group.is_last_rank:
            # The lm_head vocab shard group must match what ``LogitsProcessor``
            # gathers over, which is the attention-TP group iff
            # ``enable_dp_lm_head``. ZAYA1 ties the head to ``embed_tokens``,
            # whose shard group is the attention-TP group under DP attention, so
            # the two only line up when ``enable_dp_lm_head`` tracks
            # ``enable_dp_attention`` -- ``_zaya_overrides`` forces that for tied
            # checkpoints (otherwise ``tie_weights`` would alias a
            # ``vocab/attn_tp``-row weight into a head sharded ``vocab/tp``).
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                bias=bool(getattr(config, "lm_head_bias", False)),
                quant_config=None,
                use_attn_tp_group=get_server_args().enable_dp_lm_head,
                prefix=add_prefix("lm_head", prefix),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config)

    def get_attention_sliding_window_size(self) -> Optional[int]:
        """Global sliding-window size for SWA-enabled checkpoints (else None).

        ``ModelRunner`` calls this to size the attention backend's SWA metadata
        buffers; returning None on base checkpoints leaves the runtime in the
        plain full-attention path. The per-layer window is selected inside
        ``ZayaAttention`` via ``RadixAttention.sliding_window_size``.
        """
        return self.config.get_attention_sliding_window_size()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            inputs_embeds=inputs_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if not self.pp_group.is_last_rank:
            return hidden_states
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    # ---------------- weight loading ----------------

    _EXPERT_RE = re.compile(
        r"^(.*\.zaya_block\.experts)\.local_experts\.(\d+)\.(linear_fc1|linear_fc2)\.weight$"
    )

    # The checkpoint keeps q and k as separate projections; the runtime merges
    # them into one ``linear_qk`` (see CCA.__init__), so each maps onto a shard of
    # the merged parameter -- q is shard 0, k is shard 1.
    _MERGED_QK_RE = re.compile(r"^(.*\.qkv)\.linear_(q|k)\.(weight|bias)$")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load an HF ZAYA1 safetensors checkpoint into the SGLang module tree.

        Most keys map 1:1 because the module names already mirror the HF
        checkpoint layout. Two cases need rewriting:

        1. ``self_attn.qkv.{linear_q, linear_k, conv_qk.{0,1}, val_proj{1,2}, temp}``
           lands directly on the registered ``CCA`` submodule (which is named
           ``qkv`` exactly to keep this mapping trivial).
        2. ``zaya_block.experts.local_experts.<i>.linear_fc1.weight`` (gate
           and up projections concatenated along dim 0) is split and routed
           to FusedMoE shards ``w1`` (first half) and ``w3`` (second half);
           ``linear_fc2.weight`` becomes the FusedMoE ``w2`` shard.
        """
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())
        # ``balancing_biases`` is a persistent buffer; FusedMoE may also expose
        # buffers. Expose them all through ``params_dict`` so that the regular
        # ``default_weight_loader`` can write to them.
        for key, buf in buffers_dict.items():
            params_dict.setdefault(key, buf)

        fused_moe_modules: dict[str, nn.Module] = {}
        for name, module in self.named_modules():
            if module.__class__.__name__ == "FusedMoE" or hasattr(module, "w13_weight"):
                fused_moe_modules[name] = module

        loaded_params: set[str] = set()

        for ckpt_name, loaded_weight in weights:
            # Skip keys that have no runtime counterpart in this model.
            if ckpt_name.startswith("lm_head") and self.config.tie_word_embeddings:
                continue
            if "rotary_emb" in ckpt_name:
                continue

            qk_match = self._MERGED_QK_RE.match(ckpt_name)
            if qk_match is not None:
                cca_prefix, which, kind = qk_match.groups()
                param_name = f"{cca_prefix}.linear_qk.{kind}"
                param = params_dict.get(param_name)
                if param is None:
                    logger.warning("No param %s for %s", param_name, ckpt_name)
                    continue
                # Both the merged column-parallel loader and the replicated
                # stand-in installed for tp=1 take (param, weight, shard_id).
                param.weight_loader(param, loaded_weight, 0 if which == "q" else 1)
                loaded_params.add(param_name)
                continue

            match = self._EXPERT_RE.match(ckpt_name)
            if match is not None:
                experts_prefix = match.group(
                    1
                )  # e.g. model.layers.1.zaya_block.experts
                expert_id = int(match.group(2))
                kind = match.group(3)
                moe_module = fused_moe_modules.get(experts_prefix)
                if moe_module is None:
                    logger.warning(
                        "FusedMoE module %s not found; skipping %s",
                        experts_prefix,
                        ckpt_name,
                    )
                    continue
                weight_loader = moe_module.weight_loader
                if kind == "linear_fc1":
                    param_name = f"{experts_prefix}.w13_weight"
                    param = params_dict.get(param_name)
                    if param is None:
                        logger.warning("No param %s for %s", param_name, ckpt_name)
                        continue
                    half = loaded_weight.shape[0] // 2
                    weight_loader(
                        param,
                        loaded_weight[:half],
                        ckpt_name,
                        shard_id="w1",
                        expert_id=expert_id,
                    )
                    weight_loader(
                        param,
                        loaded_weight[half:],
                        ckpt_name,
                        shard_id="w3",
                        expert_id=expert_id,
                    )
                    loaded_params.add(param_name)
                else:  # linear_fc2
                    param_name = f"{experts_prefix}.w2_weight"
                    param = params_dict.get(param_name)
                    if param is None:
                        logger.warning("No param %s for %s", param_name, ckpt_name)
                        continue
                    weight_loader(
                        param,
                        loaded_weight,
                        ckpt_name,
                        shard_id="w2",
                        expert_id=expert_id,
                    )
                    loaded_params.add(param_name)
                continue

            # HF stores CCA tensors under ``self_attn.qkv.*``, which already
            # matches our submodule registration, so no rename is needed.
            if ckpt_name not in params_dict:
                # ``conv_qk`` is an ``nn.Sequential`` of two ``nn.Conv1d``,
                # whose keys end in ``.0.{weight,bias}`` / ``.1.{weight,bias}``
                # and are exposed through ``named_parameters()`` automatically.
                # Anything else is genuinely unknown – warn and skip.
                logger.warning(
                    "WARNING: checkpoint key %s has no matching parameter; skipping",
                    ckpt_name,
                )
                continue

            param = params_dict[ckpt_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(ckpt_name)

        self.fold_decode_constants()
        return loaded_params

    def fold_decode_constants(self) -> None:
        """Precompute the per-layer constants derived from loaded weights.

        Must run after every weight load (including reloads) and before the
        first forward, since the forward paths read the folded buffers rather
        than recomputing from the parameters. Kept separate from
        ``load_weights`` so a caller that populates weights another way can
        still refresh them.
        """
        for module in self.modules():
            if isinstance(module, ResidualScaling):
                module.fold_scales()
            elif isinstance(module, CCA):
                module.fold_decode_conv()
                module.fold_qk_scales()
            elif isinstance(module, ZayaRouter):
                module.fold_mod_reachability()


EntryClass = ZayaForCausalLM
