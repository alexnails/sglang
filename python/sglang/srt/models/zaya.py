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
- Per-request CCA state (``conv_state`` + ``prev_hs``) lives in SGLang's
  centralized ``MambaPool`` inside ``HybridReqToTokenPool``. The per-request
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
from typing import List, Optional, Tuple

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
    dp_gather_replicate,
    dp_gather_replicate_async,
    dp_reduce_scatter_tensor,
    dp_scatter,
    get_attention_dp_rank,
    get_global_dp_buffer,
    get_global_dp_buffer_len,
    get_local_dp_buffer,
    is_dp_attention_enabled,
    is_dp_max_padding,
    prewarm_dp_gather_async,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import get_global_server_args
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
# ``qk = [W_q hs || W_k hs]`` plus a one-token ``prev_hs`` lag for val_proj2.
# The per-request conv state lives in the centralized MambaPool; the backend
# (ShortConvAttnBackend) hands out the slot indices + prefix flags and CCA runs
# these functions against them. ``conv_qk`` is the module's two-stage conv;
# both functions mutate ``conv_state`` / ``prev_hs_state`` in place and return
# ``(qk_out, v2_input)`` -- the conv output ``[T, in_out_ch]`` and the (shifted)
# ``val_proj2`` input ``[T, hidden_size]``.
# ---------------------------------------------------------------------------


def cca_extend(
    qk: torch.Tensor,
    hidden_states: torch.Tensor,
    conv_qk: nn.Module,
    conv_state: torch.Tensor,
    prev_hs_state: torch.Tensor,
    slot_ids: List[int],
    has_prefix: List[bool],
    extend_seq_lens_cpu: List[int],
    total_padding: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prefill / extend conv-state step (v1, pure torch).

    Walks each request in the batch, applies ``conv_qk`` with the request's own
    initial state (zeros on a fresh first chunk, the cached ``conv_state`` slot
    otherwise), writes the updated ``conv_state`` / ``prev_hs_state`` back, and
    returns the concatenated ``(qk_out, v2_input)`` in the original token layout.

    ``slot_ids`` is the host mirror of the per-request MambaPool slot indices and
    ``has_prefix[i]`` is ``True`` when request ``i`` resumes a cached prefix.

    The Triton swap (:func:`cca_conv1d_fn`) removes this per-request loop.
    """
    dtype = hidden_states.dtype
    if total_padding is None:
        total_padding = conv_state.shape[-1]
    in_out_ch = qk.shape[-1]
    hidden_size = hidden_states.shape[-1]

    qk_out = torch.empty_like(qk)
    v2_input = torch.empty_like(hidden_states)

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

        packed_out = conv_qk(packed)  # [1, C, offsets_in[-1] - pad]

        start = 0
        for i, s in enumerate(seq_lens):
            end = start + s
            a_i = offsets_in[i]
            qk_out[start:end] = packed_out[0, :, a_i : a_i + s].transpose(0, 1)
            new_state = packed[0, :, a_i + s : a_i + s + pad]
            conv_state[slot_ids[i]] = new_state.to(conv_state.dtype)

            hs_cur = hidden_states[start:end]
            first = hidden_states.new_zeros((1, hidden_size))
            v2_input[start:end] = torch.cat([first, hs_cur[:-1]], dim=0)
            prev_hs_state[slot_ids[i]] = (
                hs_cur[-1].unsqueeze(-1).to(prev_hs_state.dtype)
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

            out = conv_qk(padded)  # [1, C, S_cur]
            qk_out[start:end] = out.squeeze(0).transpose(0, 1)

            new_state = padded[..., -total_padding:]
            conv_state[slot] = new_state.squeeze(0).to(conv_state.dtype)

            hs_cur = hidden_states[start:end]
            if prefix:
                first = prev_hs_state[slot].squeeze(-1).to(dtype).unsqueeze(0)
            else:
                first = hidden_states.new_zeros((1, hidden_size))
            v2_input[start:end] = torch.cat([first, hs_cur[:-1]], dim=0)

            prev_hs_state[slot] = hs_cur[-1].unsqueeze(-1).to(prev_hs_state.dtype)
            start = end

    return qk_out, v2_input


def cca_decode(
    qk: torch.Tensor,
    hidden_states: torch.Tensor,
    conv_qk: nn.Module,
    conv_state: torch.Tensor,
    prev_hs_state: torch.Tensor,
    mamba_indices: torch.Tensor,
    total_padding: Optional[int] = None,
    decode_conv_weight: Optional[torch.Tensor] = None,
    decode_conv_bias: Optional[torch.Tensor] = None,
    decode_conv_groups: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-token decode conv-state step (v1, pure torch).

    Gathers each request's cached ``conv_state`` / ``prev_hs_state`` via
    ``index_select``, applies the conv over the ``[T, C, total_padding + 1]``
    window, and scatters the updated state back with ``index_copy_``. All ops are
    on-device (``mamba_indices`` is a device ``long`` tensor), so this stays
    CUDA-graph capturable. Returns ``(qk_out, prev_hs)`` where ``prev_hs`` is the
    previous hidden state feeding ``val_proj2``.

    When ``decode_conv_weight`` / ``_bias`` / ``_groups`` are supplied (see
    :meth:`CCA.fold_decode_conv`) the two conv stages are evaluated as a single
    grouped matmul; otherwise ``conv_qk`` is run as-is. The Triton swap is
    :func:`cca_conv1d_update`.
    """
    dtype = hidden_states.dtype
    if total_padding is None:
        total_padding = conv_state.shape[-1]

    from sglang.kernels.ops.attention import cca_state_step as _state_step

    if _state_step.covered(
        qk, hidden_states, conv_state, prev_hs_state, mamba_indices, total_padding
    ):
        # One kernel for the two gathers, the concat and the two scatters.
        padded, prev_hs = _state_step.cca_state_step(
            qk,
            hidden_states,
            conv_state,
            prev_hs_state,
            mamba_indices,
            total_padding,
        )
        qk_out = _cca_decode_conv(
            padded,
            conv_qk,
            decode_conv_weight,
            decode_conv_bias,
            decode_conv_groups,
        )
        return qk_out, prev_hs

    left_pad = conv_state.index_select(0, mamba_indices).to(dtype)
    cur = qk.unsqueeze(-1)  # [T, C, 1]
    padded = torch.cat([left_pad, cur], dim=-1)  # [T, C, total_padding + 1]
    qk_out = _cca_decode_conv(
        padded, conv_qk, decode_conv_weight, decode_conv_bias, decode_conv_groups
    )

    new_state = padded[..., -total_padding:]
    conv_state.index_copy_(0, mamba_indices, new_state.to(conv_state.dtype))

    # Read the previous hidden state (val_proj2 input) BEFORE overwriting the
    # slot with the current token.
    prev_hs = prev_hs_state.index_select(0, mamba_indices).squeeze(-1).to(dtype)
    prev_hs_state.index_copy_(
        0, mamba_indices, hidden_states.unsqueeze(-1).to(prev_hs_state.dtype)
    )
    return qk_out, prev_hs


# Fused kernel seam (TODO) -- perf swap for the v1 torch paths above. These
# mirror the ``causal_conv1d_fn`` / ``causal_conv1d_update`` contract but for
# CCA's two-stage *grouped* conv (conv_qk[0] depthwise + conv_qk[1] grouped
# per-head), which the stock depthwise ``causal_conv1d`` cannot express. Once
# implemented they replace the per-request loop in ``cca_extend`` and the
# separate gather/conv/scatter launches in ``cca_decode`` with a single
# index-driven kernel. Same ``(qk_out, v2_input)`` return contract.


def cca_conv1d_fn(*args, **kwargs):
    raise NotImplementedError(
        "Fused CCA prefill conv-with-state kernel not implemented yet; "
        "the model uses cca_extend (v1 torch) in the meantime."
    )


def cca_conv1d_update(*args, **kwargs):
    raise NotImplementedError(
        "Fused CCA decode conv-with-state kernel not implemented yet; "
        "the model uses cca_decode (v1 torch) in the meantime. The state "
        "plumbing around the conv is already fused -- see "
        "sglang.kernels.ops.attention.cca_state_step -- so what remains here is "
        "folding the grouped matmul itself into that kernel."
    )


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
    """
    if decode_conv_weight is not None:
        # [T, C, taps] -> [T, G, Cg*taps] (the trailing (Cg, taps) dims flatten
        # in place, matching how fold_decode_conv laid out the weight) -> one
        # grouped matmul -> [T, C].
        num_tokens = padded.shape[0]
        grouped = padded.reshape(num_tokens, decode_conv_groups, -1)
        return (
            torch.einsum("tgk,gok->tgo", grouped, decode_conv_weight) + decode_conv_bias
        ).reshape(num_tokens, -1)
    return conv_qk(padded).squeeze(-1)


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
        ch_per_group = self.in_out_ch // self.decode_conv_groups
        self.register_buffer(
            "decode_conv_weight",
            torch.zeros(
                self.decode_conv_groups,
                ch_per_group,
                ch_per_group * self.decode_conv_taps,
            ),
            persistent=False,
        )
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Blend post-conv q/k with the grouped pre-conv means, then RMS-normalize.

        Prefers the fused Triton kernel and falls back to the two torch helpers
        when it cannot serve the shapes (see ``cca_qk_mix.covered``) -- notably
        before ``fold_qk_scales`` has run, so CPU unit tests keep the torch path.
        """
        from sglang.kernels.ops.attention import cca_qk_mix as _cca_qk_mix

        scale = self.qk_k_scale if self._qk_scales_folded else None
        if _cca_qk_mix.covered(
            qk_out,
            query_pre_flat,
            key_base_flat,
            scale,
            num_q_heads=self.num_q_heads,
            num_k_heads=self.num_k_heads,
            head_dim=self.head_dim,
        ):
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
            )

        query, key = self._add_grouped_qk_means(
            query_conv, key_conv, query_pre, key_base
        )
        query, key = self._normalize_qk(query, key)
        return query.to(out_dtype), key.to(out_dtype)

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

        self.decode_conv_weight.copy_(
            folded.reshape(groups, cg, cg * taps).to(self.decode_conv_weight.dtype)
        )
        self.decode_conv_bias.copy_(
            (b1 + (w1.sum(dim=3) * b0[:, None, :]).sum(dim=2)).to(
                self.decode_conv_bias.dtype
            )
        )
        # [G, Co_g, Ci_g, taps] -> [G*Co_g, Ci_g, taps] == [C, C/groups, kernel]
        self.fold_conv1d_weight.copy_(
            folded.reshape(groups * cg, cg, taps).to(self.fold_conv1d_weight.dtype)
        )
        self._decode_conv_folded = True

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

    def _compute_value_per_rank(
        self, hidden_states: torch.Tensor, v2_input: torch.Tensor
    ) -> torch.Tensor:
        """This rank's V heads, running only the projections that feed them.

        ``val_proj1`` supplies the first ``num_k_heads_full // 2`` K heads and
        ``val_proj2`` the rest (the HF layout, see their construction). When this
        rank's head range falls entirely inside one of those, the other
        projection is dead work: skipping it drops a GEMM, the ``cat`` and the
        ``contiguous`` copy that slicing the concatenated tensor needed. That is
        the common case for ZAYA1 -- ``num_query_groups`` is 2, so at
        ``attn_tp == 2`` each rank owns exactly one projection's output.

        Falls back to computing both and slicing when the range straddles the
        boundary (or the split is not head-aligned), which is also the tp=1 path.
        """
        head_dim = self.head_dim
        start, end = self.k_head_start, self.k_head_end
        v1_heads = self.num_k_heads_full // 2
        aligned = self.num_k_heads_full % 2 == 0

        if aligned and end <= v1_heads:
            value, _ = self.val_proj1(hidden_states)
            value = value[:, start * head_dim : end * head_dim]
        elif aligned and start >= v1_heads:
            value, _ = self.val_proj2(v2_input)
            value = value[
                :, (start - v1_heads) * head_dim : (end - v1_heads) * head_dim
            ]
        else:
            v1, _ = self.val_proj1(hidden_states)
            v2, _ = self.val_proj2(v2_input)
            value = torch.cat([v1, v2], dim=-1)[:, start * head_dim : end * head_dim]

        return value.reshape(value.shape[0], self.num_k_heads, head_dim)

    def _forward_no_state(
        self, hs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reference path: process the entire ``hs`` of shape ``[S, H]`` with
        a zero initial conv state and a zero ``prev_hs``.

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project ``hidden_states`` into ``(q, k, v)`` honoring per-request state.

        The per-request conv-state plumbing (slot gather/scatter, prefix mask,
        cuda-graph buffers) is owned by :class:`ShortConvAttnBackend
        <sglang.srt.layers.attention.linear.short_conv_backend.ShortConvAttnBackend>`,
        reached via ``get_attn_backend().conv_state_metadata``; CCA runs its own
        two-stage grouped conv (:func:`cca_extend` / :func:`cca_decode`) against
        that handle, so this module holds no pool access. Those functions return
        the conv output ``qk_out`` and the ``val_proj2`` input ``v2_input`` (the
        shifted / previous hidden state), updating the ``conv_state`` /
        ``prev_hs`` pool slots in place.

        ``q`` / ``k`` / ``v`` are all returned in ``hidden_states``' dtype. The
        blend and normalize still accumulate in fp32 internally, but the result is
        stored at model precision: the caller rounded it there immediately anyway,
        so materializing an fp32 copy first only cost a per-layer ``copy_``.

        Shapes::

            q : [T, num_q_heads, head_dim]
            k : [T, num_k_heads, head_dim]
            v : [T, num_k_heads, head_dim]
        """
        if hidden_states.shape[0] == 0:
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
        # conv against it and gets back the conv output + val_proj2 input, with
        # the conv_state / prev_hs pool slots updated in place.
        meta = get_attn_backend().conv_state_metadata(self.layer_id, forward_batch)
        conv_state = meta.layer_cache.conv[0]
        prev_hs_state = meta.layer_cache.conv[1]
        if forward_batch.forward_mode.is_decode_or_idle():
            qk_out, v2_input = cca_decode(
                qk,
                hidden_states,
                self.conv_qk,
                conv_state,
                prev_hs_state,
                meta.cache_indices,
                self.total_padding,
                decode_conv_weight=(
                    self.decode_conv_weight if self._decode_conv_folded else None
                ),
                decode_conv_bias=self.decode_conv_bias,
                decode_conv_groups=self.decode_conv_groups,
            )
        else:
            qk_out, v2_input = cca_extend(
                qk,
                hidden_states,
                self.conv_qk,
                conv_state,
                prev_hs_state,
                meta.slot_ids_cpu,
                meta.has_prefix_cpu,
                forward_batch.extend_seq_lens_cpu,
                self.total_padding,
            )

        query_conv = qk_out[:, : self.latent_q_dim].view(
            T, self.num_q_heads, self.head_dim
        )
        key_conv = qk_out[:, self.latent_q_dim :].view(
            T, self.num_k_heads, self.head_dim
        )

        # Emit the model dtype straight away: the caller rounded the fp32 result
        # to it immediately, so this is the same single rounding minus two
        # per-layer copies (aten::copy_ was the largest single op in the profile).
        query, key = self._mix_and_normalize_qk(
            qk_out,
            q_raw,
            k_raw,
            query_conv,
            key_conv,
            query_pre,
            key_base,
            out_dtype=hidden_states.dtype,
        )

        value = self._compute_value_per_rank(hidden_states, v2_input)
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
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
        q, k, v = self.qkv(hidden_states, forward_batch)
        target_dtype = hidden_states.dtype
        # ``flatten(1)`` rather than ``reshape(T, -1)``: under DP attention a
        # replica with no requests this step runs an idle forward with T=0, and
        # ``reshape(0, -1)`` raises (the ``-1`` is ambiguous for a 0-element
        # tensor). ``flatten`` multiplies the head dims explicitly, so it yields
        # ``[0, heads*head_dim]`` and is identical to the old reshape for T>0.
        q = q.flatten(1).to(target_dtype)
        k = k.flatten(1).to(target_dtype)
        v = v.flatten(1).to(target_dtype)

        q, k = self.rotary_emb(positions, q, k)
        # Some rotary backends (notably AITER on ROCm) hand back tensors with
        # a different stride than the input. RadixAttention's KV-store kernel
        # asserts contiguous layout, so normalize q/k/v before the attention.
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        # o_proj is RowParallel with ``reduce_results=False``; reduce the partial
        # sums across the attention-TP group (equals the global TP group unless
        # DP attention is enabled). A size-1 group makes this a no-op.
        if self.tp_size > 1:
            output = attn_tp_all_reduce(output)
        return output


# ---------------------------------------------------------------------------
# Router (EDA + MOD) and MoE block
# ---------------------------------------------------------------------------


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
        self.num_experts = (num_moe_experts + 1) if self.use_mod else num_moe_experts
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

        self.register_buffer(
            "balancing_biases",
            torch.zeros(self.num_experts, dtype=torch.float32),
            persistent=True,
        )
        if self.use_mod:
            with torch.no_grad():
                self.balancing_biases[-1] = -1.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_router_hidden_states: Optional[torch.Tensor] = None,
        gather_event: Optional[torch.cuda.Event] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ``hidden_states`` is ``[T, H]``.
        use_eda = (
            self.use_eda
            and prev_router_hidden_states is not None
            and hasattr(self, "router_states_scale")
        )
        # The EDA term reads only ``prev_router_hidden_states``, which the previous
        # MoE layer already left in the gathered layout, so it is the one piece of
        # router work that does not depend on this layer's gather. Compute it
        # first, then wait -- with ``gather_event`` set it runs while the gather is
        # still in flight on the comm stream. Same operands in the same order as
        # the un-overlapped form, so the arithmetic is unchanged.
        eda_term = (
            prev_router_hidden_states * self.router_states_scale if use_eda else None
        )
        if gather_event is not None:
            torch.cuda.current_stream().wait_event(gather_event)

        hs, _ = self.down_proj(hidden_states)
        if eda_term is not None:
            hs = hs + eda_term

        # ``hs`` is a freshly-allocated tensor (output of ``down_proj`` or the
        # EDA add above) and ``rmsnorm_eda`` is non-residual / out-of-place,
        # so we can hand the same buffer to the next layer without cloning.
        router_hidden_states_next = hs

        hs_norm = self.rmsnorm_eda(hs)

        # Step through the Sequential manually so the ``(tensor, bias)`` tuple
        # returned by each ReplicatedLinear is unpacked correctly.
        out = hs_norm
        for stage in self.router_mlp:
            if isinstance(stage, ReplicatedLinear):
                out, _ = stage(out)
            else:
                out = stage(out)
        logits = out

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
        if route_prob.dtype != hidden_states.dtype:
            route_prob = route_prob.to(hidden_states.dtype)

        return route_prob, expert_choice, router_hidden_states_next


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


class DPCombine(msgspec.Struct, frozen=True):
    """Where this rank's slice of the DP-gathered token set lives.

    Passed to :class:`ZayaBlock` when the experts ran on the gathered (global)
    tokens and the partial outputs should be combined with a reduce-scatter that
    lands only this DP replica's rows, instead of an all-reduce that replicates
    all of them and a scatter that then drops the rest.
    """

    local_start: int
    local_rows: int


def dp_combine_for(*, global_rows: int, local_rows: int) -> Optional[DPCombine]:
    """Describe the reduce-scatter combine for a gather of ``local_rows`` rows per
    replica into a ``global_rows``-row buffer, or ``None`` when the layout cannot
    support one.

    Three preconditions, all properties of the gathered buffer rather than of the
    model:

    * MAX_LEN padding -- every replica occupies an equal, contiguous block, so
      this rank's rows start at ``dp_rank * local_rows``. Under SUM_LEN the blocks
      are ragged and the even ``tensor_split`` inside ``dp_reduce_scatter_tensor``
      would straddle replica boundaries.
    * The blocks tile the buffer exactly (``local_rows * dp_size == global_rows``).
      MAX_LEN pads every replica to the batch's longest, so this holds whenever
      that padding was applied to the tensor we are handed -- but the check is
      cheap and a mismatch would silently return the wrong number of rows.
    * ``global_rows`` divisible by the TP size -- the reduce-scatter splits the
      global buffer into ``tp_size`` equal chunks, and each replica's block must be
      a whole number of them. A tp=8/dp=4 decode of one token per replica gathers
      4 rows and cannot be split 8 ways, so it keeps the all-reduce.
    """
    parallel = get_parallel()
    if not is_dp_max_padding():
        return None
    if local_rows * parallel.attn_dp_size != global_rows:
        return None
    if global_rows % parallel.tp_size != 0:
        return None
    return DPCombine(
        local_start=get_attention_dp_rank() * local_rows, local_rows=local_rows
    )


def _slice_combined_rows(
    dp_combine: Optional[DPCombine], *tensors: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Narrow global per-token tensors to the rows a reduce-scatter combine kept.

    A no-op without ``dp_combine`` (the combine was an all-reduce, so every rank
    still holds all rows). Row slices of a row-major tensor stay contiguous, so
    the fused MOD kernels' coverage checks still pass.
    """
    if dp_combine is None:
        return tensors
    rows = slice(dp_combine.local_start, dp_combine.local_start + dp_combine.local_rows)
    return tuple(t[rows] for t in tensors)


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
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_router_hidden_states: Optional[torch.Tensor] = None,
        dp_combine: Optional[DPCombine] = None,
        gather_event: Optional[torch.cuda.Event] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.shape[0] == 0:
            return hidden_states, hidden_states.new_zeros((0, self.mlp_expansion))

        probs, indices, router_hs_next = self.router(
            hidden_states, prev_router_hidden_states, gather_event
        )

        topk_out = StandardTopKOutput(
            topk_weights=probs.to(hidden_states.dtype),
            topk_ids=indices.to(torch.int32),
            router_logits=probs.to(hidden_states.dtype),
        )

        if self.config.zaya_use_mod:
            # MOD: clamp the "skip expert" id (== num_moe_experts) into the
            # valid expert range so FusedMoE never indexes out of bounds; the
            # mask below decides per-token whether to actually use experts or
            # the skip path.
            clamped_ids = torch.clamp(indices, min=0, max=self.num_moe_experts - 1).to(
                torch.int32
            )
            topk_out = topk_out._replace(topk_ids=clamped_ids)

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
                masked_experts = self._combine_experts(masked_experts, dp_combine)
                # The blend is per-token elementwise, so it is equally valid on
                # the global rows (after an all-reduce) or on just this replica's
                # rows (after a reduce-scatter) -- as long as its other operands
                # are narrowed to the same rows.
                indices, hidden_states, probs = _slice_combined_rows(
                    dp_combine, indices, hidden_states, probs
                )
                hidden_out = _mod.mod_blend(
                    masked_experts,
                    indices,
                    hidden_states,
                    probs,
                    self.num_moe_experts,
                )
            else:
                mod_mask, masked_experts = mod_premask_experts(
                    experts_out, indices, self.num_moe_experts
                )
                masked_experts = self._combine_experts(masked_experts, dp_combine)
                mod_mask, hidden_states, probs = _slice_combined_rows(
                    dp_combine, mod_mask, hidden_states, probs
                )
                hidden_out = mod_blend(masked_experts, mod_mask, hidden_states * probs)
        else:
            hidden_out = self._combine_experts(
                self.experts(hidden_states, topk_out), dp_combine
            )

        return hidden_out, router_hs_next

    def _combine_experts(
        self, experts_out: torch.Tensor, dp_combine: Optional[DPCombine]
    ) -> torch.Tensor:
        """Combine per-rank partial expert outputs, keeping only what is needed.

        With ``dp_combine`` the reduce and the DP scatter collapse into a single
        reduce-scatter that delivers just this replica's rows; without it, the
        caller reduces globally and scatters afterwards.
        """
        if dp_combine is None:
            return self._reduce_experts(experts_out)
        local = torch.empty(
            (dp_combine.local_rows, experts_out.shape[1]),
            dtype=experts_out.dtype,
            device=experts_out.device,
        )
        dp_reduce_scatter_tensor(local, experts_out)
        return local

    def _reduce_experts(self, experts_out: torch.Tensor) -> torch.Tensor:
        """Combine partial expert outputs over the MoE parallel groups.

        Mirrors the canonical SGLang MoE reduce (cf. ``qwen3_moe``): first an
        all-reduce over the expert-parallel (EP) group, then over the
        MoE-tensor-parallel (TP) group. Under plain TP this is a single reduce
        over the global TP group; under EP / DP attention it stays scoped to the
        MoE groups and never spans the DP-attention replicas.
        """
        if self.ep_size > 1:
            experts_out = moe_expert_parallel_all_reduce(experts_out)
        if self.tp_size > 1:
            experts_out = moe_tensor_model_parallel_all_reduce(experts_out)
        return experts_out


# ---------------------------------------------------------------------------
# Decoder layers
# ---------------------------------------------------------------------------


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
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        target_dtype = (
            self.input_norm.weight.dtype
            if isinstance(self.input_norm, RMSNorm)
            else hidden_states.dtype
        )
        hidden_states, residual = _residual_scale_norm(
            self.res_scale, self.input_norm, residual, hidden_states, target_dtype
        )
        hidden_states = self.self_attn(hidden_states, positions, forward_batch)
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

        self.gather_event_key = ("zaya_gather", layer_id)
        if envs.SGLANG_OPT_ZAYA_OVERLAP_DP_GATHER.get():
            prewarm_dp_gather_async(self.gather_event_key)

        # Odd layer ids are the MoE layers (see _build_layer), so layer 1 is the
        # first; it reports which combine the run settled on, once.
        self.first_moe_layer_id = 1
        self._logged_combines: set[str] = set()

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        prev_router_hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        target_dtype = (
            self.input_norm.weight.dtype
            if isinstance(self.input_norm, RMSNorm)
            else hidden_states.dtype
        )
        hidden_states, residual = _residual_scale_norm(
            self.res_scale, self.input_norm, residual, hidden_states, target_dtype
        )
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
        # The gather is needed only when the MoE-TP group is *wider* than the
        # attention-TP group, i.e. when it spans DP replicas: then a token must be
        # visible to every MoE rank for the expert shards to see it. When
        # ``moe_tp == attn_tp`` (e.g. --moe-dp-size equal to the attention DP size)
        # each replica owns a self-contained MoE over its own ranks, the token is
        # already replicated across them by ``attn_tp_all_reduce``, and the
        # gather/scatter pair is pure overhead.
        #
        # This matters a lot: profiling tp=8/dp=4 on the 74B put ~83% of decode GPU
        # time in collectives, of which the per-MoE-layer all-gather alone was
        # 29%. Skipping it when the groups coincide removes 60 all-gathers and 60
        # scatters per step.
        # Compare against the width of the group ``_reduce_experts`` actually
        # reduces over -- expert-parallel AND MoE-tensor-parallel -- not moe_tp
        # alone. Under EP (``--ep-size 8``) moe_tp collapses to 1 while the reduce
        # still spans all 8 ranks, so keying off moe_tp alone would skip a gather
        # that is required and silently drop every token the rank does not own.
        parallel = get_parallel()
        moe_reduce_width = parallel.moe_ep_size * parallel.moe_tp_size
        use_dp_gather = (
            parallel.attn_dp_size > 1
            and moe_reduce_width > parallel.attn_tp_size
            and get_moe_a2a_backend().is_none()
        )
        # The reduce-scatter combine subsumes the scatter below, but only when the
        # MoE reduce spans exactly the TP group -- that is the group
        # ``dp_reduce_scatter_tensor`` reduces over. A narrower MoE reduce (e.g.
        # --moe-dp-size) must keep the group-scoped all-reduce in
        # ``_reduce_experts``.
        dp_combine = None
        if (
            use_dp_gather
            and moe_reduce_width == parallel.tp_size
            and envs.SGLANG_OPT_ZAYA_MOE_REDUCE_SCATTER.get()
        ):
            dp_combine = dp_combine_for(
                global_rows=get_global_dp_buffer_len(),
                local_rows=hidden_states.shape[0],
            )
            if self.layer_id == self.first_moe_layer_id:
                # The combine declines itself on layouts it cannot serve, so report
                # which path a run actually took, not which one was requested, and
                # spell out each precondition -- otherwise a decline is
                # indistinguishable from the flag not being read at all.
                #
                # Deduplicate on the whole message, not just the verdict: prefill
                # runs under SUM_LEN and always declines, so keying on the verdict
                # alone would let that first line mask every later decode shape.
                global_rows = get_global_dp_buffer_len()
                local_rows = hidden_states.shape[0]
                msg = (
                    f"ZAYA1 MoE combine: "
                    f"{'reduce-scatter' if dp_combine else 'all-reduce'} "
                    f"global_rows={global_rows} local_rows={local_rows} "
                    f"max_pad={is_dp_max_padding()} "
                    f"tiles={local_rows * parallel.attn_dp_size == global_rows} "
                    f"div_tp={global_rows % parallel.tp_size == 0}"
                )
                if msg not in self._logged_combines:
                    self._logged_combines.add(msg)
                    logger.info("%s", msg)

        gather_event = None
        if use_dp_gather:
            hidden_states, local_hidden_states = (
                get_global_dp_buffer(get_tp_group()),
                hidden_states,
            )
            if envs.SGLANG_OPT_ZAYA_OVERLAP_DP_GATHER.get():
                # Key the persistent event by layer: a fresh torch.cuda.Event per
                # layer per forward exhausts the HSA signal pool within a few
                # hundred steps (see _tbo_event).
                gather_event = dp_gather_replicate_async(
                    hidden_states,
                    local_hidden_states,
                    forward_batch,
                    event_key=self.gather_event_key,
                )
            else:
                dp_gather_replicate(hidden_states, local_hidden_states, forward_batch)
        hidden_states, prev_router_hidden_states = self.zaya_block(
            hidden_states, prev_router_hidden_states, dp_combine, gather_event
        )
        if use_dp_gather and dp_combine is None:
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
        if self.pp_group.is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
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
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )

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
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
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


EntryClass = ZayaForCausalLM
