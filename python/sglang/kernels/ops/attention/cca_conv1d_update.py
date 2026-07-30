"""Fused ZAYA1 CCA decode conv: window, history shift and grouped matmul in one.

The decode path today is two launches -- ``cca_state_step`` builds the ``[T, C, W]``
conv window and shifts both pool slots, then an einsum applies the folded grouped
weight. This module folds the conv arithmetic into the window build, so the window
is never materialized and the batched GEMM is replaced by a tiled ``tl.dot``.

**This is launch-neutral, not a launch saving, and that is deliberate.** The
conv-channel work (``C`` channels, ``G`` groups) and the ``prev_hs`` carry (``H``
channels) have different natural parallel shapes, so one kernel covering both
would either run the ``H`` work ``G`` times over or collapse to a handful of
programs. Two launches with the right decomposition beat one with the wrong one.
What this can win is the ``[T, C, W]`` round-trip and, possibly, the GEMM itself:
``cca_state_step``'s note that "a hand-rolled Triton matvec measured no better"
was about a *matvec*, whereas decode at C=128 concurrency has enough tokens for a
real tiled dot. Treat that as a hypothesis to measure, not a given.

Like its siblings it gates on ``covered()`` and tolerates the negative slot ids
that batch padding writes -- those rows read and write no state and their outputs
are discarded by the caller.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _cca_conv1d_update_kernel(
    qk_ptr,  # [T, C]         this step's pre-conv row
    w_ptr,  # [G, CG, CG*W]  folded grouped weight, flattened over (ci, tap)
    bias_ptr,  # [G, CG]
    conv_state_ptr,  # [S, C, W-1]   per-slot history
    slots_ptr,  # [T]            slot per token (<0 == padding)
    out_ptr,  # [T, C]         out: conv output
    s_qk_t,
    s_w_g,
    s_w_o,
    s_cs_s,
    s_cs_c,
    s_out_t,
    num_tokens,
    CG: tl.constexpr,  # channels per group
    W: tl.constexpr,  # taps
    BLOCK_T: tl.constexpr,
):
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    g = tl.program_id(1)

    tmask = t < num_tokens
    tsafe = tl.where(tmask, t, 0)
    slot = tl.load(slots_ptr + tsafe, mask=tmask, other=-1)
    live = tmask & (slot >= 0)

    # Activations and the conv pool are indexed by global channel; the weight's
    # leading axis is the group, so its out-channel axis is group-local.
    co_local = tl.arange(0, CG)
    co = g * CG + co_local
    ci = tl.arange(0, CG)

    acc = tl.zeros([BLOCK_T, CG], dtype=tl.float32)
    for m in tl.static_range(W):
        if m == W - 1:
            # Last tap is this step's token.
            xm = tl.load(
                qk_ptr + tsafe[:, None] * s_qk_t + co[None, :],
                mask=tmask[:, None],
                other=0.0,
            )
        else:
            xm = tl.load(
                conv_state_ptr + slot[:, None] * s_cs_s + co[None, :] * s_cs_c + m,
                mask=live[:, None],
                other=0.0,
            )
        # The folded weight flattens (ci, tap) into one axis of width CG*W, so
        # tap m occupies the stride-W slice starting at m -- load it as [ci, co]
        # to contract over ci.
        wm = tl.load(
            w_ptr + g * s_w_g + co_local[None, :] * s_w_o + ci[:, None] * W + m
        )
        acc += tl.dot(xm, wm, out_dtype=tl.float32)

    acc = acc + tl.load(bias_ptr + co)[None, :]
    tl.store(
        out_ptr + tsafe[:, None] * s_out_t + co[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=tmask[:, None],
    )

    # Shift the history left by one and append this token. Each iteration reads a
    # tap strictly above the one it writes, and writes ascend, so no read sees an
    # already-written value.
    cur = tl.load(
        qk_ptr + tsafe[:, None] * s_qk_t + co[None, :],
        mask=tmask[:, None],
        other=0.0,
    )
    for w in tl.static_range(W - 2):
        nxt = tl.load(
            conv_state_ptr + slot[:, None] * s_cs_s + co[None, :] * s_cs_c + (w + 1),
            mask=live[:, None],
            other=0.0,
        )
        tl.store(
            conv_state_ptr + slot[:, None] * s_cs_s + co[None, :] * s_cs_c + w,
            nxt,
            mask=live[:, None],
        )
    tl.store(
        conv_state_ptr + slot[:, None] * s_cs_s + co[None, :] * s_cs_c + (W - 2),
        cur,
        mask=live[:, None],
    )


@triton.jit
def _prev_hs_step_kernel(
    hs_ptr,  # [T, H]
    prev_hs_ptr,  # [S, H, 1]
    prev_out_ptr,  # [T, H]  out
    slots_ptr,  # [T]
    s_hs_t,
    s_ph_s,
    s_pv_t,
    hidden_size,
    BLOCK_H: tl.constexpr,
):
    t = tl.program_id(0)
    slot = tl.load(slots_ptr + t)
    for h0 in tl.range(0, hidden_size, BLOCK_H):
        h = h0 + tl.arange(0, BLOCK_H)
        hmask = h < hidden_size
        # Read this slot's carried state before overwriting it with the current
        # token; one program owns the pair, so the two cannot race.
        prev = tl.load(
            prev_hs_ptr + slot * s_ph_s + h, mask=hmask & (slot >= 0), other=0.0
        )
        tl.store(prev_out_ptr + t * s_pv_t + h, prev, mask=hmask)
        cur = tl.load(hs_ptr + t * s_hs_t + h, mask=hmask, other=0.0)
        tl.store(prev_hs_ptr + slot * s_ph_s + h, cur, mask=hmask & (slot >= 0))


def covered(
    qk: torch.Tensor,
    hidden_states: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    conv_state: torch.Tensor,
    prev_hs_state: torch.Tensor,
    slots: Optional[torch.Tensor],
    total_padding: int,
    groups: int,
) -> bool:
    """Whether the fused decode conv can serve these inputs.

    Mirrors ``cca_state_step.covered`` and additionally requires the folded
    grouped weight in its ``[G, CG, CG*W]`` matmul layout and a group width
    ``tl.dot`` can take as a tile.
    """
    if weight is None or bias is None or slots is None:
        return False
    if total_padding < 1 or groups < 1:
        return False

    tensors = (qk, hidden_states, weight, bias, conv_state, prev_hs_state, slots)
    if not all(t.is_cuda for t in tensors):
        return False
    if qk.ndim != 2 or hidden_states.ndim != 2:
        return False
    if conv_state.ndim != 3 or prev_hs_state.ndim != 3:
        return False
    if qk.shape[0] != hidden_states.shape[0]:
        return False

    num_channels = qk.shape[-1]
    if num_channels % groups != 0:
        return False
    ch_per_group = num_channels // groups
    if ch_per_group < 16 or (ch_per_group & (ch_per_group - 1)) != 0:
        return False

    taps = total_padding + 1
    if tuple(weight.shape) != (groups, ch_per_group, ch_per_group * taps):
        return False
    if tuple(bias.shape) != (groups, ch_per_group):
        return False
    if conv_state.shape[-1] != total_padding or conv_state.shape[-2] != num_channels:
        return False
    if prev_hs_state.shape[-2] != hidden_states.shape[-1]:
        return False
    if prev_hs_state.shape[-1] != 1:
        return False
    if slots.ndim != 1 or slots.shape[0] != qk.shape[0]:
        return False
    if not slots.is_contiguous():
        return False

    if conv_state.dtype != qk.dtype or prev_hs_state.dtype != hidden_states.dtype:
        return False
    if weight.dtype != qk.dtype or bias.dtype != qk.dtype:
        return False

    return (
        qk.stride(-1) == 1
        and hidden_states.stride(-1) == 1
        and conv_state.stride(-1) == 1
        and prev_hs_state.stride(-1) == 1
        and weight.stride(-1) == 1
    )


def cca_conv1d_update(
    qk: torch.Tensor,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    conv_state: torch.Tensor,
    prev_hs_state: torch.Tensor,
    slots: torch.Tensor,
    total_padding: int,
    groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(qk_out, prev_hs)``, shifting both pool slots in place.

    ``qk_out`` is the conv output ``[T, C]`` and ``prev_hs`` the previous hidden
    state feeding ``val_proj2``. Caller must have checked :func:`covered`.
    """
    num_tokens, num_channels = qk.shape
    hidden_size = hidden_states.shape[-1]
    ch_per_group = num_channels // groups
    taps = total_padding + 1

    qk_out = torch.empty_like(qk)
    prev_out = torch.empty_like(hidden_states)
    if num_tokens == 0:
        return qk_out, prev_out

    block_t = 16 if num_tokens < 64 else 64
    _cca_conv1d_update_kernel[(triton.cdiv(num_tokens, block_t), groups)](
        qk,
        weight,
        bias,
        conv_state,
        slots,
        qk_out,
        qk.stride(0),
        weight.stride(0),
        weight.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        qk_out.stride(0),
        num_tokens,
        CG=ch_per_group,
        W=taps,
        BLOCK_T=block_t,
        num_warps=4,
    )
    _prev_hs_step_kernel[(num_tokens,)](
        hidden_states,
        prev_hs_state,
        prev_out,
        slots,
        hidden_states.stride(0),
        prev_hs_state.stride(0),
        prev_out.stride(0),
        hidden_size,
        BLOCK_H=512,
        num_warps=4,
    )
    return qk_out, prev_out
