"""Fused ZAYA1 CCA decode conv: window, history shift and grouped matmul in one.

The decode path today is two launches -- ``cca_state_step`` builds the ``[T, C, W]``
conv window and shifts both pool slots, then an einsum applies the folded grouped
weight. This module folds the conv arithmetic into the window build, so the window
is never materialized and the batched GEMM is replaced by a tiled ``tl.dot``.

**This is launch-neutral, not a launch saving, and that is deliberate.** The
conv-channel work (``C`` channels, ``G`` groups) and the val_proj2 lag carry
(``D`` channels) have different natural parallel shapes, so one kernel covering
both would either run the ``D`` work ``G`` times over or collapse to a handful of
programs. Two launches with the right decomposition beat one with the wrong one.
What it could win was the ``[T, C, W]`` round-trip and, possibly, the GEMM itself:
``cca_state_step``'s note that "a hand-rolled Triton matvec measured no better"
was about a *matvec*, whereas decode at C=128 concurrency has enough tokens for a
real tiled dot.

**Measured, and it loses.** MI350X, 74B tp8/dp4, 1k in / 1k out, global residual
and the fused prefill on in both arms, 3 reps with rep 1 discarded:

    C     TPOT ms off -> on        output tok/s
    32    15.93 -> 16.31 (+2.4%)   1954.7 -> 1901.5 (-2.7%)
    128   18.81 -> 19.40 (+3.1%)   6503.0 -> 6315.2 (-2.9%)

TPOT spread across reps was 0.05%, so that is a real regression, not noise: the
tiled ``tl.dot`` does not catch rocBLAS's batched GEMM on this shape (batch 9,
M=tokens, K=384, N=128), and there was no launch to save to pay for it. Kept
behind ``SGLANG_OPT_ZAYA_FUSED_CCA_DECODE``, default off, because the useful
artifact is the negative result -- the seam it replaced used to read as a TODO
with an expected win. Do not re-attempt without a different idea; the decode
conv is not where the time goes.

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
    WSPAN: tl.constexpr,  # per-input-channel span of the weight (W + bias col)
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
        # The folded weight flattens (ci, tap) into one axis of width
        # CG*WSPAN, so tap m occupies the stride-WSPAN slice starting at m --
        # load it as [ci, co] to contract over ci. WSPAN exceeds W by the
        # trailing bias column the einsum path folds in (see
        # ``CCA.fold_decode_conv``); this kernel never reads that column, it
        # adds the bias itself, which is already free inside the accumulator.
        wm = tl.load(
            w_ptr + g * s_w_g + co_local[None, :] * s_w_o + ci[:, None] * WSPAN + m
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
def _lag_step_kernel(
    lag_ptr,  # [T, D]
    lag_state_ptr,  # [S, D, 1]
    lag_out_ptr,  # [T, D]  out
    slots_ptr,  # [T]
    s_lag_t,
    s_ls_s,
    s_lo_t,
    lag_dim,
    BLOCK_D: tl.constexpr,
):
    t = tl.program_id(0)
    slot = tl.load(slots_ptr + t)
    for d0 in tl.range(0, lag_dim, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        dmask = d < lag_dim
        # Read this slot's carried value before overwriting it with the current
        # token's; one program owns the pair, so the two cannot race.
        prev = tl.load(
            lag_state_ptr + slot * s_ls_s + d, mask=dmask & (slot >= 0), other=0.0
        )
        tl.store(lag_out_ptr + t * s_lo_t + d, prev, mask=dmask)
        cur = tl.load(lag_ptr + t * s_lag_t + d, mask=dmask, other=0.0)
        tl.store(lag_state_ptr + slot * s_ls_s + d, cur, mask=dmask & (slot >= 0))


def covered(
    qk: torch.Tensor,
    lag_now: Optional[torch.Tensor],
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    conv_state: torch.Tensor,
    lag_state: Optional[torch.Tensor],
    slots: Optional[torch.Tensor],
    total_padding: int,
    groups: int,
) -> bool:
    """Whether the fused decode conv can serve these inputs.

    Mirrors ``cca_state_step.covered`` and additionally requires the folded
    grouped weight in its ``[G, CG, CG*(W+1)]`` matmul layout -- the einsum path's
    layout, whose trailing column carries the conv bias -- and a group width
    ``tl.dot`` can take as a tile.
    """
    if weight is None or bias is None or slots is None:
        return False
    if total_padding < 1 or groups < 1:
        return False
    if (lag_now is None) != (lag_state is None):
        return False

    tensors = [qk, weight, bias, conv_state, slots]
    if lag_now is not None:
        tensors += [lag_now, lag_state]
    if not all(t.is_cuda for t in tensors):
        return False
    if qk.ndim != 2:
        return False
    if conv_state.ndim != 3:
        return False
    if lag_now is not None:
        if lag_now.ndim != 2 or lag_state.ndim != 3:
            return False
        if qk.shape[0] != lag_now.shape[0]:
            return False

    num_channels = qk.shape[-1]
    if num_channels % groups != 0:
        return False
    ch_per_group = num_channels // groups
    if ch_per_group < 16 or (ch_per_group & (ch_per_group - 1)) != 0:
        return False

    taps = total_padding + 1
    if tuple(weight.shape) != (groups, ch_per_group, ch_per_group * (taps + 1)):
        return False
    if tuple(bias.shape) != (groups, ch_per_group):
        return False
    if conv_state.shape[-1] != total_padding or conv_state.shape[-2] != num_channels:
        return False
    if lag_now is not None:
        if lag_state.shape[-2] != lag_now.shape[-1]:
            return False
        if lag_state.shape[-1] != 1:
            return False
        if lag_state.stride(-2) != 1:
            return False
    if slots.ndim != 1 or slots.shape[0] != qk.shape[0]:
        return False
    if not slots.is_contiguous():
        return False

    if conv_state.dtype != qk.dtype:
        return False
    if weight.dtype != qk.dtype or bias.dtype != qk.dtype:
        return False

    if not (
        qk.stride(-1) == 1 and conv_state.stride(-1) == 1 and weight.stride(-1) == 1
    ):
        return False
    if lag_now is None:
        return True
    if lag_state.dtype != lag_now.dtype:
        return False
    return lag_now.stride(-1) == 1 and lag_state.stride(-1) == 1


def cca_conv1d_update(
    qk: torch.Tensor,
    lag_now: Optional[torch.Tensor],
    weight: torch.Tensor,
    bias: torch.Tensor,
    conv_state: torch.Tensor,
    lag_state: Optional[torch.Tensor],
    slots: torch.Tensor,
    total_padding: int,
    groups: int,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return ``(qk_out, lag_prev)``, shifting both pool slots in place.

    ``qk_out`` is the conv output ``[T, C]`` and ``lag_prev`` the previous step's
    ``val_proj2`` value, or ``None`` when this rank carries no lag stream (then
    this is a single launch). Caller must have checked :func:`covered`.
    """
    num_tokens, num_channels = qk.shape
    ch_per_group = num_channels // groups
    taps = total_padding + 1
    has_lag = lag_now is not None

    qk_out = torch.empty_like(qk)
    lag_prev = torch.empty_like(lag_now) if has_lag else None
    if num_tokens == 0:
        return qk_out, lag_prev

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
        WSPAN=taps + 1,
        BLOCK_T=block_t,
        num_warps=4,
    )
    if has_lag:
        _lag_step_kernel[(num_tokens,)](
            lag_now,
            lag_state,
            lag_prev,
            slots,
            lag_now.stride(0),
            lag_state.stride(0),
            lag_prev.stride(0),
            lag_now.shape[-1],
            BLOCK_D=512,
            num_warps=4,
        )
    return qk_out, lag_prev
