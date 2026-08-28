"""Fused ZAYA1 CCA decode conv-state step.

One kernel replaces the five state-plumbing launches around the decode conv in
:func:`cca_decode <sglang.srt.models.zaya.cca_decode>` -- two gathers, a
concatenate and two scatters::

    left_pad = conv_state.index_select(0, slots)          # gather
    padded   = cat([left_pad, qk.unsqueeze(-1)], dim=-1)  # concat
    conv_state.index_copy_(0, slots, padded[..., -W+1:])   # scatter (shift)
    prev_hs  = prev_hs_state.index_select(0, slots)        # gather
    prev_hs_state.index_copy_(0, slots, hidden_states)     # scatter

It emits the ``[T, C, W]`` conv window the matmul consumes, returns the previous
hidden state that feeds ``val_proj2``, and shifts both pool slots in place --
reading each slot before overwriting it, so the gather/scatter pair on the same
row is safe within one program.

The grid is ``(T, n_channel_tiles + n_hidden_tiles)``: the second axis indexes
one flat tile space where the low ``n_channel_tiles`` entries do the conv window
and history shift for their slice of ``C``, and the rest do the ``prev_hs``
read-then-overwrite for their slice of ``H``. Tiling that second axis rather
than looping it inside one program is what makes the kernel fill the GPU at
decode batch sizes: at 32 tokens a ``grid=(T,)`` launch is 32 programs on a
256-CU MI355X, each serially walking 5 channel tiles and 8 hidden tiles, so the
tiles that could run concurrently instead queue behind each other.

The split is safe because it never puts two programs on the same
``(slot, column)``: within a token the tiles partition ``C`` and ``H``
disjointly, so the "read the old value before storing the new one" ordering that
the shift and the ``prev_hs`` swap rely on stays *inside* one program, exactly
as it did with the inner loops. Two tokens sharing one positive slot id would
alias -- but they did under ``grid=(T,)`` too (two programs, same rows, no
ordering between them), so the contract is unchanged: slot ids must be distinct,
apart from the negative padding ids, which touch nothing.

The grouped matmul itself is deliberately left outside: it is a batched
``[Cg, Cg*W] x [Cg*W]`` per group, which cuBLAS/rocBLAS-backed einsum already
runs near bandwidth, and a hand-rolled Triton matvec measured no better. The
launches removed here are pure data movement.

Follows ``kda_fused_decode``'s structure -- a ``covered()`` predicate gates
supported inputs and the caller falls back to the unfused chain -- and, like it,
handles the negative slot ids that batch padding writes (those rows read and
write nothing; their outputs are discarded by the caller). Written in Triton so
it runs on ROCm, ZAYA1's reference deployment.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _cca_state_step_kernel(
    qk_ptr,  # [T, C]        this step's pre-conv q/k row
    hs_ptr,  # [T, H]        this step's hidden state
    conv_state_ptr,  # [S, C, W-1]   per-slot conv history
    prev_hs_ptr,  # [S, H, 1]     per-slot previous hidden state
    slots_ptr,  # [T]           mamba slot per token (<0 == padding)
    window_ptr,  # [T, C, W]     out: conv window
    prev_out_ptr,  # [T, H]        out: previous hidden state
    s_qk_t,
    s_hs_t,
    s_cs_s,
    s_cs_c,
    s_ph_s,
    s_win_t,
    s_win_c,
    s_pv_t,
    num_channels,
    hidden_size,
    W: tl.constexpr,  # taps == total_padding + 1
    NC_TILES: tl.constexpr,  # ceil(num_channels / BLOCK_C)
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    t = tl.program_id(0)
    tile = tl.program_id(1)
    slot = tl.load(slots_ptr + t)

    if tile < NC_TILES:
        # ---- conv window + in-place history shift ---------------------------
        c = tile * BLOCK_C + tl.arange(0, BLOCK_C)
        cmask = c < num_channels

        qk = tl.load(qk_ptr + t * s_qk_t + c, mask=cmask, other=0.0)

        # Taps [0, W-1) come from the cached history; tap W-1 is this token.
        for w in tl.static_range(W - 1):
            hist = tl.load(
                conv_state_ptr + slot * s_cs_s + c * s_cs_c + w,
                mask=cmask & (slot >= 0),
                other=0.0,
            )
            tl.store(window_ptr + t * s_win_t + c * s_win_c + w, hist, mask=cmask)
        tl.store(window_ptr + t * s_win_t + c * s_win_c + (W - 1), qk, mask=cmask)

        # Shift the history left by one: new[w] = old[w+1], new[W-2] = qk. Read
        # every old tap above before storing so the two do not alias.
        for w in tl.static_range(W - 2):
            nxt = tl.load(
                conv_state_ptr + slot * s_cs_s + c * s_cs_c + (w + 1),
                mask=cmask & (slot >= 0),
                other=0.0,
            )
            tl.store(
                conv_state_ptr + slot * s_cs_s + c * s_cs_c + w,
                nxt,
                mask=cmask & (slot >= 0),
            )
        tl.store(
            conv_state_ptr + slot * s_cs_s + c * s_cs_c + (W - 2),
            qk,
            mask=cmask & (slot >= 0),
        )
    else:
        # ---- previous hidden state: read before overwrite -------------------
        h = (tile - NC_TILES) * BLOCK_H + tl.arange(0, BLOCK_H)
        hmask = h < hidden_size
        prev = tl.load(
            prev_hs_ptr + slot * s_ph_s + h, mask=hmask & (slot >= 0), other=0.0
        )
        tl.store(prev_out_ptr + t * s_pv_t + h, prev, mask=hmask)
        cur = tl.load(hs_ptr + t * s_hs_t + h, mask=hmask, other=0.0)
        tl.store(prev_hs_ptr + slot * s_ph_s + h, cur, mask=hmask & (slot >= 0))


def covered(
    qk: torch.Tensor,
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    prev_hs_state: torch.Tensor,
    slots: Optional[torch.Tensor],
    total_padding: int,
) -> bool:
    """Whether the fused state step can serve these inputs.

    Requires everything on one accelerator, matching dtypes between each pool and
    the value written into it, a unit innermost stride on the pool views, and at
    least two taps (``total_padding >= 1``) so the shift is well defined.
    """
    if slots is None or total_padding < 1:
        return False
    tensors = (qk, hidden_states, conv_state, prev_hs_state, slots)
    if not all(t.is_cuda for t in tensors):
        return False
    if qk.ndim != 2 or hidden_states.ndim != 2:
        return False
    if conv_state.ndim != 3 or prev_hs_state.ndim != 3:
        return False
    if conv_state.shape[-1] != total_padding:
        return False
    if conv_state.shape[-2] != qk.shape[-1]:
        return False
    if prev_hs_state.shape[-2] != hidden_states.shape[-1]:
        return False
    if prev_hs_state.shape[-1] != 1:
        return False
    if slots.ndim != 1 or slots.shape[0] != qk.shape[0]:
        return False
    if not slots.is_contiguous():
        return False
    # The kernel stores qk straight into the conv pool and hidden_states into the
    # prev_hs pool, so no dtype conversion is modelled.
    if conv_state.dtype != qk.dtype or prev_hs_state.dtype != hidden_states.dtype:
        return False
    return (
        conv_state.stride(-1) == 1
        and prev_hs_state.stride(-1) == 1
        and qk.stride(-1) == 1
        and hidden_states.stride(-1) == 1
    )


def cca_state_step(
    qk: torch.Tensor,
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    prev_hs_state: torch.Tensor,
    slots: torch.Tensor,
    total_padding: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(window, prev_hs)`` and shift both pool slots in place.

    ``window`` is ``[T, C, total_padding + 1]`` in ``qk``'s dtype; ``prev_hs`` is
    ``[T, H]`` in ``hidden_states``' dtype. Caller must have checked
    :func:`covered`."""
    num_tokens, num_channels = qk.shape
    hidden_size = hidden_states.shape[-1]
    taps = total_padding + 1

    window = torch.empty(
        (num_tokens, num_channels, taps), dtype=qk.dtype, device=qk.device
    )
    prev_out = torch.empty(
        (num_tokens, hidden_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    if num_tokens == 0:
        return window, prev_out

    # One flat tile axis: [0, nc_tiles) walk the channels, the rest walk the
    # hidden dim. Tiling both instead of looping them inside one program is what
    # gets the launch off 32 programs at decode batch sizes; see the module
    # docstring for why no two programs can then share a (slot, column).
    block_c, block_h = 256, 512
    nc_tiles = triton.cdiv(num_channels, block_c)
    nh_tiles = triton.cdiv(hidden_size, block_h)

    _cca_state_step_kernel[(num_tokens, nc_tiles + nh_tiles)](
        qk,
        hidden_states,
        conv_state,
        prev_hs_state,
        slots,
        window,
        prev_out,
        qk.stride(0),
        hidden_states.stride(0),
        conv_state.stride(0),
        conv_state.stride(1),
        prev_hs_state.stride(0),
        window.stride(0),
        window.stride(1),
        prev_out.stride(0),
        num_channels,
        hidden_size,
        W=taps,
        NC_TILES=nc_tiles,
        BLOCK_C=block_c,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return window, prev_out
