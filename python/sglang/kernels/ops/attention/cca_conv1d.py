"""Fused ZAYA1 CCA prefill conv, varlen and driven entirely by device tensors.

Replaces the per-request host loop in :func:`cca_extend
<sglang.srt.models.zaya.cca_extend>`. That loop walks the batch on the CPU and
issues, per request, a pack copy, an unpack copy, a conv-state write, a
``torch.cat`` and a ``prev_hs`` write -- so its launch count grows with the batch
and its trip count comes from ``extend_seq_lens_cpu``. Reading CPU sequence
lengths is also what keeps the prefill CUDA graph uncapturable.

This path is four launches per layer whatever the batch size, and every bound it
needs (request of a token, that request's start, its pool slot, whether it
resumes a prefix) is read from device tensors the backend already publishes
(``query_start_loc``, ``cache_indices``, ``has_initial_state``):

1. a plain shifted copy for ``v2_input[1:] = hidden_states[:-1]``,
2. :func:`_boundary_state_kernel` -- fixes the ``v2_input`` row at each request
   start from that request's cached ``prev_hs`` slot, and carries the request's
   last hidden state into the slot for the next chunk,
3. :func:`_cca_conv1d_varlen_kernel` -- the conv itself, tiled over tokens,
4. :func:`_cca_conv_state_tail_kernel` -- writes each request's trailing window
   back to its ``conv_state`` slot.

Order matters and is load-bearing. Step 2 reads a slot it then overwrites within
one program, so the read/write pair on a row is safe. Step 4 must follow step 3,
because step 3 reads the *incoming* ``conv_state`` for the halo taps of the first
tokens of a resumed request; writing the outgoing window first would corrupt it.

The conv consumes the single folded grouped weight (``CCA.fold_conv1d_weight``,
see :meth:`CCA.fold_decode_conv`), not the two-stage ``conv_qk`` -- the fold is
itself a convolution, so one 3-tap grouped kernel is exactly equivalent.

Tiling over tokens rather than one program per token is what makes this
affordable: the per-group weight is ``Cg x Cg x taps`` (98 KB at ZAYA1's
Cg=128, taps=3), so a program-per-token grid would re-read it 4096 times per
layer -- gigabytes of weight traffic for a conv whose arithmetic is trivial.

Written in Triton so it runs on ROCm, ZAYA1's reference deployment. Follows
``cca_state_step``'s structure: a ``covered()`` predicate gates supported inputs
and the caller falls back to the reference torch path.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _boundary_state_kernel(
    hs_ptr,  # [T, H]      this chunk's hidden states
    prev_hs_ptr,  # [S, H, 1]   per-slot previous hidden state
    v2_ptr,  # [T, H]      in/out: already holds the shifted copy
    cu_ptr,  # [B+1]       request token offsets
    slots_ptr,  # [B]         pool slot per request
    prefix_ptr,  # [B]         request resumes a cached prefix
    s_hs_t,
    s_ph_s,
    s_v2_t,
    hidden_size,
    BLOCK_H: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    hmask = h < hidden_size

    start = tl.load(cu_ptr + b)
    end = tl.load(cu_ptr + b + 1)
    # Batch padding leaves zero-length requests; they own no tokens and their
    # slot must not be touched.
    if end > start:
        slot = tl.load(slots_ptr + b)
        prefix = tl.load(prefix_ptr + b)

        # The first token's val_proj2 input is the request's carried hidden
        # state, or zero on a fresh chunk. Read it before the store below
        # overwrites the same row.
        prev = tl.load(prev_hs_ptr + slot * s_ph_s + h, mask=hmask & prefix, other=0.0)
        tl.store(v2_ptr + start * s_v2_t + h, prev, mask=hmask)

        last = tl.load(hs_ptr + (end - 1) * s_hs_t + h, mask=hmask, other=0.0)
        tl.store(prev_hs_ptr + slot * s_ph_s + h, last, mask=hmask)


@triton.jit
def _cca_conv1d_varlen_kernel(
    qk_ptr,  # [T, C]        pre-conv q/k rows
    w_ptr,  # [C, CG, TAPS] folded grouped-conv weight
    bias_ptr,  # [C]
    conv_state_ptr,  # [S, C, PAD]   incoming per-slot history
    out_ptr,  # [T, C]        out: conv output
    cu_ptr,  # [B+1]
    slots_ptr,  # [B]
    prefix_ptr,  # [B]
    s_qk_t,
    s_w_o,
    s_w_i,
    s_cs_s,
    s_cs_c,
    s_out_t,
    num_tokens,
    num_requests,
    CG: tl.constexpr,  # channels per group
    TAPS: tl.constexpr,  # PAD + 1
    PAD: tl.constexpr,  # total_padding
    BLOCK_T: tl.constexpr,
    SEARCH: tl.constexpr,  # ceil(log2(num_requests)) binary-search steps
):
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    g = tl.program_id(1)

    tmask = t < num_tokens
    tsafe = tl.where(tmask, t, 0)

    # Which request owns each token: the largest b with cu[b] <= t. Taking the
    # *largest* is what makes zero-length (padded) requests fall out -- for them
    # cu[b] == cu[b+1], so the next request wins the tie.
    lo = tl.zeros([BLOCK_T], dtype=tl.int32)
    hi = tl.full([BLOCK_T], num_requests, dtype=tl.int32)
    for _ in tl.static_range(SEARCH):
        mid = (lo + hi) // 2
        take = tl.load(cu_ptr + mid) <= tsafe
        lo = tl.where(take, mid, lo)
        hi = tl.where(take, hi, mid)

    req_start = tl.load(cu_ptr + lo)
    slot = tl.load(slots_ptr + lo)
    prefix = tl.load(prefix_ptr + lo)
    local = tsafe - req_start

    co = g * CG + tl.arange(0, CG)  # this group's output channels
    ci = tl.arange(0, CG)  # in-channel index within the group

    acc = tl.zeros([BLOCK_T, CG], dtype=tl.float32)
    for m in tl.static_range(TAPS):
        # Tap m of the causal window reads request position local-(TAPS-1)+m.
        pos = local - (TAPS - 1) + m
        in_seq = pos >= 0
        # When the tap lies inside this chunk its global row is just
        # t-(TAPS-1)+m, independent of which request the token belongs to.
        src = tl.maximum(tsafe - (TAPS - 1) + m, 0)
        x = tl.load(
            qk_ptr + src[:, None] * s_qk_t + co[None, :],
            mask=tmask[:, None] & in_seq[:, None],
            other=0.0,
        )
        # Otherwise it is a halo tap from the carried history, whose tap index
        # is PAD+pos (pos in [-PAD, 0), so this lands in [0, PAD)).
        halo = tl.load(
            conv_state_ptr
            + slot[:, None] * s_cs_s
            + co[None, :] * s_cs_c
            + tl.maximum(PAD + pos, 0)[:, None],
            mask=tmask[:, None] & (~in_seq[:, None]) & prefix[:, None],
            other=0.0,
        )
        xm = tl.where(in_seq[:, None], x, halo)
        # Load the tap's weight slice as [ci, co] so the dot contracts over ci.
        wm = tl.load(w_ptr + co[None, :] * s_w_o + ci[:, None] * s_w_i + m)
        acc += tl.dot(xm, wm, out_dtype=tl.float32)

    acc = acc + tl.load(bias_ptr + co)[None, :]
    tl.store(
        out_ptr + tsafe[:, None] * s_out_t + co[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=tmask[:, None],
    )


@triton.jit
def _cca_conv_state_tail_kernel(
    qk_ptr,  # [T, C]
    conv_state_ptr,  # [S, C, PAD]  in/out
    cu_ptr,  # [B+1]
    slots_ptr,  # [B]
    prefix_ptr,  # [B]
    s_qk_t,
    s_cs_s,
    s_cs_c,
    num_channels,
    PAD: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = c < num_channels

    start = tl.load(cu_ptr + b)
    end = tl.load(cu_ptr + b + 1)
    if end > start:
        slot = tl.load(slots_ptr + b)
        prefix = tl.load(prefix_ptr + b)
        seq_len = end - start

        for i in tl.static_range(PAD):
            # Outgoing tap i is request position seq_len-PAD+i.
            pos = seq_len - PAD + i
            from_chunk = pos >= 0
            v_chunk = tl.load(
                qk_ptr + tl.maximum(start + pos, 0) * s_qk_t + c,
                mask=cmask & from_chunk,
                other=0.0,
            )
            # A chunk shorter than the window keeps its oldest taps from the
            # incoming history, shifted left by seq_len. Reads index seq_len+i,
            # which is always above the taps already written (0..i-1), so the
            # in-place update does not alias.
            v_carry = tl.load(
                conv_state_ptr
                + slot * s_cs_s
                + c * s_cs_c
                + tl.minimum(pos + PAD, PAD - 1),
                mask=cmask & (~from_chunk) & prefix,
                other=0.0,
            )
            tl.store(
                conv_state_ptr + slot * s_cs_s + c * s_cs_c + i,
                tl.where(from_chunk, v_chunk, v_carry),
                mask=cmask,
            )


def covered(
    qk: torch.Tensor,
    hidden_states: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    conv_state: torch.Tensor,
    prev_hs_state: torch.Tensor,
    query_start_loc: Optional[torch.Tensor],
    has_prefix: Optional[torch.Tensor],
    slots: Optional[torch.Tensor],
    total_padding: int,
    groups: int,
) -> bool:
    """Whether the fused prefill conv can serve these inputs.

    Needs the folded single grouped weight (the two-stage ``conv_qk`` is not
    expressible here), the backend's device-side request metadata, everything on
    one accelerator with unit innermost strides, and a channels-per-group that
    ``tl.dot`` can take as a tile width.
    """
    if weight is None or bias is None:
        return False
    if query_start_loc is None or has_prefix is None or slots is None:
        return False
    if total_padding < 1 or groups < 1:
        return False

    tensors = (
        qk,
        hidden_states,
        weight,
        bias,
        conv_state,
        prev_hs_state,
        query_start_loc,
        has_prefix,
        slots,
    )
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
    # tl.dot tiles are power-of-two and at least 16 wide; ZAYA1's is head_dim.
    if ch_per_group < 16 or (ch_per_group & (ch_per_group - 1)) != 0:
        return False

    taps = total_padding + 1
    if tuple(weight.shape) != (num_channels, ch_per_group, taps):
        return False
    if bias.ndim != 1 or bias.shape[0] != num_channels:
        return False
    if conv_state.shape[-1] != total_padding or conv_state.shape[-2] != num_channels:
        return False
    if prev_hs_state.shape[-2] != hidden_states.shape[-1]:
        return False
    if prev_hs_state.shape[-1] != 1:
        return False

    num_requests = query_start_loc.shape[0] - 1
    if num_requests < 1:
        return False
    if slots.shape[0] < num_requests or has_prefix.shape[0] < num_requests:
        return False
    if not (
        query_start_loc.is_contiguous()
        and slots.is_contiguous()
        and has_prefix.is_contiguous()
    ):
        return False

    # The kernels store qk straight into the conv pool and hidden_states into the
    # prev_hs pool, and read the weight at the activation dtype, so no dtype
    # conversion is modelled.
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


def cca_conv1d_fn(
    qk: torch.Tensor,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    conv_state: torch.Tensor,
    prev_hs_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    has_prefix: torch.Tensor,
    slots: torch.Tensor,
    total_padding: int,
    groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(qk_out, v2_input)``, updating both pool slots in place.

    ``qk_out`` is the conv output ``[T, C]`` and ``v2_input`` the shifted
    ``val_proj2`` input ``[T, H]``, both in the activation dtype. Caller must have
    checked :func:`covered`.
    """
    num_tokens, num_channels = qk.shape
    hidden_size = hidden_states.shape[-1]
    num_requests = query_start_loc.shape[0] - 1
    ch_per_group = num_channels // groups
    taps = total_padding + 1

    qk_out = torch.empty_like(qk)
    v2_input = torch.empty_like(hidden_states)
    if num_tokens == 0:
        return qk_out, v2_input

    # val_proj2 reads the previous token. Inside a request that is a plain shift;
    # the row at each request start is wrong here and is fixed next.
    v2_input[1:].copy_(hidden_states[:-1])

    grid_h = triton.cdiv(hidden_size, 512)
    _boundary_state_kernel[(num_requests, grid_h)](
        hidden_states,
        prev_hs_state,
        v2_input,
        query_start_loc,
        slots,
        has_prefix,
        hidden_states.stride(0),
        prev_hs_state.stride(0),
        v2_input.stride(0),
        hidden_size,
        BLOCK_H=512,
        num_warps=4,
    )

    block_t = 64
    search_steps = max(1, (num_requests - 1).bit_length())
    _cca_conv1d_varlen_kernel[(triton.cdiv(num_tokens, block_t), groups)](
        qk,
        weight,
        bias,
        conv_state,
        qk_out,
        query_start_loc,
        slots,
        has_prefix,
        qk.stride(0),
        weight.stride(0),
        weight.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        qk_out.stride(0),
        num_tokens,
        num_requests,
        CG=ch_per_group,
        TAPS=taps,
        PAD=total_padding,
        BLOCK_T=block_t,
        SEARCH=search_steps,
        num_warps=4,
    )

    # Strictly after the conv: it reads the incoming history for halo taps.
    _cca_conv_state_tail_kernel[(num_requests, triton.cdiv(num_channels, 256))](
        qk,
        conv_state,
        query_start_loc,
        slots,
        has_prefix,
        qk.stride(0),
        conv_state.stride(0),
        conv_state.stride(1),
        num_channels,
        PAD=total_padding,
        BLOCK_C=256,
        num_warps=4,
    )
    return qk_out, v2_input
