"""Fused ZAYA1 CCA q/k head-mix + normalize (+ partial-rotary RoPE).

One kernel replaces the ~26-launch elementwise tail of the CCA projection --
``_add_grouped_qk_means`` followed by ``_normalize_qk`` in
:class:`CCA <sglang.srt.models.zaya.CCA>`::

    q_out[g,j] = rms(conv_q[g,j] + 0.5*pre_q[g,j] + 0.5*base_k[g])          * sqrt_hd
    k_out[g]   = rms(conv_k[g]   + 0.5*mean_j(pre_q[g,j]) + 0.5*base_k[g])  * sqrt_hd * temp[g]

where ``rms(x) = x * rsqrt(sum(x^2) + eps)`` over the head dim and ``g`` indexes
GQA groups (one k head per group, ``gqa_groups`` q heads inside it). The blend
and the two normalizations are separate torch expression trees today, each
materializing fp32 temporaries of the full q tensor; every intermediate here
stays in registers and only the two results are written.

One program per ``(token, k head)``, holding the whole group as a ``[G, HD]``
tile. The ``G`` q-head RMS sums then reduce together along ``axis=1`` instead of
one serial ``tl.sum`` per head, and ``mean_j(pre_q)`` for the k blend is a
reduction along ``axis=0`` of that same tile rather than a separate running
accumulator -- so ``pre_q`` is read once and the group's ``G + 1`` reductions
collapse to two.

Rotary
------
``ROT_D > 0`` folds the neox partial-rotary RoPE in as well, removing the
separate ``sgl_kernel.rotary_embedding`` launch that ran immediately after this
kernel (one per attention layer, 60 per decode step on ZAYA1-74B). The head is
already in registers, so the rotation is free arithmetic on values that would
otherwise be stored, re-read and stored again.

The shared fused rope path (``models/utils.create_fused_set_kv_buffer_arg`` ->
``fused_qk_rope_reshape_and_cache``) cannot serve ZAYA1: it derives
``d_freq = cos_sin.shape[-1] // 2`` and asserts ``d_freq in (d // 2, d)``. With
``partial_rotary_factor=0.5`` the cache is ``[max_pos, 64]`` against
``head_dim=128``, so ``d_freq == 32`` and the assert fires -- that kernel has no
partial-rotary mode. Hence the rotation lives here instead.

Under ``ROT_D`` the head is loaded as three register tiles rather than one:
``lo = d[0:ROT_D/2]``, ``hi = d[ROT_D/2:ROT_D]`` and ``pass = d[ROT_D:HD]``.
Splitting on load is what makes the neox rotation

    lo' = lo*cos - hi*sin        hi' = hi*cos + lo*sin

pure lane-local arithmetic: ``lo`` and ``hi`` sit in the *same* lane of two
different tiles, so no cross-lane shuffle (``tl.flip`` / ``tl.reshape``, as
``rope_cache._get_neox_rotated_x`` needs) is required. Each element is still
loaded exactly once; only the address arithmetic is split.

ORDERING: the RMS sum runs over all ``HD`` dims and the k temperature is applied
*before* the rotation, matching today's ``_normalize_qk`` -> ``rotary_emb``
order. The rotation is norm-preserving on the rotated half so the two do not
interact numerically, but the order is pinned anyway rather than relied upon.

NOT bit-identical to the unfused chain: on ROCm the cos/sin cache is stored in
the model dtype (``RotaryEmbedding.__init__`` casts it whenever the platform is
not CUDA/XPU and ``SGLANG_ROPE_CACHE_FP32`` is off), and the separate rotary
kernel also receives ``q``/``k`` already rounded to bf16. This kernel keeps the
normalized head in fp32 across the rotation and rounds once, at the store, so
its result is *closer* to the fp64 reference than the chain it replaces -- but
it differs from it in the last bf16 bits.

KV store
--------
``HAS_STORE`` additionally scatters the post-rope ``k`` and the matching ``v``
into the paged KV buffers at ``out_cache_loc[t]``, so ``RadixAttention`` can be
called with ``save_kv_cache=False`` (the pattern ``qwen3_moe`` uses) and the
per-layer ``set_kv_buffer`` launch -- another 60 per decode step -- disappears.
``k`` is already in registers post-rotation, so the only new traffic is reading
``v``, which the store kernel had to read anyway.

The slot resolution mirrors ``rope_cache``'s: ``slot = out_cache_loc[t]``, then
``slot = full_to_swa[slot]`` on a sliding-window layer of a hybrid pool, then
skip when ``slot < 0`` (the sentinel row batch padding maps to). The caller does
the layout gating; :func:`store_covered` only accepts a 3-D ``[rows, H, D]`` NHD
buffer, which by construction excludes the 5-D SHUFFLE layout, the 4-D HND
layout and the page-major strided views. A wrong write here corrupts KV
silently, so both gates are deliberately narrower than the kernel could serve.

Follows the structure of ``kda_fused_decode`` (a ``covered()`` predicate gates
supported inputs, everything else falls back to the unfused chain), but is
written in Triton rather than CUDA-JIT so it runs on ROCm as well -- ZAYA1's
reference deployment is MI350X.

Motivation: an eager decode profile of ZAYA1-base put 56% of GPU time in tiny
elementwise kernels at ~4270 launches per step, and this tail was the second
largest cluster (~400 launches/step, 6.2% of decode).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

# The head dim is loaded as one masked block, so it must fit a single Triton
# block. ZAYA1 uses 128; 256 leaves headroom without hurting occupancy.
_MAX_HEAD_DIM = 256

# A program holds the whole group as one [G, HD] fp32 tile (three of them: the
# blend inputs and the result), so G bounds register pressure rather than just
# a loop count. 16 x 256 fp32 is already 4096 elements per tile; beyond that the
# torch fallback is the better bet. ZAYA1 uses 8.
_MAX_GQA_GROUPS = 16

_RMS_EPS = 1e-12

_INT_DTYPES = (torch.int32, torch.int64)
_FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@triton.jit
def _blend(conv_row, pre_row, half_base_k, off, mask):
    """``conv + 0.5*pre + 0.5*base_k`` for one offset tile, in fp32.

    ``conv_row`` / ``pre_row`` are already advanced to the token's row, and
    ``half_base_k`` is broadcast-compatible with ``off``. Returns the blend and
    the raw ``pre`` load, which the k blend reduces over the group.
    """
    pre = tl.load(pre_row + off, mask=mask, other=0.0).to(tl.float32)
    conv = tl.load(conv_row + off, mask=mask, other=0.0).to(tl.float32)
    return conv + 0.5 * pre + half_base_k, pre


@triton.jit
def _kv_slot(loc_ptr, swa_map_ptr, t, HAS_SWA: tl.constexpr):
    """The physical KV slot for token ``t``, or a negative sentinel to skip it.

    Mirrors ``rope_cache._fused_qk_rope_reshape_and_cache_kernel``: the allocated
    (full-pool) slot, then one indirection through ``full_to_swa`` on a
    sliding-window layer of a hybrid pool. That mapping's trailing ``-1`` entry
    is what makes a padding row map to a skip rather than to slot 0.
    """
    slot = tl.load(loc_ptr + t).to(tl.int64)
    if HAS_SWA:
        # Guard the gather itself: a negative source slot would index before the
        # mapping's base. The shared rope+store kernel reads it unguarded because
        # out_cache_loc is non-negative in practice; guarding costs one select.
        mapped = tl.load(swa_map_ptr + tl.maximum(slot, 0)).to(tl.int64)
        slot = tl.where(slot >= 0, mapped, -1)
    return slot


@triton.jit
def _store_v(value_ptr, v_cache_ptr, t, g, slot, s_v_t, s_v_h, s_vc_t, s_vc_h, off, m):
    """Copy this ``(token, k head)``'s V row into the paged V buffer."""
    v = tl.load(value_ptr + t * s_v_t + g * s_v_h + off, mask=m, other=0.0)
    tl.store(v_cache_ptr + slot * s_vc_t + g * s_vc_h + off, v, mask=m)


@triton.jit
def _cca_qk_mix_kernel(
    conv_qk_ptr,  # [T, latent_q + latent_k] conv output (q segment then k)
    pre_q_ptr,  # [T, latent_q]  raw W_q hidden_states
    base_k_ptr,  # [T, latent_k]  raw W_k hidden_states
    k_scale_ptr,  # [NK] sqrt(head_dim) * per-k-head temperature
    positions_ptr,  # [T] int, or None when ROT_D == 0
    cos_sin_ptr,  # [max_pos, ROT_D] cos half then sin half, or None
    value_ptr,  # [T, NK, HD] V heads, or None when not storing
    k_cache_ptr,  # [rows, NK, HD] NHD paged K buffer, or None
    v_cache_ptr,  # [rows, NK, HD] NHD paged V buffer, or None
    loc_ptr,  # [T] out_cache_loc, or None
    swa_map_ptr,  # [rows+1] full->SWA slot mapping, or None
    q_out_ptr,  # [T, NQ, HD] fp32
    k_out_ptr,  # [T, NK, HD] fp32
    s_conv_t,
    s_pre_t,
    s_base_t,
    s_qout_t,
    s_kout_t,
    s_cos_t,
    s_v_t,
    s_v_h,
    s_kc_t,
    s_kc_h,
    s_vc_t,
    s_vc_h,
    latent_q,
    q_scale,  # sqrt(head_dim)
    eps,
    NK: tl.constexpr,
    G: tl.constexpr,  # gqa_groups: q heads per k head
    HD: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_G: tl.constexpr,  # next_power_of_2(G)
    ROT_D: tl.constexpr,  # rotary_dim, 0 to skip the rotation
    ROT_HALF: tl.constexpr,  # ROT_D // 2
    BLOCK_H: tl.constexpr,  # next_power_of_2(ROT_HALF), 1 when ROT_D == 0
    BLOCK_P: tl.constexpr,  # next_power_of_2(HD - ROT_D), 1 when ROT_D == 0
    HAS_STORE: tl.constexpr = False,
    HAS_SWA: tl.constexpr = False,
):
    pid = tl.program_id(0)
    t = pid // NK
    g = pid % NK

    conv_row = conv_qk_ptr + t * s_conv_t
    pre_row = pre_q_ptr + t * s_pre_t
    base_row = base_k_ptr + t * s_base_t + g * HD
    conv_k_row = conv_row + latent_q + g * HD

    # The group's q heads reduce together: one [BLOCK_G, ...] tile whose rows are
    # the group's q heads, so the G RMS sums collapse to one axis=1 reduction and
    # mean_j(pre_q) to one axis=0 reduction of the same tile.
    j = tl.arange(0, BLOCK_G)[:, None]
    jmask = j < G
    q_row = (g * G + j) * HD  # [BLOCK_G, 1]

    k_scale = tl.load(k_scale_ptr + g).to(tl.float32)
    inv_g_half = 0.5 / G

    if ROT_D == 0:
        d = tl.arange(0, BLOCK)
        dmask = d < HD

        base_k = tl.load(base_row + d, mask=dmask, other=0.0).to(tl.float32)
        half_base_k = 0.5 * base_k

        off = q_row + d[None, :]
        qmask = jmask & (d[None, :] < HD)
        q, pre_q = _blend(conv_row, pre_row, half_base_k[None, :], off, qmask)
        # Masked lanes carry 0, so they add nothing to their row's sum, and the
        # row itself is never stored.
        inv = tl.rsqrt(tl.sum(q * q, axis=1) + eps) * q_scale
        tl.store(q_out_ptr + t * s_qout_t + off, q * inv[:, None], mask=qmask)

        conv_k = tl.load(conv_k_row + d, mask=dmask, other=0.0).to(tl.float32)
        k = conv_k + inv_g_half * tl.sum(pre_q, axis=0) + half_base_k
        inv_k = tl.rsqrt(tl.sum(k * k, axis=0) + eps) * k_scale
        k = k * inv_k
        tl.store(k_out_ptr + t * s_kout_t + g * HD + d, k, mask=dmask)

        if HAS_STORE:
            slot = _kv_slot(loc_ptr, swa_map_ptr, t, HAS_SWA)
            if slot >= 0:
                tl.store(k_cache_ptr + slot * s_kc_t + g * s_kc_h + d, k, mask=dmask)
                _store_v(
                    value_ptr,
                    v_cache_ptr,
                    t,
                    g,
                    slot,
                    s_v_t,
                    s_v_h,
                    s_vc_t,
                    s_vc_h,
                    d,
                    dmask,
                )
    else:
        # Three tiles: the two rotated halves and the pass-through tail. lo and
        # hi share a lane index, which is what keeps the rotation shuffle-free.
        i = tl.arange(0, BLOCK_H)
        imask = i < ROT_HALF
        p = tl.arange(0, BLOCK_P)
        pmask = p < (HD - ROT_D)

        hb_lo = 0.5 * tl.load(base_row + i, mask=imask, other=0.0).to(tl.float32)
        hb_hi = 0.5 * tl.load(base_row + ROT_HALF + i, mask=imask, other=0.0).to(
            tl.float32
        )
        hb_pa = 0.5 * tl.load(base_row + ROT_D + p, mask=pmask, other=0.0).to(
            tl.float32
        )

        off_lo = q_row + i[None, :]
        off_hi = q_row + ROT_HALF + i[None, :]
        off_pa = q_row + ROT_D + p[None, :]
        qm_h = jmask & imask[None, :]
        qm_p = jmask & pmask[None, :]

        q_lo, pre_lo = _blend(conv_row, pre_row, hb_lo[None, :], off_lo, qm_h)
        q_hi, pre_hi = _blend(conv_row, pre_row, hb_hi[None, :], off_hi, qm_h)
        q_pa, pre_pa = _blend(conv_row, pre_row, hb_pa[None, :], off_pa, qm_p)

        # RMS over the WHOLE head (all three tiles) and before the rotation --
        # the order the unfused chain uses.
        ssq = (
            tl.sum(q_lo * q_lo, axis=1)
            + tl.sum(q_hi * q_hi, axis=1)
            + tl.sum(q_pa * q_pa, axis=1)
        )
        inv = tl.rsqrt(ssq + eps) * q_scale
        q_lo = q_lo * inv[:, None]
        q_hi = q_hi * inv[:, None]
        q_pa = q_pa * inv[:, None]

        pos = tl.load(positions_ptr + t).to(tl.int64)
        cos_row = cos_sin_ptr + pos * s_cos_t
        cos = tl.load(cos_row + i, mask=imask, other=0.0).to(tl.float32)
        sin = tl.load(cos_row + ROT_HALF + i, mask=imask, other=0.0).to(tl.float32)

        q_out_row = q_out_ptr + t * s_qout_t
        cos2 = cos[None, :]
        sin2 = sin[None, :]
        tl.store(q_out_row + off_lo, q_lo * cos2 - q_hi * sin2, mask=qm_h)
        tl.store(q_out_row + off_hi, q_hi * cos2 + q_lo * sin2, mask=qm_h)
        tl.store(q_out_row + off_pa, q_pa, mask=qm_p)

        ck_lo = tl.load(conv_k_row + i, mask=imask, other=0.0).to(tl.float32)
        ck_hi = tl.load(conv_k_row + ROT_HALF + i, mask=imask, other=0.0)
        ck_hi = ck_hi.to(tl.float32)
        ck_pa = tl.load(conv_k_row + ROT_D + p, mask=pmask, other=0.0).to(tl.float32)
        k_lo = ck_lo + inv_g_half * tl.sum(pre_lo, axis=0) + hb_lo
        k_hi = ck_hi + inv_g_half * tl.sum(pre_hi, axis=0) + hb_hi
        k_pa = ck_pa + inv_g_half * tl.sum(pre_pa, axis=0) + hb_pa

        ssq_k = (
            tl.sum(k_lo * k_lo, axis=0)
            + tl.sum(k_hi * k_hi, axis=0)
            + tl.sum(k_pa * k_pa, axis=0)
        )
        inv_k = tl.rsqrt(ssq_k + eps) * k_scale
        k_lo = k_lo * inv_k
        k_hi = k_hi * inv_k
        k_pa = k_pa * inv_k
        k_lo_r = k_lo * cos - k_hi * sin
        k_hi_r = k_hi * cos + k_lo * sin

        k_out_row = k_out_ptr + t * s_kout_t + g * HD
        tl.store(k_out_row + i, k_lo_r, mask=imask)
        tl.store(k_out_row + ROT_HALF + i, k_hi_r, mask=imask)
        tl.store(k_out_row + ROT_D + p, k_pa, mask=pmask)

        if HAS_STORE:
            slot = _kv_slot(loc_ptr, swa_map_ptr, t, HAS_SWA)
            if slot >= 0:
                # The SAME fp32 expression that went to k_out is rounded into the
                # pool, so the fused store is bit-identical to the set_kv_buffer
                # copy it replaces (both round the fp32 result once).
                kc_row = k_cache_ptr + slot * s_kc_t + g * s_kc_h
                tl.store(kc_row + i, k_lo_r, mask=imask)
                tl.store(kc_row + ROT_HALF + i, k_hi_r, mask=imask)
                tl.store(kc_row + ROT_D + p, k_pa, mask=pmask)
                dv = tl.arange(0, BLOCK)
                _store_v(
                    value_ptr,
                    v_cache_ptr,
                    t,
                    g,
                    slot,
                    s_v_t,
                    s_v_h,
                    s_vc_t,
                    s_vc_h,
                    dv,
                    dv < HD,
                )


def covered(
    conv_qk: torch.Tensor,
    pre_q: torch.Tensor,
    base_k: torch.Tensor,
    k_scale: Optional[torch.Tensor],
    num_q_heads: int,
    num_k_heads: int,
    head_dim: int,
) -> bool:
    """Whether the fused kernel can serve these inputs.

    Requires a whole number of q heads per k head (ZAYA1 always splits evenly)
    and few enough of them to hold the group as one register tile, a head dim
    that fits one Triton block, row-major 2-D inputs with a unit innermost
    stride, and float inputs on an accelerator. ``k_scale`` is the folded
    ``sqrt(head_dim) * temperature`` vector, absent until weights load.

    Says nothing about the rotary fusion -- see :func:`rope_covered`, which is a
    strictly additional gate. The mix can fuse while the rotation falls back.
    """
    if k_scale is None:
        return False
    if not (conv_qk.is_cuda and pre_q.is_cuda and base_k.is_cuda):
        return False
    if head_dim > _MAX_HEAD_DIM or head_dim <= 0:
        return False
    if num_k_heads <= 0 or num_q_heads % num_k_heads != 0:
        return False
    if num_q_heads // num_k_heads > _MAX_GQA_GROUPS:
        return False
    if conv_qk.ndim != 2 or pre_q.ndim != 2 or base_k.ndim != 2:
        return False
    if conv_qk.shape[-1] != (num_q_heads + num_k_heads) * head_dim:
        return False
    if pre_q.shape[-1] != num_q_heads * head_dim:
        return False
    if base_k.shape[-1] != num_k_heads * head_dim:
        return False
    if not (
        conv_qk.stride(-1) == 1 and pre_q.stride(-1) == 1 and base_k.stride(-1) == 1
    ):
        return False
    if k_scale.numel() != num_k_heads or not k_scale.is_contiguous():
        return False
    return all(t.dtype in _FLOAT_DTYPES for t in (conv_qk, pre_q, base_k))


def _same_device(tensor: torch.Tensor, device: torch.device) -> bool:
    """Device equality that tolerates an un-indexed reference device.

    ``torch.device("cuda") != torch.device("cuda:0")`` -- the index is part of
    the identity -- while every tensor reports an indexed device. A caller that
    writes the reference device by hand rather than reading it off a tensor would
    otherwise see every check fail, and because these predicates decline by
    *falling back*, that reads as "the fusion is off" with no error anywhere. An
    un-indexed reference means "any device of this type", so honor that.
    """
    got = tensor.device
    if got.type != device.type:
        return False
    return device.index is None or got.index == device.index


def rope_geometry_decline_reason(
    rotary_dim: int, head_dim: int, is_neox_style: bool = True
) -> Optional[str]:
    """Why the head shape is not one the three-tile split can express, or ``None``.

    * ``rotary_dim == head_dim // 2`` -- ZAYA1's ``partial_rotary_factor=0.5``.
      Any other split changes the tile geometry (the pass-through tail stops
      being the same width as the rotated part) and is rejected rather than
      guessed at.
    * ``rotary_dim // 2`` even -- ``lo`` and ``hi`` are addressed as two tiles of
      ``rotary_dim // 2`` lanes and the cos/sin cache is read at the same
      granularity, so an odd half is refused instead of relying on the mask to
      paper over a half-lane.
    * neox layout. GPT-J interleaves the rotated pair *within* a lane, which is
      exactly the cross-lane case this kernel avoids by splitting on load.

    Split out from ``rope_decline_reason`` so the geometry can be pinned without
    a GPU.
    """
    if not is_neox_style:
        return "gptj layout (the rotated pair is intra-lane)"
    if rotary_dim <= 0 or head_dim <= 0:
        return f"degenerate dims rotary_dim={rotary_dim} head_dim={head_dim}"
    if rotary_dim != head_dim // 2:
        return f"rotary_dim {rotary_dim} != head_dim//2 {head_dim // 2}"
    if (rotary_dim // 2) % 2 != 0:
        return f"odd rotary half {rotary_dim // 2}"
    return None


def rope_geometry_covered(
    rotary_dim: int, head_dim: int, is_neox_style: bool = True
) -> bool:
    """Whether the head shape fits the three-tile split. See the reason variant."""
    return rope_geometry_decline_reason(rotary_dim, head_dim, is_neox_style) is None


def rope_decline_reason(
    positions: Optional[torch.Tensor],
    cos_sin_cache: Optional[torch.Tensor],
    rotary_dim: int,
    *,
    head_dim: int,
    num_tokens: int,
    is_neox_style: bool = True,
    device: Optional[torch.device] = None,
) -> Optional[str]:
    """Why the neox partial rotary cannot fold into the mix kernel, or ``None``.

    Deliberately narrow: :func:`rope_geometry_decline_reason` for the head shape,
    plus a ``[max_pos, rotary_dim]`` cache (cos half then sin half) with a unit
    innermost stride and a 1-D integer ``positions``, both on the inputs' device.

    Everything it rejects still gets the fused mix plus a separate rotary launch,
    i.e. today's behavior. Returning the *reason* rather than a bare bool is what
    keeps such a decline visible: it feeds both the once-per-outcome fusion log
    and the tests' assertion messages. The string is built only on decline, so
    the accepted path pays nothing for it.
    """
    if positions is None or cos_sin_cache is None:
        return "no rotary offered"
    geometry = rope_geometry_decline_reason(rotary_dim, head_dim, is_neox_style)
    if geometry is not None:
        return geometry
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[-1] != rotary_dim:
        return (
            f"cos_sin_cache {tuple(cos_sin_cache.shape)} is not "
            f"[max_pos, {rotary_dim}]"
        )
    if cos_sin_cache.stride(-1) != 1:
        return "cos_sin_cache innermost stride != 1"
    if cos_sin_cache.dtype not in _FLOAT_DTYPES:
        return f"cos_sin_cache dtype {cos_sin_cache.dtype}"
    if positions.ndim != 1 or positions.numel() != num_tokens:
        return f"positions {tuple(positions.shape)} is not 1-D of {num_tokens}"
    if positions.dtype not in _INT_DTYPES:
        return f"positions dtype {positions.dtype}"
    if not positions.is_contiguous():
        return "positions not contiguous"
    if not (positions.is_cuda and cos_sin_cache.is_cuda):
        return "positions / cos_sin_cache not on an accelerator"
    if device is not None:
        for name, t in (("positions", positions), ("cos_sin_cache", cos_sin_cache)):
            if not _same_device(t, device):
                return f"{name} on {t.device}, inputs on {device}"
    return None


def rope_covered(
    positions: Optional[torch.Tensor],
    cos_sin_cache: Optional[torch.Tensor],
    rotary_dim: int,
    *,
    head_dim: int,
    num_tokens: int,
    is_neox_style: bool = True,
    device: Optional[torch.device] = None,
) -> bool:
    """Whether the neox partial rotary can fold in. See the reason variant."""
    return (
        rope_decline_reason(
            positions,
            cos_sin_cache,
            rotary_dim,
            head_dim=head_dim,
            num_tokens=num_tokens,
            is_neox_style=is_neox_style,
            device=device,
        )
        is None
    )


def store_decline_reason(
    value: Optional[torch.Tensor],
    k_cache: Optional[torch.Tensor],
    v_cache: Optional[torch.Tensor],
    out_cache_loc: Optional[torch.Tensor],
    full_to_swa: Optional[torch.Tensor],
    *,
    num_k_heads: int,
    head_dim: int,
    num_tokens: int,
    out_dtype: torch.dtype,
    device: torch.device,
) -> Optional[str]:
    """Why the KV scatter cannot fold into the mix kernel, or ``None``.

    A wrong write here does not crash -- it corrupts KV and shows up as degraded
    output quality -- so this gate is deliberately narrower than the kernel could
    serve, and every check below is a hard reject rather than a fixup:

    * **3-D ``[rows, heads, dim]`` K/V buffers only.** That is the plain NHD pool
      layout, where the write target is a flat slot row and needs no page
      arithmetic at all. It is exactly the shape ``_set_kv_buffer_impl`` writes.
      Requiring 3-D is what rules out the other layouts by construction: the 5-D
      SHUFFLE (vectorized_5d) buffers, the 4-D HND ``(page, head, off, dim)``
      buffers and the page-major strided views are all rejected without this
      needing to know which one it is looking at. The caller pins the page-size
      side of that invariant by checking ``rows == pool.size + pool.page_size``,
      which is what makes the flat slot index correct for ANY page size and also
      rejects the placeholder buffers of a no-op pool.
    * **matching dtypes.** ``k_cache.dtype == v_cache.dtype == out_dtype`` is the
      bf16 gate: an fp8 or fp4 pool stores under a different ``store_dtype`` (and
      needs per-tensor scales the kernel does not apply), so it falls back.
    * unit innermost strides on both buffers and on ``value`` -- the head axis
      may be strided (a per-rank slice of the replicated V projection is), which
      is why ``s_v_h`` is passed rather than assumed.
    * a 1-D integer ``out_cache_loc`` of exactly ``num_tokens`` entries, and an
      int64 ``full_to_swa`` when one is supplied. ``full_to_swa`` is indexed by
      FULL-pool slot id, not by a row of ``k_cache`` (which is the SWA sub-pool
      here), so its length is the caller's invariant to check, not this one's.

    Like the rope gate, it answers with the reason so a decline is legible rather
    than a bare False.
    """
    for name, t in (
        ("value", value),
        ("k_cache", k_cache),
        ("v_cache", v_cache),
        ("out_cache_loc", out_cache_loc),
    ):
        if t is None:
            return f"{name} not supplied"
    if k_cache.ndim != 3 or v_cache.ndim != 3:
        return (
            f"kv buffers are not 3-D NHD (k={tuple(k_cache.shape)}, "
            f"v={tuple(v_cache.shape)}); the 5-D SHUFFLE, 4-D HND and "
            "page-major layouts are not served"
        )
    for name, buf in (("k_cache", k_cache), ("v_cache", v_cache)):
        if buf.shape[1] != num_k_heads or buf.shape[2] != head_dim:
            return f"{name} {tuple(buf.shape)} is not [rows, {num_k_heads}, {head_dim}]"
        if buf.dtype != out_dtype:
            return f"{name} dtype {buf.dtype} != out_dtype {out_dtype}"
        if buf.stride(-1) != 1:
            return f"{name} innermost stride != 1"
    if value.ndim != 3 or tuple(value.shape) != (num_tokens, num_k_heads, head_dim):
        return (
            f"value {tuple(value.shape)} is not "
            f"[{num_tokens}, {num_k_heads}, {head_dim}]"
        )
    if value.dtype != out_dtype:
        return f"value dtype {value.dtype} != out_dtype {out_dtype}"
    if value.stride(-1) != 1:
        return "value innermost stride != 1"
    if out_cache_loc.ndim != 1 or out_cache_loc.numel() != num_tokens:
        return f"out_cache_loc {tuple(out_cache_loc.shape)} is not 1-D of {num_tokens}"
    if out_cache_loc.dtype not in _INT_DTYPES:
        return f"out_cache_loc dtype {out_cache_loc.dtype}"
    if not out_cache_loc.is_contiguous():
        return "out_cache_loc not contiguous"
    if full_to_swa is not None:
        if full_to_swa.ndim != 1 or full_to_swa.dtype != torch.int64:
            return (
                f"full_to_swa {tuple(full_to_swa.shape)}/{full_to_swa.dtype} is "
                "not a 1-D int64 mapping"
            )
        if not full_to_swa.is_contiguous() or full_to_swa.numel() == 0:
            return "full_to_swa empty or not contiguous"
    # ``device`` is required rather than inferred: the accelerator check already
    # happened in ``covered()`` (which gates this one), so what is left to catch
    # is a buffer that belongs to a *different* device than the inputs -- and
    # making it explicit is also what lets these branches be tested on CPU.
    for name, t in (
        ("value", value),
        ("k_cache", k_cache),
        ("v_cache", v_cache),
        ("out_cache_loc", out_cache_loc),
        ("full_to_swa", full_to_swa),
    ):
        if t is not None and not _same_device(t, device):
            return f"{name} on {t.device}, inputs on {device}"
    return None


def store_covered(
    value: Optional[torch.Tensor],
    k_cache: Optional[torch.Tensor],
    v_cache: Optional[torch.Tensor],
    out_cache_loc: Optional[torch.Tensor],
    full_to_swa: Optional[torch.Tensor],
    *,
    num_k_heads: int,
    head_dim: int,
    num_tokens: int,
    out_dtype: torch.dtype,
    device: torch.device,
) -> bool:
    """Whether the KV scatter can fold in. See the reason variant."""
    return (
        store_decline_reason(
            value,
            k_cache,
            v_cache,
            out_cache_loc,
            full_to_swa,
            num_k_heads=num_k_heads,
            head_dim=head_dim,
            num_tokens=num_tokens,
            out_dtype=out_dtype,
            device=device,
        )
        is None
    )


def cca_qk_mix(
    conv_qk: torch.Tensor,
    pre_q: torch.Tensor,
    base_k: torch.Tensor,
    k_scale: torch.Tensor,
    *,
    num_q_heads: int,
    num_k_heads: int,
    head_dim: int,
    q_scale: float,
    eps: float = _RMS_EPS,
    out_dtype: torch.dtype = torch.float32,
    positions: Optional[torch.Tensor] = None,
    cos_sin_cache: Optional[torch.Tensor] = None,
    rotary_dim: int = 0,
    value: Optional[torch.Tensor] = None,
    k_cache: Optional[torch.Tensor] = None,
    v_cache: Optional[torch.Tensor] = None,
    out_cache_loc: Optional[torch.Tensor] = None,
    full_to_swa: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(q, k)`` as ``[T, heads, head_dim]`` in ``out_dtype``.

    Accumulation is always fp32 inside the kernel; ``out_dtype`` only picks the
    store precision. Writing the model dtype directly saves the caller a cast --
    and the cast is all it was, since the fp32 result was rounded to the model
    dtype immediately afterwards.

    Passing ``positions`` / ``cos_sin_cache`` / ``rotary_dim`` also applies the
    neox partial rotary to both outputs, removing the separate rotary launch.
    Passing ``value`` / ``k_cache`` / ``v_cache`` / ``out_cache_loc`` also
    scatters k and v into the paged buffers, removing the ``set_kv_buffer``
    launch; ``full_to_swa`` adds the hybrid-pool slot indirection.

    Caller must have checked :func:`covered`, plus :func:`rope_covered` and
    :func:`store_covered` for whichever extra arguments it passes.
    """
    num_tokens = conv_qk.shape[0]
    q_out = torch.empty(
        (num_tokens, num_q_heads, head_dim), dtype=out_dtype, device=conv_qk.device
    )
    k_out = torch.empty(
        (num_tokens, num_k_heads, head_dim), dtype=out_dtype, device=conv_qk.device
    )
    if num_tokens == 0:
        return q_out, k_out

    rot_d = int(rotary_dim) if positions is not None else 0
    if rot_d:
        rot_half = rot_d // 2
        block_h = triton.next_power_of_2(rot_half)
        block_p = triton.next_power_of_2(head_dim - rot_d)
        s_cos_t = cos_sin_cache.stride(0)
    else:
        rot_half, block_h, block_p, s_cos_t = 0, 1, 1, 0

    has_store = k_cache is not None
    if has_store:
        s_v_t, s_v_h = value.stride(0), value.stride(1)
        s_kc_t, s_kc_h = k_cache.stride(0), k_cache.stride(1)
        s_vc_t, s_vc_h = v_cache.stride(0), v_cache.stride(1)
    else:
        s_v_t = s_v_h = s_kc_t = s_kc_h = s_vc_t = s_vc_h = 0

    _cca_qk_mix_kernel[(num_tokens * num_k_heads,)](
        conv_qk,
        pre_q,
        base_k,
        k_scale,
        positions if rot_d else None,
        cos_sin_cache if rot_d else None,
        value if has_store else None,
        k_cache,
        v_cache,
        out_cache_loc if has_store else None,
        full_to_swa if has_store else None,
        q_out,
        k_out,
        conv_qk.stride(0),
        pre_q.stride(0),
        base_k.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        s_cos_t,
        s_v_t,
        s_v_h,
        s_kc_t,
        s_kc_h,
        s_vc_t,
        s_vc_h,
        num_q_heads * head_dim,
        float(q_scale),
        float(eps),
        NK=num_k_heads,
        G=num_q_heads // num_k_heads,
        HD=head_dim,
        BLOCK=triton.next_power_of_2(head_dim),
        BLOCK_G=triton.next_power_of_2(num_q_heads // num_k_heads),
        ROT_D=rot_d,
        ROT_HALF=rot_half,
        BLOCK_H=block_h,
        BLOCK_P=block_p,
        HAS_STORE=has_store,
        HAS_SWA=(has_store and full_to_swa is not None),
        # One warp, not four. The reductions are over the head dim (ZAYA1: 128
        # elements), so 4 warps is 256 ROCm lanes per 128-element row: half of
        # them idle, and each ``tl.sum`` becomes a cross-wavefront LDS reduction
        # with the barriers that implies. At one warp the block is 64 lanes and
        # the reduction stays inside the wavefront. Worth re-sweeping now that a
        # program carries a [G, HD] tile rather than a single row.
        num_warps=1,
    )
    return q_out, k_out
