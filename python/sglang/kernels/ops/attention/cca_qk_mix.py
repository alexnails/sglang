"""Fused ZAYA1 CCA q/k head-mix + normalize.

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

_RMS_EPS = 1e-12


@triton.jit
def _cca_qk_mix_kernel(
    conv_qk_ptr,  # [T, latent_q + latent_k] conv output (q segment then k)
    pre_q_ptr,  # [T, latent_q]  raw W_q hidden_states
    base_k_ptr,  # [T, latent_k]  raw W_k hidden_states
    k_scale_ptr,  # [NK] sqrt(head_dim) * per-k-head temperature
    q_out_ptr,  # [T, NQ, HD] fp32
    k_out_ptr,  # [T, NK, HD] fp32
    s_conv_t,
    s_pre_t,
    s_base_t,
    s_qout_t,
    s_kout_t,
    latent_q,
    q_scale,  # sqrt(head_dim)
    eps,
    NK: tl.constexpr,
    G: tl.constexpr,  # gqa_groups: q heads per k head
    HD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid // NK
    g = pid % NK

    d = tl.arange(0, BLOCK)
    mask = d < HD

    # base_k for this group, reused by every q head in the group and by k.
    base_k = tl.load(base_k_ptr + t * s_base_t + g * HD + d, mask=mask, other=0.0).to(
        tl.float32
    )
    half_base_k = 0.5 * base_k

    pre_q_sum = tl.zeros([BLOCK], dtype=tl.float32)

    for j in tl.static_range(G):
        head = g * G + j
        pre_q = tl.load(
            pre_q_ptr + t * s_pre_t + head * HD + d, mask=mask, other=0.0
        ).to(tl.float32)
        conv_q = tl.load(
            conv_qk_ptr + t * s_conv_t + head * HD + d, mask=mask, other=0.0
        ).to(tl.float32)

        q = conv_q + 0.5 * pre_q + half_base_k
        inv = tl.rsqrt(tl.sum(q * q, axis=0) + eps) * q_scale
        tl.store(q_out_ptr + t * s_qout_t + head * HD + d, q * inv, mask=mask)

        pre_q_sum += pre_q

    conv_k = tl.load(
        conv_qk_ptr + t * s_conv_t + latent_q + g * HD + d, mask=mask, other=0.0
    ).to(tl.float32)
    k = conv_k + (0.5 / G) * pre_q_sum + half_base_k
    k_scale = tl.load(k_scale_ptr + g).to(tl.float32)
    inv = tl.rsqrt(tl.sum(k * k, axis=0) + eps) * k_scale
    tl.store(k_out_ptr + t * s_kout_t + g * HD + d, k * inv, mask=mask)


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

    Requires a whole number of q heads per k head (ZAYA1 always splits evenly),
    a head dim that fits one Triton block, row-major 2-D inputs with a unit
    innermost stride, and float inputs on an accelerator. ``k_scale`` is the
    folded ``sqrt(head_dim) * temperature`` vector, absent until weights load.
    """
    if k_scale is None:
        return False
    if not (conv_qk.is_cuda and pre_q.is_cuda and base_k.is_cuda):
        return False
    if head_dim > _MAX_HEAD_DIM or head_dim <= 0:
        return False
    if num_k_heads <= 0 or num_q_heads % num_k_heads != 0:
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
    return all(
        t.dtype in (torch.float32, torch.float16, torch.bfloat16)
        for t in (conv_qk, pre_q, base_k)
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(q, k)`` as fp32 ``[T, heads, head_dim]``. Caller must have
    checked :func:`covered`."""
    num_tokens = conv_qk.shape[0]
    q_out = torch.empty(
        (num_tokens, num_q_heads, head_dim),
        dtype=torch.float32,
        device=conv_qk.device,
    )
    k_out = torch.empty(
        (num_tokens, num_k_heads, head_dim),
        dtype=torch.float32,
        device=conv_qk.device,
    )
    if num_tokens == 0:
        return q_out, k_out

    _cca_qk_mix_kernel[(num_tokens * num_k_heads,)](
        conv_qk,
        pre_q,
        base_k,
        k_scale,
        q_out,
        k_out,
        conv_qk.stride(0),
        pre_q.stride(0),
        base_k.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        num_q_heads * head_dim,
        float(q_scale),
        float(eps),
        NK=num_k_heads,
        G=num_q_heads // num_k_heads,
        HD=head_dim,
        BLOCK=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return q_out, k_out
