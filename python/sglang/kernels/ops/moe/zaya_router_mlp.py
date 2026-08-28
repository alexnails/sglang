"""Fused ZAYA1 router MLP: three dense stages and two GELUs in one kernel.

``ZayaRouter.router_mlp`` is ``nn.Sequential`` of five modules --
``Linear(D, D)``, ``GELU``, ``Linear(D, D)``, ``GELU``, ``Linear(D, E)`` -- which
is five launches per MoE layer. ZAYA1-74B has 60 MoE layers, so that is 300 of
~2400 launches in a decode step that is launch-bound rather than
bandwidth-bound. One kernel does all five here, removing 240 launches per step.

This is pure launch overhead, not a bandwidth problem: at ``D = 256`` and
``E = 25`` the three weight matrices plus two bias vectors are ~270 KB per layer
in bf16, and a decode step reads them once.

The activation is carried in registers between stages, so nothing round-trips
through HBM. Each program owns ``BLOCK_M`` tokens and the full ``D`` columns.

**The GELU is the erf form.** ``ZayaRouter`` builds ``nn.GELU()``, i.e.
``approximate='none'``::

    gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))

not the tanh approximation. The two differ by only ~5e-4 absolute, which is
small enough to look like ordinary numerical noise in an end-to-end eval and
large enough to move routing decisions on near-ties, in every one of the 60
MoE layers. The
structural check in ``ZayaRouter.__init__`` refuses to fuse anything whose
``approximate`` is not ``'none'`` for exactly that reason -- the flavour is not
visible from the tensors, so it cannot be checked here.

Rounding order follows the torch chain rather than maximizing precision, so the
logits stay as close as possible to the unfused path:

* each dense stage accumulates in fp32 and rounds **once**, bias included --
  which is what ``addmm``'s fp32 accumulator plus bias epilogue does;
* each GELU then reads that rounded value, computes in fp32, and rounds again --
  which is what ``at::gelu`` on a bf16 tensor does.

Keeping the fp32 accumulator alive across a stage boundary instead would be
*more* accurate than torch and therefore further from it, which is the wrong
trade when the acceptance bar is "the argmax downstream picks the same expert".

The ``K`` axis of each dense stage is split in halves and the weight tiles are
loaded one at a time. That is deliberate: a single ``tl.dot`` against the whole
``[256, 256]`` operand needs 128 KiB of LDS to stage, which does not fit on
gfx942 and crowds out occupancy even on gfx950's 160 KiB.

LDS budget at the shipping shape (``D = 256``, so ``HALF = 128``), since this is
the part most likely not to survive contact with the compiler:

===========================  =========================  ========
tile                         shape                      bf16
===========================  =========================  ========
dot operand B (per half)     ``[128, 128]``             32 KiB
dot operand A                ``[BLOCK_M=32, 128]``       8 KiB
stage-3 operand B            ``[128, NPAD=32]``          8 KiB
===========================  =========================  ========

The two B halves are written to the same variable and consumed immediately, so
liveness should let them share one 32 KiB scratch: peak ~40 KiB. If the compiler
keeps both live instead, it is ~72 KiB -- still inside gfx950's 160 KiB, over
gfx942's 64 KiB. gfx950 is the target, so that is an accepted risk rather than a
designed-around one, and ``router_mlp(..., block_m=...)`` is the lever if it
turns out to be tight: BLOCK_M only sizes the A tile and the accumulator, so
halving it costs occupancy, not correctness.

Follows ``kda_fused_decode`` / ``zaya_mod`` / ``zaya_router_tail``: a
``covered()`` predicate gates supported inputs and the caller falls back to the
unfused ``nn.Sequential``. Triton, not CUDA-JIT, so it runs on ROCm -- ZAYA1's
reference deployment is MI355X.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe.zaya_router_tail import (
    _same_device,
    debug_asserts_enabled,
)

# The K split is two halves, so a stage's dot operand is [D/2, D/2]. Capping D at
# 256 caps that at [128, 128] -- 32 KiB in bf16, which stages comfortably in LDS
# on every gfx9 part. ZAYA1 uses D = 256 exactly.
_MAX_EXPANSION = 256

# tl.dot needs every dimension at least 16 wide.
_MIN_DOT_DIM = 16

# The router MLP is left unquantized (see ZayaRouter.__init__), so only the
# float dtypes appear. fp32 is excluded on purpose: tl.dot would silently pick a
# reduced-precision fp32 mode, and the fallback is cheap enough not to care.
_FUSED_DTYPES = (torch.float16, torch.bfloat16)


@triton.jit
def _dense_bias_gelu(
    a_lo,
    a_hi,
    w_ptr,
    b_ptr,
    s_w_out,
    s_w_in,
    N_OFF: tl.constexpr,
    HALF: tl.constexpr,
):
    """One ``HALF``-wide slice of ``gelu(a @ W.T + b)``, rounded like torch.

    ``a`` arrives pre-split along K as ``(a_lo, a_hi)``. ``W`` is ``[OUT, IN]``
    row-major, so the dot operand ``b[k, n] = W[N_OFF + n, k]`` is a strided
    load -- coalesced along k, which is the unit-stride axis.
    """
    k = tl.arange(0, HALF)
    n = N_OFF + tl.arange(0, HALF)

    # These loads carry no mask, and do not need one: N_OFF is 0 or HALF, so
    # n < 2*HALF == D, and both k and k + HALF are < D. W is exactly [D, D] and
    # the bias exactly [D] (covered() checks both), so every lane is in bounds
    # by construction rather than by masking. There is deliberately no padding
    # on this axis -- HALF is a power of two because D is.
    #
    # Loaded and consumed one tile at a time so the two halves share one LDS
    # scratch buffer instead of needing two live at once.
    w = tl.load(w_ptr + n[None, :] * s_w_out + k[:, None] * s_w_in)
    acc = tl.dot(a_lo, w)
    w = tl.load(w_ptr + n[None, :] * s_w_out + (k[:, None] + HALF) * s_w_in)
    acc = tl.dot(a_hi, w, acc)

    bias = tl.load(b_ptr + n).to(tl.float32)
    # One rounding for the whole linear, bias included: addmm's fp32 accumulator
    # folds the bias in before its single store.
    z = (acc + bias[None, :]).to(b_ptr.dtype.element_ty)
    zf = z.to(tl.float32)
    # erf-form GELU -- nn.GELU()'s default, NOT the tanh approximation.
    # 0.7071067811865475 is 1/sqrt(2), the same constant the in-tree erf-GELU
    # in kernels/ops/elementwise/elementwise.py uses. Spelled out rather than
    # named so nothing depends on a module global surviving Triton's compile.
    gelu = 0.5 * zf * (1.0 + tl.erf(zf * 0.7071067811865475))
    # And one rounding for the activation, as at::gelu's store does.
    return gelu.to(b_ptr.dtype.element_ty)


@triton.jit
def _router_mlp_kernel(
    x_ptr,  # [T, D]  model dtype
    w1_ptr,  # [D, D]
    b1_ptr,  # [D]
    w2_ptr,  # [D, D]
    b2_ptr,  # [D]
    w3_ptr,  # [NUM_EXPERTS, D]
    out_ptr,  # [T, NUM_EXPERTS]  model dtype   out
    s_x_m,
    s_out_m,
    s_w1_out,
    s_w1_in,
    s_w2_out,
    s_w2_in,
    s_w3_out,
    s_w3_in,
    num_tokens,
    NUM_EXPERTS: tl.constexpr,
    HALF: tl.constexpr,  # D // 2
    NPAD: tl.constexpr,  # NUM_EXPERTS padded up to a dot-legal power of 2
    BLOCK_M: tl.constexpr,
    DEBUG_ASSERT: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < num_tokens
    k = tl.arange(0, HALF)

    # Address with a CLAMPED row so the last tile's padding lanes -- up to
    # BLOCK_M - 1 of them, and T is 1 at bs=1 decode -- point at a real row
    # instead of up to 31 rows past the end. row_mask still decides which values
    # are used and which rows are written, so this only changes what an unmasked
    # lane would have touched.
    safe_rows = tl.minimum(rows, num_tokens - 1)

    x_lo = tl.load(
        x_ptr + safe_rows[:, None] * s_x_m + k[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    x_hi = tl.load(
        x_ptr + safe_rows[:, None] * s_x_m + (k[None, :] + HALF),
        mask=row_mask[:, None],
        other=0.0,
    )

    h_lo = _dense_bias_gelu(x_lo, x_hi, w1_ptr, b1_ptr, s_w1_out, s_w1_in, 0, HALF)
    h_hi = _dense_bias_gelu(x_lo, x_hi, w1_ptr, b1_ptr, s_w1_out, s_w1_in, HALF, HALF)

    g_lo = _dense_bias_gelu(h_lo, h_hi, w2_ptr, b2_ptr, s_w2_out, s_w2_in, 0, HALF)
    g_hi = _dense_bias_gelu(h_lo, h_hi, w2_ptr, b2_ptr, s_w2_out, s_w2_in, HALF, HALF)

    # Third stage: no bias, no activation, and NUM_EXPERTS (25 on the 74B) is
    # neither a power of two nor dot-legal, so pad the output axis and mask.
    n = tl.arange(0, NPAD)
    col_mask = n < NUM_EXPERTS
    # NPAD is NUM_EXPERTS padded up to a dot-legal power of two: 32 for the 74B's
    # 25, so 7 of 32 columns are padding. Clamp the addressing index for the same
    # reason as the rows -- w3 has exactly NUM_EXPERTS rows, and a padded column
    # lane that escaped the mask would read past the smallest of the three weight
    # tensors. col_mask still zeroes their contribution to the dot.
    safe_n = tl.minimum(n, NUM_EXPERTS - 1)
    w = tl.load(
        w3_ptr + safe_n[None, :] * s_w3_out + k[:, None] * s_w3_in,
        mask=col_mask[None, :],
        other=0.0,
    )
    logits = tl.dot(g_lo, w)
    w = tl.load(
        w3_ptr + safe_n[None, :] * s_w3_out + (k[:, None] + HALF) * s_w3_in,
        mask=col_mask[None, :],
        other=0.0,
    )
    logits = tl.dot(g_hi, w, logits)

    if DEBUG_ASSERT:
        tl.device_assert(safe_n < NUM_EXPERTS, "zaya_router_mlp: expert col OOB")
        tl.device_assert(safe_rows < num_tokens, "zaya_router_mlp: row OOB")

    tl.store(
        out_ptr + safe_rows[:, None] * s_out_m + safe_n[None, :],
        logits.to(out_ptr.dtype.element_ty),
        mask=row_mask[:, None] & col_mask[None, :],
    )


def decline_reason(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: Optional[torch.Tensor],
    w2: torch.Tensor,
    b2: Optional[torch.Tensor],
    w3: torch.Tensor,
    *,
    num_experts: int,
) -> Optional[str]:
    """Why the fused router MLP cannot serve these tensors, or ``None``.

    Note what this canNOT check: that the two activations between the stages are
    the erf GELU. That is a property of the module graph, not of any tensor, so
    ``fusable_router_mlp`` checks it once at construction and refuses to call in
    here otherwise.

    Returns the reason rather than a bool for the same purpose as the tail's:
    a decline costs four launches and changes nothing else, so without a name
    for the gate that said no it is indistinguishable from the fusion working.
    """
    named = (("x", x), ("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2), ("w3", w3))
    for name, t in named:
        if t is None:
            return f"{name} is None"
    # Device last, deliberately. Everything above is a property of the tensors
    # rather than of where they live, so ordering it first is what lets a
    # CPU-only test reach every structural branch -- and a shape or dtype
    # mismatch is a configuration bug worth naming, where "on cpu" is not.
    for name, t in named:
        if t.dtype not in _FUSED_DTYPES:
            return f"{name} dtype {t.dtype}"
    for name, t in (("x", x), ("w1", w1), ("w2", w2), ("w3", w3)):
        if t.ndim != 2:
            return f"{name} is {t.ndim}-D, not 2-D"

    expansion = x.shape[1]
    if expansion % 2 != 0 or expansion > _MAX_EXPANSION:
        return f"expansion {expansion} is odd or above {_MAX_EXPANSION}"
    half = expansion // 2
    if half < _MIN_DOT_DIM:
        return f"K-split half {half} below the tl.dot minimum {_MIN_DOT_DIM}"
    if num_experts <= 0 or num_experts > _MAX_EXPANSION:
        return f"num_experts {num_experts} outside 1..{_MAX_EXPANSION}"

    if tuple(w1.shape) != (expansion, expansion):
        return f"w1 {tuple(w1.shape)} is not [{expansion}, {expansion}]"
    if tuple(w2.shape) != (expansion, expansion):
        return f"w2 {tuple(w2.shape)} is not [{expansion}, {expansion}]"
    if tuple(w3.shape) != (num_experts, expansion):
        return f"w3 {tuple(w3.shape)} is not [{num_experts}, {expansion}]"
    if tuple(b1.shape) != (expansion,):
        return f"b1 {tuple(b1.shape)} is not [{expansion}]"
    if tuple(b2.shape) != (expansion,):
        return f"b2 {tuple(b2.shape)} is not [{expansion}]"
    if not b1.is_contiguous():
        return "b1 not contiguous"
    if not b2.is_contiguous():
        return "b2 not contiguous"
    # The kernel indexes both axes of each weight explicitly, so an arbitrary
    # stride pair is fine as long as the innermost element step is unit -- which
    # is what makes the k-major dot-operand load coalesce.
    for name, t in (("x", x), ("w1", w1), ("w2", w2), ("w3", w3)):
        if t.stride(-1) != 1:
            return f"{name} innermost stride != 1"
    for name, t in named:
        if not t.is_cuda:
            return f"{name} on {t.device}"
        if not _same_device(t, x.device):
            return f"{name} on {t.device}, x on {x.device}"
    return None


def covered(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: Optional[torch.Tensor],
    w2: torch.Tensor,
    b2: Optional[torch.Tensor],
    w3: torch.Tensor,
    *,
    num_experts: int,
) -> bool:
    """Whether the fused router MLP can serve these. See the reason variant."""
    return decline_reason(x, w1, b1, w2, b2, w3, num_experts=num_experts) is None


def router_mlp(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    w3: torch.Tensor,
    *,
    num_experts: int,
    block_m: int = 32,
) -> torch.Tensor:
    """Expert logits ``[T, num_experts]`` in ``x``'s dtype.

    Caller must have checked :func:`covered` *and* that the activations being
    fused away are the erf GELU.
    """
    num_tokens, expansion = x.shape
    out = torch.empty((num_tokens, num_experts), dtype=x.dtype, device=x.device)
    if num_tokens == 0:
        return out

    _router_mlp_kernel[(triton.cdiv(num_tokens, block_m),)](
        x,
        w1,
        b1,
        w2,
        b2,
        w3,
        out,
        x.stride(0),
        out.stride(0),
        w1.stride(0),
        w1.stride(1),
        w2.stride(0),
        w2.stride(1),
        w3.stride(0),
        w3.stride(1),
        num_tokens,
        NUM_EXPERTS=num_experts,
        HALF=expansion // 2,
        NPAD=triton.next_power_of_2(max(num_experts, _MIN_DOT_DIM)),
        BLOCK_M=block_m,
        DEBUG_ASSERT=debug_asserts_enabled(),
        num_warps=4,
    )
    return out
