"""Fused ZAYA1 router tail: softmax + balancing bias + top-1 + gather + munge.

One kernel replaces the whole elementwise tail of
:meth:`ZayaRouter.forward <sglang.srt.models.zaya.ZayaRouter.forward>` *and* the
expert-id munging :class:`ZayaBlock <sglang.srt.models.zaya.ZayaBlock>` does on
its way into ``FusedMoE``. The torch chain is nine launches per MoE layer::

    expert_prob  = softmax(logits, dim=-1, dtype=fp32)   # 1
    biased       = expert_prob + balancing_biases        # 1
    choice       = biased.argmax(dim=-1, keepdim=True)   # 1
    route_prob   = gather(expert_prob, 1, choice)        # 1
    route_prob   = route_prob.to(model_dtype)            # 1
    # ... in ZayaBlock.forward:
    topk_ids     = choice.to(int32)                      # 1
    clamped      = clamp(choice, 0, num_moe_experts - 1) # 1
    topk_ids     = clamped.to(int32)                     # 1
    # ... in the aiter MoE runner:
    topk_weights = topk_weights.to(float32)              # 1

ZAYA1-74B has 60 MoE layers, so that tail alone is ~540 of ~2400 launches in a
decode step that is launch-bound, not bandwidth-bound.

Everything above lives in registers here: ``num_experts`` (25 on the 74B: 24
experts plus the MOD skip slot) is one masked column block per program, and the
grid is one program per token.

Outputs, all ``[T, 1]``:

``moe_weight`` (fp32)
    The routing probability, in the dtype every other sglang top-k emits. The
    aiter/hpc_ops runners open with ``topk_weights.to(torch.float32)``, which is
    then a no-op instead of a launch -- that is the ninth launch above, bought
    without touching any shared MoE code.
``moe_ids`` (int32)
    The chosen expert, ALREADY CLAMPED into ``[0, num_moe_experts - 1]``, ready
    to hand to ``FusedMoE`` as ``topk_ids``.
``route_prob`` (model dtype)
    The same probability rounded to the model dtype, which is what the MOD
    residual blend multiplies the hidden state by. Kept separate from
    ``moe_weight`` so the MOD arithmetic is unchanged by this fusion; both come
    out of the same program, so the extra store costs no launch.
``skip_ids`` (int32, optional)
    The *unclamped* argmax. MOD needs it: clamping maps the skip slot
    (``num_moe_experts``) onto real expert ``num_moe_experts - 1``, so the
    clamped ids can no longer tell a skipped token from one genuinely routed to
    the last expert. Emitted only when clamping actually loses information
    (``max_expert_id < num_experts - 1``); otherwise the caller aliases
    ``moe_ids``.

Two things this kernel pins that torch leaves loose:

* **Tie-breaking.** ``torch.argmax`` does not specify which index wins an exact
  tie. Here the lowest index always wins, via the
  ``tl.min(tl.where(v == best, cols, N))`` idiom used elsewhere in-tree
  (``moe_fused_gate``, ``speculative/dflash``). Ties are reachable in practice:
  ``balancing_biases`` is a constant vector and the softmax of equal logits is
  exactly equal in fp32.
* **Reduction order.** The fp32 max/sum here will not match torch's tree
  reduction bit-for-bit, so ``route_prob`` is close but not identical (~1 ULP).
  The acceptance bar is the *selected expert* matching bit-for-bit plus
  ``assert_close`` on the probabilities.

Follows ``kda_fused_decode`` / ``zaya_mod``: a ``covered()`` predicate gates
supported inputs and the caller falls back to the torch chain. Triton, not
CUDA-JIT, so it runs on ROCm -- ZAYA1's reference deployment is MI355X.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# One masked column block per program, so the expert count must fit a single
# Triton block. ZAYA1 uses 25; 1024 is far more headroom than any MoE router
# with top-1 routing plausibly needs.
_MAX_EXPERTS = 1024

_FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@triton.jit
def _router_tail_kernel(
    logits_ptr,  # [T, NUM_EXPERTS] model dtype
    biases_ptr,  # [NUM_EXPERTS] fp32
    weight_ptr,  # [T, 1] fp32           out
    prob_ptr,  # [T, 1] model dtype    out
    ids_ptr,  # [T, 1] int32          out (clamped)
    skip_ids_ptr,  # [T, 1] int32          out (raw), unused unless EMIT_SKIP_IDS
    s_logits_t,
    NUM_EXPERTS: tl.constexpr,
    MAX_EXPERT_ID: tl.constexpr,
    SOFTMAX_FP32: tl.constexpr,
    EMIT_SKIP_IDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < NUM_EXPERTS

    x = tl.load(logits_ptr + t * s_logits_t + cols, mask=mask, other=0.0).to(tl.float32)
    # Padding lanes must not win the max nor add to the exp sum.
    x = tl.where(mask, x, float("-inf"))

    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)  # padding lanes: exp(-inf) == 0
    prob = e / tl.sum(e, axis=0)

    if not SOFTMAX_FP32:
        # ``zaya_high_prec=False`` runs torch's softmax at the logits' dtype,
        # which still accumulates in fp32 internally and rounds once on store.
        # Reproduce that single rounding rather than keeping the fp32 value,
        # otherwise the biased comparison below sees different numbers.
        prob = prob.to(logits_ptr.dtype.element_ty).to(tl.float32)

    biases = tl.load(biases_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    biased = tl.where(mask, prob + biases, float("-inf"))

    best = tl.max(biased, axis=0)
    # Lowest index wins an exact tie. torch.argmax leaves this unspecified, and
    # exact ties are reachable here (constant biases, equal logits).
    idx = tl.min(tl.where(biased == best, cols, NUM_EXPERTS), axis=0)

    # Gather the *unbiased* probability of the winner. Exactly one lane is
    # non-zero, so the fp32 sum is exact.
    chosen = tl.sum(tl.where(cols == idx, prob, 0.0), axis=0)

    tl.store(weight_ptr + t, chosen)
    tl.store(prob_ptr + t, chosen.to(prob_ptr.dtype.element_ty))
    tl.store(ids_ptr + t, tl.minimum(idx, MAX_EXPERT_ID).to(tl.int32))
    if EMIT_SKIP_IDS:
        tl.store(skip_ids_ptr + t, idx.to(tl.int32))


def covered(
    logits: torch.Tensor,
    balancing_biases: torch.Tensor,
    *,
    num_experts: int,
    max_expert_id: int,
    topk: int,
    out_dtype: torch.dtype,
) -> bool:
    """Whether the fused tail can serve these inputs.

    Restricted to top-1 routing, which is what ZAYA1 ships; wider top-k needs
    the cumulative-skip rewrite in ``ZayaRouter.forward`` and falls back. Also
    requires the expert axis to fit one Triton block, row-major logits with a
    unit innermost stride, and an fp32 bias vector sized to the expert count.
    """
    if not (logits.is_cuda and balancing_biases.is_cuda):
        return False
    if topk != 1:
        return False
    if num_experts <= 0 or num_experts > _MAX_EXPERTS:
        return False
    if not (0 <= max_expert_id < num_experts):
        return False
    if logits.ndim != 2 or logits.shape[1] != num_experts:
        return False
    if logits.stride(-1) != 1:
        return False
    if balancing_biases.numel() != num_experts:
        return False
    if not balancing_biases.is_contiguous():
        return False
    # The bias vector is fp32 in the checkpoint and the kernel widens it to fp32
    # regardless, so any float dtype is served -- and served identically to the
    # torch chain, which also promotes before adding. Accepting bf16 here
    # matters: a stray ``model.to(dtype)`` would otherwise drop the whole router
    # onto the fallback with no error and no clue why decode got slower.
    if balancing_biases.dtype not in _FLOAT_DTYPES:
        return False
    return logits.dtype in _FLOAT_DTYPES and out_dtype in _FLOAT_DTYPES


def router_tail(
    logits: torch.Tensor,
    balancing_biases: torch.Tensor,
    *,
    num_experts: int,
    max_expert_id: int,
    softmax_fp32: bool,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(moe_weight, moe_ids, route_prob, skip_ids)``, each ``[T, 1]``.

    ``moe_ids`` is clamped to ``[0, max_expert_id]``; ``skip_ids`` is the raw
    argmax and aliases ``moe_ids`` when the clamp cannot change anything. Caller
    must have checked :func:`covered`.
    """
    num_tokens = logits.shape[0]
    device = logits.device
    moe_weight = torch.empty((num_tokens, 1), dtype=torch.float32, device=device)
    route_prob = torch.empty((num_tokens, 1), dtype=out_dtype, device=device)
    moe_ids = torch.empty((num_tokens, 1), dtype=torch.int32, device=device)

    emit_skip_ids = max_expert_id < num_experts - 1
    skip_ids = (
        torch.empty((num_tokens, 1), dtype=torch.int32, device=device)
        if emit_skip_ids
        else moe_ids
    )
    if num_tokens == 0:
        return moe_weight, moe_ids, route_prob, skip_ids

    _router_tail_kernel[(num_tokens,)](
        logits,
        balancing_biases,
        moe_weight,
        route_prob,
        moe_ids,
        skip_ids,
        logits.stride(0),
        NUM_EXPERTS=num_experts,
        MAX_EXPERT_ID=max_expert_id,
        SOFTMAX_FP32=bool(softmax_fp32),
        EMIT_SKIP_IDS=emit_skip_ids,
        BLOCK=triton.next_power_of_2(num_experts),
        num_warps=1,
    )
    return moe_weight, moe_ids, route_prob, skip_ids
