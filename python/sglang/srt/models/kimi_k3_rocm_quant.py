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
"""Kimi-K3 ROCm weight preparation for Quark checkpoints.

Quark serializes K3 as per-output-channel FP8, which is not the layout the
AITER kernels want. The conversions live here; ``kimi_k3.py`` calls every entry
point behind ``_is_hip`` and keeps the checkpoint's own layout otherwise.
"""

import torch
from torch import nn


def _k3_channel_fp8_to_bf16(module: nn.Module, weight: torch.Tensor) -> torch.Tensor:
    """Dequantize a per-output-channel FP8 weight to dense bf16.

    The aiter batched absorb GEMM takes ``w_scale`` as a single scalar (the
    Triton kernel documents it as "per-batch scale for WQ with shape (1,)"), so
    a per-channel vector cannot reach it. Requantizing to one per-tensor scale
    would fit, but the scale axis is the GEMM's *contraction* axis, and on real
    K3 weights that second quantization costs ~2.3% relative error against the
    checkpoint (worst channel ~3.8%). Dequantizing instead costs ~0.12% and
    lets the absorb run its bf16 path with w_scale left at the 1.0 default.
    Must run while dim 0 is still the channel axis weight_scale indexes, i.e.
    before the kv_b_proj head split.
    """
    weight_scale = module.weight_scale
    # Per-channel scale is 1D [out]; reshape so it broadcasts against [out, in].
    if weight_scale.dim() == 1:
        weight_scale = weight_scale.view(-1, 1)
    return (weight.to(torch.float32) * weight_scale.to(torch.float32)).to(
        torch.bfloat16
    )


# fp8 e4m3 decode LM head (env-gated, M <= 4 only; see logits_processor).
# The LM head is the single largest decode weight read: [vocab/TP, 7168] bf16
# per rank. Halving its bytes with per-output-channel fp8 halves the GEMM's
# energy on gfx950 (measured -60% J/call at the TP8 shard shape, incl. the
# per-token activation quant). The bf16 weight stays resident for larger
# batches; these caches are plain attributes (never enter the state_dict).
_K3_FP8_MAX = 448.0


def k3_prepare_lm_head_fp8(lm_head: nn.Module) -> bool:
    """Cache an fp8-e4m3 per-output-channel copy of the LM-head weight.

    Returns True when the cache was built. Skips -- leaving the bf16 matmul as
    the only path -- when the weight is not bf16 (e.g. a pre-quantized
    checkpoint), when a bias is present, or when aiter is unavailable. Runs
    from post_load_weights, so dummy-load benchmark flows get the same path.
    """
    if getattr(lm_head, "k3_fp8_decode_weight", None) is not None:
        return True
    weight = getattr(lm_head, "weight", None)
    if not (isinstance(weight, torch.Tensor) and weight.dtype == torch.bfloat16):
        return False
    if getattr(lm_head, "bias", None) is not None:
        return False
    try:
        from aiter import gemm_a8w8, per_token_quant_hip  # noqa: F401
    except ImportError:
        return False
    n = weight.shape[0]
    # Per-output-channel scale in fp32 without a full-matrix fp32 copy.
    scale = (
        torch.linalg.vector_norm(weight, ord=torch.inf, dim=1, dtype=torch.float32)
        .clamp_(min=1e-12)
        .div_(_K3_FP8_MAX)
    )
    weight_fp8 = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    for lo in range(0, n, 8192):
        hi = min(n, lo + 8192)
        weight_fp8[lo:hi] = (
            (weight[lo:hi].float() / scale[lo:hi, None])
            .clamp_(-_K3_FP8_MAX, _K3_FP8_MAX)
            .to(torch.float8_e4m3fn)
        )
    lm_head.k3_fp8_decode_weight = weight_fp8
    lm_head.k3_fp8_decode_scale = scale
    return True


def k3_lm_head_fp8_decode(lm_head: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Decode LM head: per-token fp8 activation quant + aiter a8w8 GEMM.

    Returns bf16 logits [M, vocab_shard], same dtype/shape as the bf16
    ``torch.matmul`` it replaces. Graph-capture safe (both aiter ops are).
    """
    from aiter import gemm_a8w8, per_token_quant_hip

    h = hidden_states.contiguous()
    m = h.shape[0]
    q, s = per_token_quant_hip(h, quant_dtype=torch.float8_e4m3fn)
    return gemm_a8w8(
        q,
        lm_head.k3_fp8_decode_weight,
        s.view(m, 1).float(),
        lm_head.k3_fp8_decode_scale[:, None],
    )
