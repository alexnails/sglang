#!/usr/bin/env bash
# Kimi-K3 on one 8x MI350X / MI355X (gfx950) node with this branch: the measured best config (README.md, config B).
# TP8, aiter MLA prefill/decode, FP8 KV, EAGLE3 (topk 1, 3 steps, 4 draft tokens, unquantized draft),
# chunked prefill 8192, 16 running requests, decode graphs for bs 1/2/4/8/16.
#
# Usage: launch_k3_eagle3.sh [extra sglang args...]
#   MODEL_PATH     target checkpoint (default moonshotai/Kimi-K3)
#   DRAFT          EAGLE3 draft (default lightseekorg/kimi-k3-eagle3-mla)
#   PORT           HTTP port (default 30000)
#   K3_GEMM_TABLE  aiter bf16 GEMM table; default is the MI350X-tuned table next to this script,
#                  empty (K3_GEMM_TABLE=) uses aiter's own tables
#   COALESCE=1     prefill coalescing for C8 and up: more tok/s/user, higher TTFT
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL_PATH=${MODEL_PATH:-moonshotai/Kimi-K3}
DRAFT=${DRAFT:-lightseekorg/kimi-k3-eagle3-mla}
PORT=${PORT:-30000}

export SGLANG_USE_AITER=1 SGLANG_AITER_K3_OPT=1 AITER_FLYDSL_FORCE=1 AITER_SITUV2_A8W4=1
export SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1 SGLANG_ROCM_K3_KDA_FUSED_BACKEND=aiter
# Off by default, on in every measurement of this config.
export SGLANG_AITER_MLA_AUTO_KV_SPLITS=1 SGLANG_AITER_MLA_MAX_SPLITS_PER_BATCH=128
export SGLANG_OPT_FUSED_KDA_VERIFY=1 AITER_MLA_REDUCE_FLYDSL=1

K3_GEMM_TABLE=${K3_GEMM_TABLE-$HERE/k3_bf16_tuned_gemm_mi350x.csv}
if [ -n "$K3_GEMM_TABLE" ]; then
  export AITER_CONFIG_GEMM_BF16=$K3_GEMM_TABLE
fi

COALESCE_ARGS=()
if [ "${COALESCE:-0}" = "1" ]; then
  COALESCE_ARGS=(--prefill-decode-interval 4 --enable-prefill-delayer --prefill-delayer-queue-min-ratio 0.25
                 --prefill-delayer-max-delay-ms 5000 --prefill-delayer-max-delay-passes 128)
fi

exec python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --trust-remote-code --served-model-name kimi-k3 \
  --tp-size 8 \
  --prefill-attention-backend aiter --decode-attention-backend aiter \
  --kv-cache-dtype fp8_e4m3 --dtype bfloat16 \
  --speculative-algorithm EAGLE3 --speculative-draft-model-path "$DRAFT" \
  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --speculative-draft-model-quantization unquant \
  --linear-attn-verify-backend triton --enable-linear-replayssm-spec \
  "${COALESCE_ARGS[@]}" \
  --chunked-prefill-size 8192 --context-length 80000 --max-running-requests 16 \
  --cuda-graph-max-bs-decode 16 --cuda-graph-bs-decode 1 2 4 8 16 \
  --mem-fraction-static 0.85 \
  --model-loader-extra-config '{"enable_multithread_load": true}' \
  --port "$PORT" \
  "$@"
