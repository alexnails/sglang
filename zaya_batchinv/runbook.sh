#!/usr/bin/env bash
# ZAYA1-74B batch-invariance investigation. ONE 8x MI355X box, cheapest first.
#
# Each arm is a single-variable change against the reference. Run the arms in
# order and STOP as soon as an arm collapses the distinct count to 1 -- that
# arm names the dominant term. Do not run the whole ladder blind.
#
# Follow the campaign's stop discipline: never `kill -9`; gate the next start
# on VRAM returning to idle on all 8 cards.
set -euo pipefail

MODEL=${MODEL:?set MODEL to the ZAYA1-74B checkpoint path}
PORT=${PORT:-30000}
URL="http://127.0.0.1:${PORT}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE_ROOT=${PROBE_ROOT:-/data/zaya_probe}

# The reference config from ZAYA_PERF_TASKS.md.
BASE_ENV=(SGLANG_OPT_ZAYA_GLOBAL_RESIDUAL=1)
BASE_ARGS=(
  --model-path "$MODEL" --trust-remote-code
  --tp 8 --dp 4 --enable-dp-attention
  --attention-backend triton
  --disable-radix-cache
  --port "$PORT"
)

serve() {  # serve <extra env...> -- <extra args...>
  local envs=() args=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do envs+=("$1"); shift; done
  shift || true
  args=("$@")
  echo "### env: ${envs[*]:-none}   args: ${args[*]:-none}"
  env "${BASE_ENV[@]}" "${envs[@]}" \
    python3 -m sglang.launch_server "${BASE_ARGS[@]}" "${args[@]}"
}

# ===========================================================================
# ARM 0 -- IS IT THE PROMPT OR THE SERVER?   ~3 min, no server restart.
# ===========================================================================
# The original measurement used one ambiguous prompt on a preview checkpoint
# whose gsm8k is 37-49% unparseable, i.e. a model with genuinely near-tied
# formatting decisions. Before blaming the numerics, check whether a decisive
# prompt is stable. If it is, the perturbation is ordinary and only the
# *readout* was pathological.
#
#   serve "${BASE_ENV[@]}" --   # reference, in another shell
arm0() {
  python3 "$HERE/probe_client.py" distinct --url "$URL" --n 8 --tokens 64 --verbose
}

# ===========================================================================
# ARM 1 -- IS IT BATCH SIZE, OR IS IT DP RANK?   ~2 min, same server.
# ===========================================================================
# 8 requests over dp=4 giving exactly 4 distinct outputs is suspiciously equal
# to the replica count. routed_dp_rank pins a request to one replica at batch
# size 1, which separates the two explanations completely.
arm1() {
  python3 "$HERE/probe_client.py" dprank --url "$URL" --dp-size 4 --tokens 64
  python3 "$HERE/probe_client.py" sweep  --url "$URL" --dp-rank 0 \
      --sizes 1,2,4,8,16,24,25,32,64
  # 24/25 is deliberate: the Triton MoE default config flips BLOCK_SIZE_K
  # 64 -> 32 exactly at M > E = 24
  # (layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py:246-259).
}

# ===========================================================================
# ARM 2 -- MEASURE THE PERTURBATION, DON'T INFER IT.   ~5 min, same server.
# ===========================================================================
# Re-scores one FIXED token sequence alone vs batched, so hundreds of
# positions are compared instead of one. Produces EPSILON in nats plus the
# top-2 vocab gap distribution -- together they predict the flip rate.
arm2() {
  python3 "$HERE/probe_client.py" epsilon --url "$URL" --batch 8 --tokens 128 \
      --prompt ambiguous --top-logprobs 4
  python3 "$HERE/probe_client.py" epsilon --url "$URL" --batch 8 --tokens 128 \
      --prompt decisive --top-logprobs 4
}

# ===========================================================================
# ARM 3 -- MEASURE EXPERT FLIPS DIRECTLY.   ~15 min, needs a probe server.
# ===========================================================================
# --enable-return-routed-experts does NOT work for ZAYA1: the capture hook
# (layers/moe/topk.py:1971) is only reached from select_experts, and
# ZayaBlock builds StandardTopKOutput by hand (models/zaya.py:2027-2031); and
# RoutedExpertsCapturer reads hf_text_config.num_experts_per_tok
# (state_capturer/routed_experts.py:70), which ZayaConfig never defines, so
# the server would raise AttributeError at startup.
#
# So: sitecustomize.py + zaya_router_probe.py wrap ZayaRouter._routing_reference
# instead. No source file is modified.
#
#   PYTHONPATH=$HERE:$PYTHONPATH ZAYA_PROBE_DIR=$PROBE_ROOT/serial \
#     serve -- --disable-cuda-graph
arm3_serial() {
  python3 "$HERE/probe_client.py" flips --url "$URL" \
      --probe-dir "$PROBE_ROOT/serial" --batch 1 --tokens 1
}
#   ...restart with ZAYA_PROBE_DIR=$PROBE_ROOT/parallel
arm3_parallel() {
  python3 "$HERE/probe_client.py" flips --url "$URL" \
      --probe-dir "$PROBE_ROOT/parallel" --batch 8 --tokens 1
}
arm3_analyse() {
  python3 "$HERE/analyze_router.py" gap "$PROBE_ROOT/serial"
  python3 "$HERE/analyze_router.py" flips "$PROBE_ROOT/serial" \
      "$PROBE_ROOT/parallel"
}
# NOTE: the gap histogram needs only ONE run, and it answers "how tied is
# top-1 routing" outright. Run arm3_serial + `analyze_router.py gap` even if
# you skip the flip diff.

# ===========================================================================
# ARM 4..7 -- SINGLE-VARIABLE ABLATIONS. One server restart each.
# After each restart, re-run arm0 and read the 'ambiguous / parallel' cell.
# ===========================================================================

# 4: pin the Triton decode split count. Removes the ONLY batch coupling in
#    attention: kernels/ops/attention/metadata.py:28-53 makes num_kv_splits[i]
#    a function of max/min seq_len over the batch AND of num_seq.
#    Cheap, and touches nothing else.
arm4() { serve -- --triton-attention-split-tile-size 256; }

# 5: take aiter out of the MoE. unquant.py:68 gates the aiter runner on
#    SGLANG_USE_AITER, which docker/rocm.Dockerfile:835 sets to 1. aiter's
#    kernel-selection key includes the token count M
#    (ZAYA_PERF_TASKS.md:94-99) and its body is not vendored here, so this is
#    the only way to see whether it is the term.
arm5() { serve SGLANG_USE_AITER=0 --; }

# 6: make the collective size-invariant. SGLANG_USE_1STAGE_ALLREDUCE=1 selects
#    sglang's CustomAllreduce, whose should_custom_ar returns True with NO
#    size check on HIP (custom_all_reduce.py:277-278) -- unlike the default
#    AiterCustomAllreduce (custom_all_reduce.py:392-403), whose thresholds
#    live in the external aiter wheel.
arm6() { serve SGLANG_USE_1STAGE_ALLREDUCE=1 --; }

# 7: drop the ZAYA global-residual dataflow. It changes the o_proj collective
#    from an attn_tp(=2) all-reduce on [local,4096] to a tp(=8) partial gather
#    on [global,4096] (zaya.py:2323-2339), i.e. a dp_size-times larger operand
#    at every one of 60 attention layers -- much more exposed to size-keyed
#    thresholds. Costs ~7% throughput; this arm is about attribution only.
arm7() { serve SGLANG_OPT_ZAYA_GLOBAL_RESIDUAL=0 --; }

# ===========================================================================
# ARM 8 -- THE CEILING. What does the built-in feature actually buy?
# ===========================================================================
# triton IS an allowed deterministic backend (server_args.py:278-288) and is
# in the radix-supported subset (:290-293), so radix cache is not force-off.
# BUT: every batch-invariance guard in the MoE lives in the *Triton* runner
# (moe_runner/triton_utils/fused_moe_triton_config.py:72-77, 190-197) --
# there is NO deterministic handling anywhere in moe_runner/aiter.py. So on
# this box the flag alone is EXPECTED to be insufficient. Run both:
arm8a() { serve -- --enable-deterministic-inference; }
arm8b() { serve SGLANG_USE_AITER=0 -- --enable-deterministic-inference; }
# If 8b still gives distinct > 1 on a decisive prompt, that is a DEFECT --
# either in ZAYA's own dataflow or in a guard that does not cover this path.
# If 8b gives 1, the model is batch-invariant when asked to be, and everything
# above is ordinary optimized-kernel behaviour.

case "${1:-help}" in
  arm0|arm1|arm2|arm3_serial|arm3_parallel|arm3_analyse) "$1" ;;
  arm4|arm5|arm6|arm7|arm8a|arm8b) "$1" ;;
  *) grep -E '^# (ARM|=====)' "$0" | head -40; echo; sed -n '/^case/,$p' "$0" ;;
esac
