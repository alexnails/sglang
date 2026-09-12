#!/usr/bin/env bash
# Chunk dose-response under DP, with the division corrected.
#
# parallel_hook.py:201 divides chunked_prefill_size by dp_size, NOT by 2, and
# memory_hook.py:136 defaults this GPU tier to 16384. So 16384 // 4 = 4096 is
# exactly the chunk every DP arm has been running, and chain20's "--chunk 16384"
# arm would have resolved to 4096 again -- the same config as its baseline. It
# was cancelled before it launched.
#
# Effective per-rank chunk = passed // dp_size, so:
#   32768 // 4 =  8192  (2x)
#   65536 // 4 = 16384  (4x)
#
# Pool stays capped at 8M tokens, matching results_chunk-4096.json, so the only
# variable is the chunk. The ~100 GB the cap freed is what pays for the larger
# activation footprint; memory_hook runs before parallel_hook and sizes its
# reserve from the PRE-division value, so a launch failure here is a
# mem-fraction problem, not a real infeasibility.
set -u
export MODEL_PATH=/scratch/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
export ACC_LEN=5.8
export LONGBENCH_CTX=65536
cd /scratch/engram
L=/scratch/logs/ab3
PLAN=/scratch/engram/plan_pooled.npz
COMMON="--arms cb-lowlat-dp-hiengram-16 --plan $PLAN --hiengram-gib 16
        --dataset longbench_v2 --concurrency 64 128 256 512
        --max-running 512 --max-total-tokens 8388608
        --prompt-factor 2 --min-prompts 16 --ready-timeout 7200"

while pgrep -f "[a]b_v3" >/dev/null; do sleep 30; done
echo "=== chain21 start $(date -u +%FT%TZ) ===" >>"$L/chain21.log"

echo "--- pass 32768 -> effective 8192 ---" >>"$L/chain21.log"
python3 ab_v3_next.py $COMMON --chunk 32768 \
    --out "$L/results_chunk-8192.json" >>"$L/chain21.log" 2>&1

echo "--- pass 65536 -> effective 16384 ---" >>"$L/chain21.log"
python3 ab_v3_next.py $COMMON --chunk 65536 \
    --out "$L/results_chunk-16384.json" >>"$L/chain21.log" 2>&1

echo "=== chain21 done $(date -u +%FT%TZ) ===" >>"$L/chain21.log"
