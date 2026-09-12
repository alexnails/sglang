#!/usr/bin/env bash
# Re-baseline the headline A/B at the corrected chunk.
#
# Every DP arm in this study ran a per-rank chunk of 4096 (16384 // dp_size),
# which the dose-response then showed costs 10-17%. The "HiEngram is free"
# conclusion was measured in that handicapped regime, so it is owed a re-test
# where the server is not throttled.
#
# This is the no-cache half. Its cache counterpart is already measured at the
# same chunk, pool cap, sweep and seeds in results_chunk-8192.json, so the two
# form a direct pair:
#   cb-lowlat-dp-hiengram-16 @ 8192: 2313.5 / 2405.5 / 2594.7 / 2599.8
set -u
export MODEL_PATH=/scratch/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
export ACC_LEN=5.8
export LONGBENCH_CTX=65536
cd /scratch/engram
L=/scratch/logs/ab3

while pgrep -f "[c]hain21.sh" >/dev/null; do sleep 30; done
while pgrep -f "[a]b_v3" >/dev/null; do sleep 30; done
echo "=== chain22 start $(date -u +%FT%TZ) ===" >>"$L/chain22.log"

echo "--- no cache, effective chunk 8192 ---" >>"$L/chain22.log"
python3 ab_v3_next.py --arms cb-lowlat-dp \
    --dataset longbench_v2 --concurrency 64 128 256 512 \
    --max-running 512 --max-total-tokens 8388608 --chunk 32768 \
    --prompt-factor 2 --min-prompts 16 --ready-timeout 7200 \
    --out "$L/results_chunk-8192-nocache.json" >>"$L/chain22.log" 2>&1

echo "=== chain22 done $(date -u +%FT%TZ) ===" >>"$L/chain22.log"
