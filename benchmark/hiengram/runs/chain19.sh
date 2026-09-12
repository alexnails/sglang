#!/usr/bin/env bash
# The no-cache partner for the BCG+DP fix-verification arm.
#
# cb-lowlat-dp-bcg-hiengram-16 cleared c=1/8/32/64 once the gather was bounded,
# so this arm is no longer a diagnosis: it is the baseline that says what the
# cache costs in the BCG+DP configuration. Same concurrencies as the fixcheck.
#
# Its previous attempt died in init_process_group on a port sglang's own
# get_free_port() had just handed it; the driver now pins --nccl-port itself.
set -u
export MODEL_PATH=/scratch/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
export ACC_LEN=5.8
cd /scratch/engram
L=/scratch/logs/ab3

while pgrep -f "[c]hain18.sh" >/dev/null; do sleep 30; done
while pgrep -f "[a]b_v3" >/dev/null; do sleep 30; done
echo "=== chain19 start $(date -u +%FT%TZ) ===" >>"$L/chain19.log"

python3 ab_v3_next.py --arms cb-lowlat-bcg-dp \
    --dataset sharegpt --concurrency 1 8 32 64 \
    --max-running 1024 --mem-fraction 0.70 --min-prompts 16 --prompt-factor 2 \
    --ready-timeout 5400 \
    --out "$L/results_isolate.json" >>"$L/chain19.log" 2>&1

echo "=== chain19 done $(date -u +%FT%TZ) ===" >>"$L/chain19.log"
