#!/usr/bin/env bash
# The agentic-lifetime run: the one regime where HiEngram's capacity can pay.
#
# Residency is Little's law, resident = input_throughput x lifetime, and context
# length cancels. Every workload tried so far had a ~22 s lifetime and left the
# pool at 2-3%, which is why every capacity arm was inconclusive. Long
# generations are the only way into the band: the pool fills from decode, one
# token per running request per step, and requests stay resident while it fills.
#
# Sized against the two pools measured at --mem-fraction 0.70:
#   no cache   30.79M tokens
#   hiengram   50.68M tokens  (1.65x)
#
# 4k in + 36k out = 40k resident per request:
#   c=512   21.0M  - CONTROL, both fit (68% / 41%), expect parity
#   c=1024  41.9M  - baseline OVER by 36%, hiengram fits at 83%
#
# At c=1024 the baseline crosses its pool once each request has generated ~26k
# of its 36k tokens, so the divergence appears in the back half of the point and
# shows up as retraction, queueing and a throughput drop the cache arm does not
# take. Hours is expected: 37.7M tokens per point, 12-37 min depending on how
# ITL degrades at 40k context, times two points times two arms.
#
# --prompt-factor 1 puts exactly one request in each slot, so residency is the
# steady state rather than a queue being worked off.
set -u
export MODEL_PATH=/scratch/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
export ACC_LEN=5.8
cd /scratch/engram
L=/scratch/logs/ab3
PLAN=/scratch/engram/plan_pooled.npz

while pgrep -f "[c]hain23.sh" >/dev/null; do sleep 30; done
while pgrep -f "[a]b_v3" >/dev/null; do sleep 30; done
echo "=== chain24 start $(date -u +%FT%TZ) ===" >>"$L/chain24.log"

python3 ab_v3_next.py \
    --arms cb-lowlat-bcg-dp cb-lowlat-dp-bcg-hiengram-16 \
    --plan "$PLAN" --hiengram-gib 16 \
    --dataset random --input-len 4096 --output-len 36864 \
    --concurrency 512 1024 \
    --max-running 1024 --mem-fraction 0.70 \
    --prompt-factor 1 --min-prompts 16 --warmup 2 \
    --ready-timeout 7200 \
    --out "$L/results_agentic.json" >>"$L/chain24.log" 2>&1

echo "=== chain24 done $(date -u +%FT%TZ) ===" >>"$L/chain24.log"
