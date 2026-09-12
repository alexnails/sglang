#!/usr/bin/env bash
# Does the DP chunk halving explain the DP ceiling?
#
# Measured on the capacity run: peak pool usage ~2% of 73.27M tokens, peak
# #running-req 48 per DP rank against a 512 ceiling, input:output 29.6:1, mean
# prefill chunk 4076 of 4096. Nothing is capacity-bound and nothing is
# request-slot-bound; the server is prefill-bound at a 4096 chunk.
#
# sglang HALVES chunked_prefill_size under DP attention (server_args.py:3728),
# so every DP arm so far ran an effective 4096 -- the setting worth up to +54%
# elsewhere in this study. Passing 16384 yields an effective 8192.
#
# Both arms cap the pool at 8M tokens (5x the measured 1.5M peak) so the only
# variable is the chunk, and so the memory a larger chunk needs is actually free
# rather than absorbed by an auto-sizing pool.
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

while pgrep -f "[c]hain19.sh" >/dev/null; do sleep 30; done
while pgrep -f "[a]b_v3" >/dev/null; do sleep 30; done
echo "=== chain20 start $(date -u +%FT%TZ) ===" >>"$L/chain20.log"

echo "--- capped pool, inherited chunk (effective 4096) ---" >>"$L/chain20.log"
python3 ab_v3_next.py $COMMON --out "$L/results_chunk-4096.json" >>"$L/chain20.log" 2>&1

echo "--- capped pool, chunk 16384 (effective 8192) ---" >>"$L/chain20.log"
python3 ab_v3_next.py $COMMON --chunk 16384 --out "$L/results_chunk-16384.json" >>"$L/chain20.log" 2>&1

echo "=== chain20 done $(date -u +%FT%TZ) ===" >>"$L/chain20.log"
