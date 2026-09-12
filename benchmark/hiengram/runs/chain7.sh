#!/usr/bin/env bash
# DP attention round appended: cb-lowlat-dp is expected to be the one that may
# not fit; if it OOMs and cb-lowlat-dp-hiengram-4 does not, that is HiEngram
# enabling a configuration rather than merely speeding one up.
set -u
export MODEL_PATH=/scratch/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
export ACC_LEN=5.8
cd /scratch/engram
L=/scratch/logs/ab3
PLAN=/scratch/engram/plan_pooled.npz
CC="1 8 16 32 64"

while pgrep -f "[a]b_v3|[c]hain6" >/dev/null; do sleep 30; done
echo "=== chain7 start $(date -u +%FT%TZ) ===" >>"$L/chain7.log"

# DP on the leading base, on the workload that stresses memory most
echo "--- dp longbench $(date -u +%FT%TZ) ---" >>"$L/chain7.log"
python3 ab_v3_next.py \
    --arms cb-lowlat-dp cb-lowlat-dp-host-shared cb-lowlat-dp-hiengram-4 cb-lowlat-bcg-dp \
    --plan "$PLAN" --hiengram-gib 4 \
    --dataset longbench_v2 --concurrency $CC \
    --prompt-factor 2 --min-prompts 8 \
    --out "$L/results_longbench.json" >>"$L/chain7.log" 2>&1

# and on chat text, where DP attention usually shows its throughput win
echo "--- dp sharegpt $(date -u +%FT%TZ) ---" >>"$L/chain7.log"
python3 ab_v3_next.py \
    --arms cb-lowlat-dp cb-lowlat-dp-hiengram-4 \
    --plan "$PLAN" --hiengram-gib 4 \
    --dataset sharegpt --concurrency $CC \
    --out "$L/results_sharegpt.json" >>"$L/chain7.log" 2>&1

echo "=== chain7 done $(date -u +%FT%TZ) ===" >>"$L/chain7.log"
