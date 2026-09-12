#!/usr/bin/env bash
# Retry the two BCG longbench arms that OOM'd in graph capture, now that
# teardown waits for HBM to drain between arms.
set -u
export MODEL_PATH=/scratch/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
export ACC_LEN=5.8
cd /scratch/engram
L=/scratch/logs/ab3
PLAN=/scratch/engram/plan_pooled.npz

while pgrep -f "[a]b_v3|[c]hain6|[c]hain7" >/dev/null; do sleep 30; done
echo "=== chain8 start $(date -u +%FT%TZ) ===" >>"$L/chain8.log"

python3 ab_v3_next.py --arms cb-lowlat-bcg cb-lowlat-bcg-hiengram-4 \
    --plan "$PLAN" --hiengram-gib 4 \
    --dataset longbench_v2 --concurrency 1 8 16 32 64 \
    --prompt-factor 2 --min-prompts 8 \
    --out "$L/results_longbench.json" >>"$L/chain8.log" 2>&1

echo "=== chain8 done $(date -u +%FT%TZ) ===" >>"$L/chain8.log"
