#!/usr/bin/env bash
# Run the remaining passes after the in-flight driver finishes.
set -u
export MODEL_PATH=/scratch/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
export ACC_LEN=5.8
cd /scratch/engram
L=/scratch/logs/ab3

while pgrep -f "[a]b_v3.py" >/dev/null; do sleep 30; done
echo "=== prior driver done, starting chain $(date -u +%FT%TZ) ===" >>"$L/chain.log"

# 1. capacity pair + chunk-size controls, random 4k1k
python3 ab_v3_next.py --arms tuned-free tuned-host-shared-free tuned-16k tuned-bcg-16k \
    --dataset random --concurrency 1 8 32 64 \
    --out "$L/results.json" >>"$L/chain.log" 2>&1

# 2. shared-prefix 4k1k: 3584 shared + 512 unique
python3 ab_v3_next.py --arms tuned tuned-host-shared tuned-bcg tuned-bcg-host-shared \
    --dataset generated-shared-prefix --concurrency 1 8 32 64 \
    --out "$L/results_gsp.json" >>"$L/chain.log" 2>&1

# 3. sharegpt: the only workload where the hot set can hit
python3 ab_v3_next.py --arms tuned tuned-host-shared tuned-hiengram-4 tuned-hiengram-8 tuned-hiengram-16 \
    --plan /scratch/engram/plan_pooled.npz --hiengram-gib 4 8 16 \
    --dataset sharegpt --concurrency 1 8 32 64 \
    --out "$L/results_sharegpt.json" >>"$L/chain.log" 2>&1

echo "=== chain done $(date -u +%FT%TZ) ===" >>"$L/chain.log"
