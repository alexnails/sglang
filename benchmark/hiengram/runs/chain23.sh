#!/usr/bin/env bash
# What is 1.37x more pool worth, in the regime where the pool actually binds?
#
# Residency is Little's law: tokens enter the pool only via prefill, so
# resident = input_throughput x lifetime, and context length cancels. At the
# measured 68.3k tok/s the real pool (53.4M) binds only once mean request
# lifetime passes 13 minutes; ours is 22 s, hence the 2.8% utilisation that has
# made every capacity arm inconclusive.
#
# Rather than manufacture 140k-token generations, shrink the pool to where the
# SAME workload binds. Measured peak residency is 1.51M tokens, so:
#   1,200,000  - below demand, must retract/queue   (stands in for baseline)
#   1,644,000  - 1.37x larger, above demand, fits   (stands in for hiengram)
# Both arms are no-cache and identical apart from the cap, so this isolates the
# value of the capacity ratio itself, with no host-fetch cost in the comparison.
set -u
export MODEL_PATH=/scratch/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
export ACC_LEN=5.8
export LONGBENCH_CTX=65536
cd /scratch/engram
L=/scratch/logs/ab3
COMMON="--arms cb-lowlat-dp --dataset longbench_v2 --concurrency 256 512
        --max-running 512 --chunk 32768
        --prompt-factor 2 --min-prompts 16 --ready-timeout 7200"

while pgrep -f "[a]b_v3" >/dev/null; do sleep 30; done
echo "=== chain23 start $(date -u +%FT%TZ) ===" >>"$L/chain23.log"

echo "--- pool 1.20M tokens (binds) ---" >>"$L/chain23.log"
python3 ab_v3_next.py $COMMON --max-total-tokens 1200000 \
    --out "$L/results_poolbind-small.json" >>"$L/chain23.log" 2>&1

echo "--- pool 1.64M tokens (1.37x, fits) ---" >>"$L/chain23.log"
python3 ab_v3_next.py $COMMON --max-total-tokens 1644000 \
    --out "$L/results_poolbind-large.json" >>"$L/chain23.log" 2>&1

echo "=== chain23 done $(date -u +%FT%TZ) ===" >>"$L/chain23.log"
