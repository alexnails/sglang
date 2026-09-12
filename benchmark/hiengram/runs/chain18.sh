#!/usr/bin/env bash
# Re-run the capacity arm lost to EADDRINUSE.
#
# cb-lowlat-dp-hiengram-16 died in init_process_group, before model load and
# before any engram code ran: the previous arm's c10d rendezvous socket on 10245
# was still bound. ab_v3_next.py now waits for that port in teardown.
#
# Its baseline half (cb-lowlat-dp, 5 points, peak pool utilisation 0.03) is
# already in results_capacity.json, so this writes alongside it and refresh.py
# merges on arm name.
#
# Waits on chain17.sh, not on ab_v3: chain17 has a gap between its two stages.
set -u
export MODEL_PATH=/scratch/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
export ACC_LEN=5.8
export LONGBENCH_CTX=65536
cd /scratch/engram
L=/scratch/logs/ab3
PLAN=/scratch/engram/plan_pooled.npz

while pgrep -f "[c]hain17.sh" >/dev/null; do sleep 30; done
while pgrep -f "[a]b_v3" >/dev/null; do sleep 30; done
echo "=== chain18 start $(date -u +%FT%TZ) ctx=$LONGBENCH_CTX ===" >>"$L/chain18.log"

python3 ab_v3_next.py \
    --arms cb-lowlat-dp-hiengram-16 \
    --plan "$PLAN" --hiengram-gib 16 \
    --dataset longbench_v2 --concurrency 32 64 128 256 512 \
    --max-running 512 --prompt-factor 2 --min-prompts 16 \
    --ready-timeout 7200 \
    --out "$L/results_capacity_b.json" >>"$L/chain18.log" 2>&1

echo "=== chain18 done $(date -u +%FT%TZ) ===" >>"$L/chain18.log"
