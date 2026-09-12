#!/usr/bin/env bash
# Force the pool to fill: loosen admission, raise the request ceiling.
#
# chain24 ran 1024 x 36k-token generations and both arms still parked the SAME
# absolute residency -- 10.47M vs 10.64M tokens, within 1.7% -- despite one
# having 65% more pool. Offered load does not set residency; the admission
# policy does, and it declined to use the headroom. So this run attacks the
# policy instead of the workload.
#
#   --schedule-conservativeness 0.1
#       new_token_ratio = INIT(0.7) x conservativeness, and DP multiplies by a
#       further 0.3, so this reserves ~2% of max_new_tokens per admitted request
#       instead of ~21%. Lower is more aggressive; larger is more conservative.
#   --context-length 65536
#       ReqToTokenPool is size x max_context_len x 4B and the server otherwise
#       runs the model's full 1M context, so a slot costs 4 MB. The workload
#       needs 41k, so capping here makes 4096 slots cost ~1 GB instead of 17 GB.
#   --max-running-requests 4096
#       affordable only because of the line above.
#
# Residency targets against the pools measured at --mem-fraction 0.70
# (no cache 30.79M, hiengram 50.68M), at 4k in + 36k out = 40,960 per request:
#   c=1024   41.9M  - inside the band: baseline over, hiengram fits
#   c=2048   83.9M  - past both: baseline should retract far harder
set -u
export MODEL_PATH=/scratch/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277
export ACC_LEN=5.8
cd /scratch/engram
L=/scratch/logs/ab3
PLAN=/scratch/engram/plan_pooled.npz

while pgrep -f "[a]b_v3" >/dev/null; do sleep 30; done
echo "=== chain25 start $(date -u +%FT%TZ) ===" >>"$L/chain25.log"

python3 ab_v3_next.py \
    --arms cb-lowlat-bcg-dp cb-lowlat-dp-bcg-hiengram-16 \
    --plan "$PLAN" --hiengram-gib 16 \
    --dataset random --input-len 4096 --output-len 36864 \
    --concurrency 1024 2048 \
    --max-running 4096 --mem-fraction 0.70 \
    --extra "--schedule-conservativeness 0.1 --context-length 65536" \
    --prompt-factor 1 --min-prompts 16 --warmup 2 \
    --ready-timeout 7200 \
    --out "$L/results_agentic-forced.json" >>"$L/chain25.log" 2>&1

echo "=== chain25 done $(date -u +%FT%TZ) ===" >>"$L/chain25.log"
