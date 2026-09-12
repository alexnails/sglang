# HiEngram A/B harness

The benchmark harness behind the HiEngram measurements: host-resident engram tables
with a device-side hot-row cache, on 4x GB300, TP4/EP4, DeepSeek-V4.1-Flash.

All numbers come from `sglang.benchmark.serving` driving a real server. An earlier
hand-rolled client sent identical prompts and the radix cache served them from the
prefix tree, which invalidated every concurrency number; don't reintroduce one.

## Layout

| path | what it is |
| --- | --- |
| `ab_v3.py` | the driver: builds the arm matrix, launches a server per arm, runs a sweep, scrapes the server log, writes JSON |
| `make_plan.py` | builds the frequency-ordered hot-row plan consumed by `SGLANG_DSV41_ENGRAM_HBM_CACHE_PLAN` |
| `show.py` | prints arms and points from a results file |
| `poolcheck.py` | reconciles scraped pool GB against `max_total_num_tokens` |
| `runs/chain*.sh` | one script per round, each carrying why it was run in its header |

## Running an arm

```
export MODEL_PATH=/path/to/DeepSeek-V4.1-Flash
export ACC_LEN=5.8
python3 ab_v3.py --arms cb-lowlat-dp cb-lowlat-dp-hiengram-16 \
    --plan plan_pooled.npz --hiengram-gib 16 \
    --dataset longbench_v2 --concurrency 64 128 256 512 \
    --max-running 512 --out results.json
```

Acceptance is pinned with `SGLANG_SIMULATE_ACC_LEN` and gated per point at +/-0.03, so
generated text is meaningless by construction: read throughput and latency, never output.
Each point also lands in `<out>.points.jsonl` as it completes, because a multi-hour arm
otherwise writes nothing until the end.

## Things that cost a run

- `chunked_prefill_size` is divided by `dp_size`, not halved. This GPU tier defaults to
  16384, so every DP arm silently runs 4096 unless told otherwise -- worth 15-18% here.
  Pass `--chunk` at `dp_size` times the per-rank value you want.
- `ReqToTokenPool` is `max_running_requests * max_context_len * 4B`. At the model's full
  1M context a slot costs 4 MB, which is what makes a high request ceiling unaffordable;
  `--context-length` sized to the workload fixes that.
- `nccl_port` defaults to `get_free_port()`, which probes with `SO_REUSEADDR` while
  torch's TCPStore binds without it, so the probe can hand back a port TCPStore then
  rejects with EADDRINUSE before the model loads. The driver pins a rotating port.
- `--mem-fraction-static 0.8` leaves no room for 1024 request slots. Use 0.70.
- Driving retraction hard reaches an unguarded `repeat_interleave(..., output_size=...)`
  in the DSV4 low-ratio indexer (`deepseek_v4_backend.py`), which fails a device-side
  assert in PyTorch's `Repeat.cu`. Not an engram path.
