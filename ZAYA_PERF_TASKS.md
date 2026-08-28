# ZAYA1-74B perf climb — task board

Hardware: ONE devbox `zaya-mi355x-b`, 8x MI355X (gfx950, ROCm 7.2). GPU arms are
therefore **strictly serial**; code tracks run in parallel in isolated worktrees.

Baseline (measured, MI355X, tp8/dp4, ISL 1024/OSL 256, rep 1 discarded):

| arm | C=1 tok/s | TPOT | C=32 tok/s | TPOT | C=128 tok/s | TPOT |
|---|---|---|---|---|---|---|
| bf16 baseline | 61.1 | 16.00 | 1706.5 | 17.58 | 4878.2 | 22.73 |
| + global residual | 65.4 | 15.00 | 1826.9 | 16.45 | 5306.2 | 20.98 |

`SGLANG_OPT_ZAYA_GLOBAL_RESIDUAL=1` is the reference config for every arm below.

Concurrency scaling on the reference config (ISL 1024/OSL 256):

| C | out tok/s | total tok/s | TPOT ms | scaling efficiency |
|---|---|---|---|---|
| 1 | 65.4 | 327 | 15.00 | - |
| 32 | 1826.9 | 9128 | 16.45 | 87% |
| 128 | 5306.2 | 26397 | 20.98 | 63% |
| 512 | 10943 | 54674 | 35.80 | 33% |

Still climbing at C=512 (2.06x throughput for 4x concurrency), so the fixed
per-layer cost is amortizing rather than saturating. max_running_requests is
170/replica = 680 total at `--max-mamba-cache-size 2048`.

Prefill / TTFT reference (`--disable-radix-cache`, since bench_serving reuses
its random prompts and the prefix cache otherwise skips prefill entirely):

| cell | TTFT off | TTFT + prefill graph | delta |
|---|---|---|---|
| ISL 128, C=128 | 303.5 | 189.2 | **-37.7%** |
| ISL 256, C=64 | 276.1 | 231.1 | **-16.3%** |
| ISL 1024, C=128 | 811.7 | 851.8 | +4.9% |
| ISL 4096, C=8 | 366.7 | - | - |
| ISL 16384, C=4 | 850.9 | - | - |

So `--cuda-graph-backend-prefill breakable` is a per-workload flag: a large win
below ~256 ISL, a loss above ~1k. Decode is neutral (-0.5%).

## Launch census (static, brackets the measured ~2382/step)

| cluster | /step | share | addressable |
|---|---|---|---|
| Router scalar soup | 840 | 35% | ~700 |
| dp_gather_partial wrapper (SUM_LEN) | 240 | 10% | 180 (fill+memcpy+copy-back) |
| CCA chain | 900 total ATT layer | 37% | 60 (conv bias-add) + 60 (rope/KV store) |
| MoE expert wrappers | ~360 | 15% | 120 (MoD) + 60 (casts) |
| MoE TP8 all-reduce | 60 | — | irreducible |

## Code tracks

### A — small, high-confidence (owner: me)
- [x] A1 MoD dead-code gate. `balancing_biases[-1] = -1.0` on all 60 layers and the
      bias is added to a *softmax probability*, so the skip slot is provably
      unreachable: `p_skip + b_skip <= 1 + b_skip <= max_j b_j <= max_j(p_j + b_j)`.
      Gate the branch at load time. **-120 launches/step, bit-identical.**
- [x] A2 `inplace=False` on the FusedMoE construction. Latent aliasing hazard on the
      triton runner (aiter allocates fresh, so not live here). Defensive.
- [~] A3 conv bias-add. DEFERRED, and the obvious form is a trap: `baddbmm` batches
      over G and yields `[G,T,O]`, so transposing back to `[T,G,O]` costs a copy and
      cancels the win. Better design: have `cca_state_step` emit the window with a
      trailing ones-column (free -- it already writes the window) and append the bias
      as the last column of `decode_conv_weight` in `fold_decode_conv`. The bias then
      lands inside the matmul's fp32 accumulator (strictly more accurate) and the
      separate add disappears. **-60 launches/step.** Blocked on track C2, which owns
      `cca_state_step.py`; hand it over once C2 lands.

### B — router fusion (owner: agent, worktree)
- [ ] B1 route-select kernel: softmax + balancing_biases + argmax + gather + 3 casts
      + clamp -> one Triton kernel emitting bf16 prob / clamped int32 ids / fp32
      weights. Kills the fp32->bf16->fp32 round trip into aiter. **~-480/step.**
- [ ] B2 EDA mul+add -> `addcmul`. **-59/step.** (two-line change)
- [ ] B3 router MLP 256->256->256->25, 3 GEMMs + 2 GELUs -> one persistent kernel.
      **-240/step.** 270 KB of weights per layer = 23 us of HBM for the whole step,
      i.e. pure launch overhead.

### C — dp_gather wrapper + Triton occupancy (owner: agent, worktree)
- [x] C1 (-120, not -180: see commit) `_dp_gather_via_all_reduce` under SUM_LEN does fill_(0) + memcpy_triton +
      all-reduce + copy-back = 4 launches/attention layer. Only the AR is
      irreducible. **-180/step.**
- [x] C2 `cca_state_step` launches 32 programs on a 256-CU GPU; `cca_qk_mix` runs
      num_warps=4 (256 threads) on a 128-element tensor. Duration, not count.

### D — CCA v2 projected cache (owner: agent, worktree)
- [ ] D1 Cache `W2 . hs` instead of raw `hs` in conv[1]: state width 4096 -> 128
      (32x). Bit-exact (val_proj2 is linear, bias-free). Per-request state
      768 KB -> 292 KB => **2.6x mamba slots**; prefill `v2_input[1:].copy_()`
      drops ~2.0 GB/step of memcpy => est. **TTFT -5 to -9%**.

## GPU arms (serial, one node)
- [x] G1 finish chain2: long prefill (ISL 4096/16384) +/- fused CCA prefill; C=512.
- [ ] G2 aiter fmoe tuned-table clone. Confirmed miss: every MoE call logs
      `[aiter] [fused_moe] using 2stage default for ('gfx950',256,M,4096,512,24,1,...)`
      and `/tmp/aiter_configs/tuned_fmoe.csv` has **zero rows with expert=24**
      (2403 rows, 1357 gfx950). Clone the E=512/513 rows at the identical
      model_dim=4096/inter_dim=512 and point `AITER_CONFIG_FMOE` at the copy.
      **No code change.**
      **RESULT: NOT ESTABLISHED -- retracted pending a clean rerun (chain 5).**
      Measured -6.0% TTFT, but the control's own rep spread was 6.3% (349.7 vs
      371.6 ms at ISL4096/C=8), i.e. the effect sits inside the noise. Root cause:
      `--num-prompts` was `conc*4`, so ISL4096/C=8 got 32 prompts and ISL16384/C=4
      got 16 -- a median TTFT over 16 samples is worthless. Fixed: floored at
      max(8*conc, 64). Confounder worth keeping in mind: the TREATMENT arms were
      stable to 0.06% at the same 32 prompts, so the control looks disturbed rather
      than the cell being inherently unmeasurable.
      Note the C=128 decode cells are unaffected -- `conc*4` gives 512 prompts
      there, and those reps agreed to 0.1-0.3%. TPOT figures throughout are solid
      (averaged over 256 output tokens/request); low-concurrency TTFT figures are
      the weak ones.
      **The original observation stands regardless:** The only *unquantized* gfx950 rows at
      our model_dim=4096 sit at inter_dim=384/E=128/topk=8, and what they encode is a
      block_m ladder (32 for token<=256, 64 at 512, 128 at 1024+, ksplit 0). At decode
      M is small so the default already picks 32 -- hence no decode effect -- but on
      long prefill:

      | cell | baseline | + cloned table | delta |
      |---|---|---|---|
      | ISL 4096 C=8 TTFT | 366.7 ms | 339.6 ms | **-7.4%** |
      | ISL 4096 C=8 total tok/s | 8292 | 8667 | **+4.5%** |
      | ISL 16384 C=4 TTFT | 850.9 ms | 796.9 ms | **-6.3%** |
      | ISL 16384 C=4 total tok/s | 13132 | 13696 | **+4.3%** |

      Rep spread 0.06%, so real. This is a HEURISTIC transfer, not a tune: the E and
      topk fields were rewritten and only block_m/ksplit carry over. A confirmed win
      means the shape deserves a real tuning run
      (`benchmark/kernels/fused_moe_triton/`-style sweep against aiter's own tuner),
      and the row should be upstreamed into aiter's table rather than shipped as a
      local CSV.
- [ ] G3 `--quantization mxfp4`. Verified `mxfp4.py:367-368`: on HIP every LinearBase
      gets UnquantizedLinearMethod, only FusedMoE is quantized. Experts
      18.1 -> 4.5 GB/GPU, zero added launches. Step 0 first: read the fused_moe
      share of step time off a profile; if <15%, the axis caps there.
- [ ] G4 A/B each landed code change against the global-residual reference.

## Refuted — do not re-propose
Reduce-scatter MoE combine; MAX_LEN padding; side-stream gather overlap; TBO;
4x tp2 servers; fused decode conv (`cca_conv1d_update`); torch.compile;
`--moe-runner-backend aiter` (identical to auto); `SGLANG_DP_USE_REDUCE_SCATTER=0`
(no-op under SUM_LEN); lifting the attn_tp=2 cap (same collective count, 4x less
KV); attn_tp=1 (o_proj reduce is already free under global residual);
`--quantization mxfp8` (gate_up_interleaved + SwiGLU-OAI mismatch); MoD `-1` ids.

## Operational
NEVER `kill -9` an sglang server on this box: ROCm leaks the ranks' device
allocations and the container PID 1 is `sleep infinity`, so ~200 of 288 GiB per
card is lost for the pod's lifetime. The harness gates `stop` on VRAM bytes
returning to idle and refuses to start another arm otherwise.

## RESULT: the landed stack (measured, MI355X, ISL 1024/OSL 256, reps 2-3)

-299 launches/step banked: A1 MoD gate (-120), C1 sum_len gather fusion (-120),
B2 EDA addcmul_ (-59), plus the CCA Triton occupancy fixes (duration, not count).

| C | reference (global residual) | landed | d tok/s | d TPOT |
|---|---|---|---|---|
| 1 | 65.4 / 15.00 ms | **74.4 / 13.11** | **+13.7%** | **-12.6%** |
| 32 | 1826.9 / 16.45 ms | **2025.7 / 14.82** | **+10.9%** | **-9.9%** |
| 128 | 5306.2 / 20.98 ms | **5801.2 / 19.12** | **+9.3%** | **-8.9%** |

Better than proportional: 299 of ~2382 launches is 12.5%, which at "launches are
~45% of the step" predicts ~5.6% TPOT, and ~10% was measured. The removed ops are
not average ones -- MoD premask/blend and the gather's fill_/copy-back each move a
full [T, 4096] buffer, so ~250 MB/step of traffic left with them.

Cumulative vs the session's bf16 start: C=1 TPOT -18.1%, C=32 -15.7%,
C=128 tok/s +18.9%.

## NOT banked, and why

- **B1/B3 router fusion (-720).** The two Triton kernels had never been compiled.
  The TAIL kernel takes a DEVICE-SIDE FAULT on gfx950: SIGABRT at the next sync,
  two tests after the offending launch, no Python error. The traceback points at
  `ReplicatedLinear.__init__` in an unrelated third test -- that is just where the
  CPU thread was. Both kernels are behind `SGLANG_OPT_ZAYA_FUSED_ROUTER` (default
  off) and their 19 tests are skipped with the reason recorded. Handed back to the
  agent with the signature and `AMD_SERIALIZE_KERNEL=3`.
- **D1 CCA v2 projected cache (2.63x state shrink).** Built before `cca_conv1d.py`
  existed on the branch, and that file also writes the 4096-wide v2 stream D1
  shrinks to 128. Needs reconciling before it is safe to merge.

## Refuted DURING this session (my own hypotheses, corrected)

- **page_size=1 costs TTFT: WRONG.** `extend_attention.py:500` states PAGE_SIZE==1
  compiles to byte-identical SASS, and `create_flashinfer_kv_indices_triton` is
  per-token at every page size. `--page-size 64` is a DECODE flag (it unlocks
  aiter's vectorized_5d layout). The `_MAMBA_EXTRA_BUFFER_ARCHS` gap still gates
  the overlap scheduler and page size, but NOT the prefill path.
- **The prefill graph corrupts conv state: WRONG.** The graph demonstrably engages
  (`Capture target prefill CUDA graph begin. backend=breakable`) and gsm8k holds
  at 0.285 vs 0.270 control. A bug giving 31 of 32 requests the wrong state would
  have been catastrophic. Real finding underneath: **ZAYA1 is not batch-invariant
  at all** -- 4 distinct outputs from 8 identical greedy prompts at bs=8 with NO
  graph -- which is what actually explains "the graph changes greedy token ids".
- **Quantization is only a capacity lever: WRONG.** mxfp4 gives TPOT -23.7% at C=1
  where weight traffic is ~3% of the step, so it cannot be bandwidth. It is KERNEL
  QUALITY: mxfp4 hits aiter's tuned per_1x32 path (976 gfx950 rows) while bf16
  falls to the untuned 2stage default. Same root cause as the table miss.
  => Highest-value cheap experiment left: tune the **bf16** E=24/topk=1/
  inter_dim=512 shape properly. It may capture much of that 20% with no
  quantization and no accuracy question.

## Measurement hygiene learned the hard way

- `--num-prompts conc*4` gave 16-32 samples in the long-prefill cells; the control
  spread 6.3% against 4-6% effects. Floored at max(8*conc, 64). TPOT figures were
  never affected (averaged over 256 output tokens, reps agree to 0.1-0.3%).
- Never rewrite a script in place while a chain is invoking it: bash reads
  incrementally and the running instance sees corrupted content ("unbound
  variable" at a shifted line). Write to .tmp and `mv`.
- Gate a stop on VRAM BYTES returning to idle, three consecutive readings, and
  refuse to serve unless every card has >=240 GiB free. A single low sample let an
  arm start against ~274 GB still in use.
- gsm8k on this preview checkpoint has 37-49% unparseable answers, so it catches
  only gross regressions.
