---
name: ttft-accounting
description: Build a reconciled, op-level TTFT ledger for an LLM serving config and turn it into ranked, gate-validated optimization levers. Hardware-generic (NVIDIA/AMD): the accounting identity, trace hygiene, roofline discipline, and validation gates are invariant; only the ceilings and occupancy granularity are measured per silicon. Use when asked to account for / decompose / map TTFT, find prefill bottlenecks, or adjudicate a claimed TTFT win.
---

# TTFT Accounting

Accounting means every millisecond is *assigned* and the assignment is *checked*. A kernel table alone is not accounting; a ledger that does not sum to the measured TTFT is a story, not a measurement.

## The identity

```
TTFT = non-GPU (tokenize + transport + scheduler/batch-build + response path)
     + GPU busy (sum of per-op kernel time inside the request window)
     + launch/idle gaps inside the window
```

**Reconciliation is mandatory**: ledger total vs measured TTFT within ±10%; a careful pass reaches <1%. Reconcile against BOTH the trace-derived request window and the client-measured TTFT (they differ by the inter-request turnaround — account for it, don't hand-wave it). If the ledger does not reconcile, the ledger is wrong — find the leak before drawing any conclusion.

## Step 0 — measure the silicon's ceilings yourself

Never quote spec sheets. On the exact GPU, measure: dense GEMM throughput at a large square shape (the compute ceiling), and a device-to-device copy (the bandwidth ceiling). All roofline verdicts in the ledger cite these numbers.

Two roofline traps that inverted conclusions in practice:
- **A kernel can be "at 41% of the GEMM ceiling" and still have zero headroom** — compute its *own* op-mix roofline first. A masked-softmax attention epilogue can cost ~10+ vector-ALU slot-equivalents per score element; a kernel at 74% of its vector-ALU issue roofline is nearly optimal no matter what the MFMA/tensor-core ceiling says. Count the issue slots before declaring headroom.
- **"X% of peak" from a trace is not a utilization measurement.** Re-derive FLOP/s from first principles (shape math / kernel time) and cross-check against a microbench of the same kernel at the same shape. A serving trace under memory pressure from interleaved layers can read 25% slower than the same kernel solo.

## Step 1 — workload construction (get the shape you think you have)

- Calibrate synthetic prompt lengths: nominal ≠ tokenized. Measure the inflation ratio once (system and question parts inflate differently), then verify realized ISL and cached-tokens per rung from the benchmark's own report.
- Cached-prefix targets must be page-aligned (cached tokens are floored to page multiples).
- **Warmup trap**: benchmark warmups typically reuse the first dataset prompts, so at small NP some measured requests are full cache hits. Use a fresh seed after every cache flush, and identify true requests by GPU-burst size when analyzing a trace.
- Cached-token *reports* inflate under concurrent chunked admission (chunk self-matches). Read cached/req at cc1 to get the truth; treat higher-cc readings as accounting noise.

## Step 2 — capture

- Profile at cc1 first: one request = one attributable window. Capture 6–10 true requests.
- Check profiler overhead by comparing profiled vs unprofiled TTFT on the same server — if they differ materially, fix that before trusting the trace.
- Prefer one-rank traces. Record whether kernels run under graph replay (they attribute to the replay call-site; a graph-off "mapping" capture recovers real python sites when needed).
- Know your graph-padding: ops inside a captured graph run at the padded bucket size; grid math planned from a padded buffer (`q.shape[0]`) instead of the real token count silently over-counts. Wins measured against padded do-nothing work are serving no-ops.

## Step 3 — trace hygiene (each of these fabricated or destroyed a top-3 finding at least once)

1. **Sum-vs-union first.** `sum(kernel durations) / union(intervals)` ≈ 1.0 means the trace is clean; >1.3 means corrupt timestamps. Only then consider outlier filtering.
2. **Never apply a ">N× median duration for this kernel name" filter to bimodal kernels.** One kernel name can legitimately span a 185× duration range in a single forward (e.g., the same attention kernel serving full-attention and sliding-window layers). A naive filter deleted 68–75% of real GPU time in two independent traces and inverted GPU-bound into launch-bound. Use mode-aware filtering: split each name's durations into log-space modes (>0.7 dex gaps), filter within modes.
3. **Averaging trap**: per-request GPU estimates diluted by cache-hit or warmup requests manufacture phantom CPU gaps (a fake ~0.9 s "gap" survived one full analysis round). Compute per-request windows from burst boundaries of *true* requests only.
4. `with_stack` inflates CPU-side durations — use stacks for attribution, kernel-event boundaries for wall time.

## Step 4 — the ledger

One row per op family, exhaustive down to ~0.05 ms/request. Columns:

```
op family | ms/req | %TTFT | launches/req | ms/launch | python call-site | assessment
```

`assessment` is exactly one of:
- **bound** — at a measured roofline (cite which: HBM %, GEMM %, issue-slot %); not a lever.
- **tuned** — config space exhausted by sweep (cite the sweep); not a lever.
- **LEVER** — plausible win, with the mechanism named and a ms estimate.

Cover the non-GPU side with the same rigor: tokenize, IPC serialize/deserialize, scheduler batch build, radix/prefix match, allocation, page-table build, sampling, detokenize, HTTP. Corroborate the CPU floor three ways: inter-request GPU gap, a full-cache-hit request's end-to-end time, and (trace TTFT − GPU busy). CPU components under ~1 ms each are almost never worth chasing; find that out *from the ledger*, not by guessing.

## Step 5 — occupancy structure (the hardware-specific 20 minutes that pays)

TTFT at fixed work can be a **staircase in the launch geometry**, not a function of token count. Detect it with a controlled length sweep (±10% around the deployed shape, fine steps): flat-within-bands + ~(N+1)/N steps at boundaries is the signature. Two mechanisms to distinguish:
- true wave quantization (programs / (CUs·occupancy) crosses an integer), and
- **co-residency generations**: if K programs are co-resident (occupancy × CU count), everything past a whole multiple runs as a nearly-empty generation whose stragglers are *latency*-bound — a flat per-layer tax regardless of remainder size. The fix class differs (tail split / persistent kernel / pipelining that hides the straggler latency), so identify which one you have by whether cost is flat or proportional past the boundary.

This is the only hardware-specific section: measure ceilings, CU/SM count, and occupancy per silicon; everything else in this skill transfers unchanged.

## Step 6 — turning levers into adopted changes (the gates)

- **Probe → confirm**: rank with cheap probes, confirm every winner with a full clean run. Deltas <5% in probes are noise until reproduced. Pair arms same-box-same-hour; box-state sensitivity of several percent is real.
- **A win that beats its own projection is guilty until cross-checked.** The one campaign result that exceeded kernel-math projection (−13% vs −5% projected) was fast-but-wrong (a gather bug real page tables exposed and synthetic `arange` layouts hid). Microbench with *realistic* layouts (churned page tables, ragged batches), not clean ones.
- **Correctness gates, in order of trustworthiness**: (1) bitwise comparison where the change claims exactness — and certify each knob *separately*: "all knobs off is exact" does not mean each knob is (a loop restructure alone reassociated an epilogue everyone believed was exactness-preserving); (2) planted-fact retrieval concordance vs a trusted reference server for long-context paths; (3) *paired* back-to-back task metric (e.g. gsm8k n=200, |Δ| gate ≈0.03 — its absolute spread is ±0.03–0.05, so unpaired absolutes prove nothing). **Greedy token agreement on random text is NOT a gate** — two correct kernels legitimately diverge on near-ties; it produced a false rejection of a correct change.
- Anomalous single-rung tails (one p99 spike with clean mean/median): rerun before believing — periodic host effects exist.
- Label numerics tiers exactly as certified, never better. If a change is 1-ULP-class, say "1-ULP-class", not "exact". Ship non-bit-exact kernel changes default-off behind a flag so adoption is a per-deployment choice.

## Deliverable

The ledger table (both GPU and non-GPU sections) + the reconciliation arithmetic + a ranked lever list where every lever carries a mechanism, an ms estimate, and a confidence — and every non-lever row carries the measured reason it is closed. The lever list is only as credible as the reconciliation line above it.
