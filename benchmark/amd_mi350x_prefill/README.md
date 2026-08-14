# AMD MI350X prefill configs

Measured optimal serving configs for five models on AMD Instinct MI350X (gfx950,
CDNA4, 256 CU, 287.7 GiB HBM3E), ROCm 7.2 userspace on a 7.0.2 host driver.

`configs.json` holds the full dataset: per-model config, per-concurrency tables
(req/s, throughput, all TTFT percentiles, cache hit rate), every config-axis
delta, build provenance, drift controls, and the mechanism findings.

## Scope

Two workloads, both at **OSL = 1** so TTFT isolates prefill:

- **W1** — 8k ISL, uncached (`--disable-radix-cache`, verified `#cached-token: 0`)
- **W2** — 64k ISL at ~93.75% nominal cache hit (`generated-shared-prefix`)

Concurrency swept 1 → 128. 319 measured points across four completed models;
Kimi K2.7 is stubbed pending its run.

## Headline

**Concurrency dominates every flag measured.** At cc=1 you already have 84–98.5%
of peak throughput; cc=128 buys another 2–6% for ~95–110× the TTFT. The useful
operating band is cc=1–4 (W1) and cc=8–32 (W2).

**Data parallelism beats tensor parallelism for prefill.** 4×TP1 behind a
prefix-affinity router gives 2.0× TP4's per-GPU throughput *and* 2.1× better
TTFT. TP scales poorly here — solo 8k prefill is 125.2 ms at TP1, 125.6 ms at
TP2 (two GPUs buy nothing), 75.5 ms at TP4. The cost is shrinking GEMM/MoE
tiles, not the all-reduce. Note round-robin instead of prefix-affinity turns
that 2× win into a 35% **loss** on cached traffic.

## Flags that are actively harmful on specific models

- `--cuda-graph-backend-prefill` — crashes GLM-5.2 (DSA) and Gemma-4 (sglang
  auto-disables the prefill graph for multimodal; passing the flag *overrides*
  that and re-enables a broken path). GLM cannot start out of the box on ROCm
  because `breakable` is the default.
- `--page-size 1024` on GLM-5.2 — hangs the server. The override reaches the
  attention path but not the memory-pool path.
- Page size generally has four different right answers: gpt-oss needs 64,
  Gemma wants 1, GLM is forced to 64, Qwen cannot exceed 256.

## Benchmarking notes

Several of these were bugs that cost real runs; see `methodology_rules` in the
JSON for the full list.

- `--random-range-ratio 1.0` is mandatory — the default 0.0 yields a mean of
  ~50% of nominal ISL.
- The W2 hit-rate ceiling is **tokenizer-dependent** (90.9%–93.75% measured), not
  a flat 93.75%. Compute realized-prefix ÷ realized-total; nominal 61440/4096
  realizes to 67.5k–76.4k tokens depending on the model.
- Prewarm must match **both** the seed and `--gsp-prompts-per-group`, and every
  point needs a `flush_cache` — a descending ladder otherwise re-runs the
  previous rung's exact requests (observed 100.0% hit, 651k tok/s: void data).
- Size the group count so the prefix working set fits the KV pool. On Gemma,
  8 groups overflowed and retained 0 of 8 prefixes, costing 3.05× throughput
  versus 4 groups.
- `in_tok/s` on cached workloads counts cached tokens and is inflated ~16×.
- `#running-req` is not residency — it tracks cumulative requests processed.
  Use `full_tok_usage` and `#new-seq`.
