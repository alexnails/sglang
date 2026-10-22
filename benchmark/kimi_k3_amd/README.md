# Kimi-K3 on AMD gfx950 (MI350X / MI355X): `kevin/k3-amd-full`

This branch is SGLang with every Kimi-K3 gfx950 optimization from the merge queue applied, plus the open upstream
PRs that the measured config depends on. It runs on one node of 8 GPUs: TP8, aiter MLA, an FP8 KV cache and EAGLE3
(topk 1, 3 steps, 4 draft tokens).

## Quick start

```bash
git clone -b kevin/k3-amd-full https://github.com/kevin-mii/sglang.git
cd sglang
PYTHONPATH=$PWD/python benchmark/kimi_k3_amd/launch_k3_eagle3.sh
```

`launch_k3_eagle3.sh` sets the environment and server arguments of the measured config.

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `moonshotai/Kimi-K3` | Target checkpoint |
| `DRAFT` | `lightseekorg/kimi-k3-eagle3-mla` | EAGLE3 draft |
| `PORT` | `30000` | HTTP port |
| `K3_GEMM_TABLE` | `k3_bf16_tuned_gemm_mi350x.csv` (next to the script) | aiter bf16 GEMM table; empty (`K3_GEMM_TABLE=`) uses aiter's own tables |
| `COALESCE` | `0` | `1` turns on prefill coalescing for C8 and up: more tok/s/user, higher TTFT |

Extra arguments are passed to `sglang.launch_server`.

## What's in the branch

The base is `main` 4eb606b2ae (2026-10-17). On top of it, in order:

1. **The twelve Kimi-K3 gfx950 PRs**, in merge-queue order:

   | Slot | PR | Change |
   |---|---|---|
   | 1 | [#44978](https://github.com/sgl-project/sglang/pull/44978) | Greedy-verify argmax split, aiter tuned bf16 rows |
   | 2 | [#44979](https://github.com/sgl-project/sglang/pull/44979) | MLA prefill: FP8 prefix K/V pack, absorbed short cached extends |
   | 3 | [#44980](https://github.com/sgl-project/sglang/pull/44980) | Streaming attention-residual read for prefill |
   | 4 | [#44981](https://github.com/sgl-project/sglang/pull/44981) | MoE prefill: MXFP4 activations, SiTU front epilogue, row-sharded tail |
   | 5 | [#44982](https://github.com/sgl-project/sglang/pull/44982) | KDA verify tuning, ReplaySSM commit, chunk-major prefill |
   | 6 | [#44983](https://github.com/sgl-project/sglang/pull/44983) | MLA decode prep, split combine, fp8 verify query |
   | 7 | [#44984](https://github.com/sgl-project/sglang/pull/44984) | Fused MLA split combine, W_vc and output gate |
   | 8 | [#44985](https://github.com/sgl-project/sglang/pull/44985) | TP8 LL column-parallel latent tail, fused with the next attention read |
   | 9 | [#44986](https://github.com/sgl-project/sglang/pull/44986) | Attention-output TP8 all-reduce fused with the FFN-side read |
   | 10 | [#44987](https://github.com/sgl-project/sglang/pull/44987) | Fused router top-k + MoE route sort, zero-copy MoE output |
   | 11 | [#44988](https://github.com/sgl-project/sglang/pull/44988) | Split MoE front, route-sort bank partials |
   | 12 | [#44989](https://github.com/sgl-project/sglang/pull/44989) | One-launch MLA q_a / kv_a norms and q_b projection |

2. **Three open upstream PRs, picked:**
   - [#43599](https://github.com/sgl-project/sglang/pull/43599): its EAGLE3 aux-capture commit only. EAGLE3 does not
     start for Kimi-K3 without it.
   - [#41867](https://github.com/sgl-project/sglang/pull/41867): the MoE front GEMM goes through aiter.
   - [#41848](https://github.com/sgl-project/sglang/pull/41848): skip the `routed_input` copy.
3. **This directory:** the launch script, the GEMM table and the aiter patch.

Every new path is gated to gfx950. The `python/` tree is byte-identical to the checkpoint tested end to end.

## Environment it was measured with

- **ROCm and torch:** ROCm 7.0, torch 2.9.0a0.
- **Triton:** 3.7.0 (the ROCm 7.2 build from pypi.amd.com). This combination needs a one-line inductor fix:
  - Why: Triton 3.7 dropped `KernelMetadata.cluster_dims`, and the EAGLE draft's `torch.compile`'d top-k reads it
    during graph capture.
  - Fix: in `torch/_inductor/runtime/triton_heuristics.py`, change `if hasattr(binary, "metadata")` to
    `if hasattr(binary, "metadata") and hasattr(binary.metadata, "cluster_dims")`.
- **aiter:** e7d2453f2 plus `aiter_e7d2453f2_k3.patch`. The patch does two things:
  - It sets the world-8 LL all-reduce cutoff to K3's joint `[latent 3584 | shared 7168]` reduction at 4 tokens.
  - It adds `torch.Stream` handling in `torch_to_c_types`.

  The branch's Dockerfile pins aiter 2bb549102, which has not been run with this config.
- **bf16 GEMM table:** `k3_bf16_tuned_gemm_mi350x.csv` is aiter's merged table plus K3 rows re-timed on MI350X at
  1000 W, with weights resident in HBM.
  - Its kernel names are aiter e7d2453f2's.
  - On another aiter version, re-tune it or run with `K3_GEMM_TABLE=`.

## Results on 8x MI350X (gfx950, 1000 W)

**Workload:** agentic replay of lightseekorg/agentic-dataset SWE-smith conversations. They are multi-turn, with
~50K-token cached prefixes, 500 output tokens and greedy decoding. TokenSpeed 7e2f1deb ran on the same machine with
the same workload, for reference.

| Concurrency | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| This branch, tok/s/user | 217.2 | 156.6 | 114.1 | 73.4 | 43.4 |
| TokenSpeed, tok/s/user | 188.1 | 129.5 | 86.1 | 59.7 | 35.5 |

The sweep ran on an earlier integration of the same code, on `main` b67785153a. That build carried two extra opt-in
paths, both off. Its C1 decode cycle was 15.8 ms, against 15.6 ms on this branch.

**Latency:**
- C1 decode cycle at a 50K cached prefix: 15.6 ms.
- Agentic turn TTFT, 50K cached plus 1.9K new tokens: 230-248 ms.
- Uncached 50K-token prefill: 3.27 s.

**Accuracy:**
- GSM8K 200: 0.990.
- GSM8K 400: 0.9825, with an accept length of 3.62.

**Coalescing:** with `COALESCE=1`, measured on an earlier build:
- C8 / C16 throughput went from 69.8 / 42.9 to 73.6 / 49.0 tok/s/user.
- TTFT went from 1.2 / 1.9 s to 1.6 / 3.1 s.

## MI355X

Not measured yet. MI355X is the same gfx950 with 256 CUs, at higher power and clocks, so the branch runs unchanged.
Two things were tuned on MI350X:
- **The GEMM table:** re-tune it, or start with `K3_GEMM_TABLE=`.
- **The kernel cutoffs:** MLA KV split counts, the <= 16-row decode paths and the 64-token absorbed extend.
