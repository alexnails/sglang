# Kimi-K3 on AMD gfx950 (MI350X / MI355X): launch configs

Both configs below run on one node of 8 GPUs at TP8, with aiter MLA and an FP8 KV cache.

- **Config A** is the best you can run on upstream `main` today.
- **Config B** is this branch: the best measured config.

## A. Upstream `main` (no unmerged PRs)

**Why DSPARK:** EAGLE3 does not start for K3 on `main`; it needs #43599. That leaves DSPARK, the cookbook's default
K3 speculator.

**Not measured on our pod:** the DSpark draft isn't downloaded there. Upstream's MI350X rows in
`docs/src/snippets/configs/moonshotai/kimi-k3-benchmarks.jsx` use random 8K/1K prompts with simulated acceptance.

```bash
SGLANG_USE_AITER=1 SGLANG_AITER_K3_OPT=1 AITER_FLYDSL_FORCE=1 AITER_SITUV2_A8W4=1 \
SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION=1 \
python3 -m sglang.launch_server --model-path moonshotai/Kimi-K3 --trust-remote-code --tp-size 8 \
  --prefill-attention-backend aiter --decode-attention-backend aiter \
  --kv-cache-dtype fp8_e4m3 --dtype bfloat16 \
  --speculative-algorithm DSPARK --speculative-draft-model-path RadixArk/Kimi-K3-DSpark \
  --speculative-dspark-block-size 7 --enable-linear-replayssm-spec \
  --chunked-prefill-size 8192 --context-length 80000 --max-running-requests 16 \
  --cuda-graph-max-bs-decode 16 --cuda-graph-bs-decode 1 2 4 8 16 \
  --mem-fraction-static 0.85 --model-loader-extra-config '{"enable_multithread_load": true}'
```

**Same serving setup as B:** chunked prefill, context length, running requests and graph sizes match B, so the two
compare directly.

**Differences from the cookbook's MI35x cell:**
- The cookbook adds `--dcp-size 8 --dcp-comm-backend a2a` and decode graphs up to bs 256.
- On MI350X (Oct 14 tree), the a2a DCP exchange broke HIP graph capture.
- DCP8 decode was also 11-13% slower per step at 1-4 requests.
- So A leaves DCP out for C1-C16 latency. Add it back for KV capacity at high concurrency.

## B. This branch (best measured)

**What's in it:**
- Base: `main` 4eb606b2ae (2026-10-17).
- The twelve Kimi-K3 gfx950 PRs #44978-#44989.
- Picks: #43599's EAGLE3 aux capture, #41867 and #41848.
- The tree is byte-identical to the checkpoint tested end to end.

```bash
PYTHONPATH=$PWD/python benchmark/kimi_k3_amd/launch_k3_eagle3.sh
```

### Environment it was measured with

- **ROCm and torch:** ROCm 7.0, torch 2.9.0a0.
- **Triton:** 3.7.0 (the ROCm 7.2 build from pypi.amd.com).
  - This combination needs an inductor guard: Triton 3.7 dropped `KernelMetadata.cluster_dims`, and the EAGLE draft's
    `torch.compile`'d top-k reads it during graph capture.
  - Fix: in `torch/_inductor/runtime/triton_heuristics.py`, change `if hasattr(binary, "metadata")` to
    `if hasattr(binary, "metadata") and hasattr(binary.metadata, "cluster_dims")`.
- **aiter:** e7d2453f2 plus `aiter_e7d2453f2_k3.patch`.
  - The patch moves the world-8 LL all-reduce cutoff to K3's joint `[latent 3584 | shared 7168]` reduction at
    4 tokens.
  - It also adds `torch.Stream` handling in `torch_to_c_types`.
  - This branch's Dockerfile pins aiter 2bb549102, which has not been run with this config.
- **bf16 GEMM table:** `k3_bf16_tuned_gemm_mi350x.csv` (`AITER_CONFIG_GEMM_BF16`) is aiter's merged table plus K3
  rows re-timed with HBM-resident weights on MI350X at 1000 W.
  - Its kernel names are aiter e7d2453f2's.
  - On another aiter or on MI355X, re-tune it or run with `K3_GEMM_TABLE=`.

### Measured on 8x MI350X (gfx950, 1000 W)

**Workload:** agentic replay of lightseekorg/agentic-dataset SWE-smith conversations: multi-turn, ~50K-token cached
prefixes, 500 output tokens, greedy decoding. TokenSpeed 7e2f1deb ran on the same pod with the same workload.

**Which build the sweep used:** the integration of the same code on `main` b67785153a. That build also carried two
opt-in paths, both off. Its C1 decode cycle was 15.8 ms, against 15.6 ms on this tree.

| Concurrency | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| This config, tok/s/user | 217.2 | 156.6 | 114.1 | 73.4 | 43.4 |
| TokenSpeed, tok/s/user | 188.1 | 129.5 | 86.1 | 59.7 | 35.5 |

**Latency:**
- C1 decode cycle at a 50K cached prefix: 15.6 ms on this tree.
- Agentic turn TTFT: 230-248 ms.
- Uncached 50K-token prefill: 3.27 s.

**Accuracy and coalescing:**
- GSM8K 200: 0.990 on this tree.
- GSM8K 400: 0.9825 with an accept length of 3.62.
- `COALESCE=1`, measured on an earlier build: C8 / C16 went from 69.8 / 42.9 to 73.6 / 49.0 tok/s/user, with TTFT
  going from 1.2 / 1.9 s to 1.6 / 3.1 s.

**MI355X:** not run yet. It is the same gfx950 with 256 CUs at higher power and clocks. Besides the GEMM table, the
kernel cutoffs were measured on MI350X: MLA KV split counts, the <= 16-row decode paths and the 64-token absorbed
extend.
