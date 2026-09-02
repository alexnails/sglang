# Weight cache findings register

Everything measured while working the "can the daemon serve an engine in another
pod" question from [#33522](https://github.com/sgl-project/sglang/issues/33522).
Status is verified against main at `1109e443` (0.5.18) unless a row says
otherwise. Reproduce with the probes in this directory.

## Blockers for separate pods

| # | Finding | Status |
| --- | --- | --- |
| 1 | `VmmFdTransportBackend` is a placeholder: `can_export_state` returns `False`, every other entry point raises `NotImplementedError`. #33279 merged but changed no behaviour, and no PR implements the backend. | open |
| 2 | The daemon cannot export torch caching-allocator tensors to a shareable fd; it must own VMM reservations. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is not a shortcut — it changes the share format (38-byte handle) and still fails to import cross-container. | open, design work |
| 3 | Liveness is `os.kill(daemon_pid, 0)` -> self-SIGKILL. Meaningless without a shared PID namespace, which is exactly what `vmm_fd` drops; PID reuse is a hazard even today. #36156 is open and still PID-based. | open, prerequisite for 1 |
| 4 | The shipped transport requires a shared IPC **and** PID namespace. Each alone fails, with different driver errors (`invalid device context` vs `mapping of buffer object failed`); neither has to be the host namespace. | inherent |

## Failure semantics

| # | Finding | Status |
| --- | --- | --- |
| 5 | A dead daemon SIGKILLs every attached engine in 7.09 s. A DaemonSet rollout is therefore a node-wide outage. | open (Phase 6) |
| 6 | `PR_SET_PDEATHSIG` is SIGKILL, so the launcher dying kills the real daemon in 0.51 s and its cleanup handler never runs. | inherent |
| 7 | Every daemon shutdown, SIGTERM included, leaves a stale socket and `.ready`. | open |
| 8 | A stale socket makes the next client-mode start fail hard with `ConnectionRefused` — no fallback, so every engine on the node crash-loops. Nothing in the engine path calls `cleanup_stale_daemon_files`. | open |
| 9 | A *missing* socket is a silent disk fallback, and no API field reports how weights were actually loaded, so a 400 s disk load looks identical to a 0.09 s attach. The transport name is logged but not exposed. | open |

## Correctness and coverage

| # | Finding | Status |
| --- | --- | --- |
| 10 | MoE expert weights arrive transposed (`w13_weight` `[128,720,2880]` vs model `[128,2880,720]`); gpt-oss-120b-bf16 fails the shape check. | fixed on main, broken in the published image |
| 11 | `lmsysorg/sglang:latest` (rev `71de97b`) has no `transport.py`, no UUID keying and no KV-sizing fix, so anyone evaluating from it hits bugs fixed weeks ago. | open, release hygiene |
| 12 | MXFP4 is not on the IPC allowlist, so the cookbook's default gpt-oss command cannot use the weight cache at all. | open |
| 13 | The allowlist is only `""` and block-wise fp8 — per-tensor fp8, awq, gptq and nvfp4 are all rejected. | open (#32398) |
| 14 | `_assert_ipc_compatible_allocator` rejects `expandable_segments` claiming the export "would fail mid-way". In test the export succeeded and produced a valid handle; the cross-container *import* is what fails, and it fails for the legacy allocator too. The guard may be right for other reasons, but the stated reason did not reproduce. | open, comment bug |
| 15 | The `CacheConfig` fingerprint hashes the literal `model_path`, so a hub id and its resolved snapshot path are a hard mismatch. | inherent, undocumented |

## Fixed during this investigation

| # | Finding | Fixed by |
| --- | --- | --- |
| 16 | KV over-sizing — the daemon's resident weights were invisible to sizing, halving headroom at 62 GB. Verified correct at 120B. | #34053, #36583 |
| 17 | Socket path hardcoded to `/tmp/sglang_weight_cache_rank{N}.sock`. | #36299 |
| 18 | Rank-keyed sockets: one instance per node, and cross-container ordinal mismatch. Both sides now derive the path from the device UUID. | #36101 |
| 19 | Socket node not validated on a world-writable `/tmp`. The client now rejects a symlink, a plain file, or another user's socket. | #36101 |

## Not sglang's to fix

| # | Finding |
| --- | --- |
| 20 | The device plugin will not allocate one GPU to two pods. Needs the `NVIDIA_VISIBLE_DEVICES` escape hatch (often disabled by policy) or time-slicing / MPS. |
| 21 | Separate pods cannot share an `emptyDir`, so the socket needs a `hostPath`; the socket is mode 0600, so both pods run as the same uid. |
| 22 | ~~One managed-Kubernetes platform fails the legacy transport despite meeting the namespace contract.~~ **Root-caused: an `emptyDir` mounted at `/dev/shm`.** See below. |

## Two pods, verified end to end

Run on k3s v1.36.4 / 8xB300 with `two-pods.yaml`. The daemon pod stayed up
untouched — same pid (313668) before and after — while the engine pod was
deleted outright and recreated:

| Step | Result |
| --- | --- |
| daemon pod ready | model loaded in 10.52 s, 228 tensors exported, 2006 MiB resident |
| engine pod attaches | 228 handles, 0.016 s map, 0.04 s weight load, `mem usage=0.00 GB` |
| generation | correct |
| **engine pod deleted** | daemon pod still Running, GPU memory unchanged |
| replacement engine pod | re-attached in 0.016 s, serving 105.7 s after the delete |
| generation after restart | correct, new pod IP |

The 105.7 s is container start plus CUDA-graph capture; the weight load inside
it is 0.03 s.

**The trap that made this look impossible: an `emptyDir` at `/dev/shm`.** torch's
CUDA IPC keeps a per-block reference-count file there. `hostIPC: true` gives a
pod the host's `/dev/shm`, but mounting an `emptyDir` over it — the standard fix
for PyTorch's "insufficient shared memory" dataloader error, and therefore
present in most GPU pod specs — silently substitutes a per-pod tmpfs, and the
import dies with `cudaErrorMapBufferObjectFailed`. Two otherwise identical
manifests, measured both ways: with the mount the engine pod fails, without it
the same pod attaches in 0.016 s.

This also explains the managed-Kubernetes failure recorded above as unexplained.
An earlier attempt to rule `/dev/shm` out by symlinking the ref-count file in
through `/proc/<pid>/root` gave a false negative.

## Process and docs

| # | Finding |
| --- | --- |
| 23 | #33522 has 0 of 39 boxes checked despite roughly six items having merged; #36042 was closed unmerged, likely a duplicate of #36101. |
| 24 | `--weight-cache-mode daemon` is slower than `off` (42.7 s vs 31.1 s) — stated in a docstring, nowhere user-facing. |
| 25 | The daemon needs `--ep-size` matching the engine's `--ep`, which no document mentions. |
| 26 | DP>1, speculative decoding, and `release`/`resume_memory_occupation` plus weight updates are all rejected while the cache is active. |
