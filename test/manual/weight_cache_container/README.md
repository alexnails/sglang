# Weight cache across a container boundary

Manual probes for the question the [fast-recovery roadmap](https://github.com/sgl-project/sglang/issues/33522)
leaves open under *"systemd unit + k8s DaemonSet recipes; document container
requirements for cross-container IPC"*: can the weight-cache daemon serve an
engine that lives in a different container, so that the daemon survives the
engine's restart?

These are host-level probes, not CI tests. They need docker and a GPU, so they
live under `test/manual/`.

## Files

| File | What it does |
| --- | --- |
| `torch_ipc_probe.py` | Exercises the shipped transport (`MultiprocessingSerializer` → `cudaIpcGetMemHandle`) on a single tensor. |
| `vmm_fd_probe.py` | Working prototype of the `vmm_fd` transport: `cuMemCreate` → fd over `SCM_RIGHTS` → `cuMemImportFromShareableHandle`. |
| `layered_diag.py` | Splits a failure into the driver call vs. torch's `/dev/shm` ref-count file, so a bare CUDA error becomes a diagnosis. |
| `transport_matrix.sh` | Runs both probes across a grid of docker isolation settings. |
| `two_container_e2e.sh` | The real daemon and real server in separate containers, plus an engine restart across the boundary. |

## What these measured

Measured 2026-08-21 on `lmsysorg/sglang:latest` (v0.5.17): bare-metal H100 with
docker 28.5.2, and a managed-Kubernetes H200 pod with two containers.

### The shipped transport needs two namespaces

Each is necessary, together they are sufficient, and they fail with *different*
driver errors — which is what lets you tell them apart:

| Isolation | IPC ns | PID ns | Result |
| --- | --- | --- | --- |
| one container (control) | shared | shared | zero-copy verified |
| docker defaults | private | private | `invalid device context` |
| `--ipc=host` only | shared | private | `invalid device context` |
| `--pid=host` only | private | shared | `mapping of buffer object failed` |
| `--ipc=host --pid=host` | shared | shared | **zero-copy verified** |
| `--pid=container:<ref>` | shared | shared (not host) | **zero-copy verified** |

Neither namespace has to be the *host* one, which matters because a shared
private PID namespace is what `shareProcessNamespace: true` gives you inside a
pod. Asymmetric device injection is **not** a factor: `{0,1}` against `{0}`
passes in both directions, as does by-UUID against by-index injection.

With those two namespaces, `two_container_e2e.sh` works end to end: daemon
container ready in 25.2 s, engine container attaches 228 handles in **0.04 s**
with `mem usage=0.00 GB`, engine container destroyed, daemon stays resident, and
the replacement engine re-attaches in **0.04 s** with identical output.

### One platform refuses anyway

A managed-Kubernetes pod failed with `cudaErrorMapBufferObjectFailed` in both
directions while sharing the host IPC namespace, a shared PID namespace, the
same physical GPU by UUID, and the same uid — i.e. meeting the contract above.
Same-container controls passed there.

`layered_diag.py` narrows it. `/dev/shm` is genuinely *not* shared between those
containers (a 64 GiB tmpfs on one side, the host's 1 TiB on the other, neither
seeing the other's marker), and torch keeps its IPC ref-count file there — but
bridging that file in via `/proc/<pid>/root/dev/shm` (`BRIDGE=1`) produces a
byte-identical error. The driver call fails first, so `/dev/shm` is a separate
problem rather than the cause. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
changes the share format (a 38-byte handle instead of 66) and fails identically.

### vmm_fd needs no namespace sharing at all

| Environment | Isolation | torch_ipc | vmm_fd |
| --- | --- | --- | --- |
| managed k8s pod, 2 containers | host IPC + shared PID | **fails** | **verified** |
| bare metal, 2 containers | no namespace sharing | **fails** | **verified** |
| bare metal, 2 containers | `--ipc=host --pid=host` | verified | verified |

An fd received over `SCM_RIGHTS` is duplicated into the receiver by the kernel,
so nothing has to resolve the exporting process the way `cuIpcOpenMemHandle`
does. Private IPC namespace, private PID namespace, nothing shared but the bind
mount carrying the socket — and it still imports and reads back correctly,
including on the platform where the shipped transport does not work at all.

## Why this is a probe and not a patch

`vmm_fd_probe.py` proves the primitive, not a drop-in backend.
[#33279](https://github.com/sgl-project/sglang/pull/33279) lands the transport
abstraction with every `VmmFdTransportBackend` method stubbed as `pass`
(including `can_export_state`, so the daemon always selects torch IPC — merging
it changes no behaviour). Writing the backend is more than porting this ctypes
code: an ordinary torch tensor lives in a `cudaMalloc`'d caching-allocator block
and **cannot** be exported to a shareable fd at all, so the daemon has to own
VMM-backed allocations for the weights. The stub's `_export_fd_for_tensor(tensor)`
signature implies otherwise.

One consequence worth pairing with that work: `vmm_fd` exists to drop the shared
PID namespace, and `IpcModelLoader._start_daemon_liveness_watchdog` is
`os.kill(daemon_pid, 0)`. Without a shared PID namespace that number is
meaningless in the engine's namespace — either it matches nothing and the
watchdog SIGKILLs a healthy engine seconds after startup, or it matches an
unrelated process and a dead daemon goes undetected while the engine reads freed
GPU memory. A namespace-independent liveness check (socket EOF, or `SO_PEERCRED`
plus a generation counter) is a prerequisite for `vmm_fd`, not a follow-up.

## Other measured behaviour worth knowing

- A dead daemon SIGKILLs every attached engine in **7.09 s** (5 s watchdog poll);
  killing the daemon's launcher kills the real daemon in **0.51 s** via
  `PR_SET_PDEATHSIG`.
- Every daemon shutdown — SIGTERM included, because PDEATHSIG is SIGKILL and the
  cleanup handler never runs — leaves a stale socket, and a stale socket makes
  the next client-mode start a hard `ConnectionRefused` failure. A *missing*
  socket, by contrast, is a silent disk fallback, and no API field distinguishes
  the two.
- The `CacheConfig` fingerprint hashes the literal `model_path` string, so
  `Qwen/Qwen3-8B` and its resolved snapshot path are a hard mismatch.
- `--weight-cache-mode daemon` measured *slower* than `off` (42.7 s vs 31.1 s on
  Qwen3-8B), as its own docstring warns.
