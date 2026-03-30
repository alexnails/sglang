# PR #21567 Analysis: CurveZMQ Security for SGLang

## Summary

PR #21567 adds CurveZMQ (Curve25519-based encryption and authentication) to all
ZMQ sockets in SGLang. This analysis evaluates two things:

1. **Whether the PR could stall the `multimodal-gen-test-2-gpu-amd` CI job**
2. **Performance implications of Curve25519 vs HMAC in SGLang's context**

---

## 1. CI Stall Risk Assessment: `multimodal-gen-test-2-gpu-amd`

### Verdict: **YES — High risk of stall/failure**

There are several concrete issues that could cause the multimodal-gen 2-GPU AMD
CI test to stall or fail.

### Issue A: CURVE keypair mismatch across `mp.Process` boundaries (Critical)

The multimodal_gen server launches worker processes via `mp.Process` with
`spawn` start method (set in `diffusion_generator.py:52`). The launch flow is:

```
launch_server() [parent process]
  └─ mp.Process(target=run_scheduler_process, ...) × num_gpus
       └─ Scheduler.__init__()
            └─ get_zmq_socket(ctx, zmq.ROUTER, endpoint, bind=True)  # gpu_id==0 only
```

When CURVE is enabled (the default in this PR when `zmq.has("curve")` is true),
`get_zmq_socket` calls `get_curve_config()`, which **auto-generates a new
Curve25519 keypair** on first call per process.

Because `mp.Process(spawn)` creates fresh Python interpreters, the ROUTER
server socket in the rank-0 scheduler process generates **keypair A**, while
the `SchedulerClient` / `AsyncSchedulerClient` in the HTTP server process
(or the broker coroutine) generates **keypair B**.

The client side calls `connect_with_curve()`, which uses keypair B's public
key as `curve_serverkey` (the shared-key fallback, step 3 in
`apply_curve_client`). This means the client presents `B.public_key` as the
expected server identity, but the server is actually running with
`A.public_key`. **CurveZMQ will silently reject the handshake** — the
connection will appear to succeed at the TCP level but no messages will flow.
The client will hang until its timeout (100 minutes for generation, 2 seconds
for ping).

In practice this means:
- `diffusion_server` fixture calls `_wait_for_ready()` polling `Application startup complete.`
- The server process starts up fine (Scheduler binds with keypair A)
- The HTTP server / broker tries to talk to the scheduler with keypair B
- Messages are silently dropped by CurveZMQ authentication
- The test hangs until the 80-minute CI timeout

### Issue B: `get_zmq_socket` return type changed for multimodal_gen

The old `get_zmq_socket` in `multimodal_gen/runtime/utils/common.py` returned
`tuple[zmq.Socket, str]` (socket + actual endpoint). The PR deletes that
function and switches the import to `sglang.srt.utils.network.get_zmq_socket`,
which returns only `zmq.Socket` when an endpoint is provided.

The multimodal_gen scheduler code on the **PR branch** already adapts to this:

```python
self.receiver = get_zmq_socket(
    self.context, zmq.ROUTER, endpoint, bind=True
)
```

But the **base branch** code was:

```python
self.receiver, actual_endpoint = get_zmq_socket(
    self.context, zmq.ROUTER, endpoint, True
)
```

This is handled correctly in the PR diff — no stall from this specifically.
However, it confirms the PR touches the exact code path exercised by the 2-GPU
CI test.

### Issue C: `config_socket` did not support ROUTER before this PR

The `config_socket` function in `network.py` previously only handled PUSH,
PULL, DEALER, REQ, and REP socket types. It would raise `ValueError` for
ROUTER sockets. This PR adds ROUTER to the supported list:

```python
elif socket_type in [zmq.DEALER, zmq.REQ, zmq.REP, zmq.ROUTER]:
```

This is a prerequisite fix for multimodal_gen, which uses ROUTER sockets.
However, if the old `common.py` `get_zmq_socket` had its own buffer
configuration that included ROUTER and the import switch happened without this
fix, the scheduler would crash on startup.

**On this PR branch, the fix is present**, so this is not a stall risk in
isolation. But it shows the tight coupling — reverting the `config_socket` fix
while keeping the import switch would break 2-GPU tests.

### Issue D: ROCm Docker image may lack libsodium

The `get_curve_config()` function falls back gracefully when
`zmq.has("curve")` is false (returns None, CURVE disabled). However, the ROCm
Docker images used for AMD CI are custom-built and may or may not include
libsodium. If libsodium **is** present (likely in modern pyzmq wheels), CURVE
auto-enables and Issue A becomes active. If it's **not** present, CURVE is
silently disabled and everything works as before — but the security feature
provides no protection on AMD.

Either way, the behavior is **non-deterministic from the PR author's
perspective** and creates a fragile CI dependency on the Docker image contents.

### Issue E: The multimodal_gen `ServerArgs` has no `--no-zmq-curve` flag

The `--no-zmq-curve` and `--zmq-curve-keys-dir` flags are only added to
`sglang.srt.server_args.ServerArgs`, not to
`sglang.multimodal_gen.runtime.server_args.ServerArgs`. The multimodal_gen
server is launched via `sglang serve` which routes through the multimodal_gen
CLI. There is no way for the multimodal_gen test suite to disable CURVE or
supply a shared keypair via CLI arguments.

The only workaround is setting `SGLANG_NO_ZMQ_CURVE=1` as an environment
variable, but the CI workflow does not set this.

---

## 2. Performance Analysis: Curve25519 vs HMAC

### Context

The PR description claims ~25µs overhead per ZMQ roundtrip from CurveZMQ.
Let's evaluate this claim and compare against HMAC as an alternative.

### What CurveZMQ actually does

CurveZMQ implements the CurveCP protocol on top of ZMQ:

1. **Handshake** (one-time per connection): Curve25519 ECDH key agreement +
   crypto_box (XSalsa20-Poly1305) for the handshake messages. This involves:
   - 2 Curve25519 scalar multiplications (~100µs each on modern x86)
   - 4 crypto_box operations (~2µs each)
   - Total: **~210µs per new connection**

2. **Per-message** (ongoing): Each ZMQ frame is encrypted with
   crypto_secretbox (XSalsa20-Poly1305):
   - XSalsa20 stream cipher: ~3 cycles/byte
   - Poly1305 MAC: ~1.3 cycles/byte
   - Total: **~4.3 cycles/byte**, or **~1.3µs per 1KB message** at 3GHz

### What HMAC would provide

HMAC (e.g., HMAC-SHA256) provides only **authentication**, not encryption:

1. **Per-message**: HMAC-SHA256 computation:
   - SHA-256: ~11 cycles/byte (without hardware acceleration)
   - With SHA-NI (Intel/AMD): ~3.5 cycles/byte
   - Total: **~1.1µs per 1KB** (with SHA-NI), **~3.5µs per 1KB** (without)

2. **No handshake overhead** (pre-shared key model)
3. **No confidentiality** — messages are readable by any network observer

### Comparison for SGLang's workload

| Metric | CurveZMQ (Curve25519 + XSalsa20-Poly1305) | HMAC-SHA256 |
|---|---|---|
| Handshake cost | ~210µs (one-time per connection) | None (PSK) |
| Per-message cost (1KB) | ~1.3µs | ~1.1µs (SHA-NI) / ~3.5µs (no SHA-NI) |
| Per-message cost (100KB) | ~130µs | ~110µs (SHA-NI) / ~350µs (no SHA-NI) |
| Provides encryption | Yes | No |
| Provides authentication | Yes (mutual) | Yes (symmetric) |
| Key management | Per-node keypairs, public key exchange | Pre-shared secret |
| Forward secrecy | Yes (per-connection ephemeral keys) | No |
| AMD ROCm CPU support | Depends on libsodium build | Universally available |

### SGLang's ZMQ message sizes

SGLang uses ZMQ for:

1. **Control plane** (scheduler signals, health checks, tokenizer→scheduler
   metadata): Typically **< 1KB**. Overhead: ~1-2µs with either approach.
   Negligible vs GPU kernel times (milliseconds).

2. **Token IDs / batch metadata** (tokenizer→scheduler PUSH/PULL): Typically
   **1-100KB** depending on batch size. Overhead: 1-130µs with CurveZMQ.
   Still negligible — a single GPU attention kernel takes 0.5-5ms.

3. **Disaggregation aux data** (prefill→decode metadata): **1-10KB**. Same
   analysis as above.

4. **KV cache data**: Transferred via **RDMA/NCCL**, not ZMQ. Zero CURVE
   overhead.

### Performance verdict

**CurveZMQ's overhead is negligible for SGLang's workload.** The key
observations:

- The hot path (GPU compute) is 3-4 orders of magnitude slower than ZMQ crypto
- ZMQ is used for control plane and small metadata, not bulk data transfer
- KV cache (the only large data) goes through NCCL/RDMA, bypassing ZMQ entirely
- The ~25µs per roundtrip claim in the PR is reasonable for small messages
- Even for larger messages (100KB), 130µs of crypto overhead is dwarfed by GPU
  kernel execution time

**HMAC would not meaningfully outperform CurveZMQ** for SGLang because:

- Per-message cost difference is minimal (~0.2µs per KB with SHA-NI, and
  CurveZMQ is actually *faster* without SHA-NI hardware)
- HMAC loses encryption (confidentiality), forward secrecy, and per-node
  identity verification
- The one-time 210µs handshake cost is amortized over the connection lifetime
  (connections are long-lived in SGLang)
- HMAC would require a separate key distribution mechanism (the PR's
  public-key bootstrap is more elegant than PSK distribution)

### Where Curve25519 is actually slower than HMAC

The only scenario where HMAC wins meaningfully is **extremely high connection
churn** — if SGLang were creating and tearing down thousands of ZMQ
connections per second, the 210µs handshake would add up. But SGLang
connections are long-lived (created at startup, persist for the server
lifetime), so this is irrelevant.

On **AMD MI325X specifically** (the CI runner), the CPUs are EPYC processors
with SHA-NI support, so HMAC-SHA256 would be hardware-accelerated. However,
libsodium's XSalsa20 is also extremely well-optimized with SIMD on x86-64,
and the per-byte cost difference is marginal.

---

## 3. Recommendations

### For CI stability (Critical)

1. **The multimodal_gen process topology needs a shared-key mechanism.** The
   parent process should generate the CURVE keypair once and propagate it to
   child processes via environment variables before spawning them. Currently,
   each `mp.Process(spawn)` child generates its own keypair independently.

2. **Set `SGLANG_NO_ZMQ_CURVE=1` in the CI workflow** as a short-term fix to
   unblock the AMD CI tests while the key-sharing issue is resolved.

3. **Add `--no-zmq-curve` to multimodal_gen's ServerArgs** for parity with
   the main srt ServerArgs.

### For the PR itself

4. The `scheduler_endpoint` in multimodal_gen is always `tcp://...`, so CURVE
   will always be applied when available. The broker, scheduler clients, and
   ping sockets all need to share the same keypair as the scheduler's ROUTER
   socket.

5. Consider adding a test that exercises the multimodal_gen ZMQ path with
   CURVE enabled (even a unit test that creates a ROUTER + REQ pair in
   separate threads with the same CurveConfig).

### For performance

6. **Curve25519 is the right choice over HMAC.** The performance difference is
   negligible for SGLang's message sizes and connection patterns. Curve25519
   provides strictly better security properties (encryption + authentication +
   forward secrecy) with comparable or better throughput on the actual hardware.
