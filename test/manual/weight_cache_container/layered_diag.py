"""Split a cross-container CUDA IPC failure into the layers that can cause it.

A torch CUDA IPC import needs two independent things to work:

  layer 1  ``cudaIpcOpenMemHandle`` -- the driver call, which requires the
           exporting process to be reachable through a shared IPC and PID
           namespace.
  layer 2  a per-block reference-count file that torch opens by POSIX shm
           name, i.e. out of the importer's own ``/dev/shm``.

Containers routinely share the IPC namespace while having separate ``/dev/shm``
mounts, so layer 2 can fail on its own. This probe reports which layer is at
fault instead of leaving a bare CUDA error.

Handles travel over localhost TCP because the point of the probe is platforms
with no shared writable mount between the containers (the network namespace is
shared far more often than the filesystem).

    python layered_diag.py export 47700
    python layered_diag.py import 47700

With ``BRIDGE=1`` the importer also tries to make layer 2 succeed on its own:
when the PID namespace is shared, the exporter's ``/dev/shm`` is reachable at
``/proc/<pid>/root/dev/shm``, so the ref-count file can be symlinked into the
importer's own ``/dev/shm``. If the error is unchanged with the file bridged in,
layer 1 is failing first and ``/dev/shm`` is a separate problem.
"""

import base64
import json
import os
import socket
import struct
import sys
import time

import torch
from torch.multiprocessing.reductions import reduce_tensor

from sglang.srt.utils import MultiprocessingSerializer

NEL = 1024


def namespaces():
    return {k: os.readlink(f"/proc/self/ns/{k}") for k in ("ipc", "pid", "mnt")}


def gpu_uuid():
    # torch does not expose .uuid on every backend build (it is absent on ROCm),
    # so this attribute is genuinely optional rather than defensively accessed.
    return getattr(torch.cuda.get_device_properties(0), "uuid", None)


def run_export(port):
    t = torch.arange(0, NEL, dtype=torch.float32, device="cuda:0") * 2.0
    torch.cuda.synchronize()
    _, a = reduce_tensor(t)
    handle, storage_size_bytes, ref_counter_handle = bytes(a[7]), a[8], a[11]
    if isinstance(ref_counter_handle, bytes):
        ref_counter_handle = ref_counter_handle.decode()

    meta = {
        "pid": os.getpid(),
        "uuid": str(gpu_uuid()),
        "handle_len": len(handle),
        "handle_prefix": handle[:2].hex(),
        "storage_size_bytes": storage_size_bytes,
        "ref_counter_handle": ref_counter_handle,
        "sum": t.sum().item(),
        "payload_b64": base64.b64encode(
            MultiprocessingSerializer.serialize({"w": t})
        ).decode(),
        **namespaces(),
    }
    blob = json.dumps(meta).encode()
    print(
        json.dumps({k: v for k, v in meta.items() if k != "payload_b64"}, indent=1),
        flush=True,
    )

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(8)
    print(f"[export] serving on 127.0.0.1:{port}", flush=True)
    while True:
        conn, _ = srv.accept()
        conn.sendall(struct.pack("!I", len(blob)) + blob)
        conn.close()
        time.sleep(0.2)


def run_import(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", port))
    (n,) = struct.unpack("!I", s.recv(4))
    buf = b""
    while len(buf) < n:
        chunk = s.recv(min(65536, n - len(buf)))
        if not chunk:
            break
        buf += chunk
    s.close()
    meta = json.loads(buf)

    mine = namespaces()
    print(
        f"[import] same_gpu={meta['uuid'] == str(gpu_uuid())} "
        f"ipc_ns_match={meta['ipc'] == mine['ipc']} "
        f"pid_ns_match={meta['pid'] == mine['pid']} "
        f"mnt_ns_match={meta['mnt'] == mine['mnt']}"
    )
    print(f"[import] handle_len={meta['handle_len']} prefix={meta['handle_prefix']}")

    shm_name = meta["ref_counter_handle"].lstrip("/")
    local = f"/dev/shm/{shm_name}"
    peer = f"/proc/{meta['pid']}/root/dev/shm/{shm_name}"
    print(
        f"[import] LAYER2 ref_counter={shm_name!r} "
        f"in_my_shm={os.path.exists(local)} via_proc={os.path.exists(peer)}"
    )

    linked = False
    if (
        os.environ.get("BRIDGE") == "1"
        and not os.path.exists(local)
        and os.path.exists(peer)
    ):
        try:
            os.symlink(peer, local)
            linked = True
            print(f"[import] LAYER2 bridged {local} -> {peer}")
        except OSError as e:
            print(f"[import] LAYER2 bridge failed: {e}")

    try:
        state = MultiprocessingSerializer.deserialize(
            base64.b64decode(meta["payload_b64"])
        )
        got = state["w"].sum().item()
        ok = abs(got - meta["sum"]) < 1e-3
        print(f"[import] RESULT=OK sum={got} match={ok}")
        rc = 0
    except Exception as e:  # noqa: BLE001 - the failure mode is the result
        print(f"[import] RESULT=FAIL {type(e).__name__}: {str(e).splitlines()[0]}")
        rc = 4
    finally:
        if linked and os.path.islink(local):
            os.unlink(local)
    return rc


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("export", "import"):
        print(__doc__)
        return 2
    torch.cuda.set_device(0)
    torch.zeros(1, device="cuda:0")
    port = int(sys.argv[2])
    return run_export(port) if sys.argv[1] == "export" else run_import(port)


if __name__ == "__main__":
    sys.exit(main())
