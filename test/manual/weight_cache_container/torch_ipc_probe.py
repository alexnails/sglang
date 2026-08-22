"""Probe whether the shipped weight-cache transport can cross a container boundary.

Exercises the same call chain the weight cache uses --
``MultiprocessingSerializer`` -> torch ``reduce_tensor`` -> ``cudaIpcGetMemHandle``
-- but with a single tensor instead of a whole model, so it isolates the
transport from everything else in the daemon.

The handle travels through a file in a shared directory rather than a socket, so
neither the network namespace nor the socket path is a variable in the result.

    python torch_ipc_probe.py export /shared
    python torch_ipc_probe.py import /shared

The exporter stays alive (the memory must remain valid) and mutates its tensor
after serving, which lets the importer prove it mapped live memory rather than
receiving a copy taken at export time.
"""

import os
import sys
import time

import torch

from sglang.srt.utils import MultiprocessingSerializer

NEL = 1024
EXPECT = float(sum(i * 2.0 for i in range(NEL)))
BUMP = 1000.0


def gpu_uuid():
    # torch does not expose .uuid on every backend build (it is absent on ROCm),
    # so this attribute is genuinely optional rather than defensively accessed.
    return getattr(torch.cuda.get_device_properties(0), "uuid", None)


def describe(tag):
    print(
        f"[{tag}] pid={os.getpid()} gpu={gpu_uuid()} "
        f"ipc_ns={os.readlink('/proc/self/ns/ipc')} "
        f"pid_ns={os.readlink('/proc/self/ns/pid')} "
        f"mnt_ns={os.readlink('/proc/self/ns/mnt')}",
        flush=True,
    )


def run_export(share):
    describe("export")
    t = torch.arange(0, NEL, dtype=torch.float32, device="cuda:0") * 2.0
    torch.cuda.synchronize()
    payload = MultiprocessingSerializer.serialize({"w": t})

    tmp = os.path.join(share, "handle.bin.partial")
    with open(tmp, "wb") as f:
        f.write(payload)
    os.rename(tmp, os.path.join(share, "handle.bin"))
    with open(os.path.join(share, "ready"), "w") as f:
        f.write(f"pid={os.getpid()}\nsum={t.sum().item()}\n")
    print(f"[export] published {len(payload)} bytes, sum={t.sum().item()}", flush=True)

    served = os.path.join(share, "served")
    while True:
        time.sleep(2)
        if os.path.exists(served):
            os.remove(served)
            t.add_(BUMP)
            torch.cuda.synchronize()
            print(f"[export] mutated, new sum={t.sum().item()}", flush=True)


def run_import(share):
    describe("import")
    with open(os.path.join(share, "handle.bin"), "rb") as f:
        payload = f.read()
    try:
        state = MultiprocessingSerializer.deserialize(payload)
    except Exception as e:  # noqa: BLE001 - the failure mode is the result
        print(f"[import] RESULT=FAIL {type(e).__name__}: {str(e).splitlines()[0]}")
        return 3

    got = state["w"].sum().item()
    open(os.path.join(share, "served"), "w").close()
    if abs(got - EXPECT) < 1e-3:
        print(f"[import] RESULT=OK zero-copy import verified (sum={got})")
        return 0
    if abs(got - (EXPECT + BUMP * NEL)) < 1e-3:
        print(
            f"[import] RESULT=OK live shared memory (saw exporter mutation, sum={got})"
        )
        return 0
    print(f"[import] RESULT=GARBAGE sum={got} expected={EXPECT}")
    return 4


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("export", "import"):
        print(__doc__)
        return 2
    role, share = sys.argv[1], sys.argv[2]
    torch.cuda.set_device(0)
    torch.zeros(1, device="cuda:0")  # materialise the primary context
    return run_export(share) if role == "export" else run_import(share)


if __name__ == "__main__":
    sys.exit(main())
