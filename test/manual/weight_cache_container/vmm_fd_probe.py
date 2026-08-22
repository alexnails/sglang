"""Prototype of the vmm_fd weight-cache transport, as a standalone probe.

#27310 §4.2 and #33279 propose a second transport for the weight cache: allocate
with ``cuMemCreate`` requesting a POSIX file-descriptor handle type, export the
fd with ``cuMemExportToShareableHandle``, pass it over the daemon's Unix socket
with ``SCM_RIGHTS``, and reserve plus map it on import. #33279 lands the
abstraction with every ``VmmFdTransportBackend`` method stubbed as ``pass``; this
file is a working end-to-end version of the primitive it needs, driven straight
at ``libcuda`` via ctypes so it can be run without touching the daemon.

An fd received over SCM_RIGHTS is duplicated into the receiver by the kernel, so
unlike ``cuIpcOpenMemHandle`` nothing has to resolve the exporting process. That
is why this path does not need a shared IPC or PID namespace -- only a shared
filesystem path for the socket itself.

    python vmm_fd_probe.py export /shared/vmm.sock
    python vmm_fd_probe.py import /shared/vmm.sock

Caveat, and the reason this is a probe rather than a patch: an ordinary torch
tensor lives in a cudaMalloc'd caching-allocator block and cannot be exported
to a shareable fd at all. A real backend needs the daemon to own VMM-backed
allocations for the weights.
"""

import ctypes
import os
import socket
import sys

import torch  # only to create the primary CUDA context

CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0

NEL = 1024
PAYLOAD = [i * 2.0 for i in range(NEL)]
EXPECT = float(sum(PAYLOAD))


class CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _AllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", CUmemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _AllocFlags),
    ]


class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", CUmemLocation), ("flags", ctypes.c_int)]


cuda = ctypes.CDLL("libcuda.so.1")
cuda.cuMemCreate.argtypes = [
    ctypes.POINTER(ctypes.c_ulonglong),
    ctypes.c_size_t,
    ctypes.POINTER(CUmemAllocationProp),
    ctypes.c_ulonglong,
]
cuda.cuMemcpyHtoD_v2.argtypes = [ctypes.c_ulonglong, ctypes.c_void_p, ctypes.c_size_t]
cuda.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_size_t]


def check(rc, what):
    if rc != 0:
        name = ctypes.c_char_p()
        cuda.cuGetErrorName(ctypes.c_int(rc), ctypes.byref(name))
        label = name.value.decode() if name.value else "?"
        raise RuntimeError(f"{what} -> rc={rc} ({label})")


def make_prop(dev=0):
    p = CUmemAllocationProp()
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED
    p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    p.location.id = dev
    return p


def rounded_size(prop, nbytes):
    g = ctypes.c_size_t(0)
    check(
        cuda.cuMemGetAllocationGranularity(
            ctypes.byref(g),
            ctypes.byref(prop),
            ctypes.c_int(CU_MEM_ALLOC_GRANULARITY_MINIMUM),
        ),
        "cuMemGetAllocationGranularity",
    )
    gran = g.value
    return ((nbytes + gran - 1) // gran) * gran


def reserve_and_map(handle, size, dev=0):
    ptr = ctypes.c_ulonglong(0)
    check(
        cuda.cuMemAddressReserve(
            ctypes.byref(ptr),
            ctypes.c_size_t(size),
            ctypes.c_size_t(0),
            ctypes.c_ulonglong(0),
            ctypes.c_ulonglong(0),
        ),
        "cuMemAddressReserve",
    )
    check(
        cuda.cuMemMap(
            ctypes.c_ulonglong(ptr.value),
            ctypes.c_size_t(size),
            ctypes.c_size_t(0),
            ctypes.c_ulonglong(handle),
            ctypes.c_ulonglong(0),
        ),
        "cuMemMap",
    )
    desc = CUmemAccessDesc()
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = dev
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    check(
        cuda.cuMemSetAccess(
            ctypes.c_ulonglong(ptr.value),
            ctypes.c_size_t(size),
            ctypes.byref(desc),
            ctypes.c_size_t(1),
        ),
        "cuMemSetAccess",
    )
    return ptr.value


def run_export(sock_path):
    prop = make_prop()
    size = rounded_size(prop, NEL * 4)
    handle = ctypes.c_ulonglong(0)
    check(
        cuda.cuMemCreate(
            ctypes.byref(handle),
            ctypes.c_size_t(size),
            ctypes.byref(prop),
            ctypes.c_ulonglong(0),
        ),
        "cuMemCreate",
    )
    dptr = reserve_and_map(handle.value, size)
    host = (ctypes.c_float * NEL)(*PAYLOAD)
    check(
        cuda.cuMemcpyHtoD_v2(
            ctypes.c_ulonglong(dptr), ctypes.byref(host), ctypes.c_size_t(NEL * 4)
        ),
        "cuMemcpyHtoD",
    )

    fd = ctypes.c_int(0)
    check(
        cuda.cuMemExportToShareableHandle(
            ctypes.byref(fd),
            ctypes.c_ulonglong(handle.value),
            ctypes.c_int(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
            ctypes.c_ulonglong(0),
        ),
        "cuMemExportToShareableHandle",
    )
    print(
        f"[export] pid={os.getpid()} size={size} fd={fd.value} sum={EXPECT}", flush=True
    )

    if os.path.exists(sock_path):
        os.unlink(sock_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    os.chmod(sock_path, 0o600)
    srv.listen(8)
    print(f"[export] listening on {sock_path}", flush=True)
    while True:
        conn, _ = srv.accept()
        socket.send_fds(conn, [size.to_bytes(8, "little")], [fd.value])
        conn.close()
        print("[export] sent fd", flush=True)


def run_import(sock_path):
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    c.connect(sock_path)
    msg, fds, _, _ = socket.recv_fds(c, 8, 1)
    c.close()
    if not fds:
        print("[import] RESULT=FAIL no fd received over SCM_RIGHTS")
        return 3
    size = int.from_bytes(msg, "little")
    print(f"[import] pid={os.getpid()} fd={fds[0]} size={size}", flush=True)

    handle = ctypes.c_ulonglong(0)
    try:
        check(
            cuda.cuMemImportFromShareableHandle(
                ctypes.byref(handle),
                ctypes.c_void_p(fds[0]),
                ctypes.c_int(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
            ),
            "cuMemImportFromShareableHandle",
        )
        dptr = reserve_and_map(handle.value, size)
        host = (ctypes.c_float * NEL)()
        check(
            cuda.cuMemcpyDtoH_v2(
                ctypes.byref(host), ctypes.c_ulonglong(dptr), ctypes.c_size_t(NEL * 4)
            ),
            "cuMemcpyDtoH",
        )
    except RuntimeError as e:
        print(f"[import] RESULT=FAIL {e}")
        return 4

    got = float(sum(host))
    if abs(got - EXPECT) < 1e-3:
        print(f"[import] RESULT=OK vmm_fd import verified (sum={got})")
        return 0
    print(f"[import] RESULT=GARBAGE sum={got} expected={EXPECT}")
    return 5


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("export", "import"):
        print(__doc__)
        return 2
    torch.cuda.set_device(0)
    torch.zeros(1, device="cuda:0")  # materialise the primary context
    return (
        run_export(sys.argv[2]) if sys.argv[1] == "export" else run_import(sys.argv[2])
    )


if __name__ == "__main__":
    sys.exit(main())
