"""engram_gather with a device-resident cache of hot rows in front of the table.

The n_heads rows of one (token, n-gram size) group share a rolling hash value and
are always looked up together, so the cache is keyed per group: one probe serves
the whole group, and group g owns the n_heads consecutive slots n_heads * g + h.
A hit reads those slots, a miss reads the big (usually host-resident) table
exactly as engram_gather does. Both paths dequantize through engram_dequant_row,
so which side a row came from cannot change a single output bit.

The kernel never sees the rolling value, only the row ids it was modded into, so
a group is keyed on the first three of them. Distinct rolling values cannot share
three ids: that would need them congruent modulo p0 * p1 * p2 ~ 4.1e21, which
exceeds the 2**64 range the rolling value differs over. The kernel's probe and
build_group_probe_table's insert walk the same sequence: bucket
mix(id0, id1, id2) & (M - 1), then forward until the keys match or a bucket is
empty.
"""

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.embeddings.engram_gather import _E8M0_ZERO, engram_dequant_row

_MIX_MUL_A = 0xBF58476D1CE4E5B9
_MIX_MUL_B = 0x94D049BB133111EB
PROBE_KEYS = 3
MAX_PROBE = 32


@triton.jit
def _engram_finalize(z, MIX_A: tl.constexpr, MIX_B: tl.constexpr):
    z = (z ^ ((z >> 30) & 0x3FFFFFFFF)) * MIX_A
    z = (z ^ ((z >> 27) & 0x1FFFFFFFFF)) * MIX_B
    return z ^ ((z >> 31) & 0x1FFFFFFFF)


@triton.jit
def _engram_bucket(k0, k1, k2, MIX_A: tl.constexpr, MIX_B: tl.constexpr):
    z = _engram_finalize(k0, MIX_A, MIX_B)
    z = _engram_finalize(z ^ k1, MIX_A, MIX_B)
    return _engram_finalize(z ^ k2, MIX_A, MIX_B)


@triton.jit
def _engram_gather_cached_kernel(
    hw_ptr,
    hs_ptr,
    cw_ptr,
    cs_ptr,
    keys_ptr,
    groups_ptr,
    ids_ptr,
    out_ptr,
    ids_stride,
    probe_mask,
    row_lo,
    row_hi,
    DIM: tl.constexpr,
    BLK: tl.constexpr,
    H: tl.constexpr,
    HPAD: tl.constexpr,
    COLS: tl.constexpr,
    PROBE: tl.constexpr,
    MIX_A: tl.constexpr,
    MIX_B: tl.constexpr,
    E8M0_ZERO: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    ngram = tl.program_id(1).to(tl.int64)
    base = ids_ptr + token * ids_stride + ngram * H
    lanes = tl.arange(0, HPAD)
    live = lanes < H
    idx = tl.load(base + lanes, mask=live, other=0).to(tl.int64)
    owned = live & (idx >= row_lo) & (idx < row_hi)
    probing = tl.max(owned.to(tl.int32), axis=0) > 0
    k0 = tl.load(base).to(tl.int64)
    k1 = tl.load(base + 1).to(tl.int64)
    k2 = tl.load(base + 2).to(tl.int64)
    bucket = _engram_bucket(k0, k1, k2, MIX_A, MIX_B) & probe_mask
    last = bucket + PROBE - 1
    entry = keys_ptr + bucket * 3
    s0 = tl.load(entry, mask=probing, other=-1).to(tl.int64)
    s1 = tl.load(entry + 1, mask=probing, other=0).to(tl.int64)
    s2 = tl.load(entry + 2, mask=probing, other=0).to(tl.int64)
    while (s0 != -1) & ((s0 != k0) | (s1 != k1) | (s2 != k2)) & (bucket < last):
        bucket += 1
        entry = keys_ptr + (bucket & probe_mask) * 3
        s0 = tl.load(entry).to(tl.int64)
        s1 = tl.load(entry + 1).to(tl.int64)
        s2 = tl.load(entry + 2).to(tl.int64)
    hit = probing & (s0 == k0) & (s1 == k1) & (s2 == k2)
    group = tl.load(groups_ptr + (bucket & probe_mask), mask=hit, other=0).to(tl.int64)
    w_ptr = tl.where(hit, cw_ptr.to(tl.int64), hw_ptr.to(tl.int64))
    s_ptr = tl.where(hit, cs_ptr.to(tl.int64), hs_ptr.to(tl.int64))
    local = tl.where(hit, group * H + lanes, tl.where(owned, idx - row_lo, 0))
    offs = tl.arange(0, DIM)
    out = engram_dequant_row(
        w_ptr,
        s_ptr,
        local[:, None],
        offs[None, :],
        owned[:, None] & (offs[None, :] >= 0),
        DIM=DIM,
        BLK=BLK,
        E8M0_ZERO=E8M0_ZERO,
    )
    rows = token * COLS + ngram * H + lanes
    tl.store(
        out_ptr + rows[:, None] * DIM + offs[None, :],
        out.to(tl.bfloat16),
        mask=live[:, None] & (offs[None, :] >= 0),
    )


def engram_gather_cached(
    host_weight_ptr: int,
    host_scale_ptr: int,
    cache_weight_ptr: int,
    cache_scale_ptr: int,
    probe_keys: torch.Tensor,
    probe_groups: torch.Tensor,
    ids: torch.Tensor,
    out: torch.Tensor,
    dim: int,
    block_size: int,
    n_heads: int,
    n_ngram: int,
    row_lo: int = 0,
    row_hi: int = 2**62,
) -> torch.Tensor:
    """Gather rows ``ids`` ([T, n_ngram * n_heads] int) into ``out`` ([T * n_ngram *
    n_heads, dim] bf16, contiguous).

    ``ids`` column ``i * n_heads + h`` is head h of the i-th n-gram size, so each
    row of ``ids`` holds n_ngram groups of n_heads rows; its rows may be strided.
    ``host_weight_ptr`` / ``host_scale_ptr`` address the table of global rows
    [row_lo, row_hi) as engram_gather takes them. ``cache_weight_ptr`` /
    ``cache_scale_ptr`` address [n_heads * groups, dim] fp8 e4m3 and
    [n_heads * groups, dim // block_size] e8m0 device rows, slot
    ``n_heads * g + h`` holding head h of the group build_group_probe_table
    numbered g. Ids outside [row_lo, row_hi) produce zero rows, including inside
    a group that straddles the bound.
    """
    assert dim & (dim - 1) == 0 and dim % block_size == 0, (dim, block_size)
    assert out.dtype == torch.bfloat16 and out.is_contiguous()
    assert out.numel() == ids.numel() * dim, (out.shape, ids.shape, dim)
    assert n_heads >= PROBE_KEYS, (n_heads, PROBE_KEYS)
    assert ids.ndim == 2 and ids.shape[1] == n_ngram * n_heads, ids.shape
    assert ids.stride(1) == 1, ids.stride()
    capacity = probe_groups.numel()
    assert capacity and capacity & (capacity - 1) == 0, capacity
    assert probe_keys.shape == (capacity, PROBE_KEYS), probe_keys.shape
    assert probe_keys.dtype == torch.int32 and probe_keys.is_contiguous()
    assert probe_groups.dtype == torch.int32 and probe_groups.is_contiguous()
    tokens = ids.shape[0]
    if tokens:
        _engram_gather_cached_kernel[(tokens, n_ngram)](
            host_weight_ptr,
            host_scale_ptr,
            cache_weight_ptr,
            cache_scale_ptr,
            probe_keys,
            probe_groups,
            ids,
            out,
            ids.stride(0),
            capacity - 1,
            row_lo,
            row_hi,
            DIM=dim,
            BLK=block_size,
            H=n_heads,
            HPAD=triton.next_power_of_2(n_heads),
            COLS=n_ngram * n_heads,
            PROBE=MAX_PROBE,
            MIX_A=_MIX_MUL_A - 2**64,
            MIX_B=_MIX_MUL_B - 2**64,
            E8M0_ZERO=_E8M0_ZERO,
        )
    return out


def _mix_host(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.uint64)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(_MIX_MUL_A)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(_MIX_MUL_B)
    return z ^ (z >> np.uint64(31))


def _bucket_host(keys: np.ndarray) -> np.ndarray:
    """Bucket of each [G, PROBE_KEYS] key row, the kernel's _engram_bucket."""
    z = _mix_host(keys[:, 0])
    z = _mix_host(z ^ keys[:, 1].astype(np.uint64))
    return _mix_host(z ^ keys[:, 2].astype(np.uint64))


def build_group_probe_table(
    member_ids: torch.Tensor, load_factor: float = 0.6
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Open-addressing table of the cached groups, on ``member_ids``' device.

    ``member_ids`` [G, n_heads] are the row ids of each cached group; group g is
    the g-th row, so the caller fills cache slots ``n_heads * g + h`` with the
    table rows of ``member_ids[g]``. Returns (probe_keys [M, PROBE_KEYS] int32
    with -1 in column 0 for an empty bucket, probe_groups [M] int32, M); M is a
    power of two greater than G, so every probe ends on a match or on an empty
    bucket.
    """
    assert 0.0 < load_factor <= 1.0, load_factor
    assert member_ids.ndim == 2 and member_ids.shape[1] >= PROBE_KEYS, member_ids.shape
    ids = member_ids.to(torch.int64).cpu().numpy()
    count = ids.shape[0]
    assert count < 2**31, count
    assert count == 0 or (ids.min() >= 0 and ids.max() < 2**31), "row ids must be int32"
    keys = np.ascontiguousarray(ids[:, :PROBE_KEYS])
    assert np.unique(keys, axis=0).shape[0] == count, "group keys must be unique"
    capacity = 1
    while capacity <= count or capacity * load_factor < count:
        capacity *= 2
    table = np.full((capacity, PROBE_KEYS), -1, dtype=np.int32)
    groups = np.zeros(capacity, dtype=np.int32)
    bucket = (_bucket_host(keys) & np.uint64(capacity - 1)).astype(np.int64)
    placed = np.zeros(count, dtype=bool)
    while not placed.all():
        pending = np.nonzero(~placed)[0]
        free = table[bucket[pending], 0] == -1
        if free.any():
            candidates = pending[free]
            won = candidates[np.unique(bucket[candidates], return_index=True)[1]]
            table[bucket[won]] = keys[won]
            groups[bucket[won]] = won
            placed[won] = True
        rest = np.nonzero(~placed)[0]
        bucket[rest] = (bucket[rest] + 1) & (capacity - 1)
    device = member_ids.device
    return (
        torch.from_numpy(table).to(device),
        torch.from_numpy(groups).to(device),
        capacity,
    )
