"""The group-cached gather must reproduce engram_gather's rows bit for bit."""

import unittest

import numpy as np
import torch

from sglang.kernels.ops.embeddings.engram_gather import engram_gather
from sglang.kernels.ops.embeddings.engram_gather_cached import (
    MAX_PROBE,
    PROBE_KEYS,
    _bucket_host,
    build_group_probe_table,
    engram_gather_cached,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

H = 8
NGRAM = 3
COLS = NGRAM * H
DIM = 256
BLK = 32
ROWS = 4800
BAND = ROWS // COLS
POISON = 0x3C


def _table(seed: int, *, pinned: bool = False):
    """[ROWS, DIM] fp8 e4m3 payload and [ROWS, DIM // BLK] e8m0 scales, no NaN bytes."""
    g = torch.Generator().manual_seed(seed)
    weight = (torch.randn(ROWS, DIM, generator=g) * 3).to(torch.float8_e4m3fn)
    scale = torch.randint(
        100, 140, (ROWS, DIM // BLK), dtype=torch.uint8, generator=g
    ).view(torch.float8_e8m0fnu)
    where = {"pin_memory": True} if pinned else {"device": "cuda"}
    w = torch.empty(ROWS, DIM, dtype=torch.float8_e4m3fn, **where)
    s = torch.empty(ROWS, DIM // BLK, dtype=torch.float8_e8m0fnu, **where)
    w.view(torch.uint8).copy_(weight.view(torch.uint8))
    s.view(torch.uint8).copy_(scale.view(torch.uint8))
    return w, s


def _zero_exponent(w: torch.Tensor, s: torch.Tensor, local_row: int) -> None:
    """Payload 1.0 under an exponent byte of 0, which decodes to 2**-127."""
    w.view(torch.uint8)[local_row, 0] = (
        torch.tensor(1.0).to(torch.float8_e4m3fn).view(torch.uint8)
    )
    s.view(torch.uint8)[local_row, 0] = 0


def _pool(count: int, ngram: int, seed: int) -> torch.Tensor:
    """`count` groups for n-gram size `ngram`, head h drawn from its own row band."""
    g = torch.Generator().manual_seed(seed)
    cols = torch.arange(ngram * H, ngram * H + H)
    draws = (cols * BAND) + torch.randint(0, BAND, (count * 2 + 8, H), generator=g)
    keep = np.unique(draws[:, :PROBE_KEYS].numpy(), axis=0, return_index=True)[1]
    groups = draws[torch.from_numpy(np.sort(keep))][:count]
    assert groups.shape[0] == count, "not enough groups with distinct keys"
    return groups


def _query(pools, choices, row_lo: int) -> torch.Tensor:
    """[T, COLS] global ids, taking n-gram size i's group from pools[i][choices[i]]."""
    return torch.cat(
        [pools[i][choices[i]] + row_lo for i in range(NGRAM)], dim=1
    ).cuda()


def _mixed(seed: int, row_lo: int):
    """Cached, uncached and out-of-range groups interleaved over the n-gram sizes."""
    pools = [_pool(200, i, seed + i) for i in range(NGRAM)]
    members = torch.cat([pool[:64] for pool in pools]) + row_lo
    g = torch.Generator().manual_seed(seed)
    choices = [torch.randint(0, 200, (512,), generator=g) for _ in range(NGRAM)]
    for pick in choices:
        pick[0] = 0
        pick[1] = 199
    ids = _query(pools, choices, row_lo)
    beyond = _query(pools, [pick[:128] for pick in choices], row_lo + ROWS)
    return pools, members, torch.cat([ids, beyond])


def _cache(w, s, members: torch.Tensor, row_lo: int):
    """Slot H * g + h holds the table row of members[g, h]; unowned slots stay poison."""
    slots = max(members.numel(), 1)
    cw = torch.empty(slots, DIM, dtype=torch.float8_e4m3fn, device="cuda")
    cs = torch.empty(slots, DIM // BLK, dtype=torch.float8_e8m0fnu, device="cuda")
    cw.view(torch.uint8).fill_(POISON)
    cs.view(torch.uint8).fill_(POISON)
    local = members.reshape(-1) - row_lo
    owned = (local >= 0) & (local < ROWS)
    src = local[owned].to(w.device)
    dst = owned.nonzero().flatten().cuda()
    cw.view(torch.uint8)[dst] = w.view(torch.uint8)[src].cuda()
    cs.view(torch.uint8)[dst] = s.view(torch.uint8)[src].cuda()
    return cw, cs


def _uncached(w, s, ids, row_lo, row_hi):
    out = torch.empty(ids.numel(), DIM, dtype=torch.bfloat16, device="cuda")
    return engram_gather(
        w.data_ptr(),
        s.data_ptr(),
        ids.reshape(-1),
        out,
        DIM,
        BLK,
        row_lo=row_lo,
        row_hi=row_hi,
    )


def _cached(w, s, cw, cs, keys, groups, ids, row_lo, row_hi):
    out = torch.empty(ids.numel(), DIM, dtype=torch.bfloat16, device="cuda")
    return engram_gather_cached(
        w.data_ptr(),
        s.data_ptr(),
        cw.data_ptr(),
        cs.data_ptr(),
        keys,
        groups,
        ids,
        out,
        DIM,
        BLK,
        H,
        NGRAM,
        row_lo=row_lo,
        row_hi=row_hi,
    )


def _probe_host(keys: np.ndarray, groups: np.ndarray, member: np.ndarray, budget: int):
    """The kernel's probe sequence, in numpy."""
    capacity = groups.shape[0]
    bucket = int(_bucket_host(member.reshape(1, -1))[0] & np.uint64(capacity - 1))
    for _ in range(budget):
        entry = keys[bucket]
        if (entry == member[:PROBE_KEYS]).all():
            return int(groups[bucket])
        if int(entry[0]) == -1:
            return -1
        bucket = (bucket + 1) & (capacity - 1)
    return -1


class TestEngramGatherCached(CustomTestCase):
    def _check(self, w, s, members, ids, *, row_lo=0, load_factor=0.6):
        row_hi = row_lo + ROWS
        cw, cs = _cache(w, s, members, row_lo)
        keys, groups, capacity = build_group_probe_table(members, load_factor)
        self.assertEqual(capacity & (capacity - 1), 0)
        self.assertGreater(capacity, members.shape[0])
        want = _uncached(w, s, ids, row_lo, row_hi)
        got = _cached(w, s, cw, cs, keys.cuda(), groups.cuda(), ids, row_lo, row_hi)
        self.assertTrue(torch.equal(got, want))
        return got

    def _assert_zero_exponent(self, got, ids, row_id):
        seen = (ids.reshape(-1) == row_id).nonzero()
        self.assertTrue(seen.numel(), f"row {row_id} was never looked up")
        self.assertEqual(got[int(seen[0, 0]), 0].item(), 2.0**-127)

    def _run_mixed(self, w, s, seed, row_lo):
        pools, members, ids = _mixed(seed, row_lo)
        cached_row = int(pools[0][0, 0])
        uncached_row = int(pools[1][199, 0])
        _zero_exponent(w, s, cached_row)
        _zero_exponent(w, s, uncached_row)
        got = self._check(w, s, members, ids, row_lo=row_lo)
        self._assert_zero_exponent(got, ids, cached_row + row_lo)
        self._assert_zero_exponent(got, ids, uncached_row + row_lo)

    def test_bitwise_equal_device_table(self):
        for row_lo in (0, 1000):
            with self.subTest(row_lo=row_lo):
                w, s = _table(seed=1)
                self._run_mixed(w, s, seed=2, row_lo=row_lo)

    def test_bitwise_equal_pinned_host_table(self):
        w, s = _table(seed=3, pinned=True)
        self._run_mixed(w, s, seed=4, row_lo=0)

    def test_edge_cases(self):
        w, s = _table(seed=5)
        pools = [_pool(96, i, 6 + i) for i in range(NGRAM)]
        every = torch.cat(pools)
        one = [torch.arange(96) for _ in range(NGRAM)]
        dup = [torch.zeros(64, dtype=torch.int64) for _ in range(NGRAM)]
        cases = {
            "empty_ids": (every, torch.zeros(0, COLS, dtype=torch.int64).cuda()),
            "all_cached": (every, _query(pools, one, 0)),
            "none_cached": (every[:0], _query(pools, one, 0)),
            "duplicates": (every, _query(pools, dup, 0)),
            "unowned_only": (every, _query(pools, one, ROWS)),
        }
        for name, (members, ids) in cases.items():
            with self.subTest(case=name):
                got = self._check(w, s, members, ids)
                if name == "unowned_only":
                    self.assertTrue(torch.equal(got, torch.zeros_like(got)))

    def test_group_straddling_the_shard_bound(self):
        row_lo = 1000
        w, s = _table(seed=7)
        pools = [_pool(64, i, 8 + i) for i in range(NGRAM)]
        members = torch.cat([pool[:32] for pool in pools]) + row_lo
        members[0, :PROBE_KEYS] = torch.tensor([5, 6, 7])
        ids = torch.cat([members[0], members[33], members[65]]).view(1, COLS).cuda()
        got = self._check(w, s, members, ids, row_lo=row_lo)
        self.assertTrue(
            torch.equal(got[:PROBE_KEYS], torch.zeros_like(got[:PROBE_KEYS]))
        )
        self.assertFalse(torch.equal(got[PROBE_KEYS], torch.zeros_like(got[0])))

    def test_key_collision_is_rejected(self):
        w, s = _table(seed=9)
        other_w, other_s = _table(seed=10)
        pools = [_pool(256, i, 11 + i) for i in range(NGRAM)]
        members = torch.cat([pool[:64] for pool in pools])
        keys, groups, capacity = build_group_probe_table(members)
        cached_buckets = set(
            (_bucket_host(members[:, :PROBE_KEYS].numpy()) & np.uint64(capacity - 1))
            .astype(np.int64)
            .tolist()
        )
        collided = []
        for i in range(NGRAM):
            uncached = pools[i][64:]
            bucket = (
                _bucket_host(uncached[:, :PROBE_KEYS].numpy()) & np.uint64(capacity - 1)
            ).astype(np.int64)
            same = [j for j, b in enumerate(bucket.tolist()) if b in cached_buckets]
            self.assertTrue(same, f"no bucket collision for n-gram size {i}")
            collided.append(uncached[same])
        rows = min(group.shape[0] for group in collided)
        ids = torch.cat([group[:rows] for group in collided], dim=1).cuda()
        cw, cs = _cache(other_w, other_s, members, 0)
        got = _cached(w, s, cw, cs, keys.cuda(), groups.cuda(), ids, 0, ROWS)
        self.assertTrue(torch.equal(got, _uncached(w, s, ids, 0, ROWS)))

    def test_strided_id_rows(self):
        """The hasher hands over one layer of a [T, layers, cols], so the id rows
        arrive strided and the kernel has to address them by that stride."""
        w, s = _table(seed=21)
        pools = [_pool(96, i, 22 + i) for i in range(NGRAM)]
        members = torch.cat([pool[:48] for pool in pools])
        ids = _query(pools, [torch.arange(96) for _ in range(NGRAM)], 0)
        strided = torch.stack([ids, ids.flip(0), ids], dim=1)[:, 1]
        self.assertFalse(strided.is_contiguous())
        keys, groups, _ = build_group_probe_table(members)
        cw, cs = _cache(w, s, members, 0)
        got = _cached(w, s, cw, cs, keys.cuda(), groups.cuda(), strided, 0, ROWS)
        self.assertTrue(torch.equal(got, _uncached(w, s, strided, 0, ROWS)))

    def test_third_key_decides(self):
        """A group on a cached group's bucket sharing its first two keys and differing
        only in the third must miss: two keys do not identify a group, three do."""
        w, s = _table(seed=17)
        other_w, other_s = _table(seed=18)
        pools = [_pool(64, i, 19 + i) for i in range(NGRAM)]
        members = torch.cat([pool[:32] for pool in pools])
        keys, groups, capacity = build_group_probe_table(members)
        cached = {tuple(key) for key in members[:, :PROBE_KEYS].tolist()}
        bucket = _bucket_host(members.numpy()) & np.uint64(capacity - 1)
        twin = None
        for candidate in range(ROWS):
            probe = members[0].clone()
            probe[2] = candidate
            key = tuple(probe[:PROBE_KEYS].tolist())
            if key in cached:
                continue
            same = _bucket_host(probe.numpy().reshape(1, -1)) & np.uint64(capacity - 1)
            if int(same[0]) == int(bucket[0]):
                twin = probe
                break
        self.assertIsNotNone(twin, "no third-key twin lands on the cached bucket")
        ids = torch.cat([twin, pools[1][40], pools[2][40]]).view(1, COLS).cuda()
        cw, cs = _cache(other_w, other_s, members, 0)
        got = _cached(w, s, cw, cs, keys.cuda(), groups.cuda(), ids, 0, ROWS)
        self.assertTrue(torch.equal(got, _uncached(w, s, ids, 0, ROWS)))

    def test_hits_read_their_cache_slots(self):
        w, s = _table(seed=12)
        other_w, other_s = _table(seed=13)
        pools = [_pool(128, i, 14 + i) for i in range(NGRAM)]
        members = torch.cat([pool[:64] for pool in pools])
        keys, groups, _ = build_group_probe_table(members)
        ids = _query(pools, [torch.arange(128) for _ in range(NGRAM)], 0)
        cw, cs = _cache(other_w, other_s, members, 0)
        got = _cached(w, s, cw, cs, keys.cuda(), groups.cuda(), ids, 0, ROWS)
        keys_np, groups_np = keys.numpy(), groups.numpy()
        found = torch.zeros(ids.numel(), dtype=torch.bool)
        member_ids = ids.cpu().numpy()
        for token in range(ids.shape[0]):
            for i in range(NGRAM):
                group = member_ids[token, i * H : (i + 1) * H]
                if _probe_host(keys_np, groups_np, group, MAX_PROBE) >= 0:
                    start = token * COLS + i * H
                    found[start : start + H] = True
        self.assertGreater(int(found.sum()), members.numel() * 9 // 10)
        want = torch.where(
            found.cuda().unsqueeze(-1),
            _uncached(other_w, other_s, ids, 0, ROWS),
            _uncached(w, s, ids, 0, ROWS),
        )
        self.assertTrue(torch.equal(got, want))

    def test_probe_table_agreement(self):
        for count, load_factor in ((0, 0.6), (1, 0.6), (1500, 0.6), (1500, 1.0)):
            with self.subTest(count=count, load_factor=load_factor):
                pool = _pool(max(count, 1), 0, 15)[:count]
                keys, groups, capacity = build_group_probe_table(pool, load_factor)
                self.assertEqual(groups.numel(), capacity)
                self.assertEqual(int((keys[:, 0] != -1).sum()), count)
                keys_np, groups_np = keys.numpy(), groups.numpy()
                budget = MAX_PROBE if load_factor <= 0.6 else capacity
                for group, member in enumerate(pool.numpy()):
                    self.assertEqual(
                        _probe_host(keys_np, groups_np, member, budget), group
                    )
                for member in _pool(64, 1, 16).numpy():
                    self.assertEqual(
                        _probe_host(keys_np, groups_np, member, capacity), -1
                    )


if __name__ == "__main__":
    unittest.main()
