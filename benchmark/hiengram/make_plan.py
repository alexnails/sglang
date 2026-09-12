"""Emit the shippable HBM hot-set plan for _HbmCache.

One .npz with `layer_1` and `layer_14`, each [G, 8] int32: row g holds the 8
member row ids of group g, column h = head h. Groups are ordered by frequency
over the whole pooled corpus, descending, across all three n-gram sizes
interleaved, so keeping a byte-capped PREFIX is optimal.

The two layers are aligned row-for-row: row g is the SAME n-gram in both. The
frequency distribution is identical across layers (the rolling hash is injective
in practice, so group frequency is n-gram frequency and does not depend on the
layer's multipliers) but the rolling VALUES differ, so the ids must be computed
per layer. The alignment comes from the dense group-id streams: at a given token
position, dense_L0 and dense_L1 refer to the same n-gram, so one scatter/gather
recovers each layer-0 group's layer-14 rolling value.
"""

import json
import os

import numpy as np

WORK = "/scratch/engram_reuse"
GRP, ROLL = os.path.join(WORK, "grp"), os.path.join(WORK, "roll")
DEST = "/scratch/engram"
CORPUS = "pooled"
ROW_BYTES, GROUP_ROWS = 264, 8
GROUP_BYTES = ROW_BYTES * GROUP_ROWS  # 2112
GIB = 1 << 30
G_MIN = int(32 * GIB // GROUP_BYTES)  # 16,290,--- groups covers 32 GiB
G = 17_000_000  # ~4.4% headroom over G_MIN

z = np.load(os.path.join(WORK, "out", "layout.npz"))
PRIMES, OFFSETS = z["primes"], z["offsets"]  # [2,3,8], [2,24]
LAYER_IDS = [int(x) for x in z["layer_ids"]]  # [1, 14]
NUM_EMB = [int(x) for x in z["num_embeddings"]]

print("G_MIN for 32 GiB =", G_MIN, "-> emitting G =", G, flush=True)

# ---------------------------------------------------------------- load streams
keys0, keys1, cnts = {}, {}, {}
for j in range(3):
    k0 = np.load(os.path.join(GRP, "keys_%s_L0_g%d.npy" % (CORPUS, j)))
    c = np.load(os.path.join(GRP, "cntall_%s_L0_g%d.npy" % (CORPUS, j)))
    dense0 = np.load(
        os.path.join(GRP, "dense_%s_L0_g%d.npy" % (CORPUS, j)), mmap_mode="r"
    )
    roll0 = np.load(
        os.path.join(ROLL, "roll_%s_L0_g%d.npy" % (CORPUS, j)), mmap_mode="r"
    )
    roll1 = np.load(
        os.path.join(ROLL, "roll_%s_L1_g%d.npy" % (CORPUS, j)), mmap_mode="r"
    )
    D, T = k0.shape[0], dense0.shape[0]
    # any token position per group: scatter, last write wins
    pos = np.empty(D, dtype=np.int64)
    pos[np.asarray(dense0)] = np.arange(T, dtype=np.int64)
    k1 = np.asarray(roll1)[pos]
    # the alignment must be exact, so check it on a large sample
    s = np.linspace(0, D - 1, 200000).astype(np.int64)
    got = np.asarray(roll0)[pos[s]]
    assert np.array_equal(got, k0[s]), "layer-0 rolling value mismatch at ngram %d" % j
    keys0[j], keys1[j], cnts[j] = k0, k1, c
    print(
        "stream g=%d  D=%d  alignment verified on %d samples" % (j, D, len(s)),
        flush=True,
    )

# ------------------------------------------------- pick the top G by frequency
# Walk the distinct count values downward to find the smallest threshold whose
# >= set already covers G, then rank only that candidate set.
maxc = max(int(c.max()) for c in cnts.values())
hist = np.zeros(maxc + 2, dtype=np.int64)
for j in range(3):
    hist[: int(cnts[j].max()) + 1] += np.bincount(
        cnts[j], minlength=int(cnts[j].max()) + 1
    )
ge = np.cumsum(hist[::-1])[::-1]  # ge[v] = groups with count >= v
thr = int(np.flatnonzero(ge >= G)[-1])
print(
    "threshold count = %d (groups with count >= thr: %d)" % (thr, ge[thr]), flush=True
)

cj, ci, cc = [], [], []
for j in range(3):
    idx = np.flatnonzero(cnts[j] >= thr).astype(np.int64)
    cj.append(np.full(len(idx), j, dtype=np.int8))
    ci.append(idx)
    cc.append(cnts[j][idx].astype(np.int64))
cj, ci, cc = np.concatenate(cj), np.concatenate(ci), np.concatenate(cc)
print("candidates:", len(cc), flush=True)
# stable sort on -count: ties break by (stream, index), so the plan is deterministic
rank = np.argsort(-cc, kind="stable")[:G]
sel_j, sel_i, sel_c = cj[rank], ci[rank], cc[rank]
assert len(sel_j) == G, (len(sel_j), G)
print(
    "selected G=%d groups; count range %d..%d; marginal count %d"
    % (G, sel_c[-1], sel_c[0], sel_c[-1]),
    flush=True,
)

# ------------------------------------------------------------- member row ids
out = {}
for li, lid in enumerate(LAYER_IDS):
    arr = np.empty((G, GROUP_ROWS), dtype=np.int32)
    src = keys0 if li == 0 else keys1
    for j in range(3):
        m = sel_j == j
        r = src[j][sel_i[m]]
        for h in range(GROUP_ROWS):
            ids = r % int(PRIMES[li, j, h]) + int(OFFSETS[li, j * GROUP_ROWS + h])
            arr[m, h] = ids.astype(np.int32)
    assert arr.min() >= 0 and arr.max() < NUM_EMB[li], (
        arr.min(),
        arr.max(),
        NUM_EMB[li],
    )
    out["layer_%d" % lid] = arr
    print(
        "layer_%d: ids in [%d, %d), table rows %d"
        % (lid, arr.min(), arr.max() + 1, NUM_EMB[li]),
        flush=True,
    )

# ------------------------------------------------------------------- integrity
for lid, arr in out.items():
    probe = np.ascontiguousarray(arr[:, :3])
    u = np.unique(
        probe.view([("a", np.int32), ("b", np.int32), ("c", np.int32)]).ravel()
    )
    assert len(u) == G, "%s: probe key (first 3 member ids) not unique: %d of %d" % (
        lid,
        len(u),
        G,
    )
    print("%s: probe key unique across all %d groups" % (lid, G), flush=True)
# row g must be the same n-gram in both layers: the per-row n-gram size must agree
for lid, arr in out.items():
    li = LAYER_IDS.index(int(lid.split("_")[1]))
    bounds = [int(OFFSETS[li, j * GROUP_ROWS]) for j in range(3)] + [NUM_EMB[li]]
    which = np.searchsorted(np.asarray(bounds[1:]), arr[:, 0], side="right")
    assert np.array_equal(which.astype(np.int8), sel_j), (
        "%s: n-gram size misaligned" % lid
    )
print("both layers aligned row-for-row to the same n-gram", flush=True)

os.makedirs(DEST, exist_ok=True)
pth = os.path.join(DEST, "plan_pooled.npz")
np.savez(pth, **out)
print("wrote", pth, os.path.getsize(pth), "bytes", flush=True)

# --------------------------------------------------------------- sidecar meta
res = json.load(open(os.path.join(WORK, "results.json")))
ci_ = res["corpus"]["pooled"]
gl = res["curves"]["pooled"]["scopes"]["global"]
budgets = [1, 2, 4, 8, 16, 32]
comp = {}
for b, c in zip(budgets, gl["budget_composition_heldout"]):
    agg = {}
    for k, v in c.items():
        n = int(k.split("_g")[1]) + 2
        agg["%d-gram" % n] = agg.get("%d-gram" % n, 0) + v
    tot = sum(agg.values())
    comp["%d GiB" % b] = {
        k: dict(groups=int(round(v)), share_of_budget=round(v / tot, 4))
        for k, v in sorted(agg.items())
    }
meta = dict(
    format=dict(
        arrays=["layer_%d" % l for l in LAYER_IDS],
        dtype="int32",
        shape=["G", GROUP_ROWS],
        G=G,
        G_min_for_32GiB=G_MIN,
        row_bytes=ROW_BYTES,
        group_rows=GROUP_ROWS,
        group_bytes=GROUP_BYTES,
        ordering="group frequency over the whole pooled corpus, descending; "
        "all three n-gram sizes interleaved; keep a prefix to apply a byte cap",
        tie_break="stable sort on -count, ties ordered by (n-gram size, ascending group id) "
        "-- deterministic and reproducible",
        rows_aligned_across_layers=True,
        member_id_formula="member[g][h] = (rolling_g % primes[layer][j_g][h]) "
        "+ offsets[layer][j_g*8 + h]",
    ),
    calibration=dict(
        corpus="pooled, token-balanced interleave of four domains",
        basis="FULL pooled corpus (not the train half) -- the train/test split was the "
        "measurement protocol; a shipped plan should not discard half its data",
        n_tokens=int(ci_["n_tokens"]),
        n_documents=int(ci_["n_docs"]),
        group_lookups=int(ci_["n_tokens"]) * 3 * len(LAYER_IDS),
        domains=ci_["domains"],
        domain_token_shares={
            d: round(res["corpus"][d]["n_tokens"] / ci_["n_tokens"], 4)
            for d in ci_["domains"]
        },
        document_order="random.Random(0) shuffle within each domain",
        datasets=res["corpus_manifest"],
        tokenizer=res["meta"]["tokenizer"],
    ),
    selection=dict(
        marginal_group_count=int(sel_c[-1]),
        top_group_count=int(sel_c[0]),
        threshold_count=thr,
        candidates_considered=int(len(cc)),
        groups_per_ngram_size={
            "%d-gram" % (j + 2): int((sel_j == j).sum()) for j in range(3)
        },
    ),
    expected_hit_rate_heldout=dict(
        note="measured held-out (calibrate on first half, score on second half) on the "
        "same pooled corpus; the shipped plan uses 2x that calibration data so "
        "these are mildly conservative",
        by_budget_gib={
            str(b): round(h, 4) for b, h in zip(budgets, gl["budget_heldout"])
        },
        oracle_upper_bound={
            str(b): round(h, 4) for b, h in zip(budgets, gl["budget_oracle"])
        },
        hard_ceiling=round(1 - gl["heldout_cold_share"], 4),
        cold_share_unseen_in_calibration=round(gl["heldout_cold_share"], 4),
        composition_by_budget=comp,
    ),
    validity="A plan is only valid for the stated workload. See REPORT.md section 8: a "
    "single-domain plan does NOT transfer; this pooled plan does.",
    provenance=dict(
        sglang=res["meta"]["sglang"],
        deepseek_reference=res["meta"]["deepseek_reference"],
        engram_config={
            k: v
            for k, v in res["meta"]["engram_config"].items()
            if k.startswith("engram_") or k == "vocab_size"
        },
        study="/scratch/engram_reuse (REPORT.md, results.json, scripts/)",
    ),
)
jp = os.path.join(DEST, "plan_pooled.json")
json.dump(meta, open(jp, "w"), indent=2)
print("wrote", jp, flush=True)
print("groups per n-gram size:", meta["selection"]["groups_per_ngram_size"], flush=True)
