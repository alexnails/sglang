# Two-pass decoupled KDA/GDN chunk-state scan (gfx950, K=V=128, kg chunk-major).
#
# Drop-in replacement for chunk_gated_delta_rule_fwd_h (prefill path):
#   S_out = S_in @ M_c + L_c   per chunk c (exact affine form of the h-kernel step)
#   M_c = diag(exp2(g_last_c)) - R_c, R_c = W_c^T KG_c ;  L_c = U_c^T KG_c
# P1 folds GS chunks/group into (D_g, E_g, L_g) with M_g = diag(D_g) + E_g,
#    one CTA per (role in {M,L}, group, head). The per-iteration chain is a
#    single [K,K]@[K,K] fold dot against pre-materialized R_c (the R build is
#    independent of the running operator and overlaps across iterations); the
#    exact gamma scaling stays in fp32 so the diagonal (dominant term) never
#    sees bf16. P1 splits M_g into fp32 diagonal + bf16 off-diagonal on store.
# P2 scans the G group operators sequentially (G = NT/GS steps instead of NT):
#    S = S*diag(D) + S_bf16 @ E_bf16 + L — the bf16 dot only ever touches the
#    off-diagonal correction, so state error stays at production's own floor.
# P3: existing fused scan_o kernel with each group as a pseudo-sequence
#    starting from its scanned group state (chunk_gated_delta_rule_fwd_o_3p),
#    or the existing h-kernel for the 2p variant (chunk_gated_delta_rule_fwd_h_2p).
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices
from sglang.kernels.ops.attention.fla.op import exp2


@triton.jit(do_not_specialize=["T", "G"])
def _scan2p_compose_kernel(
    w,  # [B,T,H,K] bf16 (UT-transformed w)
    u,  # [B,T,H,V] bf16
    kg,  # [chunks,H,K,BT] bf16 chunk-major gated key (zero-padded tails)
    gk,  # [B,T,H,K] fp32 chunk-local log2 cumsum
    Dg,  # out [G,H,K] fp32 diagonal of M_g
    Eg,  # out [G,H,K,K] bf16 off-diagonal of M_g
    Lg,  # out [G,H,V,K] fp32
    cu_seqlens,
    g_seq,  # [G] int32 sequence of each group
    g_tok0,  # [G] int32 absolute token offset of group start
    g_chunk0,  # [G] int32 global chunk index of group start
    T,
    G,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    GS: tl.constexpr,
):
    i_p, i_h = tl.program_id(0), tl.program_id(1)
    role = i_p // G  # 0 -> M_g, 1 -> L_g
    i_g = i_p % G
    i_n = tl.load(g_seq + i_g)
    t0g = tl.load(g_tok0 + i_g)
    c0 = tl.load(g_chunk0 + i_g)
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    Ts = eos - bos

    o_k = tl.arange(0, K)
    wb = w + (bos * H + i_h).to(tl.int64) * K
    gb = gk + (bos * H + i_h).to(tl.int64) * K

    if role == 0:
        # ---- M_g: X = X*diag(g_j) - (X W_j^T) KG_j^T (two chained small dots;
        # the R-fold's extra [K,K] build measured slower: 69us vs 53us) ----
        p_wT = tl.make_block_ptr(wb, (K, Ts), (1, H * K), (0, t0g - bos), (K, BT), (0, 1))
        b_wT = tl.load(p_wT, boundary_check=(0, 1))
        p_kg = tl.make_block_ptr(
            kg + (c0 * H + i_h).to(tl.int64) * K * BT,
            (K, BT), (BT, 1), (0, 0), (K, BT), (1, 0),
        )
        b_kg = tl.load(p_kg)
        last = min(t0g - bos + BT, Ts) - 1
        b_g = exp2(tl.load(gb + last * H * K + o_k))
        b_X = -tl.dot(b_wT, tl.trans(b_kg))
        b_X = tl.where(o_k[:, None] == o_k[None, :], b_g[None, :], 0.0) + b_X
        for j in range(1, GS):
            t0 = t0g - bos + j * BT
            p_wT = tl.make_block_ptr(wb, (K, Ts), (1, H * K), (0, t0), (K, BT), (0, 1))
            b_wT = tl.load(p_wT, boundary_check=(0, 1))
            p_kg = tl.make_block_ptr(
                kg + ((c0 + j) * H + i_h).to(tl.int64) * K * BT,
                (K, BT), (BT, 1), (0, 0), (K, BT), (1, 0),
            )
            b_kg = tl.load(p_kg)
            last = min(t0 + BT, Ts) - 1
            b_g = exp2(tl.load(gb + last * H * K + o_k))
            b_A = tl.dot(b_X.to(w.dtype.element_ty), b_wT)  # [K,BT]
            b_X = b_X * b_g[None, :] - tl.dot(b_A.to(w.dtype.element_ty), tl.trans(b_kg))
        # split store: fp32 diagonal + bf16 off-diagonal
        b_d = tl.sum(tl.where(o_k[:, None] == o_k[None, :], b_X, 0.0), axis=1)
        b_E = tl.where(o_k[:, None] == o_k[None, :], 0.0, b_X)
        tl.store(Dg + (i_g * H + i_h).to(tl.int64) * K + o_k, b_d)
        p_E = tl.make_block_ptr(
            Eg + (i_g * H + i_h).to(tl.int64) * K * K,
            (K, K), (K, 1), (0, 0), (K, K), (1, 0),
        )
        tl.store(p_E, b_E.to(Eg.dtype.element_ty))
    else:
        # ---- L_g: X = X * (diag(g_j) - R_j) + U_j^T KG_j ----
        ub = u + (bos * H + i_h).to(tl.int64) * V
        p_uT = tl.make_block_ptr(ub, (V, Ts), (1, H * V), (0, t0g - bos), (V, BT), (0, 1))
        b_uT = tl.load(p_uT, boundary_check=(0, 1))
        p_wT = tl.make_block_ptr(wb, (K, Ts), (1, H * K), (0, t0g - bos), (K, BT), (0, 1))
        b_wT = tl.load(p_wT, boundary_check=(0, 1))
        p_kg = tl.make_block_ptr(
            kg + (c0 * H + i_h).to(tl.int64) * K * BT,
            (K, BT), (BT, 1), (0, 0), (K, BT), (1, 0),
        )
        b_kg = tl.load(p_kg)
        last = min(t0g - bos + BT, Ts) - 1
        b_g = exp2(tl.load(gb + last * H * K + o_k))
        b_X = tl.dot(b_uT, tl.trans(b_kg))
        for j in range(1, GS):
            t0 = t0g - bos + j * BT
            p_uT = tl.make_block_ptr(ub, (V, Ts), (1, H * V), (0, t0), (V, BT), (0, 1))
            b_uT = tl.load(p_uT, boundary_check=(0, 1))
            p_wT = tl.make_block_ptr(wb, (K, Ts), (1, H * K), (0, t0), (K, BT), (0, 1))
            b_wT = tl.load(p_wT, boundary_check=(0, 1))
            p_kg = tl.make_block_ptr(
                kg + ((c0 + j) * H + i_h).to(tl.int64) * K * BT,
                (K, BT), (BT, 1), (0, 0), (K, BT), (1, 0),
            )
            b_kg = tl.load(p_kg)
            last = min(t0 + BT, Ts) - 1
            b_g = exp2(tl.load(gb + last * H * K + o_k))
            b_B = tl.dot(b_X.to(u.dtype.element_ty), b_wT)  # [V,BT]
            b_X = b_X * b_g[None, :] + tl.dot(
                (b_uT.to(tl.float32) - b_B).to(u.dtype.element_ty), tl.trans(b_kg)
            )
        p_L = tl.make_block_ptr(
            Lg + (i_g * H + i_h).to(tl.int64) * V * K,
            (V, K), (K, 1), (0, 0), (V, K), (1, 0),
        )
        tl.store(p_L, b_X.to(Lg.dtype.element_ty))


@triton.jit
def _scan2p_group_scan_kernel(
    Dg,  # [G,H,K] fp32 diagonal of M_g
    Eg,  # [G,H,K,K] bf16 off-diagonal of M_g
    Lg,  # [G,H,V,K] fp32
    Sg,  # out [G,H,V,K] fp32 group-initial states
    initial_state,
    initial_state_indices,
    stride_init_state,
    seq_group0,  # [N] int32
    seq_ngroups,  # [N] int32
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    g0 = tl.load(seq_group0 + i_n)
    ng = tl.load(seq_ngroups + i_n)

    index = tl.load(initial_state_indices + i_n).to(tl.int64)
    valid = index >= 0
    h0 = initial_state + index * stride_init_state + i_h * V * K

    o_v = i_v * BV + tl.arange(0, BV)
    o_k = tl.arange(0, K)
    b_S = tl.zeros([BV, K], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if valid:
            b_S += tl.load(h0 + o_v[:, None] * K + o_k[None, :])

    for g in range(g0, g0 + ng):
        tl.store(
            Sg + ((g * H + i_h) * V).to(tl.int64) * K + o_v[:, None] * K + o_k[None, :],
            b_S,
        )
        b_d = tl.load(Dg + (g * H + i_h).to(tl.int64) * K + o_k)
        p_E = tl.make_block_ptr(
            Eg + (g * H + i_h).to(tl.int64) * K * K, (K, K), (K, 1), (0, 0), (K, K), (1, 0)
        )
        b_E = tl.load(p_E)
        b_L = tl.load(
            Lg + ((g * H + i_h) * V).to(tl.int64) * K + o_v[:, None] * K + o_k[None, :]
        )
        # exact fp32 diagonal scaling + bf16 dot on the off-diagonal only:
        # M_g's dominant term never sees bf16, so the carried state keeps
        # production's own error floor (bf16 M_g storage measured 1.6e-3).
        b_S = b_S * b_d[None, :] + tl.dot(b_S.to(tl.bfloat16), b_E) + b_L

    if USE_INITIAL_STATE:
        if valid:
            tl.store(h0 + o_v[:, None] * K + o_k[None, :], b_S)


_TABLE_CACHE = {}
_PTR_ALIAS = {}


def _tables(cu_seqlens, T_total, GS, device):
    """Group tables aligned to sequence boundaries, cached so no host<->device
    sync happens inside HIP graph capture. First (non-capture) call builds.

    Cache is keyed on cu_seqlens VALUES (SkepticKDA T5: a pointer/shape/T_total
    key returns stale tables when the allocator recycles a cu buffer with a
    different varlen split -> silently wrong outputs). Inside graph capture the
    values cannot be read (no D2H), so capture uses a pointer alias registered
    at build time; a recycled pointer re-registered by a later eager warm-up
    overwrites the alias, so the alias always tracks the most recent warm-up.
    """
    if torch.cuda.is_current_stream_capturing():
        ent = _PTR_ALIAS.get(cu_seqlens.data_ptr())
        assert ent is not None, (
            "scan2p: group-table cache miss during graph capture "
            "(warm up eagerly with the same cu_seqlens values first)"
        )
        return ent
    cu = [int(x) for x in cu_seqlens.tolist()]
    key = (tuple(cu), T_total, GS)
    ent = _TABLE_CACHE.get(key)
    if ent is not None:
        _PTR_ALIAS[cu_seqlens.data_ptr()] = ent
        return ent
    BT = 64
    N = len(cu) - 1
    g_seq, g_tok0, g_chunk0 = [], [], []
    seq_group0, seq_ngroups = [], []
    chunk0 = 0
    for n in range(N):
        Tn = cu[n + 1] - cu[n]
        NTn = triton.cdiv(Tn, BT)
        assert NTn % GS == 0, f"seq {n}: NT={NTn} not divisible by GS={GS}"
        ngn = NTn // GS
        seq_group0.append(len(g_seq))
        seq_ngroups.append(ngn)
        for gi in range(ngn):
            g_seq.append(n)
            g_tok0.append(cu[n] + gi * GS * BT)
            g_chunk0.append(chunk0 + gi * GS)
        chunk0 += NTn
    to = lambda x: torch.tensor(x, dtype=torch.int32, device=device)
    cu_pseudo = to(g_tok0 + [cu[-1]])
    co_pseudo = to(g_chunk0)
    G = len(g_seq)
    ent = dict(
        g_seq=to(g_seq),
        g_tok0=to(g_tok0),
        g_chunk0=co_pseudo,
        seq_group0=to(seq_group0),
        seq_ngroups=to(seq_ngroups),
        cu_pseudo=cu_pseudo,
        co_pseudo=co_pseudo,
        ci_pseudo=prepare_chunk_indices(cu_pseudo, BT),
        gidx=torch.arange(G, dtype=torch.long, device=device),
        G=G,
    )
    _TABLE_CACHE[key] = ent
    _PTR_ALIAS[cu_seqlens.data_ptr()] = ent
    return ent


def _run_2p_scan(k, w, u, gk, initial_state, initial_state_indices, cu_seqlens,
                 GS=8, scan_warps=4, scan_stages=1):
    """P1 operator compose + P2 group scan. Returns (Sg, tb): Sg [G,H,V,K]
    fp32 group-initial states (final state written into initial_state)."""
    B, T, H, V = u.shape
    K = k.shape[2]
    BT = 64
    device = u.device
    N = len(cu_seqlens) - 1

    tb = _tables(cu_seqlens, T, GS, device)
    G = tb["G"]

    Dg = torch.empty(G, H, K, dtype=torch.float32, device=device)
    Eg = torch.empty(G, H, K, K, dtype=torch.bfloat16, device=device)
    Lg = torch.empty(G, H, V, K, dtype=torch.float32, device=device)
    _scan2p_compose_kernel[(2 * G, H)](
        w=w,
        u=u,
        kg=k,
        gk=gk,
        Dg=Dg,
        Eg=Eg,
        Lg=Lg,
        cu_seqlens=cu_seqlens,
        g_seq=tb["g_seq"],
        g_tok0=tb["g_tok0"],
        g_chunk0=tb["g_chunk0"],
        T=T,
        G=G,
        H=H,
        K=K,
        V=V,
        BT=BT,
        GS=GS,
        num_warps=scan_warps,
        num_stages=scan_stages,
    )

    Sg = torch.empty(G, H, V, K, dtype=torch.float32, device=device)
    BV2 = 16
    _scan2p_group_scan_kernel[(V // BV2, N * H)](
        Dg=Dg,
        Eg=Eg,
        Lg=Lg,
        Sg=Sg,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        stride_init_state=(initial_state.stride(0) if initial_state is not None else 0),
        seq_group0=tb["seq_group0"],
        seq_ngroups=tb["seq_ngroups"],
        H=H,
        K=K,
        V=V,
        BV=BV2,
        USE_INITIAL_STATE=initial_state is not None,
        num_warps=4,
        num_stages=3,
    )
    return Sg, tb


def chunk_gated_delta_rule_fwd_h_2p(
    k,  # kg chunk-major [chunks,H,K,BT] bf16
    w,
    u,
    g=None,
    gk=None,
    initial_state=None,
    initial_state_indices=None,
    save_new_value=True,
    cu_seqlens=None,
    chunk_indices=None,
    use_exp2=False,
    inplace_update=True,
    track_state=None,
    track_chunk_idx=None,
    chunk_offsets=None,
    kg_chunk_major=False,
    tile=None,
    GS=8,
    scan_warps=4,
    scan_stages=1,
):
    """Same signature/contract as chunk_gated_delta_rule_fwd_h (production
    gfx950 KDA configuration only). Returns (h, v_new); final state is written
    into initial_state in place when inplace_update=True."""
    assert kg_chunk_major, "2p scan requires chunk-major gated key"
    assert g is None and gk is not None and use_exp2, "KDA gk/exp2 path only"
    assert track_state is None, "track_state not supported by 2p prototype"
    assert save_new_value, "v_new is required downstream"
    assert cu_seqlens is not None, "prototype is varlen-only (harness shape)"
    assert u.shape[-1] == k.shape[2] == 128

    Sg, tb = _run_2p_scan(
        k, w, u, gk, initial_state, initial_state_indices, cu_seqlens,
        GS=GS, scan_warps=scan_warps, scan_stages=scan_stages,
    )

    # P3: existing h-kernel, each group a pseudo-sequence starting from Sg.
    # inplace_update=False: P2 already wrote the final state.
    h, v_new = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        gk=gk,
        initial_state=Sg,
        initial_state_indices=tb["gidx"],
        save_new_value=True,
        cu_seqlens=tb["cu_pseudo"],
        chunk_indices=tb["ci_pseudo"],
        chunk_offsets=tb["co_pseudo"],
        use_exp2=True,
        inplace_update=False,
        kg_chunk_major=True,
        tile=tile if tile is not None else (16, 4, 3),
    )
    return h, v_new


def chunk_gated_delta_rule_fwd_o_3p(
    q,  # l2-normalized [B,T,H,K] bf16
    kgc,  # kg chunk-major [chunks,H,K,BT] bf16
    w,
    u,
    gk,
    A,  # Aqk [B,T,H,BT] bf16
    o,  # out [B,T,H,V] (written in place)
    scale,
    initial_state,
    initial_state_indices,
    cu_seqlens,
    GS=8,
    scan_warps=4,
    scan_stages=1,
    scan_o_block_v=128,
    scan_o_warps=4,
    scan_o_stages=1,
):
    """2p scan + fused-output scan_o: replaces chunk_gated_delta_rule_fwd_h +
    chunk_gla_fwd_o_gk. Each group of GS chunks runs as a pseudo-sequence from
    its scanned state; scan_o computes o inline with no h/v_new round trip."""
    from sglang.kernels.ops.attention.fla.kda_chunk_scan import (
        chunk_kda_fwd_scan_o,
    )

    Sg, tb = _run_2p_scan(
        kgc, w, u, gk, initial_state, initial_state_indices, cu_seqlens,
        GS=GS, scan_warps=scan_warps, scan_stages=scan_stages,
    )
    # scan_o also stores each pseudo-sequence's group-final state back into
    # Sg (harmless: buffer is discarded; the true final state came from P2).
    return chunk_kda_fwd_scan_o(
        q=q,
        kgc=kgc,
        w=w,
        u=u,
        gk=gk,
        A=A,
        o=o,
        scale=scale,
        initial_state=Sg,
        initial_state_indices=tb["gidx"],
        cu_seqlens=tb["cu_pseudo"],
        chunk_offsets=tb["co_pseudo"],
        block_v=scan_o_block_v,
        num_warps=scan_o_warps,
        num_stages=scan_o_stages,
    )


def chunk_kda_fwd_3p(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    initial_state_indices,
    cu_seqlens=None,
    A_log=None,
    dt_bias=None,
    lower_bound=None,
    output_intermediate_states=False,
    track_state=None,
    track_chunk_idx=None,
    beta_is_raw=False,
    chunk_indices=None,
    chunk_offsets=None,
    GS=8,
    **scan_kw,
):
    """chunk_kda_fwd drop-in for the gfx950 KDA prefill shape: identical
    prefix (cumsum + intra), then 2p scan + fused scan_o instead of
    fwd_h + gla_o."""
    from sglang.kernels.ops.attention.fla.chunk_intra import chunk_kda_fwd_intra
    from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
    from sglang.kernels.ops.attention.fla.kda import (
        RCP_LN2,
        kda_gate_chunk_cumsum,
    )

    assert initial_state is not None and cu_seqlens is not None
    assert not output_intermediate_states
    assert track_state is None and track_chunk_idx is None
    assert q.shape[-1] == v.shape[-1] == 128
    chunk_size = 64
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    if A_log is not None:
        g = kda_gate_chunk_cumsum(
            g,
            A_log=A_log,
            chunk_size=chunk_size,
            scale=RCP_LN2,
            dt_bias=dt_bias,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            lower_bound=lower_bound,
            beta=beta if beta_is_raw else None,
        )
        if beta_is_raw:
            g, beta = g
    else:
        if beta_is_raw:
            beta = beta.float().sigmoid().contiguous()
        g = chunk_local_cumsum(
            g,
            chunk_size=chunk_size,
            scale=RCP_LN2,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    _small_grid = q.shape[0] * chunk_indices.shape[0] * q.shape[-2] <= 256
    w, u, _, kg, Aqk, _ = chunk_kda_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=lower_bound is not None,
        fuse_diagonal=_small_grid,
        fuse_recompute=_small_grid,
        kg_chunk_major=True,
    )

    return chunk_gated_delta_rule_fwd_o_3p(
        q=q,
        kgc=kg,
        w=w,
        u=u,
        gk=g,
        A=Aqk,
        o=v,
        scale=scale,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        GS=GS,
        **scan_kw,
    )
