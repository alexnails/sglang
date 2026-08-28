"""Numerical and state-cache correctness tests for the ZAYA1 CCA module.

The CCA per-request conv-state cache must satisfy the following invariants,
which are each exercised by a dedicated test case:

1. A single-chunk extend forward (no prefix) is numerically equivalent to the
   reference torch implementation that processes the whole sequence at once.
2. Splitting a sequence into one prefill of ``S0`` tokens and ``S1`` single-
   token decode steps produces the same q / k / v tensors as the equivalent
   single-chunk run.
3. A batched two-request decode for request 0 yields identical q / k / v to a
   single-request decode of request 0 at the same step.
4. Multi-request prefills update only the conv state and lag slots for each
   request and leave unused slots zero.
5. A simulated tensor-parallel (TP=2) CCA produces per-rank q / k / v slices
   that match the corresponding head slices of a TP=1 reference, both for
   prefill (``_forward_extend``) and for decode (``_forward_decode``).
6. A prefill resumed from a cached prefix -- alone or batched alongside a fresh
   request -- matches a single-chunk run, which is what pins the *kind* of value
   parked at a chunk boundary (``conv[1]`` holds ``val_proj2 . hs``, not ``hs``).
7. A biased checkpoint, where ``W . 0 != 0``, falls back to caching the raw
   hidden state and is still numerically right.

All tests run on CPU with a tiny configuration so they stay fast and have no
GPU dependency. State is stored in a mock centralized pool that mirrors the
``HybridReqToTokenPool`` / ``MambaPool`` interface used at serving time.
"""

import os
import unittest
import unittest.mock
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _ensure_dist_initialized() -> None:
    """Set up a minimal single-rank gloo distributed environment plus the
    SGLang model-parallel groups (TP=1, PP=1, EP=1). The CCA module reads
    ``get_tensor_model_parallel_rank()`` / ``get_tensor_model_parallel_world_size()``
    inside ``__init__`` to size its head-parallel projections, so the world
    group and model parallel groups must both be initialized before any CCA
    construction.
    """
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29632")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    if not torch.distributed.is_initialized():
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="gloo",
        )

    if not model_parallel_is_initialized():
        # WARNING, on a machine with a GPU: ``backend="gloo"`` does not keep this
        # on the CPU. ``GroupCoordinator.__init__`` sets
        # ``self.device = cuda:{local_rank}`` whenever ``is_cuda_alike()``, which
        # is fixed at import from the platform and not from the backend argument,
        # and ``init_model_parallel_group`` then defaults ``use_pynccl=True`` and
        # ``use_custom_allreduce=_ENABLE_CUSTOM_ALL_REDUCE``. So this builds CUDA
        # device communicators -- RCCL, custom-allreduce IPC -- over a gloo
        # process group. On gfx950 that aborts the process, and because the abort
        # is asynchronous it lands wherever the CPU thread happens to be.
        #
        # Only call this from tests that genuinely need model-parallel state
        # (CCA reads the TP rank in __init__). ZayaRouter does not -- see
        # ``_make_tiny_router``.
        #
        # Pass arguments as kwargs because ``ensure_model_parallel_initialized``
        # forwards positional ``backend`` into the ``attention_data_parallel_size``
        # slot of ``initialize_model_parallel``, which then explodes on
        # ``int // str``. Using kwargs avoids that footgun.
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            backend="gloo",
        )


# ---------------------------------------------------------------------------
# Mock centralized pool
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MockLayerCache:
    conv: List[torch.Tensor]
    temporal: torch.Tensor


class _MockReqToTokenPool:
    """Minimal stand-in for ``HybridReqToTokenPool`` providing the two methods
    that CCA calls: ``mamba2_layer_cache`` and ``get_mamba_indices``.

    For TP-aware tests, ``tp_size`` controls the per-rank ``in_out_ch`` of the
    ``conv[0]`` state. ``conv[1]`` (the ``val_proj2`` lag) is replicated and
    takes its width from ``ZayaConfig.cca_v2_state_dim`` -- the projected value
    width when the checkpoint allows it, ``hidden_size`` otherwise -- exactly as
    ``mamba2_cache_params`` sizes it at serving time.
    """

    def __init__(self, pool_size: int, cca_config, tp_size: int = 1):
        in_out_ch_full = (
            cca_config.num_attention_heads + cca_config.num_key_value_heads
        ) * cca_config.head_dim
        assert in_out_ch_full % tp_size == 0
        in_out_ch_per_rank = in_out_ch_full // tp_size
        total_padding = (cca_config.cca_time0 - 1) + (cca_config.cca_time1 - 1)
        num_layers = len(cca_config.linear_layer_ids)

        self.conv_state = torch.zeros(
            num_layers, pool_size + 1, in_out_ch_per_rank, total_padding
        )
        self.prev_hs_state = torch.zeros(
            num_layers, pool_size + 1, cca_config.cca_v2_state_dim, 1
        )
        self.temporal = torch.zeros(num_layers, pool_size + 1, 1, 1, 0)
        self._layer_map = {lid: i for i, lid in enumerate(cca_config.linear_layer_ids)}
        self._identity_map = torch.arange(pool_size + 1, dtype=torch.int32)

    def mamba2_layer_cache(self, layer_id: int):
        idx = self._layer_map[layer_id]
        return _MockLayerCache(
            conv=[self.conv_state[idx], self.prev_hs_state[idx]],
            temporal=self.temporal[idx],
        )

    def get_mamba_indices(self, req_pool_indices: torch.Tensor) -> torch.Tensor:
        return req_pool_indices.to(torch.int32)


class _MockShortConvBackend:
    """Stand-in for ``ShortConvHybridAttnBackend`` in the CPU unit tests.

    The CCA module reaches the conv-state plumbing via
    ``get_attn_backend().conv_state_metadata(...)`` and runs its own conv
    kernel. This mock exposes that accessor over a ``_MockReqToTokenPool``,
    mirroring ``ShortConvAttnBackend``: the req -> slot mapping (and, for extend,
    its host ``.tolist()`` mirror) is resolved once per step and shared across
    all conv layers, while the decode path stays entirely on-device.
    """

    def __init__(self, pool: "_MockReqToTokenPool"):
        self.req_to_token_pool = pool
        self.token_to_kv_pool = None
        # Per-forward-step memoization keyed on the ForwardBatch identity,
        # mirroring ShortConvAttnBackend.init_forward_metadata.
        self._step_indices = {}  # id(forward_batch) -> device index tensor
        self._step_slot_ids = {}  # id(forward_batch) -> host list (extend only)
        self.extend_track_calls = []
        self.decode_track_calls = []

    def _resolve_indices(self, forward_batch):
        key = id(forward_batch)
        indices = self._step_indices.get(key)
        if indices is None:
            indices = self.req_to_token_pool.get_mamba_indices(
                forward_batch.req_pool_indices
            ).to(torch.long)
            self._step_indices[key] = indices
        return indices

    def _resolve_slot_ids(self, forward_batch, indices):
        key = id(forward_batch)
        slot_ids = self._step_slot_ids.get(key)
        if slot_ids is None:
            slot_ids = indices.tolist()
            self._step_slot_ids[key] = slot_ids
        return slot_ids

    def conv_state_metadata(self, layer_id, forward_batch):
        from sglang.srt.layers.attention.linear.short_conv_backend import (
            ShortConvMetadata,
        )

        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        indices = self._resolve_indices(forward_batch)  # already int64
        if forward_batch.forward_mode.is_decode_or_idle():
            return ShortConvMetadata(layer_cache=layer_cache, cache_indices=indices)

        slot_ids = self._resolve_slot_ids(forward_batch, indices)
        has_prefix = [int(p) > 0 for p in forward_batch.extend_prefix_lens_cpu]
        return ShortConvMetadata(
            layer_cache=layer_cache,
            cache_indices=indices,
            slot_ids_cpu=slot_ids,
            has_prefix_cpu=has_prefix,
        )

    # The radix mamba-cache track hooks. Inert here (these tests run the
    # no_buffer equivalent), but CCA calls them unconditionally, so the mock
    # must carry them and record that it was reached.
    def track_conv_states_extend(self, layer_cache, conv_inputs):
        self.extend_track_calls.append((layer_cache, tuple(conv_inputs)))

    def track_conv_states_decode(self, forward_batch):
        self.decode_track_calls.append(forward_batch)


@contextmanager
def _mock_pool_context(pool: _MockReqToTokenPool):
    """Install a mock ``ForwardContext`` whose ``attn_backend`` exposes both
    ``req_to_token_pool`` and ``conv_state_metadata`` over ``pool``."""
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        set_forward_context,
    )

    backend = _MockShortConvBackend(pool)
    ctx = ForwardContext(attn_backend=backend)
    prev = set_forward_context(ctx)
    try:
        yield backend
    finally:
        set_forward_context(prev)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_forward_batch(
    *,
    is_decode: bool,
    extend_seq_lens_cpu,
    extend_prefix_lens_cpu,
    req_pool_indices,
    input_ids: torch.Tensor,
):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    mode = ForwardMode.DECODE if is_decode else ForwardMode.EXTEND

    forward_batch = SimpleNamespace()
    forward_batch.forward_mode = mode
    forward_batch.input_ids = input_ids
    forward_batch.req_pool_indices = torch.as_tensor(
        req_pool_indices, dtype=torch.int32
    )
    forward_batch.extend_seq_lens_cpu = list(extend_seq_lens_cpu)
    forward_batch.extend_prefix_lens_cpu = list(extend_prefix_lens_cpu)
    return forward_batch


def _make_tiny_config(num_hidden_layers: int = 2):
    from sglang.srt.configs.zaya import ZayaConfig

    return ZayaConfig(
        hidden_size=16,
        ffn_hidden_size=32,
        num_hidden_layers=num_hidden_layers,
        num_experts=2,
        num_attention_heads=4,
        num_query_groups=2,
        num_key_value_heads=2,
        head_dim=8,
        cca_time0=2,
        cca_time1=2,
        max_position_embeddings=64,
        moe_router_topk=1,
        zaya_mlp_expansion=8,
        attention_bias=False,
    )


def _make_tiny_cca(
    seed: int = 0,
    tp_rank: Optional[int] = None,
    tp_size: Optional[int] = None,
    layer_id: int = 0,
    config=None,
):
    from sglang.srt.models.zaya import CCA

    if config is None:
        config = _make_tiny_config()
    torch.manual_seed(seed)
    cca = CCA(
        config=config,
        cca_num_k_heads=config.num_query_groups,
        cca_num_q_heads=config.num_attention_heads,
        hidden_size=config.hidden_size,
        head_dim=config.head_dim,
        cca_time0=config.cca_time0,
        cca_time1=config.cca_time1,
        layer_id=layer_id,
        tp_rank=tp_rank,
        tp_size=tp_size,
    )
    cca.eval()

    with torch.no_grad():
        for p in cca.parameters():
            p.data.normal_(mean=0.0, std=0.05)
        cca.temp.data.zero_()

    return cca, config


class TestZayaCCA(CustomTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_dist_initialized()

    def test_single_chunk_matches_reference(self):
        """A single-chunk extend with empty prefix matches the no-state path."""
        cca, config = _make_tiny_cca(seed=1)
        cca_ref, _ = _make_tiny_cca(seed=1)
        with torch.no_grad():
            cca_ref.load_state_dict(cca.state_dict())

        S = 5
        hs = torch.randn(S, cca.hidden_size, dtype=torch.float32) * 0.1

        q_ref, k_ref, v_ref = cca_ref._forward_no_state(hs)

        pool = _MockReqToTokenPool(pool_size=8, cca_config=config)
        fb = _make_forward_batch(
            is_decode=False,
            extend_seq_lens_cpu=[S],
            extend_prefix_lens_cpu=[0],
            req_pool_indices=[0],
            input_ids=torch.arange(S, dtype=torch.int64),
        )
        with _mock_pool_context(pool):
            q, k, v = cca.forward(hs, fb)

        torch.testing.assert_close(q, q_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(k, k_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(v, v_ref, atol=1e-5, rtol=1e-5)

    def test_chunked_prefill_across_request_boundary(self):
        """A resumed chunk must read the PROJECTED boundary value from its slot.

        Two requests in one extend batch: request 0 resumes a cached prefix,
        request 1 starts fresh. That is the mixed (non-``all_fresh``) path, where
        the lag row preceding each request's first token comes from a different
        place -- the pool slot for the resumed request, zero for the fresh one.

        This is the case that fails silently if a chunk boundary parks the raw
        hidden state while the read side expects the projected value: same shape,
        same dtype, no error, just wrong ``v`` on every resumed token. Both
        requests are compared against their own one-shot reference so a mix-up
        between the two slots also shows up.
        """
        cca, config = _make_tiny_cca(seed=31)
        cca_ref, _ = _make_tiny_cca(seed=31)
        with torch.no_grad():
            cca_ref.load_state_dict(cca.state_dict())

        A0, A1, S_b = 3, 4, 5
        S_a = A0 + A1
        torch.manual_seed(311)
        hs_a = torch.randn(S_a, cca.hidden_size, dtype=torch.float32) * 0.1
        hs_b = torch.randn(S_b, cca.hidden_size, dtype=torch.float32) * 0.1

        qa_ref, ka_ref, va_ref = cca_ref._forward_no_state(hs_a)
        qb_ref, kb_ref, vb_ref = cca_ref._forward_no_state(hs_b)

        pool = _MockReqToTokenPool(pool_size=8, cca_config=config)
        # Hold both batches alive: the mock memoizes its per-step slot mirror on
        # id(forward_batch), and a GC'd-then-reused address would false-collide.
        fb0 = _make_forward_batch(
            is_decode=False,
            extend_seq_lens_cpu=[A0],
            extend_prefix_lens_cpu=[0],
            req_pool_indices=[1],
            input_ids=torch.arange(A0, dtype=torch.int64),
        )
        fb1 = _make_forward_batch(
            is_decode=False,
            extend_seq_lens_cpu=[A1, S_b],
            extend_prefix_lens_cpu=[A0, 0],
            req_pool_indices=[1, 4],
            input_ids=torch.arange(A1 + S_b, dtype=torch.int64),
        )
        with _mock_pool_context(pool):
            # Chunk 0 of request 0 only.
            cca.forward(hs_a[:A0], fb0)
            # Chunk 1 of request 0 (resumes) batched with a fresh request 1.
            q, k, v = cca.forward(torch.cat([hs_a[A0:], hs_b], dim=0), fb1)

        torch.testing.assert_close(q[:A1], qa_ref[A0:], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k[:A1], ka_ref[A0:], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v[:A1], va_ref[A0:], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(q[A1:], qb_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k[A1:], kb_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v[A1:], vb_ref, atol=1e-4, rtol=1e-4)

    def test_chunked_prefill_then_decode_matches_full_sequence(self):
        """Two prefill chunks then decode steps, all against one reference.

        Extends the boundary case above past the prefill: after a resumed chunk
        the request keeps decoding, so a boundary row that is wrong in *kind*
        (raw instead of projected) would also poison the first decode step.
        """
        cca, config = _make_tiny_cca(seed=32)
        cca_ref, _ = _make_tiny_cca(seed=32)
        with torch.no_grad():
            cca_ref.load_state_dict(cca.state_dict())

        S0, S1, S2 = 3, 3, 2
        S_total = S0 + S1 + S2
        torch.manual_seed(321)
        hs = torch.randn(S_total, cca.hidden_size, dtype=torch.float32) * 0.1
        q_ref, k_ref, v_ref = cca_ref._forward_no_state(hs)

        pool = _MockReqToTokenPool(pool_size=8, cca_config=config)
        # ``batches`` keeps every ForwardBatch alive: the mock memoizes its
        # per-step slot mirror on id(forward_batch), so a recycled address would
        # hand a later step the wrong request list.
        batches = []
        for chunk, prefix in ((slice(0, S0), 0), (slice(S0, S0 + S1), S0)):
            n = chunk.stop - chunk.start
            batches.append(
                (
                    hs[chunk],
                    _make_forward_batch(
                        is_decode=False,
                        extend_seq_lens_cpu=[n],
                        extend_prefix_lens_cpu=[prefix],
                        req_pool_indices=[3],
                        input_ids=torch.arange(n, dtype=torch.int64),
                    ),
                )
            )
        for t in range(S2):
            idx = S0 + S1 + t
            batches.append(
                (
                    hs[idx : idx + 1],
                    _make_forward_batch(
                        is_decode=True,
                        extend_seq_lens_cpu=[],
                        extend_prefix_lens_cpu=[],
                        req_pool_indices=[3],
                        input_ids=torch.tensor([0], dtype=torch.int64),
                    ),
                )
            )

        outs = []
        with _mock_pool_context(pool):
            for chunk_hs, fb in batches:
                outs.append(cca.forward(chunk_hs, fb))

        for got, ref in zip(
            (torch.cat([o[i] for o in outs], dim=0) for i in range(3)),
            (q_ref, k_ref, v_ref),
        ):
            torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)

    def test_folded_prefill_then_decode_matches_the_two_stage_reference(self):
        """End-to-end with the folded decode conv, including the bias column.

        The other equivalence tests leave ``_decode_conv_folded`` False, so decode
        runs the real two-stage ``conv_qk`` and never touches the folded weight.
        This one folds the module under test and compares it against an UNFOLDED
        reference, so one assertion covers the fold itself, the bias now riding in
        a trailing weight column, and the state plumbing around them.

        Without a GPU the window arrives from the unfused gather/concat, so this
        exercises the ``F.pad`` arm that appends the constant-1.0 tap; the fused
        arm gets it from ``cca_state_step`` and is pinned on GPU instead.
        """
        cca, config = _make_tiny_cca(seed=35)
        cca_ref, _ = _make_tiny_cca(seed=35)
        with torch.no_grad():
            cca_ref.load_state_dict(cca.state_dict())
        cca.fold_decode_conv()
        self.assertTrue(cca._decode_conv_folded)
        self.assertFalse(cca_ref._decode_conv_folded)

        S0, S1 = 4, 3
        torch.manual_seed(351)
        hs = torch.randn(S0 + S1, cca.hidden_size, dtype=torch.float32) * 0.1
        q_ref, k_ref, v_ref = cca_ref._forward_no_state(hs)

        pool = _MockReqToTokenPool(pool_size=8, cca_config=config)
        batches = [
            (
                hs[:S0],
                _make_forward_batch(
                    is_decode=False,
                    extend_seq_lens_cpu=[S0],
                    extend_prefix_lens_cpu=[0],
                    req_pool_indices=[0],
                    input_ids=torch.arange(S0, dtype=torch.int64),
                ),
            )
        ] + [
            (
                hs[S0 + t : S0 + t + 1],
                _make_forward_batch(
                    is_decode=True,
                    extend_seq_lens_cpu=[],
                    extend_prefix_lens_cpu=[],
                    req_pool_indices=[0],
                    input_ids=torch.tensor([0], dtype=torch.int64),
                ),
            )
            for t in range(S1)
        ]
        outs = []
        with _mock_pool_context(pool):
            for chunk_hs, fb in batches:
                outs.append(cca.forward(chunk_hs, fb))

        for i, ref in enumerate((q_ref, k_ref, v_ref)):
            got = torch.cat([o[i] for o in outs], dim=0)
            torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)

    def test_projected_lag_is_narrower_than_the_hidden_state(self):
        """The whole point: conv[1] shrinks to the val_proj2 output width.

        Guards the arithmetic the pool sizing depends on -- that the cached
        quantity really is ``latent_k_dim / 2`` wide and that the model and the
        config agree on it. If these drift apart the pool entry and the value
        written into it disagree in width.
        """
        cca, config = _make_tiny_cca(seed=33)
        self.assertTrue(cca.cache_projected_v2)
        expected = (config.num_query_groups * config.head_dim) // 2
        self.assertEqual(cca.v2_lag_dim, expected)
        self.assertEqual(config.cca_v2_state_dim, expected)
        self.assertLess(cca.v2_lag_dim, cca.hidden_size)

    def test_biased_projection_falls_back_to_the_raw_hidden_state_lag(self):
        """With a bias, ``W . 0 != 0``, so the projected cache is refused.

        A freshly allocated slot is zero and the first token's ``val_proj2``
        input is defined to be zero. Caching the projection only reproduces that
        when there is no bias term; otherwise the zero slot would have to stand
        for ``b``. The gate must therefore fall back to caching the raw hidden
        state -- and the fallback must still be numerically right, which is what
        the reference comparison here pins.
        """
        from sglang.srt.configs.zaya import ZayaConfig

        cfg = _make_tiny_config()
        biased = ZayaConfig(
            hidden_size=cfg.hidden_size,
            ffn_hidden_size=cfg.ffn_hidden_size,
            num_hidden_layers=cfg.num_hidden_layers,
            num_experts=cfg.num_experts,
            num_attention_heads=cfg.num_attention_heads,
            num_query_groups=cfg.num_query_groups,
            num_key_value_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            cca_time0=cfg.cca_time0,
            cca_time1=cfg.cca_time1,
            max_position_embeddings=cfg.max_position_embeddings,
            moe_router_topk=cfg.moe_router_topk,
            zaya_mlp_expansion=cfg.zaya_mlp_expansion,
            attention_bias=True,
        )
        self.assertFalse(biased.cca_cache_projected_v2)
        self.assertEqual(biased.cca_v2_state_dim, biased.hidden_size)

        cca, _ = _make_tiny_cca(seed=34, config=biased)
        cca_ref, _ = _make_tiny_cca(seed=34, config=biased)
        with torch.no_grad():
            cca_ref.load_state_dict(cca.state_dict())
        self.assertFalse(cca.cache_projected_v2)
        self.assertEqual(cca.v2_lag_dim, biased.hidden_size)
        self.assertIsNotNone(cca.val_proj2.bias)

        S0, S1 = 4, 2
        torch.manual_seed(341)
        hs = torch.randn(S0 + S1, cca.hidden_size, dtype=torch.float32) * 0.1
        q_ref, k_ref, v_ref = cca_ref._forward_no_state(hs)

        pool = _MockReqToTokenPool(pool_size=8, cca_config=biased)
        self.assertEqual(pool.prev_hs_state.shape[-2], biased.hidden_size)
        # Held for the id(forward_batch) memo (see the chunked-prefill tests).
        batches = [
            (
                hs[:S0],
                _make_forward_batch(
                    is_decode=False,
                    extend_seq_lens_cpu=[S0],
                    extend_prefix_lens_cpu=[0],
                    req_pool_indices=[0],
                    input_ids=torch.arange(S0, dtype=torch.int64),
                ),
            )
        ] + [
            (
                hs[S0 + t : S0 + t + 1],
                _make_forward_batch(
                    is_decode=True,
                    extend_seq_lens_cpu=[],
                    extend_prefix_lens_cpu=[],
                    req_pool_indices=[0],
                    input_ids=torch.tensor([0], dtype=torch.int64),
                ),
            )
            for t in range(S1)
        ]
        outs = []
        with _mock_pool_context(pool):
            for chunk_hs, fb in batches:
                outs.append(cca.forward(chunk_hs, fb))

        for i, ref in enumerate((q_ref, k_ref, v_ref)):
            got = torch.cat([o[i] for o in outs], dim=0)
            torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)

    def test_prefill_then_decode_matches_full_sequence(self):
        """Prefill(S0) followed by ``S1`` single-token decode steps matches a
        one-shot reference over ``S0 + S1`` tokens."""
        cca, config = _make_tiny_cca(seed=2)
        cca_ref, _ = _make_tiny_cca(seed=2)
        with torch.no_grad():
            cca_ref.load_state_dict(cca.state_dict())

        S0, S1 = 4, 2
        S_total = S0 + S1
        torch.manual_seed(77)
        hs = torch.randn(S_total, cca.hidden_size, dtype=torch.float32) * 0.1

        q_ref, k_ref, v_ref = cca_ref._forward_no_state(hs)

        pool = _MockReqToTokenPool(pool_size=8, cca_config=config)
        with _mock_pool_context(pool):
            fb_prefill = _make_forward_batch(
                is_decode=False,
                extend_seq_lens_cpu=[S0],
                extend_prefix_lens_cpu=[0],
                req_pool_indices=[0],
                input_ids=torch.arange(S0, dtype=torch.int64),
            )
            q0, k0, v0 = cca.forward(hs[:S0], fb_prefill)

            q_decodes = [q0]
            k_decodes = [k0]
            v_decodes = [v0]
            for t in range(S1):
                fb_decode = _make_forward_batch(
                    is_decode=True,
                    extend_seq_lens_cpu=[],
                    extend_prefix_lens_cpu=[],
                    req_pool_indices=[0],
                    input_ids=torch.tensor([0], dtype=torch.int64),
                )
                qd, kd, vd = cca.forward(hs[S0 + t : S0 + t + 1], fb_decode)
                q_decodes.append(qd)
                k_decodes.append(kd)
                v_decodes.append(vd)

        q_cat = torch.cat(q_decodes, dim=0)
        k_cat = torch.cat(k_decodes, dim=0)
        v_cat = torch.cat(v_decodes, dim=0)

        torch.testing.assert_close(q_cat, q_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k_cat, k_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(v_cat, v_ref, atol=1e-4, rtol=1e-4)

    def test_batched_decode_matches_single_decode(self):
        """A two-request batched decode of request 0 must produce the same
        q / k / v tensors as a single-request decode of request 0."""
        cca_single, config = _make_tiny_cca(seed=11)
        cca_batched, _ = _make_tiny_cca(seed=11)
        with torch.no_grad():
            cca_batched.load_state_dict(cca_single.state_dict())

        S0 = 4
        torch.manual_seed(202)
        hs0 = torch.randn(S0, cca_single.hidden_size, dtype=torch.float32) * 0.1
        hs1 = torch.randn(S0, cca_single.hidden_size, dtype=torch.float32) * 0.1
        decode0 = torch.randn(cca_single.hidden_size, dtype=torch.float32) * 0.1
        decode1 = torch.randn(cca_single.hidden_size, dtype=torch.float32) * 0.1

        pool_single = _MockReqToTokenPool(pool_size=8, cca_config=config)
        with _mock_pool_context(pool_single):
            cca_single.forward(
                hs0,
                _make_forward_batch(
                    is_decode=False,
                    extend_seq_lens_cpu=[S0],
                    extend_prefix_lens_cpu=[0],
                    req_pool_indices=[0],
                    input_ids=torch.arange(S0, dtype=torch.int64),
                ),
            )
            q_solo, k_solo, v_solo = cca_single.forward(
                decode0.unsqueeze(0),
                _make_forward_batch(
                    is_decode=True,
                    extend_seq_lens_cpu=[],
                    extend_prefix_lens_cpu=[],
                    req_pool_indices=[0],
                    input_ids=torch.tensor([0], dtype=torch.int64),
                ),
            )

        pool_batched = _MockReqToTokenPool(pool_size=8, cca_config=config)
        with _mock_pool_context(pool_batched):
            cca_batched.forward(
                torch.cat([hs0, hs1], dim=0),
                _make_forward_batch(
                    is_decode=False,
                    extend_seq_lens_cpu=[S0, S0],
                    extend_prefix_lens_cpu=[0, 0],
                    req_pool_indices=[0, 1],
                    input_ids=torch.arange(2 * S0, dtype=torch.int64),
                ),
            )
            q_batch, k_batch, v_batch = cca_batched.forward(
                torch.stack([decode0, decode1], dim=0),
                _make_forward_batch(
                    is_decode=True,
                    extend_seq_lens_cpu=[],
                    extend_prefix_lens_cpu=[],
                    req_pool_indices=[0, 1],
                    input_ids=torch.tensor([0, 1], dtype=torch.int64),
                ),
            )

        torch.testing.assert_close(q_batch[0:1], q_solo, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(k_batch[0:1], k_solo, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(v_batch[0:1], v_solo, atol=1e-5, rtol=1e-5)

    def test_two_requests_state_isolation(self):
        """A batched prefill of two requests must update only the requests'
        own slots in the centralized pool."""
        cca, config = _make_tiny_cca(seed=4)

        S0, S1 = 3, 2
        hs0 = torch.randn(S0, cca.hidden_size, dtype=torch.float32) * 0.1
        hs1 = torch.randn(S1, cca.hidden_size, dtype=torch.float32) * 0.1
        hs = torch.cat([hs0, hs1], dim=0)

        pool = _MockReqToTokenPool(pool_size=8, cca_config=config)
        fb = _make_forward_batch(
            is_decode=False,
            extend_seq_lens_cpu=[S0, S1],
            extend_prefix_lens_cpu=[0, 0],
            req_pool_indices=[2, 5],
            input_ids=torch.arange(S0 + S1, dtype=torch.int64),
        )
        with _mock_pool_context(pool):
            cca.forward(hs, fb)

        layer_cache = pool.mamba2_layer_cache(0)
        conv_state = layer_cache.conv[0]
        lag_state = layer_cache.conv[1]

        self.assertTrue(torch.any(conv_state[2] != 0))
        self.assertTrue(torch.any(conv_state[5] != 0))

        # conv[1] holds the PROJECTED boundary value ``val_proj2 . hs[-1]``, not
        # the raw hidden state. Pinning the projected quantity is what catches a
        # chunk boundary that parks the wrong tensor: the widths agree with the
        # raw hidden state only by accident of a tiny config, and a mismatch
        # there degrades resumed prefixes silently rather than raising.
        self.assertTrue(cca.cache_projected_v2)
        self.assertEqual(lag_state.shape[-2], config.cca_v2_state_dim)
        with torch.no_grad():
            expected0 = cca.val_proj2(hs0[-1:])[0]
            expected1 = cca.val_proj2(hs1[-1:])[0]
        torch.testing.assert_close(
            lag_state[2].squeeze(-1).to(torch.float32),
            expected0.squeeze(0).to(torch.float32),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            lag_state[5].squeeze(-1).to(torch.float32),
            expected1.squeeze(0).to(torch.float32),
            atol=1e-5,
            rtol=1e-5,
        )

        for idx in (0, 1, 3, 4):
            self.assertTrue(torch.all(conv_state[idx] == 0))
            self.assertTrue(torch.all(lag_state[idx] == 0))

    def test_mamba_indices_resolved_once_per_forward_step(self):
        """The req -> MambaPool-slot mapping is identical for every CCA layer in
        a step, so it (and its GPU->CPU ``.tolist()`` sync) must be resolved once
        per forward step and shared across layers, not recomputed per layer.

        Regression guard for the per-layer mamba-sync fix: two CCA layers driven
        by a single ForwardBatch must trigger exactly one ``get_mamba_indices``
        lookup and one host materialization for the whole step.
        """

        class _CountingPool(_MockReqToTokenPool):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.get_mamba_indices_calls = 0

            def get_mamba_indices(self, req_pool_indices):
                self.get_mamba_indices_calls += 1
                return super().get_mamba_indices(req_pool_indices)

        # num_hidden_layers=4 -> CCA (even) layers live at ids 0 and 2.
        config = _make_tiny_config(num_hidden_layers=4)
        self.assertEqual(config.linear_layer_ids, [0, 2])
        cca0, _ = _make_tiny_cca(seed=5, layer_id=0, config=config)
        cca2, _ = _make_tiny_cca(seed=6, layer_id=2, config=config)

        S = 4
        hs = torch.randn(S, config.hidden_size, dtype=torch.float32) * 0.1

        def _fresh_fb():
            return _make_forward_batch(
                is_decode=False,
                extend_seq_lens_cpu=[S],
                extend_prefix_lens_cpu=[0],
                req_pool_indices=[0],
                input_ids=torch.arange(S, dtype=torch.int64),
            )

        pool = _CountingPool(pool_size=8, cca_config=config)
        with _mock_pool_context(pool) as backend:
            fb = _fresh_fb()
            cca0.forward(hs, fb)
            cca2.forward(hs, fb)

            # Two CCA layers, one forward step -> one shared lookup, both the
            # device tensor and its host mirror memoized once per step on the
            # backend (ShortConvAttnBackend does this in init_forward_metadata).
            self.assertEqual(pool.get_mamba_indices_calls, 1)
            self.assertIn(id(fb), backend._step_indices)
            self.assertEqual(backend._step_slot_ids[id(fb)], [0])

            # A new forward step (fresh ForwardBatch) resolves the mapping again.
            cca0.forward(hs, _fresh_fb())
            self.assertEqual(pool.get_mamba_indices_calls, 2)

    def test_decode_path_does_not_sync_indices_to_host(self):
        """The decode path indexes the pool entirely on-device, so it must not
        populate the host-side index cache (keeping it CUDA-graph friendly)."""
        cca, config = _make_tiny_cca(seed=7)

        pool = _MockReqToTokenPool(pool_size=8, cca_config=config)
        with _mock_pool_context(pool) as backend:
            # Keep a reference to the extend batch so its id() cannot be recycled
            # by the later decode batch (the mock keys its per-step memo on
            # id(forward_batch); a GC'd-then-reused address would false-collide).
            fb_extend = _make_forward_batch(
                is_decode=False,
                extend_seq_lens_cpu=[3],
                extend_prefix_lens_cpu=[0],
                req_pool_indices=[0],
                input_ids=torch.arange(3, dtype=torch.int64),
            )
            cca.forward(
                torch.randn(3, config.hidden_size, dtype=torch.float32) * 0.1,
                fb_extend,
            )
            fb_decode = _make_forward_batch(
                is_decode=True,
                extend_seq_lens_cpu=[],
                extend_prefix_lens_cpu=[],
                req_pool_indices=[0],
                input_ids=torch.tensor([0], dtype=torch.int64),
            )
            cca.forward(
                torch.randn(1, config.hidden_size, dtype=torch.float32) * 0.1,
                fb_decode,
            )

            # Decode resolves device indices, but the host ``.tolist()`` mirror
            # is only built by the extend path -- so the decode step stays
            # entirely on-device (CUDA-graph friendly).
            self.assertIn(id(fb_decode), backend._step_indices)
            self.assertNotIn(id(fb_decode), backend._step_slot_ids)


class TestCCAStateStepKernel(CustomTestCase):
    """The fused decode state step must be bit-identical to the torch chain.

    Derived property. ``cca_state_step`` re-derives the conv history shift
    (``new[w] = old[w+1]``, last tap = this token) and the read-before-overwrite
    ordering of the ``prev_hs`` gather/scatter, while mutating the pools in place.
    An off-by-one in the shift, or writing the slot before reading it, corrupts
    only the *next* step for that request -- which surfaces as gradual output
    drift rather than an error, so pin exactness on both the returned tensors and
    the mutated pools.

    Requires a GPU (Triton); on CPU ``covered()`` selects the torch chain, which
    the rest of this file already exercises.
    """

    @unittest.skipUnless(torch.cuda.is_available(), "fused kernel requires a GPU")
    def test_bit_identical_to_torch_chain(self):
        from sglang.kernels.ops.attention import cca_state_step as kernel

        dev = "cuda"
        torch.manual_seed(5)
        # total_padding 3 exercises a shift longer than ZAYA1's 2, where an
        # off-by-one in the history roll would otherwise be invisible.
        # (600, 1100) spans several channel AND hidden tiles of the 2-D grid
        # (BLOCK_C=256, BLOCK_H=512): with the small shapes alone every launch is
        # a single tile per axis, so a wrong tile offset would never show up. 40
        # tokens likewise puts more than one token behind each tile index.
        shapes = ((64, 32, 2), (48, 16, 3), (600, 1100, 2))
        for num_channels, hidden_size, total_padding in shapes:
            for num_tokens in (1, 6, 40):
                with self.subTest(
                    c=num_channels, h=hidden_size, p=total_padding, t=num_tokens
                ):
                    slots = (
                        torch.randperm(63, device=dev)[:num_tokens].to(torch.long) + 1
                    )
                    qk = torch.randn(
                        num_tokens, num_channels, device=dev, dtype=torch.bfloat16
                    )
                    hs = torch.randn(
                        num_tokens, hidden_size, device=dev, dtype=torch.bfloat16
                    )
                    conv0 = torch.randn(
                        64,
                        num_channels,
                        total_padding,
                        device=dev,
                        dtype=torch.bfloat16,
                    )
                    prev0 = torch.randn(
                        64, hidden_size, 1, device=dev, dtype=torch.bfloat16
                    )

                    conv_ref, prev_ref = conv0.clone(), prev0.clone()
                    conv_got, prev_got = conv0.clone(), prev0.clone()

                    with torch.no_grad():
                        left = conv_ref.index_select(0, slots).to(hs.dtype)
                        window_ref = torch.cat([left, qk.unsqueeze(-1)], dim=-1)
                        conv_ref.index_copy_(
                            0,
                            slots,
                            window_ref[..., -total_padding:].to(conv_ref.dtype),
                        )
                        prevhs_ref = (
                            prev_ref.index_select(0, slots).squeeze(-1).to(hs.dtype)
                        )
                        prev_ref.index_copy_(
                            0, slots, hs.unsqueeze(-1).to(prev_ref.dtype)
                        )

                        self.assertTrue(
                            kernel.covered(
                                qk, hs, conv_got, prev_got, slots, total_padding
                            )
                        )
                        window_got, prevhs_got = kernel.cca_state_step(
                            qk, hs, conv_got, prev_got, slots, total_padding
                        )

                    # Bit-exact: the kernel only moves data, no arithmetic.
                    self.assertTrue(torch.equal(window_got, window_ref))
                    self.assertTrue(torch.equal(prevhs_got, prevhs_ref))
                    self.assertTrue(torch.equal(conv_got, conv_ref))
                    self.assertTrue(torch.equal(prev_got, prev_ref))

    @unittest.skipUnless(torch.cuda.is_available(), "fused kernel requires a GPU")
    def test_lag_free_variant_matches_the_conv_only_chain(self):
        """``HAS_LAG=False`` must leave the conv half byte-identical.

        A rank whose V heads all come from ``val_proj1`` passes no lag tensors, so
        the lag block compiles out. The conv window and the conv-history shift
        must be unchanged by that -- a constexpr that accidentally gated the conv
        loop too would return plausible-looking zeros.
        """
        from sglang.kernels.ops.attention import cca_state_step as kernel

        dev = "cuda"
        torch.manual_seed(7)
        num_channels, total_padding, num_tokens = 48, 2, 5
        slots = torch.randperm(32, device=dev)[:num_tokens].to(torch.long) + 1
        qk = torch.randn(num_tokens, num_channels, device=dev, dtype=torch.bfloat16)
        conv0 = torch.randn(
            64, num_channels, total_padding, device=dev, dtype=torch.bfloat16
        )
        prev0 = torch.randn(64, 16, 1, device=dev, dtype=torch.bfloat16)
        hs = torch.randn(num_tokens, 16, device=dev, dtype=torch.bfloat16)

        conv_ref, prev_ref = conv0.clone(), prev0.clone()
        conv_got, prev_got = conv0.clone(), prev0.clone()
        with torch.no_grad():
            # Reference: run WITH the lag, then discard the lag results.
            self.assertTrue(
                kernel.covered(qk, hs, conv_ref, prev_ref, slots, total_padding)
            )
            window_ref, _ = kernel.cca_state_step(
                qk, hs, conv_ref, prev_ref, slots, total_padding
            )
            self.assertTrue(
                kernel.covered(qk, None, conv_got, None, slots, total_padding)
            )
            window_got, lag_got = kernel.cca_state_step(
                qk, None, conv_got, None, slots, total_padding
            )

        self.assertIsNone(lag_got)
        self.assertTrue(torch.equal(window_got, window_ref))
        self.assertTrue(torch.equal(conv_got, conv_ref))
        # The lag pool the lag-free call was never handed stays untouched.
        self.assertTrue(torch.equal(prev_got, prev0))

    def test_uncovered_inputs_fall_back(self):
        # covered() gates an in-place pool mutation, so its negative branches
        # matter more than usual: a mismatched dtype would have the kernel write
        # raw bits into the pool.
        from sglang.kernels.ops.attention import cca_state_step as kernel

        qk = torch.randn(4, 8)
        hs = torch.randn(4, 6)
        conv = torch.randn(16, 8, 2)
        prev = torch.randn(16, 6, 1)
        slots = torch.arange(4)
        # CPU tensors are not served.
        self.assertFalse(kernel.covered(qk, hs, conv, prev, slots, 2))
        # Missing slot indices (before the backend resolves them).
        self.assertFalse(kernel.covered(qk, hs, conv, prev, None, 2))
        # A single tap leaves no history to shift.
        self.assertFalse(kernel.covered(qk, hs, conv, prev, slots, 0))
        # Pool dtype must match the value written into it.
        self.assertFalse(
            kernel.covered(qk.to(torch.bfloat16), hs, conv, prev, slots, 2)
        )
        # Same for the lag pool: it is written from lag_now with no conversion.
        self.assertFalse(
            kernel.covered(qk, hs, conv, prev.to(torch.bfloat16), slots, 2)
        )
        # The lag stream is either fully specified or fully absent. A
        # half-specified pair must be refused rather than guessed at: guessing
        # "no lag" skips the pool write and the next step reads a stale value,
        # guessing "lag" writes the wrong width into the pool. Neither raises.
        self.assertFalse(kernel.covered(qk, hs, conv, None, slots, 2))
        self.assertFalse(kernel.covered(qk, None, conv, prev, slots, 2))
        # Widths must agree between the value and the pool entry it lands in.
        self.assertFalse(kernel.covered(qk, torch.randn(4, 5), conv, prev, slots, 2))


class TestCCAQKMixKernel(CustomTestCase):
    """The fused q/k head-mix kernel must match the torch chain it replaces.

    Derived property. ``cca_qk_mix`` collapses ``_add_grouped_qk_means`` +
    ``_normalize_qk`` into one kernel, re-deriving the GQA group indexing
    (q head == g * gqa_groups + j), the 0.5 blend weights, the per-k-head
    temperature and the two RMS normalizations from scratch. Any of those can be
    subtly wrong -- a transposed group index or a temperature applied to q
    instead of k still produces plausible tensors -- so pin the equivalence.

    Requires a GPU (Triton); the CPU suite exercises the torch fallback instead,
    which ``covered()`` selects when the folded scale vector is absent.
    """

    def _run(
        self,
        num_q_heads: int,
        num_k_heads: int,
        num_tokens: int,
        head_dim: int = 32,
    ):
        import torch as _torch

        from sglang.kernels.ops.attention import cca_qk_mix as kernel

        dev = "cuda"
        _torch.manual_seed(11)
        cca = _make_tiny_cca(seed=2)[0]
        # Drive the reference through the real module helpers so the test tracks
        # them rather than a re-derivation of the same formula.
        cca.num_q_heads = num_q_heads
        cca.num_k_heads = num_k_heads
        cca.gqa_groups = num_q_heads // num_k_heads
        cca.head_dim = head_dim
        cca.sqrt_head_dim = head_dim**0.5
        cca.clamp_temp = False
        cca.temp = torch.nn.Parameter(_torch.rand(num_k_heads) + 0.5)

        conv_qk = (
            _torch.randn(
                num_tokens,
                (num_q_heads + num_k_heads) * head_dim,
                dtype=_torch.bfloat16,
            )
            * 0.3
        )
        pre_q = (
            _torch.randn(num_tokens, num_q_heads * head_dim, dtype=_torch.bfloat16)
            * 0.3
        )
        base_k = (
            _torch.randn(num_tokens, num_k_heads * head_dim, dtype=_torch.bfloat16)
            * 0.3
        )

        with _torch.no_grad():
            q_ref, k_ref = cca._add_grouped_qk_means(
                conv_qk[:, : num_q_heads * head_dim].view(
                    num_tokens, num_q_heads, head_dim
                ),
                conv_qk[:, num_q_heads * head_dim :].view(
                    num_tokens, num_k_heads, head_dim
                ),
                pre_q.view(num_tokens, num_q_heads, head_dim),
                base_k.view(num_tokens, num_k_heads, head_dim),
            )
            q_ref, k_ref = cca._normalize_qk(q_ref, k_ref)

            k_scale = (cca.temp.detach().float() * cca.sqrt_head_dim).to(dev)
            args = (conv_qk.to(dev), pre_q.to(dev), base_k.to(dev), k_scale)
            self.assertTrue(
                kernel.covered(*args, num_q_heads, num_k_heads, head_dim),
                "kernel should cover these shapes",
            )
            q_got, k_got = kernel.cca_qk_mix(
                *args,
                num_q_heads=num_q_heads,
                num_k_heads=num_k_heads,
                head_dim=head_dim,
                q_scale=cca.sqrt_head_dim,
            )

        torch.testing.assert_close(
            q_got.cpu(), q_ref, rtol=2e-3, atol=2e-3, check_dtype=False
        )
        torch.testing.assert_close(
            k_got.cpu(), k_ref, rtol=2e-3, atol=2e-3, check_dtype=False
        )

    @unittest.skipUnless(torch.cuda.is_available(), "fused kernel requires a GPU")
    def test_matches_torch_chain_across_gqa_shapes(self):
        _ensure_dist_initialized()
        # 8:1 is ZAYA1-74B at attn_tp=2; 4:1 is 8B at tp=1; 1:1 exercises the
        # degenerate group where the k blend reduces to a single q head. 3:1 is
        # the only one whose group is not a power of two, so it is the case where
        # the [G, HD] tile is padded and the masked rows must contribute nothing
        # to either reduction.
        for num_q_heads, num_k_heads in ((8, 1), (8, 2), (2, 2), (6, 2)):
            for num_tokens in (1, 5):
                with self.subTest(q=num_q_heads, k=num_k_heads, t=num_tokens):
                    self._run(num_q_heads, num_k_heads, num_tokens)

    @unittest.skipUnless(torch.cuda.is_available(), "fused kernel requires a GPU")
    def test_matches_torch_chain_at_the_serving_shape(self):
        """head_dim 128 and a decode-sized batch, i.e. the launch config in prod.

        The other case pins the algebra at head_dim 32, which fits one ROCm
        wavefront twice over; 128 is what ZAYA1 actually runs and is the shape the
        block size and warp count are chosen for, so any reduction that only
        happens to be right at 32 lanes shows up here. 40 tokens puts several
        programs on every CU instead of one.
        """
        _ensure_dist_initialized()
        for num_tokens in (1, 40):
            with self.subTest(t=num_tokens):
                self._run(8, 1, num_tokens, head_dim=128)

    def test_uncovered_inputs_fall_back(self):
        # covered() is the only thing standing between an unsupported shape and a
        # wrong-answer kernel launch, so its negative branches must hold. Most
        # importantly a missing scale vector (weights not folded yet) must report
        # False so the torch path runs.
        from sglang.kernels.ops.attention import cca_qk_mix as kernel

        conv_qk = torch.randn(4, 9 * 32)
        pre_q = torch.randn(4, 8 * 32)
        base_k = torch.randn(4, 1 * 32)
        scale = torch.ones(1)
        # No folded scales yet.
        self.assertFalse(kernel.covered(conv_qk, pre_q, base_k, None, 8, 1, 32))
        # CPU tensors are not served by the Triton path.
        self.assertFalse(kernel.covered(conv_qk, pre_q, base_k, scale, 8, 1, 32))
        # Head dim beyond one block.
        self.assertFalse(kernel.covered(conv_qk, pre_q, base_k, scale, 8, 1, 4096))
        # q heads not divisible by k heads.
        self.assertFalse(kernel.covered(conv_qk, pre_q, base_k, scale, 8, 3, 32))
        # A group too wide to hold as one [G, HD] register tile.
        wide_q = torch.randn(4, 64 * 32)
        wide_conv = torch.randn(4, 65 * 32)
        self.assertFalse(kernel.covered(wide_conv, wide_q, base_k, scale, 64, 1, 32))


class TestCCAQKMixRope(CustomTestCase):
    """The fused partial-rotary RoPE inside ``cca_qk_mix``.

    ZAYA1 runs ``partial_rotary_factor=0.5`` with ``head_dim=128``, so the cos/sin
    cache is ``[max_pos, 64]``. That is the exact shape the shared fused rope
    path refuses (``rope_cache`` derives ``d_freq = cos_sin.shape[-1] // 2 == 32``
    and asserts it equals ``d`` or ``d // 2``), which is why the rotation lives in
    ``cca_qk_mix`` instead. This class pins that it computes the same thing as the
    ``_normalize_qk`` -> ``rotary_emb`` chain it replaces.

    Not bit-identical, deliberately: the chain rounds q/k to bf16 before rotating
    and (on ROCm) reads a bf16 cos/sin cache, while the fused kernel keeps the
    normalized head in fp32 across the rotation and rounds once at the store. The
    bf16 comparison is therefore ``allclose`` at bf16 tolerance, and a separate
    fp64 reference shows which side the difference falls on: the fused result must
    be at least as close to exact as the chain.
    """

    @classmethod
    def setUpClass(cls) -> None:
        _ensure_dist_initialized()
        from sglang.srt.runtime_context import get_context

        cls._server_args_override = get_context().override_server_args(
            model_path="dummy"
        )
        cls._server_args_override.install()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._server_args_override.restore()

    # -- reference chains ---------------------------------------------------

    @staticmethod
    def _reference_chain(cca, conv_qk, pre_q, base_k, rotary_emb, positions, dtype):
        """``_add_grouped_qk_means`` -> ``_normalize_qk`` -> cast -> ``rotary_emb``.

        Exactly what ``ZayaAttention.forward`` did before the fusion, driven
        through the real module helpers so the test tracks them.
        """
        T = conv_qk.shape[0]
        nq, nk, hd = cca.num_q_heads, cca.num_k_heads, cca.head_dim
        with torch.no_grad():
            q, k = cca._add_grouped_qk_means(
                conv_qk[:, : nq * hd].view(T, nq, hd),
                conv_qk[:, nq * hd :].view(T, nk, hd),
                pre_q.view(T, nq, hd),
                base_k.view(T, nk, hd),
            )
            q, k = cca._normalize_qk(q, k)
            q = q.to(dtype).flatten(1).contiguous()
            k = k.to(dtype).flatten(1).contiguous()
            q, k = rotary_emb(positions, q, k)
        return q.view(T, nq, hd), k.view(T, nk, hd)

    @staticmethod
    def _reference_fp64(cca, conv_qk, pre_q, base_k, cos_sin_cache, positions):
        """The same function evaluated in fp64 from the stored cos/sin cache.

        The cache is used as stored (bf16 on ROCm) because both the fused kernel
        and the chain read that same table -- its quantization is common-mode and
        is not what this reference measures.
        """
        T = conv_qk.shape[0]
        nq, nk, hd = cca.num_q_heads, cca.num_k_heads, cca.head_dim
        g = cca.gqa_groups
        eps = 1e-12
        cq = conv_qk[:, : nq * hd].reshape(T, nk, g, hd).double()
        ck = conv_qk[:, nq * hd :].reshape(T, nk, hd).double()
        pq = pre_q.reshape(T, nk, g, hd).double()
        bk = base_k.reshape(T, nk, hd).double()

        q = cq + 0.5 * pq + 0.5 * bk.unsqueeze(-2)
        k = ck + 0.5 * pq.mean(dim=-2) + 0.5 * bk
        sqrt_hd = float(cca.sqrt_head_dim)
        q = q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + eps) * sqrt_hd
        k = k * torch.rsqrt(k.pow(2).sum(-1, keepdim=True) + eps) * sqrt_hd
        temp = cca.temp.detach().double().view(1, nk, 1)
        k = k * temp
        q = q.reshape(T, nq, hd)

        rot = cos_sin_cache.shape[-1]
        half = rot // 2
        cs = cos_sin_cache.double()[positions.long()]
        cos, sin = cs[:, :half], cs[:, half:]

        def _rot(x):
            lo = x[..., :half]
            hi = x[..., half:rot]
            c = cos.unsqueeze(1)
            s = sin.unsqueeze(1)
            return torch.cat([lo * c - hi * s, hi * c + lo * s, x[..., rot:]], dim=-1)

        return _rot(q), _rot(k)

    # -- the fused run ------------------------------------------------------

    def _run(self, *, rope_base: int, num_tokens: int, dtype=torch.bfloat16):
        from sglang.kernels.ops.attention import cca_qk_mix as kernel
        from sglang.srt.layers.rotary_embedding import get_rope
        from sglang.srt.models.zaya import CCARope

        dev = "cuda"
        num_q_heads, num_k_heads, head_dim = 8, 1, 128
        torch.manual_seed(7)

        cca = _make_tiny_cca(seed=3)[0]
        cca.num_q_heads = num_q_heads
        cca.num_k_heads = num_k_heads
        cca.gqa_groups = num_q_heads // num_k_heads
        cca.head_dim = head_dim
        cca.sqrt_head_dim = head_dim**0.5
        cca.clamp_temp = False
        cca.temp = torch.nn.Parameter(torch.rand(num_k_heads) + 0.5)

        rotary_emb = get_rope(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position=256,
            base=rope_base,
            is_neox_style=True,
            partial_rotary_factor=0.5,
        )
        # ZAYA1's split: the cache is half the head dim wide, which is exactly
        # what the shared fused-rope kernel cannot express.
        self.assertEqual(rotary_emb.rotary_dim, head_dim // 2)
        self.assertEqual(rotary_emb.cos_sin_cache.shape[-1], head_dim // 2)

        conv_qk = (
            torch.randn(num_tokens, (num_q_heads + num_k_heads) * head_dim, dtype=dtype)
            * 0.3
        )
        pre_q = torch.randn(num_tokens, num_q_heads * head_dim, dtype=dtype) * 0.3
        base_k = torch.randn(num_tokens, num_k_heads * head_dim, dtype=dtype) * 0.3
        positions = torch.arange(num_tokens, dtype=torch.int64)

        rotary_dev = rotary_emb.to(dev)
        cache = rotary_dev.cos_sin_cache
        q_ref, k_ref = self._reference_chain(
            cca,
            conv_qk.to(dev),
            pre_q.to(dev),
            base_k.to(dev),
            rotary_dev,
            positions.to(dev),
            dtype,
        )
        q_f64, k_f64 = self._reference_fp64(
            cca,
            conv_qk.to(dev).float(),
            pre_q.to(dev).float(),
            base_k.to(dev).float(),
            cache,
            positions.to(dev),
        )

        rope = CCARope.of(rotary_dev, positions.to(dev))
        self.assertIsNotNone(rope, "the plain RotaryEmbedding must be fusable")
        self.assertTrue(
            kernel.rope_covered(
                rope.positions,
                rope.cos_sin_cache,
                rope.rotary_dim,
                head_dim=head_dim,
                num_tokens=num_tokens,
                is_neox_style=rope.is_neox_style,
                device=torch.device(dev),
            )
        )

        k_scale = (cca.temp.detach().float() * cca.sqrt_head_dim).to(dev)
        q_got, k_got = kernel.cca_qk_mix(
            conv_qk.to(dev),
            pre_q.to(dev),
            base_k.to(dev),
            k_scale,
            num_q_heads=num_q_heads,
            num_k_heads=num_k_heads,
            head_dim=head_dim,
            q_scale=cca.sqrt_head_dim,
            out_dtype=dtype,
            positions=rope.positions,
            cos_sin_cache=rope.cos_sin_cache,
            rotary_dim=rope.rotary_dim,
        )

        # bf16 equivalence with the chain. The elements are O(1) (RMS-normalized
        # then scaled by sqrt(head_dim), so the per-element RMS is 1), and the
        # chain rounds twice more than the fused path does, so the gap is a
        # couple of bf16 ulps: 2 * 2^-8 ~= 8e-3 absolute.
        tol = 8e-3 if dtype is torch.bfloat16 else 1e-5
        torch.testing.assert_close(q_got, q_ref, rtol=tol, atol=tol)
        torch.testing.assert_close(k_got, k_ref, rtol=tol, atol=tol)

        # Direction of the error: the fused path must be at least as close to the
        # fp64 evaluation as the chain it replaces, because it carries the
        # rotation in fp32 and rounds once instead of three times.
        for got, ref, exact, name in (
            (q_got, q_ref, q_f64, "q"),
            (k_got, k_ref, k_f64, "k"),
        ):
            err_fused = (got.double() - exact).abs().max().item()
            err_chain = (ref.double() - exact).abs().max().item()
            self.assertLessEqual(
                err_fused,
                err_chain * 1.05 + 1e-6,
                f"fused {name} drifted further from fp64 than the chain "
                f"({err_fused:.3e} vs {err_chain:.3e})",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "fused kernel requires a GPU")
    def test_matches_the_unfused_chain_on_both_rope_bases(self):
        # ZAYA1-74B builds two caches and picks per layer: rope_theta 1e7 for the
        # full-attention layers, swa_rotary_base 10000 for the sliding ones. The
        # fused path must read the layer's OWN cache, and a base mix-up is
        # invisible at position 0 -- so exercise both, over several positions.
        for rope_base in (10_000, 10_000_000):
            for num_tokens in (1, 33):
                with self.subTest(base=rope_base, t=num_tokens):
                    self._run(rope_base=rope_base, num_tokens=num_tokens)

    @unittest.skipUnless(torch.cuda.is_available(), "fused kernel requires a GPU")
    def test_fp32_output_matches_the_chain_tightly(self):
        # With an fp32 store the only remaining difference from the chain is the
        # intermediate rounding it does and this path does not, so the gap
        # collapses by three orders of magnitude.
        self._run(rope_base=10_000_000, num_tokens=17, dtype=torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "fused kernel requires a GPU")
    def test_the_two_layer_caches_give_distinct_results(self):
        # Guard against the fused path latching a model-wide cache pointer: with
        # different bases the same positions must produce different rotations.
        from sglang.kernels.ops.attention import cca_qk_mix as kernel
        from sglang.srt.layers.rotary_embedding import get_rope

        head_dim, nq, nk, T = 128, 8, 1, 8
        torch.manual_seed(5)
        conv_qk = torch.randn(T, (nq + nk) * head_dim, dtype=torch.bfloat16).cuda()
        pre_q = torch.randn(T, nq * head_dim, dtype=torch.bfloat16).cuda()
        base_k = torch.randn(T, nk * head_dim, dtype=torch.bfloat16).cuda()
        positions = torch.arange(1, T + 1, dtype=torch.int64).cuda()
        scale = torch.ones(nk, device="cuda")

        outs = []
        for base in (10_000, 10_000_000):
            rope = get_rope(
                head_size=head_dim,
                rotary_dim=head_dim,
                max_position=256,
                base=base,
                is_neox_style=True,
                partial_rotary_factor=0.5,
            ).cuda()
            outs.append(
                kernel.cca_qk_mix(
                    conv_qk,
                    pre_q,
                    base_k,
                    scale,
                    num_q_heads=nq,
                    num_k_heads=nk,
                    head_dim=head_dim,
                    q_scale=head_dim**0.5,
                    out_dtype=torch.bfloat16,
                    positions=positions,
                    cos_sin_cache=rope.cos_sin_cache,
                    rotary_dim=rope.rotary_dim,
                )
            )
        self.assertFalse(torch.equal(outs[0][0], outs[1][0]))
        self.assertFalse(torch.equal(outs[0][1], outs[1][1]))


class TestCCAQKMixRopeCoverage(CustomTestCase):
    """``rope_covered`` is the only thing between an unsupported head shape and a
    silently wrong rotation, so pin its negative branches.

    These run without a GPU: the head-geometry half is split out as
    ``rope_geometry_covered`` precisely so it can be checked on CPU, and the
    device half is checked by handing ``rope_covered`` CPU tensors.
    """

    def test_geometry_rejects_a_non_half_rotary_dim(self):
        from sglang.kernels.ops.attention import cca_qk_mix as kernel

        # ZAYA1's shape: partial_rotary_factor 0.5 over head_dim 128.
        self.assertTrue(kernel.rope_geometry_covered(64, 128))
        # Full rotary (the common case for other models) is NOT covered: the
        # pass-through tail would be empty and the tile widths no longer match.
        self.assertFalse(kernel.rope_geometry_covered(128, 128))
        # A quarter-rotary split leaves a 3/4-width tail.
        self.assertFalse(kernel.rope_geometry_covered(32, 128))
        # And a rotary wider than the head is nonsense.
        self.assertFalse(kernel.rope_geometry_covered(192, 128))
        self.assertFalse(kernel.rope_geometry_covered(0, 128))

    def test_geometry_rejects_an_odd_half(self):
        from sglang.kernels.ops.attention import cca_qk_mix as kernel

        # head_dim 12 -> rotary_dim 6 -> half 3, an odd per-lane half.
        self.assertFalse(kernel.rope_geometry_covered(6, 12))
        # head_dim 20 -> rotary_dim 10 -> half 5.
        self.assertFalse(kernel.rope_geometry_covered(10, 20))
        # The neighbouring even halves are fine.
        self.assertTrue(kernel.rope_geometry_covered(8, 16))
        self.assertTrue(kernel.rope_geometry_covered(4, 8))

    def test_geometry_rejects_gptj_layout(self):
        from sglang.kernels.ops.attention import cca_qk_mix as kernel

        # GPT-J interleaves the rotated pair inside a lane -- the cross-lane case
        # the three-tile split exists to avoid.
        self.assertFalse(kernel.rope_geometry_covered(64, 128, is_neox_style=False))

    def test_rope_covered_rejects_cpu_and_malformed_inputs(self):
        from sglang.kernels.ops.attention import cca_qk_mix as kernel

        hd, rot, T = 128, 64, 4
        pos = torch.arange(T, dtype=torch.int64)
        cache = torch.randn(256, rot)
        common = dict(head_dim=hd, num_tokens=T)

        # Missing either argument (no rope offered at all).
        self.assertFalse(kernel.rope_covered(None, cache, rot, **common))
        self.assertFalse(kernel.rope_covered(pos, None, rot, **common))
        # CPU tensors are not served by the Triton path.
        self.assertFalse(kernel.rope_covered(pos, cache, rot, **common))
        # A cache whose last dim is not rotary_dim (e.g. a full-rotary cache
        # handed in with a partial rotary_dim) must be refused, not sliced.
        self.assertFalse(kernel.rope_covered(pos, torch.randn(256, 128), rot, **common))
        # Float positions, a token-count mismatch and a 2-D (mrope) position
        # layout are all refused.
        self.assertFalse(kernel.rope_covered(pos.float(), cache, rot, **common))
        self.assertFalse(
            kernel.rope_covered(pos[:2], cache, rot, head_dim=hd, num_tokens=T)
        )
        self.assertFalse(kernel.rope_covered(pos.view(2, 2), cache, rot, **common))

    def test_cca_rope_snapshot_declines_non_plain_rotaries(self):
        _ensure_dist_initialized()
        from sglang.srt.layers.rotary_embedding import RotaryEmbedding
        from sglang.srt.models.zaya import CCARope

        pos = torch.arange(4, dtype=torch.int64)

        # A stand-in for a scaled / mrope subclass: not the exact base class, so
        # the snapshot declines rather than reading a cache it does not model.
        class _Subclass(RotaryEmbedding):
            pass

        sub = _Subclass(128, 64, 64, 10000, True, torch.float32)
        self.assertIsNone(CCARope.of(sub, pos))

        base = RotaryEmbedding(128, 64, 64, 10000, True, torch.float32)
        snap = CCARope.of(base, pos)
        self.assertIsNotNone(snap)
        self.assertEqual(snap.rotary_dim, 64)
        self.assertTrue(snap.is_neox_style)
        # The buffer is handed over by reference, not copied: the kernel must
        # read whatever the module currently holds.
        self.assertIs(snap.cos_sin_cache, base.cos_sin_cache)
        # A 2-D (multimodal) position layout declines too.
        self.assertIsNone(CCARope.of(base, pos.view(2, 2)))

    def test_cpu_fallback_leaves_rope_to_the_caller(self):
        # On the torch fallback (CPU, or before fold_qk_scales) nothing is
        # rotated inside CCA, so ``rope_fused`` must stay False or
        # ``ZayaAttention`` would skip the rotary entirely.
        _ensure_dist_initialized()
        from sglang.srt.layers.rotary_embedding import RotaryEmbedding
        from sglang.srt.models.zaya import CCARope

        cca = _make_tiny_cca(seed=1)[0]
        T = 3
        hd, nq, nk = cca.head_dim, cca.num_q_heads, cca.num_k_heads
        qk_out = torch.randn(T, (nq + nk) * hd)
        q_raw = torch.randn(T, nq * hd)
        k_raw = torch.randn(T, nk * hd)
        rope = CCARope.of(
            RotaryEmbedding(hd, hd // 2, 64, 10000, True, torch.float32),
            torch.arange(T, dtype=torch.int64),
        )
        query, key = cca._mix_and_normalize_qk(
            qk_out,
            q_raw,
            k_raw,
            qk_out[:, : nq * hd].view(T, nq, hd),
            qk_out[:, nq * hd :].view(T, nk, hd),
            q_raw.view(T, nq, hd),
            k_raw.view(T, nk, hd),
            out_dtype=torch.float32,
            rope=rope,
        )
        self.assertFalse(cca.rope_fused)
        # And the fallback still matches the two torch helpers exactly.
        ref_q, ref_k = cca._add_grouped_qk_means(
            qk_out[:, : nq * hd].view(T, nq, hd),
            qk_out[:, nq * hd :].view(T, nk, hd),
            q_raw.view(T, nq, hd),
            k_raw.view(T, nk, hd),
        )
        ref_q, ref_k = cca._normalize_qk(ref_q, ref_k)
        torch.testing.assert_close(query, ref_q)
        torch.testing.assert_close(key, ref_k)


class TestCCAQKMixKVStore(CustomTestCase):
    """The fused KV scatter must write exactly what ``set_kv_buffer`` would.

    Correctness here is not self-announcing: a wrong slot, a wrong head stride or
    a missed ``full_to_swa`` indirection does not crash, it corrupts KV and shows
    up much later as degraded output. So this compares the pool contents against
    a second, identical pool driven through the real ``set_kv_buffer`` -- on both
    a sliding-window layer (where the write goes through ``full_to_swa`` into the
    SWA sub-pool) and a full-attention layer (where it does not).
    """

    @classmethod
    def setUpClass(cls) -> None:
        _ensure_dist_initialized()
        from sglang.srt.runtime_context import get_context

        cls._server_args_override = get_context().override_server_args(
            model_path="dummy"
        )
        cls._server_args_override.install()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._server_args_override.restore()

    @staticmethod
    def _make_pool(*, size, size_swa, head_num, head_dim, swa_ids, full_ids, device):
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        return SWAKVPool(
            size=size,
            size_swa=size_swa,
            page_size=1,
            dtype=torch.bfloat16,
            head_num=head_num,
            head_dim=head_dim,
            swa_attention_layer_ids=swa_ids,
            full_attention_layer_ids=full_ids,
            device=device,
            enable_memory_saver=False,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "fused kernel requires a GPU")
    def test_fused_store_matches_set_kv_buffer_on_both_layer_kinds(self):
        from sglang.kernels.ops.attention import cca_qk_mix as kernel
        from sglang.srt.layers.radix_attention import RadixAttention
        from sglang.srt.layers.rotary_embedding import get_rope
        from sglang.srt.mem_cache.memory_pool import KVWriteLoc

        dev = "cuda"
        # 8:1 GQA at head_dim 128 -- ZAYA1-74B at attn_tp=2.
        nq, nk, hd, T = 8, 1, 128, 6
        # Both sub-pools the same size so a *missed* full->SWA indirection would
        # write an in-range but wrong row (caught by the comparison) rather than
        # running off the end of a smaller SWA pool.
        size, size_swa = 64, 64
        # Layer 0 sliding, layer 2 full: the hybrid_layer_pattern _make_swa_config
        # produces for [4096, 0, 0, 0].
        swa_ids, full_ids = [0], [2]

        torch.manual_seed(13)
        rope = get_rope(
            head_size=hd,
            rotary_dim=hd,
            max_position=256,
            base=10_000,
            is_neox_style=True,
            partial_rotary_factor=0.5,
        ).to(dev)

        for layer_id in (0, 2):
            with self.subTest(layer=layer_id, sliding=layer_id in swa_ids):
                fused_pool = self._make_pool(
                    size=size,
                    size_swa=size_swa,
                    head_num=nk,
                    head_dim=hd,
                    swa_ids=swa_ids,
                    full_ids=full_ids,
                    device=dev,
                )
                ref_pool = self._make_pool(
                    size=size,
                    size_swa=size_swa,
                    head_num=nk,
                    head_dim=hd,
                    swa_ids=swa_ids,
                    full_ids=full_ids,
                    device=dev,
                )
                # Mirror SWATokenToKVPoolAllocator's mapping: one entry per full
                # slot plus the trailing -1 sentinel. A random PERMUTATION, so it
                # is injective (no two tokens race for one row) and non-identity
                # (an unapplied indirection cannot pass by luck).
                n_full = size + 1
                mapping = torch.empty(n_full + 1, dtype=torch.int64, device=dev)
                mapping[:n_full] = torch.randperm(n_full, device=dev)
                mapping[-1] = -1
                fused_pool.register_mapping(mapping)
                ref_pool.register_mapping(mapping)

                conv_qk = (
                    torch.randn(T, (nq + nk) * hd, dtype=torch.bfloat16, device=dev)
                    * 0.3
                )
                pre_q = torch.randn(T, nq * hd, dtype=torch.bfloat16, device=dev) * 0.3
                base_k = torch.randn(T, nk * hd, dtype=torch.bfloat16, device=dev) * 0.3
                value = torch.randn(T, nk, hd, dtype=torch.bfloat16, device=dev)
                k_scale = torch.rand(nk, device=dev) + 0.5
                positions = torch.arange(3, 3 + T, dtype=torch.int64, device=dev)
                # Spread the write locations so a "slot == token index" bug fails.
                out_cache_loc = torch.tensor(
                    [5, 1, 40, 17, 2, 63], dtype=torch.int64, device=dev
                )

                is_sliding = layer_id in swa_ids
                full_to_swa = mapping if is_sliding else None
                k_cache = fused_pool.get_key_buffer(layer_id)
                v_cache = fused_pool.get_value_buffer(layer_id)
                self.assertTrue(
                    kernel.store_covered(
                        value,
                        k_cache,
                        v_cache,
                        out_cache_loc,
                        full_to_swa,
                        num_k_heads=nk,
                        head_dim=hd,
                        num_tokens=T,
                        out_dtype=torch.bfloat16,
                        device=torch.device(dev),
                    )
                )

                q_got, k_got = kernel.cca_qk_mix(
                    conv_qk,
                    pre_q,
                    base_k,
                    k_scale,
                    num_q_heads=nq,
                    num_k_heads=nk,
                    head_dim=hd,
                    q_scale=hd**0.5,
                    out_dtype=torch.bfloat16,
                    positions=positions,
                    cos_sin_cache=rope.cos_sin_cache,
                    rotary_dim=rope.rotary_dim,
                    value=value,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    out_cache_loc=out_cache_loc,
                    full_to_swa=full_to_swa,
                )

                # The reference: hand the SAME k/v to the real set_kv_buffer, via
                # the same KVWriteLoc bundle the attention backend builds.
                layer = RadixAttention(
                    num_heads=nq,
                    head_dim=hd,
                    scaling=hd**-0.5,
                    num_kv_heads=nk,
                    layer_id=layer_id,
                    sliding_window_size=4095 if is_sliding else -1,
                )
                swa_loc = (
                    ref_pool.translate_loc_from_full_to_swa(out_cache_loc)
                    if is_sliding
                    else None
                )
                ref_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc(out_cache_loc, swa_loc),
                    k_got.contiguous(),
                    value.contiguous(),
                )

                torch.testing.assert_close(
                    fused_pool.get_key_buffer(layer_id),
                    ref_pool.get_key_buffer(layer_id),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    fused_pool.get_value_buffer(layer_id),
                    ref_pool.get_value_buffer(layer_id),
                    rtol=0,
                    atol=0,
                )
                # And the pool actually received something -- an all-zero buffer
                # would trivially match an all-zero reference.
                self.assertGreater(
                    fused_pool.get_key_buffer(layer_id).abs().sum().item(), 0.0
                )
                self.assertEqual(q_got.shape, (T, nq, hd))

    @unittest.skipUnless(torch.cuda.is_available(), "fused kernel requires a GPU")
    def test_negative_slots_are_skipped(self):
        # The full->SWA mapping's trailing -1 is how a padding row says "write
        # nothing". Writing it as slot -1 (or as slot 0) would corrupt a live
        # entry, so pin the skip.
        from sglang.kernels.ops.attention import cca_qk_mix as kernel
        from sglang.srt.layers.rotary_embedding import get_rope

        dev = "cuda"
        nq, nk, hd, T, rows = 8, 1, 128, 3, 8
        torch.manual_seed(4)
        rope = get_rope(
            head_size=hd,
            rotary_dim=hd,
            max_position=64,
            base=10_000,
            is_neox_style=True,
            partial_rotary_factor=0.5,
        ).to(dev)
        k_cache = torch.zeros(rows, nk, hd, dtype=torch.bfloat16, device=dev)
        v_cache = torch.zeros(rows, nk, hd, dtype=torch.bfloat16, device=dev)
        # Only token 1 has a live slot; tokens 0 and 2 map to the sentinel.
        mapping = torch.full((rows + 1,), -1, dtype=torch.int64, device=dev)
        mapping[1] = 4
        out_cache_loc = torch.tensor([0, 1, 2], dtype=torch.int64, device=dev)

        kernel.cca_qk_mix(
            torch.randn(T, (nq + nk) * hd, dtype=torch.bfloat16, device=dev),
            torch.randn(T, nq * hd, dtype=torch.bfloat16, device=dev),
            torch.randn(T, nk * hd, dtype=torch.bfloat16, device=dev),
            torch.ones(nk, device=dev),
            num_q_heads=nq,
            num_k_heads=nk,
            head_dim=hd,
            q_scale=hd**0.5,
            out_dtype=torch.bfloat16,
            positions=torch.arange(T, dtype=torch.int64, device=dev),
            cos_sin_cache=rope.cos_sin_cache,
            rotary_dim=rope.rotary_dim,
            value=torch.randn(T, nk, hd, dtype=torch.bfloat16, device=dev),
            k_cache=k_cache,
            v_cache=v_cache,
            out_cache_loc=out_cache_loc,
            full_to_swa=mapping,
        )
        written = (k_cache.abs().sum(dim=(1, 2)) > 0).nonzero().flatten().tolist()
        self.assertEqual(written, [4])
        written_v = (v_cache.abs().sum(dim=(1, 2)) > 0).nonzero().flatten().tolist()
        self.assertEqual(written_v, [4])


class TestCCAQKMixStoreCoverage(CustomTestCase):
    """``store_covered`` negatives.

    A rejected input costs one extra ``set_kv_buffer`` launch; an input that
    should have been rejected corrupts KV silently. So the gate is the part that
    matters, and it is written to be checkable without a GPU: the device test is
    an explicit equality against the caller's device rather than ``is_cuda``, so
    every branch can be exercised on CPU tensors.
    """

    @staticmethod
    def _ok_args(nk=2, hd=8, T=3, rows=17, dtype=torch.bfloat16):
        return dict(
            value=torch.zeros(T, nk, hd, dtype=dtype),
            k_cache=torch.zeros(rows, nk, hd, dtype=dtype),
            v_cache=torch.zeros(rows, nk, hd, dtype=dtype),
            out_cache_loc=torch.zeros(T, dtype=torch.int64),
            full_to_swa=None,
        )

    def _covered(self, **over):
        from sglang.kernels.ops.attention import cca_qk_mix as kernel

        args = self._ok_args()
        args.update(over)
        return kernel.store_covered(
            args["value"],
            args["k_cache"],
            args["v_cache"],
            args["out_cache_loc"],
            args["full_to_swa"],
            num_k_heads=2,
            head_dim=8,
            num_tokens=3,
            out_dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )

    def test_the_baseline_is_covered(self):
        self.assertTrue(self._covered())

    def test_rejects_non_flash_kv_layouts(self):
        # The 5-D SHUFFLE (vectorized_5d) layout: (blocks, H, D//x, page, x).
        self.assertFalse(self._covered(k_cache=torch.zeros(4, 2, 4, 1, 2)))
        self.assertFalse(self._covered(v_cache=torch.zeros(4, 2, 1, 8, 2)))
        # The 4-D HND layout: (pages, H, page_size, D). Its per-slot row is not a
        # contiguous [H, D] block, so a flat slot index would land elsewhere.
        self.assertFalse(self._covered(k_cache=torch.zeros(17, 2, 1, 8)))
        # And the paged 4-D flash view a caller might reshape into.
        self.assertFalse(self._covered(k_cache=torch.zeros(17, 1, 2, 8)))

    def test_rejects_a_non_bf16_pool(self):
        # An fp8 pool stores under a different store_dtype and needs per-tensor
        # scales the kernel does not apply.
        fp8 = torch.zeros(17, 2, 8, dtype=torch.float8_e4m3fn)
        self.assertFalse(self._covered(k_cache=fp8))
        self.assertFalse(self._covered(v_cache=torch.zeros(17, 2, 8).float()))
        self.assertFalse(self._covered(value=torch.zeros(3, 2, 8).float()))

    def test_rejects_shape_and_stride_mismatches(self):
        # Wrong head count / head dim on either buffer.
        self.assertFalse(
            self._covered(k_cache=torch.zeros(17, 4, 8, dtype=torch.bfloat16))
        )
        self.assertFalse(
            self._covered(v_cache=torch.zeros(17, 2, 16, dtype=torch.bfloat16))
        )
        # A non-unit innermost stride (a transposed or strided view).
        strided = torch.zeros(17, 8, 2, dtype=torch.bfloat16).transpose(1, 2)
        self.assertFalse(self._covered(k_cache=strided))
        # V with the wrong token count.
        self.assertFalse(
            self._covered(value=torch.zeros(2, 2, 8, dtype=torch.bfloat16))
        )
        # A 2-D V (not yet reshaped into heads).
        self.assertFalse(self._covered(value=torch.zeros(3, 16, dtype=torch.bfloat16)))

    def test_rejects_bad_locs_and_mappings(self):
        # Float locs, the wrong length, a 2-D loc.
        self.assertFalse(self._covered(out_cache_loc=torch.zeros(3)))
        self.assertFalse(self._covered(out_cache_loc=torch.zeros(4, dtype=torch.int64)))
        self.assertFalse(
            self._covered(out_cache_loc=torch.zeros(3, 1, dtype=torch.int64))
        )
        # A non-int64 or empty full->SWA mapping.
        self.assertFalse(self._covered(full_to_swa=torch.zeros(18, dtype=torch.int32)))
        self.assertFalse(self._covered(full_to_swa=torch.zeros(0, dtype=torch.int64)))
        # A well-formed one is fine.
        self.assertTrue(self._covered(full_to_swa=torch.zeros(18, dtype=torch.int64)))

    def test_rejects_missing_arguments(self):
        for missing in ("value", "k_cache", "v_cache", "out_cache_loc"):
            with self.subTest(missing=missing):
                self.assertFalse(self._covered(**{missing: None}))

    def test_rejects_a_cross_device_buffer(self):
        from sglang.kernels.ops.attention import cca_qk_mix as kernel

        args = self._ok_args()
        # A "meta" tensor stands in for a buffer that belongs to another device;
        # the check is device equality, so it fires without needing two real ones.
        self.assertFalse(
            kernel.store_covered(
                args["value"],
                args["k_cache"].to("meta"),
                args["v_cache"],
                args["out_cache_loc"],
                None,
                num_k_heads=2,
                head_dim=8,
                num_tokens=3,
                out_dtype=torch.bfloat16,
                device=torch.device("cpu"),
            )
        )

    def test_store_is_only_offered_when_the_rope_fused(self):
        # Storing an un-rotated k would be silent KV corruption, and the rotation
        # happens inside the same kernel -- so the store must never be enabled
        # while the rope fell back. Pinned on the CPU fallback, where neither
        # fuses.
        _ensure_dist_initialized()
        cca = _make_tiny_cca(seed=6)[0]
        T = 3
        hd, nq, nk = cca.head_dim, cca.num_q_heads, cca.num_k_heads
        qk_out = torch.randn(T, (nq + nk) * hd)
        q_raw = torch.randn(T, nq * hd)
        k_raw = torch.randn(T, nk * hd)
        from sglang.srt.models.zaya import CCAKVStore

        store = CCAKVStore(
            k_cache=torch.zeros(8, nk, hd),
            v_cache=torch.zeros(8, nk, hd),
            out_cache_loc=torch.arange(T, dtype=torch.int64),
            full_to_swa=None,
        )
        cca._mix_and_normalize_qk(
            qk_out,
            q_raw,
            k_raw,
            qk_out[:, : nq * hd].view(T, nq, hd),
            qk_out[:, nq * hd :].view(T, nk, hd),
            q_raw.view(T, nq, hd),
            k_raw.view(T, nk, hd),
            out_dtype=torch.float32,
            rope=None,
            value=torch.randn(T, nk, hd),
            kv_store=store,
        )
        self.assertFalse(cca.rope_fused)
        self.assertFalse(cca.kv_store_fused)
        # Nothing was written into the stand-in pool.
        self.assertEqual(store.k_cache.abs().sum().item(), 0.0)


class TestZayaAttentionKVStoreGate(CustomTestCase):
    """``ZayaAttention._kv_store`` must decline every pool it cannot address.

    The resolver is pure Python over the live pool objects, so its rejects are
    checkable on CPU with stand-ins: an unrecognized pool type is refused before
    any tensor is touched, which is the branch that keeps an unfamiliar layout
    from reaching the kernel.
    """

    @classmethod
    def setUpClass(cls) -> None:
        _ensure_dist_initialized()
        from sglang.srt.runtime_context import get_context

        cls._server_args_override = get_context().override_server_args(
            model_path="dummy"
        )
        cls._server_args_override.install()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._server_args_override.restore()

    def _attention(self, layer_id=0):
        from sglang.srt.models.zaya import ZayaAttention

        config = _make_swa_config(num_hidden_layers=4, swa_layers=[4096, 0, 0, 0])
        return ZayaAttention(config=config, layer_id=layer_id)

    def test_unknown_pool_types_decline(self):
        import sglang.srt.models.zaya as zaya_mod

        attn = self._attention()
        fb = SimpleNamespace(out_cache_loc=torch.zeros(2, dtype=torch.int64))

        class _SomeOtherPool:
            pass

        with unittest.mock.patch.object(
            zaya_mod, "get_token_to_kv_pool", lambda: _SomeOtherPool()
        ):
            self.assertIsNone(attn._kv_store(fb))

    def test_missing_out_cache_loc_declines(self):
        attn = self._attention()
        self.assertIsNone(attn._kv_store(SimpleNamespace(out_cache_loc=None)))

    def test_a_scaled_attention_layer_declines(self):
        import sglang.srt.models.zaya as zaya_mod

        attn = self._attention()
        attn.attn.k_scale = torch.tensor(1.0)
        fb = SimpleNamespace(out_cache_loc=torch.zeros(2, dtype=torch.int64))

        called = []

        def _boom():
            called.append(1)
            raise AssertionError("must not reach the pool")

        with unittest.mock.patch.object(zaya_mod, "get_token_to_kv_pool", _boom):
            self.assertIsNone(attn._kv_store(fb))
        self.assertEqual(called, [])

    def test_a_dcp_masked_batch_declines(self):
        import sglang.srt.models.zaya as zaya_mod

        attn = self._attention()
        fb = SimpleNamespace(
            out_cache_loc=torch.zeros(2, dtype=torch.int64),
            dcp_kv_mask=torch.ones(2, dtype=torch.bool),
        )
        with unittest.mock.patch.object(zaya_mod, "get_token_to_kv_pool", lambda: None):
            self.assertIsNone(attn._kv_store(fb))


class TestCCADecodeConvFold(CustomTestCase):
    """``CCA.fold_decode_conv`` must reproduce the two-stage conv exactly.

    Derived property. At decode the window is ``[T, C, total_padding + 1]`` and
    only one output timestep is needed, so conv_qk[0] (depthwise, k=cca_time0)
    composed with conv_qk[1] (grouped, k=cca_time1) collapses to a single grouped
    matmul whose weight is precomputable. The tap-index bookkeeping
    (``A[..., j+k] += w1[..., j] * w0[..., k]``) and the depthwise bias
    pass-through are easy to get subtly wrong -- off-by-one in the tap offset
    still produces plausible-looking output -- so pin the identity directly.
    """

    @classmethod
    def setUpClass(cls) -> None:
        _ensure_dist_initialized()

    def _check(self, cca, T: int):
        from sglang.srt.models.zaya import _cca_decode_conv

        taps = cca.total_padding + 1
        torch.manual_seed(7)
        window = torch.randn(T, cca.in_out_ch, taps, dtype=torch.float32) * 0.3
        with torch.no_grad():
            # Reference: the real two-stage conv, which yields exactly one step.
            ref = cca.conv_qk(window)
            self.assertEqual(ref.shape, (T, cca.in_out_ch, 1))
            ref = ref.squeeze(-1)

            cca.fold_decode_conv()
            # Drive the production consumer rather than re-deriving the einsum:
            # the folded weight now carries the conv bias in a trailing column
            # activated by a constant-1.0 tap, and re-deriving it here would let
            # the two layouts drift apart unnoticed.
            got = _cca_decode_conv(
                window,
                cca.conv_qk,
                cca.decode_conv_weight,
                cca.decode_conv_bias,
                cca.decode_conv_groups,
            )
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)

        # The window that ``cca_state_step`` emits already carries the ones
        # column; the one built by the unfused fallback does not and gets it
        # appended. Both must give the same answer, or the fused and fallback
        # decode paths disagree only on GPU, where the fused one runs.
        with torch.no_grad():
            widened = torch.cat([window, torch.ones(T, cca.in_out_ch, 1)], dim=-1)
            got_widened = _cca_decode_conv(
                widened,
                cca.conv_qk,
                cca.decode_conv_weight,
                cca.decode_conv_bias,
                cca.decode_conv_groups,
            )
        torch.testing.assert_close(got_widened, got, rtol=0, atol=0)

    def test_folded_weight_carries_the_bias_column(self):
        """The bias rides in the weight so the separate add can go.

        ``_cca_decode_conv`` used to do ``einsum(...) + decode_conv_bias``, one
        launch per attention layer -- 60 per decode step on ZAYA1-74B. Folding
        the bias into a trailing weight column, activated by a constant-1.0 tap
        the window kernel writes for free, puts it inside the matmul's fp32
        accumulator instead. Pin both halves: the column holds the bias on the
        last input channel, and zero on every other, so each output picks it up
        exactly once.
        """
        cca, _ = _make_tiny_cca(seed=9)
        cca.fold_decode_conv()
        groups = cca.decode_conv_groups
        cg = cca.in_out_ch // groups
        taps = cca.total_padding + 1

        self.assertEqual(cca.decode_conv_taps_ext, taps + 1)
        self.assertEqual(
            tuple(cca.decode_conv_weight.shape), (groups, cg, cg * (taps + 1))
        )
        w = cca.decode_conv_weight.reshape(groups, cg, cg, taps + 1)
        torch.testing.assert_close(
            w[:, :, cg - 1, taps], cca.decode_conv_bias, rtol=0, atol=0
        )
        self.assertTrue(torch.all(w[:, :, : cg - 1, taps] == 0))

    def test_fold_matches_two_stage_conv(self):
        cca, _ = _make_tiny_cca(seed=3)
        for T in (1, 4):
            self._check(cca, T)

    def test_fold_matches_under_tensor_parallel_slicing(self):
        # conv_qk is TP-sliced per rank, so the fold must be built from the live
        # per-rank weights -- a rank that folded the full-width weight would
        # silently mix in another rank's heads.
        for rank in (0, 1):
            cca, _ = _make_tiny_cca(seed=4, tp_rank=rank, tp_size=2)
            self._check(cca, T=2)

    def test_unfolded_cca_does_not_use_the_zero_buffers(self):
        # Regression guard. The folded buffers are zero-initialized and only
        # valid after fold_decode_conv() runs against loaded weights. A forward
        # that consumed them unconditionally emitted bias-only garbage with no
        # error -- silent wrongness for any path that populates weights without
        # going through ZayaForCausalLM.load_weights.
        cca, _ = _make_tiny_cca(seed=6)
        self.assertFalse(cca._decode_conv_folded)
        cca.fold_decode_conv()
        self.assertTrue(cca._decode_conv_folded)

    def test_fold_is_refreshed_not_stale(self):
        # The buffers are non-persistent and derived, so a weight change must be
        # picked up by re-folding; a cached-once implementation would go stale on
        # weight reload (the RL / update_weights path).
        cca, _ = _make_tiny_cca(seed=5)
        cca.fold_decode_conv()
        before = cca.decode_conv_weight.clone()
        with torch.no_grad():
            cca.conv_qk[0].weight.mul_(2.0)
        cca.fold_decode_conv()
        self.assertFalse(torch.allclose(before, cca.decode_conv_weight))
        self._check(cca, T=1)


class TestShortConvPaddingSlotClamp(CustomTestCase):
    """``ShortConvAttnBackend`` must clamp the -1 batch-padding sentinel.

    Regression guard. Batch padding poisons unused rows' mamba slot ids to -1
    (``MambaAttnBackendBase._forward_metadata``); DP attention hits this on every
    step where a replica's batch is padded. CCA feeds the shared index view
    straight into ``index_select`` / ``index_copy_``, and a negative index there
    is an out-of-bounds device gather -- on ROCm it aborts the queue with
    HSA_STATUS_ERROR_EXCEPTION 0x1016 instead of raising, which crashed ZAYA1 at
    attn_tp > 1 under DP attention. Clamping to 0 is safe because
    ``MambaSlotAllocator`` reserves slot 0 (it hands out 1..size), so padded rows
    land on a scratch slot they cannot corrupt.
    """

    @staticmethod
    def _bare_backend(buf: Optional[torch.Tensor], idx: Optional[torch.Tensor]):
        # _refresh_cache_indices touches only these three attributes, so a bare
        # instance exercises the clamping contract without a live ModelRunner.
        from sglang.srt.layers.attention.linear.short_conv_backend import (
            ShortConvAttnBackend,
        )

        backend = object.__new__(ShortConvAttnBackend)
        backend._cache_indices_buf = buf
        backend._cache_indices = None
        backend.forward_metadata = (
            None if idx is None else SimpleNamespace(mamba_cache_indices=idx)
        )
        return backend

    def test_graph_buffer_path_clamps_padding(self):
        # Buffered path (cuda/cpu graph): the persistent buffer is refilled in
        # place, so the clamp must land in the buffer the captured graph reads.
        buf = torch.empty(8, dtype=torch.int64)
        idx = torch.tensor([3, 5, -1, -1], dtype=torch.int64)
        backend = self._bare_backend(buf, idx)
        backend._refresh_cache_indices()

        out = backend._cache_indices
        self.assertEqual(out.tolist(), [3, 5, 0, 0])
        self.assertGreaterEqual(int(out.min()), 0)
        # Must be a view of the persistent buffer, not a fresh allocation.
        self.assertIs(out.untyped_storage(), buf.untyped_storage())

    def test_fallback_path_clamps_without_mutating_source(self):
        # No buffer (eager, or bs beyond the buffer): the clamp must be
        # out-of-place. ``to(torch.long)`` aliases an already-int64 tensor, so an
        # in-place clamp would corrupt the backend's own mamba_cache_indices --
        # destroying the -1 sentinel other consumers rely on to skip padded rows.
        idx = torch.tensor([2, -1], dtype=torch.int64)
        backend = self._bare_backend(None, idx)
        backend._refresh_cache_indices()

        self.assertEqual(backend._cache_indices.tolist(), [2, 0])
        self.assertEqual(idx.tolist(), [2, -1])

    def test_no_metadata_yields_no_indices(self):
        # Before the first step forward_metadata is None; the view must stay None
        # rather than clamping a nonexistent tensor.
        backend = self._bare_backend(torch.empty(4, dtype=torch.int64), None)
        backend._refresh_cache_indices()
        self.assertIsNone(backend._cache_indices)


# ---------------------------------------------------------------------------
# Mamba radix cache -- extra_buffer track snapshot
# ---------------------------------------------------------------------------


def _torch_track_reference(
    state_a,
    state_b,
    src_rows,
    mask_rows,
    dst_rows,
    total_rows,
    check_freed_slots=False,
):
    """CPU stand-in for ``track_mamba_states_if_needed`` (Triton, CUDA only).

    Same contract: for every row ``i < total_rows`` whose mask is set, copy
    ``state[src_rows[i]] -> state[dst_rows[i]]`` in BOTH state tensors, and
    skip rows whose src/dst is negative when ``check_freed_slots``.
    """
    for i in range(total_rows):
        if not bool(mask_rows[i]):
            continue
        src = int(src_rows[i])
        dst = int(dst_rows[i])
        if check_freed_slots and (src < 0 or dst < 0):
            continue
        for state in (state_a, state_b):
            if state[0].numel() == 0:
                continue
            state[dst] = state[src].clone()


class _TrackHarness:
    """Bare ``ShortConvAttnBackend`` wired for the track paths, CPU only.

    ``object.__new__`` keeps this free of a live ModelRunner: the track code
    reads only the attributes set here plus ``mamba_cache_chunk_size`` off the
    server args, which the tests patch on the module.
    """

    def __init__(
        self,
        *,
        num_layers=2,
        num_slots=6,
        num_channels=3,
        hidden_size=4,
        windows=(2, 1),
        extra_buffer=True,
        chunk_size=8,
        track_interval=16,
    ):
        from sglang.srt.layers.attention.linear.short_conv_backend import (
            ShortConvAttnBackend,
        )

        conv = [
            torch.zeros(num_layers, num_slots, ch, w)
            for ch, w in zip((num_channels, hidden_size), windows)
        ]
        temporal = torch.zeros(num_layers, num_slots, 1, 1, 0)
        self.mamba_cache = SimpleNamespace(conv=conv, temporal=temporal)
        self.num_layers = num_layers
        self.num_slots = num_slots
        self.server_args = SimpleNamespace(
            mamba_cache_chunk_size=chunk_size,
            mamba_track_interval=track_interval,
            speculative_algorithm=None,
        )

        backend = object.__new__(ShortConvAttnBackend)
        backend.device = torch.device("cpu")
        backend.enable_unified_memory = False
        backend.conv_states_shape = conv[0].shape
        backend.conv_window_lens = [int(c.shape[-1]) for c in conv]
        backend._cache_indices = None
        backend._cache_indices_buf = None
        backend.forward_metadata = None
        backend._track_conv_indices = None
        backend._track_dst = None
        backend._track_layer_row_base = None
        backend._track_pairs = None
        backend.enable_mamba_extra_buffer = extra_buffer
        if extra_buffer:
            backend._init_track_state(self.server_args, self.mamba_cache)
        self.backend = backend

    def layer_cache(self, layer_idx):
        return _MockLayerCache(
            conv=[c[layer_idx] for c in self.mamba_cache.conv],
            temporal=self.mamba_cache.temporal[layer_idx],
        )


def _track_forward_batch(*, extend_seq_lens, prefix_lens, track_mask, track_indices):
    """Extend ForwardBatch stub carrying exactly the track inputs.

    ``mamba_track_seqlens`` mirrors what the scheduler builds in
    ``_mamba_radix_cache_v2_req_prepare_for_extend``: prefix + extend length
    for tracked rows, -1 for untracked ones.
    """
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    seqlens = [
        (p + e) if m else -1
        for p, e, m in zip(prefix_lens, extend_seq_lens, track_mask)
    ]
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        extend_prefix_lens=torch.tensor(prefix_lens, dtype=torch.int64),
        mamba_track_seqlens=torch.tensor(seqlens, dtype=torch.int64),
        mamba_track_mask=torch.tensor(track_mask, dtype=torch.bool),
        mamba_track_indices=torch.tensor(track_indices, dtype=torch.int64),
    )


def _query_start_loc(extend_seq_lens):
    out = [0]
    for s in extend_seq_lens:
        out.append(out[-1] + s)
    return torch.tensor(out, dtype=torch.int64)


class TestShortConvTrackIndices(CustomTestCase):
    """``_init_track_conv_indices``: where the extend snapshot reads from.

    The radix tree is handed a state checkpoint keyed on
    ``mamba_last_track_seqlen = prefix + floor(extend_len / chunk) * chunk``,
    NOT on the end of the extend. For a conv, the state at length L is exactly
    the last ``window`` input rows ending at L, so the snapshot is a gather at
    flattened positions ``[qsl_i + aligned - window, qsl_i + aligned)``. Get
    this wrong by one and every prefix hit resumes from a shifted conv window.
    """

    def _indices(self, harness, forward_batch, extend_seq_lens):
        from sglang.srt.layers.attention.linear import short_conv_backend

        with unittest.mock.patch.object(
            short_conv_backend,
            "get_server_args",
            lambda: harness.server_args,
        ):
            return harness.backend._init_track_conv_indices(
                _query_start_loc(extend_seq_lens), forward_batch
            )

    def test_indices_land_on_the_chunk_boundary(self):
        # chunk == 8. Request 0: 20 fresh tokens -> aligned 16, window [14, 16).
        # Request 1: 10 tokens on a 32-token prefix -> aligned 8 within the
        # extend, i.e. flattened [20 + 6, 20 + 8).
        extend = [20, 10]
        harness = _TrackHarness(chunk_size=8, windows=(2, 1))
        fb = _track_forward_batch(
            extend_seq_lens=extend,
            prefix_lens=[0, 32],
            track_mask=[True, True],
            track_indices=[4, 5],
        )
        conv_idx, lag_idx = self._indices(harness, fb, extend)

        self.assertEqual(conv_idx.tolist(), [[14, 15], [26, 27]])
        # The one-token lag entry ends on the same column as the conv window.
        self.assertEqual(lag_idx.tolist(), [[15], [27]])

    def test_untracked_rows_are_dropped(self):
        # mamba_track_mask is False for an extend shorter than one chunk; that
        # row must not appear in the gather at all (its ping-pong slot is not
        # the one the radix cache will read).
        extend = [16, 3]
        harness = _TrackHarness(chunk_size=8, windows=(2, 1))
        fb = _track_forward_batch(
            extend_seq_lens=extend,
            prefix_lens=[0, 0],
            track_mask=[True, False],
            track_indices=[4, 5],
        )
        conv_idx, lag_idx = self._indices(harness, fb, extend)
        self.assertEqual(list(conv_idx.shape), [1, 2])
        self.assertEqual(conv_idx.tolist(), [[14, 15]])
        self.assertEqual(lag_idx.tolist(), [[15]])

    def test_every_entry_shares_the_end_column(self):
        # Both ZAYA1 conv entries snapshot the SAME sequence position; only the
        # window depth differs. If they ever diverge, conv[0] and conv[1] would
        # describe different prefix lengths in one cached node.
        extend = [24]
        harness = _TrackHarness(chunk_size=8, windows=(4, 1))
        fb = _track_forward_batch(
            extend_seq_lens=extend,
            prefix_lens=[0],
            track_mask=[True],
            track_indices=[3],
        )
        entries = self._indices(harness, fb, extend)
        ends = {int(e[0, -1]) for e in entries}
        self.assertEqual(ends, {23})

    def test_matches_the_single_conv_base_implementation(self):
        # The override must be a faithful generalization: for a model with one
        # conv entry it has to reproduce MambaAttnBackendBase byte-for-byte,
        # or GDN/Mamba2 semantics would have quietly forked.
        import sglang.srt.layers.attention.hybrid_linear_attn_backend as hb
        from sglang.srt.layers.attention.linear import short_conv_backend

        extend = [20, 10]
        harness = _TrackHarness(chunk_size=8, windows=(3,))
        fb = _track_forward_batch(
            extend_seq_lens=extend,
            prefix_lens=[0, 32],
            track_mask=[True, True],
            track_indices=[4, 5],
        )
        qsl = _query_start_loc(extend)
        with unittest.mock.patch.object(
            short_conv_backend, "get_server_args", lambda: harness.server_args
        ):
            mine = harness.backend._init_track_conv_indices(qsl, fb)[0]
        # The base reads the derived accessor, not the record:
        # hybrid_linear_attn_backend imports mamba_cache_chunk_size, whereas
        # short_conv_backend imports get_server_args. Patch what each one binds.
        with unittest.mock.patch.object(
            hb,
            "mamba_cache_chunk_size",
            lambda: harness.server_args.mamba_cache_chunk_size,
        ):
            theirs = hb.MambaAttnBackendBase._init_track_conv_indices(
                harness.backend, qsl, fb
            )
        self.assertTrue(torch.equal(mine, theirs))


class TestShortConvTrackExtendSnapshot(CustomTestCase):
    """``track_conv_states_extend``: what actually lands in the track slot.

    ZAYA1 keeps TWO conv entries -- ``conv[0]`` is the conv_qk left padding
    (window == total_padding, over ``qk``) and ``conv[1]`` is the one-token
    ``prev_hs`` lag (window == 1, over ``hidden_states``). Both must be
    snapshotted, from their own input tensor, or a prefix hit restores half a
    state.
    """

    def _prepare(self, harness, fb, extend):
        from sglang.srt.layers.attention.linear import short_conv_backend

        with unittest.mock.patch.object(
            short_conv_backend, "get_server_args", lambda: harness.server_args
        ):
            indices = harness.backend._init_track_conv_indices(
                _query_start_loc(extend), fb
            )
        harness.backend._track_conv_indices = indices
        harness.backend._track_dst = fb.mamba_track_indices[fb.mamba_track_mask]

    def test_both_conv_entries_are_snapshotted(self):
        extend = [20]
        harness = _TrackHarness(chunk_size=8, windows=(2, 1))
        fb = _track_forward_batch(
            extend_seq_lens=extend,
            prefix_lens=[0],
            track_mask=[True],
            track_indices=[4],
        )
        self._prepare(harness, fb, extend)

        torch.manual_seed(0)
        qk = torch.randn(20, 3)
        hs = torch.randn(20, 4)
        layer_cache = harness.layer_cache(1)
        harness.backend.track_conv_states_extend(layer_cache, (qk, hs))

        # conv[0] slot 4 == qk rows [14, 16) laid out channel-major.
        self.assertTrue(
            torch.allclose(layer_cache.conv[0][4], qk[14:16].transpose(0, 1))
        )
        # conv[1] slot 4 == the single hidden_states row at the aligned point.
        self.assertTrue(torch.allclose(layer_cache.conv[1][4], hs[15].unsqueeze(-1)))
        # Nothing else moved, and only this layer was touched.
        self.assertEqual(float(layer_cache.conv[0][3].abs().sum()), 0.0)
        self.assertEqual(float(harness.mamba_cache.conv[0][0].abs().sum()), 0.0)

    def test_resumed_prefix_reads_only_this_extends_tokens(self):
        # A request resuming a 32-token cached prefix contributes 10 new
        # tokens; the snapshot must sit at prefix + 8 and gather from THIS
        # request's slice of the flattened batch, never from the neighbour's.
        extend = [20, 10]
        harness = _TrackHarness(chunk_size=8, windows=(2, 1))
        fb = _track_forward_batch(
            extend_seq_lens=extend,
            prefix_lens=[0, 32],
            track_mask=[False, True],
            track_indices=[4, 5],
        )
        self._prepare(harness, fb, extend)

        torch.manual_seed(1)
        qk = torch.randn(30, 3)
        hs = torch.randn(30, 4)
        layer_cache = harness.layer_cache(0)
        harness.backend.track_conv_states_extend(layer_cache, (qk, hs))

        self.assertTrue(
            torch.allclose(layer_cache.conv[0][5], qk[26:28].transpose(0, 1))
        )
        self.assertTrue(torch.allclose(layer_cache.conv[1][5], hs[27].unsqueeze(-1)))
        # The untracked row's ping-pong slot stays untouched.
        self.assertEqual(float(layer_cache.conv[0][4].abs().sum()), 0.0)

    def test_snapshot_equals_the_state_after_the_aligned_prefix(self):
        # The invariant a prefix hit relies on: the tracked state is what the
        # conv would hold after exactly `mamba_last_track_seqlen` tokens, i.e.
        # that many rows' worth of history, not the end-of-extend state.
        extend = [20]
        harness = _TrackHarness(chunk_size=8, windows=(2, 1))
        fb = _track_forward_batch(
            extend_seq_lens=extend,
            prefix_lens=[0],
            track_mask=[True],
            track_indices=[4],
        )
        self._prepare(harness, fb, extend)

        torch.manual_seed(2)
        qk = torch.randn(20, 3)
        hs = torch.randn(20, 4)
        layer_cache = harness.layer_cache(0)
        harness.backend.track_conv_states_extend(layer_cache, (qk, hs))

        aligned = 16
        self.assertTrue(
            torch.allclose(
                layer_cache.conv[0][4], qk[aligned - 2 : aligned].transpose(0, 1)
            )
        )
        self.assertTrue(
            torch.allclose(
                layer_cache.conv[1][4], hs[aligned - 1 : aligned].transpose(0, 1)
            )
        )
        # ... and NOT the end-of-extend state, which is what a plain row copy
        # of the live slot would have given.
        self.assertFalse(
            torch.allclose(layer_cache.conv[0][4], qk[18:20].transpose(0, 1))
        )

    def test_no_track_this_step_is_a_no_op(self):
        harness = _TrackHarness(chunk_size=8, windows=(2, 1))
        layer_cache = harness.layer_cache(0)
        harness.backend.track_conv_states_extend(
            layer_cache, (torch.randn(4, 3), torch.randn(4, 4))
        )
        self.assertEqual(float(harness.mamba_cache.conv[0].abs().sum()), 0.0)
        self.assertEqual(float(harness.mamba_cache.conv[1].abs().sum()), 0.0)


class TestShortConvTrackDecode(CustomTestCase):
    """``track_conv_states_decode``: the all-layers row copy.

    Exercised against a torch stand-in for the Triton scatter (the real kernel
    is CUDA-only); what is under test here is the row addressing into the
    flattened ``[n_layers * n_slots, ...]`` pool view and the mask, which is
    where a two-conv-entry model can go wrong.
    """

    @staticmethod
    def _decode_batch(mask):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        return SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            mamba_track_mask=torch.tensor(mask, dtype=torch.bool),
        )

    @staticmethod
    def _run(harness, forward_batch, cache_indices, track_indices, record=None):
        from sglang.srt.layers.attention.linear import short_conv_backend

        backend = harness.backend
        backend._cache_indices = torch.tensor(cache_indices, dtype=torch.int64)
        backend.forward_metadata = SimpleNamespace(
            mamba_track_indices=torch.tensor(track_indices, dtype=torch.int64),
            # The unmutated slot ids (pre-clamp), which the unified-memory
            # branch reads to spot freed-slot tombstones.
            mamba_cache_indices=torch.tensor(cache_indices, dtype=torch.int64),
        )

        def _spy(*args, **kwargs):
            if record is not None:
                record.append((args, kwargs))
            return _torch_track_reference(*args, **kwargs)

        with unittest.mock.patch.object(
            short_conv_backend, "track_mamba_states_if_needed", _spy
        ):
            backend.track_conv_states_decode(forward_batch)

    def test_copies_every_layer_for_both_conv_entries(self):
        harness = _TrackHarness(num_layers=3, num_slots=6, windows=(2, 1))
        conv0, conv1 = harness.mamba_cache.conv
        torch.manual_seed(3)
        conv0.normal_()
        conv1.normal_()
        live = [1, 2]
        track = [4, 5]
        before0 = conv0.clone()
        before1 = conv1.clone()

        self._run(harness, self._decode_batch([True, True]), live, track)

        for layer in range(3):
            for src, dst in zip(live, track):
                self.assertTrue(torch.equal(conv0[layer, dst], before0[layer, src]))
                self.assertTrue(torch.equal(conv1[layer, dst], before1[layer, src]))
                # The live slot itself is left alone.
                self.assertTrue(torch.equal(conv0[layer, src], before0[layer, src]))

    def test_masked_rows_are_not_copied(self):
        harness = _TrackHarness(num_layers=2, num_slots=6, windows=(2, 1))
        conv0, _ = harness.mamba_cache.conv
        torch.manual_seed(4)
        conv0.normal_()
        before0 = conv0.clone()

        self._run(harness, self._decode_batch([True, False]), [1, 2], [4, 5])

        for layer in range(2):
            self.assertTrue(torch.equal(conv0[layer, 4], before0[layer, 1]))
            # Row 1 is untracked this step: its ping-pong slot must not move.
            self.assertTrue(torch.equal(conv0[layer, 5], before0[layer, 5]))

    def test_launch_happens_even_with_nothing_to_track(self):
        # The cuda-graph inert-buffer contract. Capture runs with an all-False
        # mask buffer; if the launch were skipped then, the replayed graph
        # would never contain the scatter and every snapshot for the life of
        # that graph would be silently lost.
        harness = _TrackHarness(num_layers=2, num_slots=6, windows=(2, 1))
        record = []
        self._run(
            harness,
            self._decode_batch([False, False]),
            [1, 2],
            [0, 0],
            record=record,
        )
        self.assertEqual(len(record), 1)
        args, _ = record[0]
        # Both conv entries ride one launch, over 2 layers x 2 rows.
        self.assertEqual(args[5], 4)
        self.assertEqual(float(harness.mamba_cache.conv[0].abs().sum()), 0.0)

    def test_row_ids_are_layer_major(self):
        harness = _TrackHarness(num_layers=3, num_slots=6, windows=(2, 1))
        record = []
        self._run(
            harness, self._decode_batch([True, True]), [1, 2], [4, 5], record=record
        )
        args, _ = record[0]
        src_rows, mask_rows, dst_rows, total = args[2:6]
        self.assertEqual(src_rows.tolist(), [1, 2, 7, 8, 13, 14])
        self.assertEqual(dst_rows.tolist(), [4, 5, 10, 11, 16, 17])
        self.assertEqual(mask_rows.tolist(), [True] * 6)
        self.assertEqual(total, 6)

    def test_freed_slot_tombstone_is_masked_off(self):
        # The unified pool's virtual->physical translate emits -1 for a freed
        # slot. `layer_base + -1` is a VALID row of the previous layer, so the
        # kernel's own negative-index check cannot save us; the row has to be
        # masked out before the base is added.
        harness = _TrackHarness(num_layers=2, num_slots=6, windows=(2, 1))
        harness.backend.enable_unified_memory = True
        record = []
        self._run(
            harness, self._decode_batch([True, True]), [1, 2], [4, -1], record=record
        )
        args, _ = record[0]
        self.assertEqual(args[3].tolist(), [True, False, True, False])

    def test_freed_source_slot_is_masked_off(self):
        # _cache_indices has already clamped its -1s to the scratch slot, so
        # the source tombstone has to be read off the untouched metadata
        # tensor or the snapshot would copy slot 0's garbage.
        from sglang.srt.layers.attention.linear import short_conv_backend

        harness = _TrackHarness(num_layers=2, num_slots=6, windows=(2, 1))
        harness.backend.enable_unified_memory = True
        backend = harness.backend
        backend._cache_indices = torch.tensor([0, 2], dtype=torch.int64)
        backend.forward_metadata = SimpleNamespace(
            mamba_track_indices=torch.tensor([4, 5], dtype=torch.int64),
            mamba_cache_indices=torch.tensor([-1, 2], dtype=torch.int64),
        )
        record = []
        with unittest.mock.patch.object(
            short_conv_backend,
            "track_mamba_states_if_needed",
            lambda *a, **k: record.append(a),
        ):
            backend.track_conv_states_decode(self._decode_batch([True, True]))
        self.assertEqual(record[0][3].tolist(), [False, True, False, True])

    def test_extend_mode_does_not_take_the_decode_path(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        harness = _TrackHarness(num_layers=2, num_slots=6, windows=(2, 1))
        record = []
        fb = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            mamba_track_mask=torch.tensor([True], dtype=torch.bool),
        )
        self._run(harness, fb, [1], [4], record=record)
        self.assertEqual(record, [])

    @unittest.skipUnless(
        torch.cuda.is_available(), "the real track scatter is a Triton CUDA kernel"
    )
    def test_real_triton_scatter_matches_the_torch_reference(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        harness = _TrackHarness(num_layers=2, num_slots=6, windows=(2, 1))
        harness.mamba_cache = SimpleNamespace(
            conv=[c.cuda() for c in harness.mamba_cache.conv],
            temporal=harness.mamba_cache.temporal.cuda(),
        )
        harness.backend.device = torch.device("cuda")
        harness.backend._init_track_state(harness.server_args, harness.mamba_cache)
        conv0, conv1 = harness.mamba_cache.conv
        torch.manual_seed(5)
        conv0.normal_()
        conv1.normal_()
        expected0 = conv0.clone()
        expected1 = conv1.clone()
        for layer in range(2):
            expected0[layer, 4] = conv0[layer, 1]
            expected1[layer, 4] = conv1[layer, 1]

        backend = harness.backend
        backend._cache_indices = torch.tensor([1, 2], dtype=torch.int64, device="cuda")
        backend.forward_metadata = SimpleNamespace(
            mamba_track_indices=torch.tensor([4, 5], dtype=torch.int64, device="cuda")
        )
        backend.track_conv_states_decode(
            SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                mamba_track_mask=torch.tensor(
                    [True, False], dtype=torch.bool, device="cuda"
                ),
            )
        )
        self.assertTrue(torch.equal(conv0, expected0))
        self.assertTrue(torch.equal(conv1, expected1))


class TestShortConvNoBufferUnchanged(CustomTestCase):
    """``no_buffer`` must be exactly what it was before extra_buffer existed."""

    def test_no_track_state_is_built(self):
        harness = _TrackHarness(extra_buffer=False)
        self.assertIsNone(harness.backend._track_pairs)
        self.assertIsNone(harness.backend._track_layer_row_base)

    def test_both_track_entrypoints_are_inert(self):
        from sglang.srt.layers.attention.linear import short_conv_backend
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        harness = _TrackHarness(extra_buffer=False)
        record = []
        harness.backend._cache_indices = torch.tensor([1], dtype=torch.int64)
        harness.backend.forward_metadata = SimpleNamespace(
            mamba_track_indices=torch.tensor([4], dtype=torch.int64)
        )
        with unittest.mock.patch.object(
            short_conv_backend,
            "track_mamba_states_if_needed",
            lambda *a, **k: record.append(a),
        ):
            harness.backend.track_conv_states_decode(
                SimpleNamespace(
                    forward_mode=ForwardMode.DECODE,
                    mamba_track_mask=torch.tensor([True], dtype=torch.bool),
                )
            )
        harness.backend.track_conv_states_extend(
            harness.layer_cache(0), (torch.randn(4, 3), torch.randn(4, 4))
        )
        self.assertEqual(record, [])
        self.assertEqual(float(harness.mamba_cache.conv[0].abs().sum()), 0.0)
        self.assertEqual(float(harness.mamba_cache.conv[1].abs().sum()), 0.0)

    def test_zaya_resolves_to_extra_buffer_only_under_auto(self):
        # The arch is now on the allowlist, but an explicit strategy still
        # wins and `auto` without overlap/paging still lands on no_buffer.
        from sglang.srt.arg_groups.overrides import (
            ResolvedView,
            _mamba_radix_cache_resolution,
        )

        def _view(**kw):
            defaults = dict(
                disable_radix_cache=False,
                mamba_radix_cache_strategy="auto",
                disable_overlap_schedule=False,
                page_size=None,
                linear_attn_backend="triton",
            )
            defaults.update(kw)
            hf = SimpleNamespace(architectures=["ZayaForCausalLM"])
            return ResolvedView(
                SimpleNamespace(
                    get_model_config=lambda: SimpleNamespace(hf_config=hf), **defaults
                )
            )

        self.assertEqual(
            _mamba_radix_cache_resolution(_view()),
            {
                "uses_mamba_radix_cache": True,
                "mamba_radix_cache_strategy": "extra_buffer",
            },
        )
        declared = _mamba_radix_cache_resolution(
            _view(disable_overlap_schedule=True, page_size=1)
        )
        self.assertEqual(declared["mamba_radix_cache_strategy"], "no_buffer")
        self.assertIs(declared["disable_overlap_schedule"], True)
        pinned = _mamba_radix_cache_resolution(
            _view(mamba_radix_cache_strategy="no_buffer")
        )
        self.assertEqual(pinned, {"uses_mamba_radix_cache": True})

    def test_lfm2_is_deliberately_left_on_no_buffer(self):
        from sglang.srt.arg_groups.overrides import _MAMBA_EXTRA_BUFFER_ARCHS

        self.assertIn("ZayaForCausalLM", _MAMBA_EXTRA_BUFFER_ARCHS)
        self.assertNotIn("Lfm2ForCausalLM", _MAMBA_EXTRA_BUFFER_ARCHS)

    def test_ssm_backends_keep_their_temporal_track(self):
        # has_temporal_state is the only base-class behaviour change; every
        # SSM backend must still build its SSM track indices.
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            Mamba2AttnBackend,
            MambaAttnBackendBase,
        )
        from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
        from sglang.srt.layers.attention.linear.short_conv_backend import (
            ShortConvAttnBackend,
        )

        self.assertTrue(MambaAttnBackendBase.has_temporal_state)
        self.assertTrue(Mamba2AttnBackend.has_temporal_state)
        self.assertTrue(GDNAttnBackend.has_temporal_state)
        self.assertFalse(ShortConvAttnBackend.has_temporal_state)


class TestShortConvTrackStateGuards(CustomTestCase):
    """``_init_track_state`` refuses configurations it cannot snapshot."""

    def test_window_must_fit_inside_a_chunk(self):
        # The gather reaches `window` rows back from the chunk-aligned point;
        # a window >= chunk_size would have to read the cached prefix, which
        # is not in the current extend's token buffer at all.
        with self.assertRaises(AssertionError):
            _TrackHarness(chunk_size=2, windows=(2, 1))

    def test_track_interval_must_cover_a_chunk(self):
        with self.assertRaises(AssertionError):
            _TrackHarness(chunk_size=8, track_interval=4, windows=(2, 1))

    def test_single_conv_entry_pairs_with_the_empty_temporal(self):
        # LFM2 shape: one conv entry. The shared scatter takes two state
        # tensors per launch, so the zero-element temporal rides along.
        harness = _TrackHarness(windows=(3,))
        self.assertEqual(len(harness.backend._track_pairs), 1)
        state_a, state_b = harness.backend._track_pairs[0]
        self.assertEqual(state_a.shape[0], harness.num_layers * harness.num_slots)
        self.assertEqual(state_b[0].numel(), 0)

    def test_two_conv_entries_ride_one_launch(self):
        harness = _TrackHarness(windows=(2, 1))
        self.assertEqual(len(harness.backend._track_pairs), 1)
        state_a, state_b = harness.backend._track_pairs[0]
        self.assertEqual(state_a.shape[-1], 2)
        self.assertEqual(state_b.shape[-1], 1)

    def test_flattened_views_alias_the_pool(self):
        # The snapshot must land in the live pool, not a copy of it.
        harness = _TrackHarness(windows=(2, 1))
        state_a, state_b = harness.backend._track_pairs[0]
        self.assertIs(
            state_a.untyped_storage(),
            harness.mamba_cache.conv[0].untyped_storage(),
        )
        self.assertIs(
            state_b.untyped_storage(),
            harness.mamba_cache.conv[1].untyped_storage(),
        )

    def test_prefill_cuda_graph_is_refused(self):
        # The extend gather's row count is mamba_track_mask.sum(), so a
        # captured prefill graph would freeze it at whatever capture saw
        # (zero) and never snapshot again.
        from sglang.srt.layers.attention.linear.short_conv_backend import (
            ShortConvAttnBackend,
        )

        backend = object.__new__(ShortConvAttnBackend)
        backend.device = torch.device("cpu")
        backend.conv_window_lens = [2, 1]
        mamba_cache = SimpleNamespace(
            conv=[torch.zeros(2, 6, 3, 2), torch.zeros(2, 6, 4, 1)],
            temporal=torch.zeros(2, 6, 1, 1, 0),
        )
        with self.assertRaises(NotImplementedError):
            backend._init_track_state(
                SimpleNamespace(
                    mamba_cache_chunk_size=8,
                    mamba_track_interval=16,
                    speculative_algorithm=None,
                    cuda_graph_config=SimpleNamespace(
                        prefill=SimpleNamespace(backend="full")
                    ),
                ),
                mamba_cache,
            )
        # A disabled prefill graph is the supported configuration.
        backend._init_track_state(
            SimpleNamespace(
                mamba_cache_chunk_size=8,
                mamba_track_interval=16,
                speculative_algorithm=None,
                cuda_graph_config=SimpleNamespace(
                    prefill=SimpleNamespace(backend="disabled")
                ),
            ),
            mamba_cache,
        )
        self.assertEqual(len(backend._track_pairs), 1)

    def test_speculative_decoding_is_refused(self):
        # The decode graph runner drops its mamba-track buffers when a spec
        # algorithm is set, so the snapshot would silently never fire.
        from sglang.srt.layers.attention.linear.short_conv_backend import (
            ShortConvAttnBackend,
        )

        backend = object.__new__(ShortConvAttnBackend)
        backend.device = torch.device("cpu")
        backend.conv_window_lens = [2, 1]
        mamba_cache = SimpleNamespace(
            conv=[torch.zeros(2, 6, 3, 2), torch.zeros(2, 6, 4, 1)],
            temporal=torch.zeros(2, 6, 1, 1, 0),
        )
        with self.assertRaises(NotImplementedError):
            backend._init_track_state(
                SimpleNamespace(
                    mamba_cache_chunk_size=8,
                    mamba_track_interval=16,
                    speculative_algorithm="EAGLE",
                ),
                mamba_cache,
            )

    def test_strided_conv_layout_is_rejected(self):
        # Page-major / envelope conv views are strided, so `flatten(0, 1)` row
        # addressing is invalid. Fail loudly rather than snapshot wrong bytes.
        from sglang.srt.layers.attention.linear.short_conv_backend import (
            ShortConvAttnBackend,
        )

        backend = object.__new__(ShortConvAttnBackend)
        backend.device = torch.device("cpu")
        strided = torch.zeros(2, 6, 3, 4).transpose(2, 3)
        backend.conv_window_lens = [int(strided.shape[-1])]
        mamba_cache = SimpleNamespace(
            conv=[strided], temporal=torch.zeros(2, 6, 1, 1, 0)
        )
        with self.assertRaises(NotImplementedError):
            backend._init_track_state(
                SimpleNamespace(
                    mamba_cache_chunk_size=8,
                    mamba_track_interval=16,
                    speculative_algorithm=None,
                ),
                mamba_cache,
            )


class TestShortConvCacheChunkSize(CustomTestCase):
    """``mamba_cache_chunk_size`` must not be the model's scan length.

    Two different quantities share the word "chunk".
    ``hf_config.mamba_chunk_size`` is the SSM chunk-scan length; ZAYA1 and LFM2
    honestly report ``1`` because they have no chunked recurrence at all.
    ``ServerArgs.mamba_cache_chunk_size`` is the radix caching-point
    granularity -- how often a prefill checkpoints state and at what boundary a
    cached prefix may be trimmed. Taking the first as the second gave a
    granularity of 1, which is below ZAYA1's conv window and made
    ``ShortConvAttnBackend`` refuse to start.
    """

    @staticmethod
    def _lfm2_config():
        from sglang.srt.configs.lfm2 import Lfm2Config

        return Lfm2Config(
            hidden_size=16,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            conv_L_cache=3,
            layer_types=["conv", "full_attention", "conv", "full_attention"],
        )

    def test_zaya_window_is_the_two_conv_time_constants(self):
        # The window is (cca_time0 - 1) + (cca_time1 - 1) for conv[0] and 1 for
        # conv[1] -- a pure function of the conv time constants, with no TP term
        # (only the channel axis shards). That is what lets the memoized
        # property read it from either side of distributed init.
        from sglang.srt.server_args import _max_conv_state_window

        config = _make_tiny_config()
        expected = (config.cca_time0 - 1) + (config.cca_time1 - 1)
        self.assertEqual(_max_conv_state_window(config), expected)
        self.assertEqual(expected, 2)

    def test_conv_window_axis_is_tp_invariant(self):
        # Direct proof of the property the memoization relies on: sharding
        # moves the channel axis and leaves the window axis alone.
        from sglang.srt.configs.mamba_utils import Mamba2StateShape

        shapes = [
            Mamba2StateShape.create(
                tp_world_size=tp,
                intermediate_size=64,
                n_groups=1,
                num_heads=tp,
                head_dim=64,
                state_size=0,
                conv_kernel=4,
            )
            for tp in (1, 2, 4)
        ]
        windows = {s.conv[0][-1] for s in shapes}
        channels = {s.conv[0][0] for s in shapes}
        self.assertEqual(windows, {3})
        self.assertEqual(len(channels), 3)

    def test_zaya_gets_the_generic_chunk_not_its_scan_length(self):
        from sglang.srt.server_args import (
            FLA_CHUNK_SIZE,
            _short_conv_cache_chunk_size,
        )

        config = _make_tiny_config()
        self.assertEqual(config.mamba_chunk_size, 1)
        self.assertEqual(_short_conv_cache_chunk_size(config), FLA_CHUNK_SIZE)

    def test_lfm2_hits_the_same_wall_and_the_same_fix(self):
        # LFM2 is on no_buffer today, so this never fires -- but its config
        # reports mamba_chunk_size == 1 with a conv window of conv_L_cache - 1,
        # exactly ZAYA1's shape, so promoting it would have hit the identical
        # assertion. The derivation cures both.
        from sglang.srt.server_args import (
            FLA_CHUNK_SIZE,
            _max_conv_state_window,
            _short_conv_cache_chunk_size,
        )

        config = self._lfm2_config()
        self.assertEqual(config.mamba_chunk_size, 1)
        self.assertEqual(_max_conv_state_window(config), config.conv_L_cache - 1)
        self.assertEqual(_short_conv_cache_chunk_size(config), FLA_CHUNK_SIZE)

    def test_wide_window_rounds_up_and_stays_page_divisible(self):
        # A window at or above the generic chunk raises the granularity, but to
        # a whole number of FLA_CHUNK_SIZE so the caller's
        # max(chunk, page) % min(chunk, page) assert still holds for every page
        # size in use.
        from sglang.srt.server_args import (
            FLA_CHUNK_SIZE,
            _short_conv_cache_chunk_size,
        )

        wide = SimpleNamespace(
            mamba_chunk_size=1,
            mamba2_cache_params=SimpleNamespace(
                shape=SimpleNamespace(conv=[(8, FLA_CHUNK_SIZE + 36), (4, 1)])
            ),
        )
        chunk = _short_conv_cache_chunk_size(wide)
        self.assertEqual(chunk, 2 * FLA_CHUNK_SIZE)
        self.assertGreater(chunk, FLA_CHUNK_SIZE + 36)
        for page_size in (1, 16, 32, 64, 128):
            self.assertEqual(max(chunk, page_size) % min(chunk, page_size), 0)

    def test_unknown_conv_shape_falls_back_up_never_down(self):
        # An unreadable shape must not silently produce a granularity below the
        # generic chunk; the backend guard is then the thing that fails loudly.
        from sglang.srt.server_args import (
            FLA_CHUNK_SIZE,
            _max_conv_state_window,
            _short_conv_cache_chunk_size,
        )

        bare = SimpleNamespace(mamba_chunk_size=1)
        self.assertEqual(_max_conv_state_window(bare), 0)
        self.assertEqual(_short_conv_cache_chunk_size(bare), FLA_CHUNK_SIZE)

        class _Exploding:
            mamba_chunk_size = 1

            @property
            def mamba2_cache_params(self):
                raise AssertionError("tp size not divisible")

        self.assertEqual(_max_conv_state_window(_Exploding()), 0)
        self.assertEqual(_short_conv_cache_chunk_size(_Exploding()), FLA_CHUNK_SIZE)

    def test_multimodal_wrapper_window_is_found_via_text_config(self):
        from sglang.srt.server_args import _max_conv_state_window

        inner = self._lfm2_config()
        wrapper = SimpleNamespace(mamba_chunk_size=1, text_config=inner)
        self.assertEqual(_max_conv_state_window(wrapper), inner.conv_L_cache - 1)

    def test_ssm_models_keep_the_historical_derivation(self):
        # Only the mamba_chunk_size <= 1 branch changed. A real scan length, and
        # the FLA_CHUNK_SIZE default for a config that declares none, must both
        # go through max(scan_chunk, page_size) untouched -- so the short-conv
        # helper is never consulted for them.
        from sglang.srt.server_args import FLA_CHUNK_SIZE

        def resolve(hf_config, page_size):
            scan = getattr(hf_config, "mamba_chunk_size", FLA_CHUNK_SIZE)
            self.assertGreater(scan, 1)  # takes the untouched branch
            return max(scan, page_size)

        self.assertEqual(resolve(SimpleNamespace(mamba_chunk_size=256), 1), 256)
        self.assertEqual(resolve(SimpleNamespace(mamba_chunk_size=256), 128), 256)
        self.assertEqual(resolve(SimpleNamespace(), 1), FLA_CHUNK_SIZE)
        self.assertEqual(resolve(SimpleNamespace(), 128), 128)
        # Inkling already worked around this by pinning 64 in its config; the
        # new branch must leave that answer alone.
        self.assertEqual(resolve(SimpleNamespace(mamba_chunk_size=64), 1), 64)

    def test_derived_chunk_satisfies_the_backend_guard(self):
        # The regression this fixes end to end: with the old chunk of 1 the
        # short-conv backend refused to build its track state; with the derived
        # chunk it builds.
        from sglang.srt.server_args import _short_conv_cache_chunk_size

        config = _make_tiny_config()
        derived = _short_conv_cache_chunk_size(config)

        with self.assertRaises(AssertionError):
            _TrackHarness(chunk_size=1, windows=(2, 1))
        # mamba_track_interval defaults to 256, comfortably above the derived
        # 64 -- the backend's other precondition.
        harness = _TrackHarness(chunk_size=derived, track_interval=256, windows=(2, 1))
        self.assertEqual(len(harness.backend._track_pairs), 1)
        self.assertGreaterEqual(256, derived)

    def test_guard_names_the_minimum_viable_chunk(self):
        # The failure an operator sees has to say what value would work.
        with self.assertRaises(AssertionError) as caught:
            _TrackHarness(chunk_size=1, windows=(2, 1))
        message = str(caught.exception)
        self.assertIn("minimum viable chunk here is 3", message)
        self.assertIn("mamba_cache_chunk_size", message)


def _make_swa_config(
    *,
    num_hidden_layers: int,
    swa_layers,
    swa_rotary_base=10000,
    rope_theta: float = 10_000_000.0,
):
    """ZAYA1-74B-style config: a per-layer ``swa_layers`` window list (aligned
    with the global layer index, 0 == full attention) plus a dedicated RoPE
    base for the sliding-window layers."""
    from sglang.srt.configs.zaya import ZayaConfig

    return ZayaConfig(
        hidden_size=16,
        ffn_hidden_size=32,
        num_hidden_layers=num_hidden_layers,
        num_experts=2,
        num_attention_heads=4,
        num_query_groups=2,
        num_key_value_heads=2,
        head_dim=8,
        cca_time0=2,
        cca_time1=2,
        max_position_embeddings=64,
        moe_router_topk=1,
        zaya_mlp_expansion=8,
        attention_bias=False,
        rope_theta=rope_theta,
        swa_layers=swa_layers,
        swa_rotary_base=swa_rotary_base,
    )


class TestZayaSlidingWindowAttention(CustomTestCase):
    """ZAYA1-74B interleaves sliding-window attention with full attention and
    gives the sliding layers their own RoPE base (``swa_rotary_base``). These
    tests verify that ``ZayaConfig`` resolves the per-layer window from
    ``swa_layers`` and that ``ZayaAttention`` wires the matching
    ``RadixAttention.sliding_window_size`` and rotary base into each layer.
    """

    @classmethod
    def setUpClass(cls) -> None:
        _ensure_dist_initialized()
        # Building a RoPE cache reads a published config leaf, so this class
        # needs a published context; ``override_server_args`` is the sanctioned
        # way for a test to get one (it publishes and projects the bags).
        from sglang.srt.runtime_context import get_context

        cls._server_args_override = get_context().override_server_args(
            model_path="dummy"
        )
        cls._server_args_override.install()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._server_args_override.restore()

    def _build_attention(self, config, layer_id: int):
        from sglang.srt.models.zaya import ZayaAttention

        return ZayaAttention(config=config, layer_id=layer_id)

    def test_config_reports_per_layer_window(self):
        # 8 layers: attention layers live at the even ids 0/2/4/6 (zaya_layers
        # is None), and ``swa_layers`` marks 0 and 4 as sliding (window 4096).
        config = _make_swa_config(
            num_hidden_layers=8,
            swa_layers=[4096, 0, 0, 0, 4096, 0, 0, 0],
        )
        self.assertEqual(config.sliding_window_for_layer(0), 4096)
        self.assertEqual(config.sliding_window_for_layer(4), 4096)
        self.assertEqual(config.sliding_window_for_layer(2), 0)
        self.assertEqual(config.sliding_window_for_layer(6), 0)
        self.assertEqual(config.swa_window_size, 4096)
        # window - 1: the exclusive convention shared with the attention
        # backends (matches the Gemma reference models).
        self.assertEqual(config.get_attention_sliding_window_size(), 4095)

    def test_non_uniform_window_is_rejected(self):
        # The runtime tracks a single global window, so mixed window sizes
        # across SWA layers are unsupported and must fail loudly -- and must do
        # so while *constructing* the config (the hybrid-SWA opt-in resolves the
        # window in __init__), not lazily on first attribute read, so an
        # unsupported checkpoint is rejected at load rather than mid-serving.
        with self.assertRaises(AssertionError):
            _make_swa_config(
                num_hidden_layers=8,
                swa_layers=[4096, 0, 0, 0, 2048, 0, 0, 0],
            )

    def test_attention_selects_window_and_rope_base_per_layer(self):
        config = _make_swa_config(
            num_hidden_layers=8,
            swa_layers=[4096, 0, 0, 0, 4096, 0, 0, 0],
            swa_rotary_base=10000,
            rope_theta=10_000_000.0,
        )
        sliding = self._build_attention(config, layer_id=0)
        full = self._build_attention(config, layer_id=2)

        # Sliding layer: window-1 handed to RadixAttention, SWA rope base.
        self.assertTrue(sliding.is_sliding)
        self.assertEqual(sliding.attn.sliding_window_size, 4095)
        self.assertEqual(sliding.rotary_emb.base, 10000)

        # Full layer: no window (-1) and the global rope base.
        self.assertFalse(full.is_sliding)
        self.assertEqual(full.attn.sliding_window_size, -1)
        self.assertEqual(full.rotary_emb.base, 10_000_000)

        # Distinct rope bases must not collapse onto a shared rotary cache entry.
        self.assertIsNot(sliding.rotary_emb, full.rotary_emb)

    def test_base_checkpoint_has_no_sliding_window(self):
        # No ``swa_layers`` -> every attention layer is full attention and the
        # model reports no global window, preserving base-model behavior.
        config = _make_swa_config(num_hidden_layers=4, swa_layers=None)
        self.assertEqual(config.sliding_window_for_layer(0), 0)
        self.assertIsNone(config.swa_window_size)
        self.assertIsNone(config.get_attention_sliding_window_size())

        attn = self._build_attention(config, layer_id=0)
        self.assertFalse(attn.is_sliding)
        self.assertEqual(attn.attn.sliding_window_size, -1)
        self.assertEqual(attn.rotary_emb.base, int(config.rope_theta))

    def test_hybrid_layer_pattern_is_indexed_by_global_layer_id(self):
        # The pattern is indexed by *global* layer id and covers every layer so
        # get_hybrid_layer_ids can index it directly. Note the two same-named
        # properties mean different things: ZayaConfig.full_attention_layer_ids is
        # the mamba interface's "every attention layer" (each carries a CCA conv
        # state), whereas ModelConfig.full_attention_layer_ids after the SWA split
        # means "non-sliding attention layers" only.
        config = _make_swa_config(
            num_hidden_layers=8,
            swa_layers=[4096, 0, 0, 0, 4096, 0, 0, 0],
        )
        self.assertEqual(config.hybrid_layer_pattern, [1, -1, 0, -1, 1, -1, 0, -1])
        self.assertEqual(config.swa_attention_layer_ids, [0, 4])
        self.assertEqual(config.full_attention_layer_ids, [0, 2, 4, 6])

    def test_base_checkpoint_reports_no_hybrid_pattern(self):
        # Without swa_layers the model must not opt in to the hybrid-SWA KV
        # pool: a non-None pattern here would make get_hybrid_layer_ids split a
        # uniformly full-attention model into empty/complete halves.
        config = _make_swa_config(num_hidden_layers=4, swa_layers=None)
        self.assertIsNone(config.hybrid_layer_pattern)
        self.assertEqual(config.swa_attention_layer_ids, [])

    def test_window_size_conventions(self):
        # Two different conventions coexist and must not be conflated:
        # ``sliding_window_size`` is the inclusive window that ModelConfig reads
        # from the HF config, while the attention backends take the exclusive
        # ``window - 1`` via the model's get_attention_sliding_window_size.
        swa = _make_swa_config(
            num_hidden_layers=8, swa_layers=[4096, 0, 0, 0, 4096, 0, 0, 0]
        )
        self.assertEqual(swa.sliding_window_size, 4096)
        self.assertEqual(swa.get_attention_sliding_window_size(), 4095)

        base = _make_swa_config(num_hidden_layers=4, swa_layers=None)
        self.assertIsNone(base.sliding_window_size)

    def test_hybrid_swa_optin_is_per_checkpoint(self):
        # is_hybrid_swa is the generic escape from ModelConfig's arch allowlist, so
        # it must key off the checkpoint (swa_layers) rather than the architecture:
        # base ZAYA1 checkpoints have no sliding-window layers and must stay on the
        # single-pool path.
        from sglang.srt.configs.model_config import is_hybrid_swa_model

        swa = _make_swa_config(
            num_hidden_layers=8, swa_layers=[4096, 0, 0, 0, 4096, 0, 0, 0]
        )
        self.assertTrue(swa.is_hybrid_swa)
        self.assertTrue(is_hybrid_swa_model(["ZayaForCausalLM"], swa))

        base = _make_swa_config(num_hidden_layers=4, swa_layers=None)
        self.assertFalse(base.is_hybrid_swa)
        self.assertFalse(is_hybrid_swa_model(["ZayaForCausalLM"], base))

    def test_moe_layers_are_excluded_from_both_kv_layer_lists(self):
        # Regression guard. ZAYA1 alternates attention (even) and MoE (odd) layers,
        # and the MoE layers hold no KV at all. get_hybrid_layer_ids derives
        # swa_attention_layer_ids from `pattern == 1` and full_attention_layer_ids
        # from `pattern == 0`, and those lists SIZE the SWA sub-pools rather than
        # merely indexing them -- so reporting MoE layers as 0 made the full
        # sub-pool 90 layers wide instead of 30 on the 74B, tripling its per-token
        # cost (23039 vs 7680 bytes/token of K) and turning hybrid-SWA into a
        # capacity regression. MoE layers must therefore be in NEITHER list.
        from sglang.srt.configs.model_config import get_hybrid_layer_ids

        config = _make_swa_config(
            num_hidden_layers=8, swa_layers=[4096, 0, 0, 0, 4096, 0, 0, 0]
        )
        self.assertEqual(config.hybrid_layer_pattern, [1, -1, 0, -1, 1, -1, 0, -1])

        swa_ids, full_ids = get_hybrid_layer_ids(["ZayaForCausalLM"], config)
        self.assertEqual(swa_ids, [0, 4])
        self.assertEqual(full_ids, [2, 6])
        self.assertEqual(set(swa_ids) & set(full_ids), set())
        # Every KV-bearing layer is an attention layer, and no MoE layer appears.
        moe_layers = {1, 3, 5, 7}
        self.assertEqual(set(swa_ids + full_ids) & moe_layers, set())
        self.assertEqual(
            sorted(swa_ids + full_ids), sorted(config.full_attention_layer_ids)
        )


class TestZayaCCATensorParallel(CustomTestCase):
    """Head-parallel TP equivalence:

    For each TP rank, the CCA's q / k / v output must equal the head slice of
    the TP=1 reference's output that corresponds to that rank's heads. This
    verifies that the grouped-mean step and ``conv_qk.1`` (groups = num_q_heads
    + num_k_heads) are correctly partitioned across heads with no cross-rank
    leakage.
    """

    TP_SIZE = 2

    @classmethod
    def setUpClass(cls) -> None:
        _ensure_dist_initialized()

    def _slice_full_state_dict_into_rank(self, ref_cca, tp_cca, tp_rank: int):
        """Copy the reference's full weights into the per-rank CCA, using the
        per-parameter ``weight_loader`` that the CCA installs on its own
        parameters during ``__init__``. This mirrors what
        ``ZayaForCausalLM.load_weights`` does at serving time and is the
        only way TP correctness is exercised end-to-end.
        """
        ref_state = dict(ref_cca.state_dict())
        from sglang.srt.model_loader.weight_utils import default_weight_loader

        with torch.no_grad():
            for name, param in tp_cca.named_parameters():
                full_weight = ref_state[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, full_weight)

    def _check_per_rank_outputs(
        self,
        full_q: torch.Tensor,
        full_k: torch.Tensor,
        full_v: torch.Tensor,
        rank_q: torch.Tensor,
        rank_k: torch.Tensor,
        rank_v: torch.Tensor,
        tp_rank: int,
        cfg,
    ):
        """Compare a TP=2 rank's output against the corresponding head slice
        of the TP=1 reference output. Q heads and K heads are partitioned
        contiguously across ranks: rank ``r`` owns
        ``[r*Q_per_rank, (r+1)*Q_per_rank)`` for Q and similarly for K.
        """
        q_heads_per_rank = cfg.num_attention_heads // self.TP_SIZE
        k_heads_per_rank = cfg.num_query_groups // self.TP_SIZE
        q_lo, q_hi = tp_rank * q_heads_per_rank, (tp_rank + 1) * q_heads_per_rank
        k_lo, k_hi = tp_rank * k_heads_per_rank, (tp_rank + 1) * k_heads_per_rank
        torch.testing.assert_close(
            rank_q, full_q[:, q_lo:q_hi, :], atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            rank_k, full_k[:, k_lo:k_hi, :], atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            rank_v, full_v[:, k_lo:k_hi, :], atol=1e-5, rtol=1e-5
        )

    def test_tp2_extend_matches_full(self):
        """Single-chunk extend with TP=2 produces the same q / k / v slices
        as a TP=1 reference, verified rank-by-rank.
        """
        ref_cca, cfg = _make_tiny_cca(seed=21, tp_rank=0, tp_size=1)
        S = 6
        torch.manual_seed(901)
        hs = torch.randn(S, ref_cca.hidden_size, dtype=torch.float32) * 0.1

        ref_pool = _MockReqToTokenPool(pool_size=8, cca_config=cfg, tp_size=1)
        ref_fb = _make_forward_batch(
            is_decode=False,
            extend_seq_lens_cpu=[S],
            extend_prefix_lens_cpu=[0],
            req_pool_indices=[0],
            input_ids=torch.arange(S, dtype=torch.int64),
        )
        with _mock_pool_context(ref_pool):
            full_q, full_k, full_v = ref_cca.forward(hs, ref_fb)

        for tp_rank in range(self.TP_SIZE):
            rank_cca, _ = _make_tiny_cca(
                seed=21 + tp_rank, tp_rank=tp_rank, tp_size=self.TP_SIZE
            )
            self._slice_full_state_dict_into_rank(ref_cca, rank_cca, tp_rank)
            rank_pool = _MockReqToTokenPool(
                pool_size=8, cca_config=cfg, tp_size=self.TP_SIZE
            )
            rank_fb = _make_forward_batch(
                is_decode=False,
                extend_seq_lens_cpu=[S],
                extend_prefix_lens_cpu=[0],
                req_pool_indices=[0],
                input_ids=torch.arange(S, dtype=torch.int64),
            )
            with _mock_pool_context(rank_pool):
                rank_q, rank_k, rank_v = rank_cca.forward(hs, rank_fb)
            self._check_per_rank_outputs(
                full_q, full_k, full_v, rank_q, rank_k, rank_v, tp_rank, cfg
            )

    def test_tp2_decode_matches_full(self):
        """Prefill(S0) + decode(1 token) with TP=2 produces the same q / k / v
        slices as a TP=1 reference, verifying that the per-rank conv state
        and prev_hs cache (which is replicated on every rank) agree.
        """
        ref_cca, cfg = _make_tiny_cca(seed=22, tp_rank=0, tp_size=1)
        S0 = 5
        torch.manual_seed(902)
        hs_prefill = torch.randn(S0, ref_cca.hidden_size, dtype=torch.float32) * 0.1
        hs_decode = torch.randn(1, ref_cca.hidden_size, dtype=torch.float32) * 0.1

        ref_pool = _MockReqToTokenPool(pool_size=8, cca_config=cfg, tp_size=1)
        with _mock_pool_context(ref_pool):
            ref_cca.forward(
                hs_prefill,
                _make_forward_batch(
                    is_decode=False,
                    extend_seq_lens_cpu=[S0],
                    extend_prefix_lens_cpu=[0],
                    req_pool_indices=[0],
                    input_ids=torch.arange(S0, dtype=torch.int64),
                ),
            )
            full_q, full_k, full_v = ref_cca.forward(
                hs_decode,
                _make_forward_batch(
                    is_decode=True,
                    extend_seq_lens_cpu=[],
                    extend_prefix_lens_cpu=[],
                    req_pool_indices=[0],
                    input_ids=torch.tensor([0], dtype=torch.int64),
                ),
            )

        for tp_rank in range(self.TP_SIZE):
            rank_cca, _ = _make_tiny_cca(
                seed=22 + tp_rank, tp_rank=tp_rank, tp_size=self.TP_SIZE
            )
            self._slice_full_state_dict_into_rank(ref_cca, rank_cca, tp_rank)
            rank_pool = _MockReqToTokenPool(
                pool_size=8, cca_config=cfg, tp_size=self.TP_SIZE
            )
            with _mock_pool_context(rank_pool):
                rank_cca.forward(
                    hs_prefill,
                    _make_forward_batch(
                        is_decode=False,
                        extend_seq_lens_cpu=[S0],
                        extend_prefix_lens_cpu=[0],
                        req_pool_indices=[0],
                        input_ids=torch.arange(S0, dtype=torch.int64),
                    ),
                )
                rank_q, rank_k, rank_v = rank_cca.forward(
                    hs_decode,
                    _make_forward_batch(
                        is_decode=True,
                        extend_seq_lens_cpu=[],
                        extend_prefix_lens_cpu=[],
                        req_pool_indices=[0],
                        input_ids=torch.tensor([0], dtype=torch.int64),
                    ),
                )
            self._check_per_rank_outputs(
                full_q, full_k, full_v, rank_q, rank_k, rank_v, tp_rank, cfg
            )

    def test_tp2_conv_state_is_per_rank_sliced(self):
        """After a TP=2 prefill, each rank's conv state must equal the head
        slice of the TP=1 conv state corresponding to that rank's heads.
        """
        ref_cca, cfg = _make_tiny_cca(seed=23, tp_rank=0, tp_size=1)
        S = 4
        torch.manual_seed(903)
        hs = torch.randn(S, ref_cca.hidden_size, dtype=torch.float32) * 0.1

        ref_pool = _MockReqToTokenPool(pool_size=4, cca_config=cfg, tp_size=1)
        with _mock_pool_context(ref_pool):
            ref_cca.forward(
                hs,
                _make_forward_batch(
                    is_decode=False,
                    extend_seq_lens_cpu=[S],
                    extend_prefix_lens_cpu=[0],
                    req_pool_indices=[0],
                    input_ids=torch.arange(S, dtype=torch.int64),
                ),
            )
        full_state = ref_pool.mamba2_layer_cache(0).conv[0][0]  # [in_out_ch_full, pad]

        head_dim = cfg.head_dim
        num_q_heads_full = cfg.num_attention_heads
        num_k_heads_full = cfg.num_query_groups
        latent_q_full = num_q_heads_full * head_dim
        q_per_rank = num_q_heads_full // self.TP_SIZE
        k_per_rank = num_k_heads_full // self.TP_SIZE

        for tp_rank in range(self.TP_SIZE):
            rank_cca, _ = _make_tiny_cca(
                seed=23 + tp_rank, tp_rank=tp_rank, tp_size=self.TP_SIZE
            )
            self._slice_full_state_dict_into_rank(ref_cca, rank_cca, tp_rank)
            rank_pool = _MockReqToTokenPool(
                pool_size=4, cca_config=cfg, tp_size=self.TP_SIZE
            )
            with _mock_pool_context(rank_pool):
                rank_cca.forward(
                    hs,
                    _make_forward_batch(
                        is_decode=False,
                        extend_seq_lens_cpu=[S],
                        extend_prefix_lens_cpu=[0],
                        req_pool_indices=[0],
                        input_ids=torch.arange(S, dtype=torch.int64),
                    ),
                )
            rank_state = rank_pool.mamba2_layer_cache(0).conv[0][0]

            q_lo = tp_rank * q_per_rank * head_dim
            q_hi = q_lo + q_per_rank * head_dim
            k_lo = latent_q_full + tp_rank * k_per_rank * head_dim
            k_hi = k_lo + k_per_rank * head_dim
            expected = torch.cat([full_state[q_lo:q_hi], full_state[k_lo:k_hi]], dim=0)

            torch.testing.assert_close(rank_state, expected, atol=1e-5, rtol=1e-5)

    def test_tp2_lag_stream_only_exists_on_the_val_proj2_rank(self):
        """At attn_tp == 2 with two K heads, only rank 1 reads ``val_proj2``.

        The HF layout gives rank 0 its V heads entirely from ``val_proj1``, so it
        needs no lag at all: no projection, no pool read, no pool write. Rank 1
        needs the whole of ``val_proj2``'s output. Pinning both halves matters
        because the write side and the read side derive the range separately --
        if they disagree, rank 1 reads a slot nothing filled (silently zero) or
        rank 0 pays a GEMM it never uses.
        """
        ref_cca, cfg = _make_tiny_cca(seed=24, tp_rank=0, tp_size=1)
        S = 4
        torch.manual_seed(904)
        hs = torch.randn(S, ref_cca.hidden_size, dtype=torch.float32) * 0.1

        # tp=1 straddles the boundary and uses all of val_proj2's output.
        self.assertTrue(ref_cca.v_uses_val1)
        self.assertTrue(ref_cca.v_uses_val2)
        self.assertEqual(ref_cca.v2_lag_dim, cfg.cca_v2_state_dim)

        expected = {0: (True, False, 0), 1: (False, True, cfg.cca_v2_state_dim)}
        for tp_rank in range(self.TP_SIZE):
            rank_cca, _ = _make_tiny_cca(
                seed=24 + tp_rank, tp_rank=tp_rank, tp_size=self.TP_SIZE
            )
            self._slice_full_state_dict_into_rank(ref_cca, rank_cca, tp_rank)
            uses1, uses2, lag_dim = expected[tp_rank]
            self.assertEqual(rank_cca.v_uses_val1, uses1)
            self.assertEqual(rank_cca.v_uses_val2, uses2)
            self.assertEqual(rank_cca.v2_lag_dim, lag_dim)

            rank_pool = _MockReqToTokenPool(
                pool_size=4, cca_config=cfg, tp_size=self.TP_SIZE
            )
            with _mock_pool_context(rank_pool):
                rank_cca.forward(
                    hs,
                    _make_forward_batch(
                        is_decode=False,
                        extend_seq_lens_cpu=[S],
                        extend_prefix_lens_cpu=[0],
                        req_pool_indices=[0],
                        input_ids=torch.arange(S, dtype=torch.int64),
                    ),
                )
            lag_slot = rank_pool.mamba2_layer_cache(0).conv[1][0]
            # The pool entry stays rank-uniform (a per-rank size would desync
            # max_mamba_cache_size across the TP group); rank 0 just leaves it at
            # the zeros MambaPool handed out.
            self.assertEqual(lag_slot.shape[-2], cfg.cca_v2_state_dim)
            if uses2:
                self.assertTrue(torch.any(lag_slot != 0))
                with torch.no_grad():
                    expected_row = rank_cca.val_proj2(hs[-1:])[0].squeeze(0)
                torch.testing.assert_close(
                    lag_slot.squeeze(-1), expected_row, atol=1e-5, rtol=1e-5
                )
            else:
                self.assertTrue(torch.all(lag_slot == 0))

    def test_tp_assertions_reject_indivisible_head_counts(self):
        """The CCA constructor must reject TP sizes that don't evenly divide
        both num_q_heads and num_k_heads, since both grouped-mean and
        conv_qk.1 require each rank to hold whole K-head groups.
        """
        from sglang.srt.models.zaya import CCA

        cfg = _make_tiny_config()
        # tiny config has num_query_groups=2; TP=4 cannot divide it cleanly.
        with self.assertRaises(AssertionError):
            CCA(
                config=cfg,
                cca_num_k_heads=cfg.num_query_groups,
                cca_num_q_heads=cfg.num_attention_heads,
                hidden_size=cfg.hidden_size,
                head_dim=cfg.head_dim,
                cca_time0=cfg.cca_time0,
                cca_time1=cfg.cca_time1,
                layer_id=0,
                tp_rank=0,
                tp_size=4,
            )


@contextmanager
def _dp_layout(sizes: List[int], rank: int, is_max_len: bool):
    """Install ``sizes`` as the per-rank padded token counts, viewed from ``rank``.

    Mirrors what ``prepare_mlp_sync_batch`` publishes for a real forward: the DP
    rank globals plus the process-wide DP buffer metadata.
    """
    from sglang.srt.layers import dp_attention as dpa

    saved = (dpa._ATTN_DP_RANK, dpa._ATTN_DP_SIZE)
    saved_buffer = (
        dpa._DpGatheredBufferWrapper._global_dp_buffer_len,
        dpa._DpGatheredBufferWrapper._local_dp_buffer_len,
        dpa._DpGatheredBufferWrapper._dp_max_padding,
        dpa._DpGatheredBufferWrapper._global_num_tokens,
    )
    dpa._ATTN_DP_RANK, dpa._ATTN_DP_SIZE = rank, len(sizes)
    dpa.set_dp_buffer_len(sum(sizes), sizes[rank], is_max_len, list(sizes))
    try:
        yield
    finally:
        dpa._ATTN_DP_RANK, dpa._ATTN_DP_SIZE = saved
        dpa.set_dp_buffer_len(*saved_buffer)


class TestZayaGlobalResidualLayout(CustomTestCase):
    """Row arithmetic for the global-residual DP dataflow.

    On that dataflow the residual stream lives in the global DP layout and each
    attention layer slices its own rows back out of it, so the CPU offsets in
    ``GlobalResidualLayout`` and the device-side offsets that ``dp_gather_partial``
    writes at (``get_dp_local_info``, a cumsum over ``global_num_tokens_gpu``) must
    describe the same rows. Nothing at runtime cross-checks them: if they drift,
    attention silently reads another replica's tokens.
    """

    def _layout(self, sizes: List[int], rank: int, is_max_len: bool):
        from unittest import mock

        from sglang.srt.environ import envs
        from sglang.srt.models import zaya

        # The parallel-layout precondition is orthogonal to the arithmetic under
        # test and needs a live runtime context, so stub it out.
        with _dp_layout(sizes, rank, is_max_len), mock.patch.object(
            zaya, "dp_gather_required", return_value=True
        ), envs.SGLANG_OPT_ZAYA_GLOBAL_RESIDUAL.override(True):
            return zaya.global_residual_layout()

    def _device_slice(self, sizes: List[int], rank: int):
        from sglang.srt.layers.dp_attention import get_dp_local_info

        forward_batch = SimpleNamespace(
            dp_local_start_pos=None,
            dp_local_num_tokens=None,
            global_num_tokens_gpu=torch.tensor(sizes, dtype=torch.int32),
        )
        with _dp_layout(sizes, rank, is_max_len=False):
            start, length = get_dp_local_info(forward_batch)
        return int(start), int(length)

    def test_offsets_match_the_gather_destination(self):
        """SUM_LEN: unequal per-rank counts, so a wrong cumsum shows up as a shift."""
        sizes = [3, 7, 1, 5]
        for rank in range(len(sizes)):
            layout = self._layout(sizes, rank, is_max_len=False)
            self.assertEqual(
                (layout.local_start, layout.local_len),
                self._device_slice(sizes, rank),
                f"CPU layout disagrees with get_dp_local_info at rank {rank}",
            )

    def test_offsets_match_under_max_len_padding(self):
        """MAX_LEN: every rank padded to the same width (the cuda-graph layout)."""
        sizes = [8, 8, 8, 8]
        for rank in range(len(sizes)):
            layout = self._layout(sizes, rank, is_max_len=True)
            self.assertEqual(
                (layout.local_start, layout.local_len),
                (rank * 8, 8),
            )

    def test_slices_tile_the_buffer_without_gaps_or_overlap(self):
        """Every row of the global buffer belongs to exactly one replica.

        The MoE layers run over the whole buffer, so a gap would feed the experts
        uninitialized rows and an overlap would double-count a token.
        """
        sizes = [3, 7, 1, 5]
        covered = torch.zeros(sum(sizes), dtype=torch.int32)
        for rank in range(len(sizes)):
            layout = self._layout(sizes, rank, is_max_len=False)
            covered[layout.local_start : layout.local_start + layout.local_len] += 1
        self.assertTrue(torch.equal(covered, torch.ones_like(covered)))

    def test_idle_replica_gets_an_empty_slice(self):
        """A replica with no requests must slice to zero rows, not to its neighbour's.

        Its attention still runs (and returns early on the empty batch) and it must
        still join the gather, so the layout has to survive a zero-width block.
        """
        sizes = [4, 0, 4]
        layout = self._layout(sizes, 1, is_max_len=False)
        self.assertEqual((layout.local_start, layout.local_len), (4, 0))
        hidden = torch.arange(8 * 2, dtype=torch.float32).reshape(8, 2)
        self.assertEqual(layout.local_view(hidden).shape, (0, 2))

    def test_local_view_is_a_contiguous_alias(self):
        """Attention and the gather both assert contiguity on what they are handed."""
        layout = self._layout([3, 7, 1, 5], 1, is_max_len=False)
        hidden = torch.randn(16, 4)
        view = layout.local_view(hidden)
        self.assertTrue(view.is_contiguous())
        self.assertEqual(view.data_ptr(), hidden[3].data_ptr())

    def test_disabled_by_default_without_touching_parallel_state(self):
        """Flag off must yield the DP-local dataflow, and decide that first.

        The env check has to short-circuit ahead of the parallel-layout probe, or
        merely importing the model on a machine with no runtime context breaks.
        """
        from sglang.srt.models import zaya

        with _dp_layout([3, 7, 1, 5], 0, is_max_len=False):
            self.assertIsNone(zaya.global_residual_layout())


class TestZayaPartialGatherFoldsTheAttnReduce(CustomTestCase):
    """The algebra that lets one collective replace two.

    The global-residual dataflow drops ``attn_tp_all_reduce`` from the attention
    layer and gathers the *unreduced* o_proj partials instead: every attention-TP
    rank of a replica memcpys its partial into the same slot of the global buffer,
    so the all-reduce that gathers across replicas also sums within them. These
    cases pin that equivalence, and the failure mode of getting it wrong (using the
    replicate gather, which takes rank 0's rows and discards the rest).
    """

    HIDDEN = 4

    def _gather(self, partials: List[List[torch.Tensor]], sizes: List[int], is_partial):
        """Replay ``_dp_gather_via_all_reduce`` over all ranks on host tensors.

        Each rank zero-fills the buffer and memcpys its own rows into its replica's
        slot -- only attention-TP rank 0 does so on the replicate gather -- and the
        all-reduce leaves the sum of those per-rank buffers behind.
        """
        from sglang.srt.layers.dp_attention import get_dp_local_info

        total = torch.zeros(sum(sizes), self.HIDDEN)
        for replica, rank_partials in enumerate(partials):
            forward_batch = SimpleNamespace(
                dp_local_start_pos=None,
                dp_local_num_tokens=None,
                global_num_tokens_gpu=torch.tensor(sizes, dtype=torch.int32),
            )
            with _dp_layout(sizes, replica, is_max_len=False):
                start, length = get_dp_local_info(forward_batch)
            for attn_tp_rank, partial in enumerate(rank_partials):
                if not is_partial and attn_tp_rank != 0:
                    continue
                contribution = torch.zeros(sum(sizes), self.HIDDEN)
                contribution[int(start) : int(start) + int(length)] = partial
                total = total + contribution
        return total

    def _partials(self, sizes: List[int], attn_tp_size: int):
        torch.manual_seed(0)
        return [
            [torch.randn(rows, self.HIDDEN) for _ in range(attn_tp_size)]
            for rows in sizes
        ]

    def test_partial_gather_equals_reduce_then_replicate_gather(self):
        sizes = [3, 7, 1, 5]
        partials = self._partials(sizes, attn_tp_size=2)
        # Baseline: reduce within the replica, then gather the reduced rows.
        reduced = [[sum(rank_partials)] for rank_partials in partials]
        baseline = self._gather(reduced, sizes, is_partial=False)
        folded = self._gather(partials, sizes, is_partial=True)
        torch.testing.assert_close(folded, baseline)

    def test_replicate_gather_of_partials_drops_a_rank(self):
        """Guards the swap this campaign has already made in the other direction.

        ``dp_gather_replicate`` is correct for already-reduced rows and wrong for
        partials: it keeps attention-TP rank 0's contribution and silently discards
        every other rank's, which at attn_tp=1 is invisible and at attn_tp=2 is
        garbage output.
        """
        sizes = [3, 7, 1, 5]
        partials = self._partials(sizes, attn_tp_size=2)
        folded = self._gather(partials, sizes, is_partial=True)
        wrong = self._gather(partials, sizes, is_partial=False)
        self.assertFalse(torch.allclose(folded, wrong))

    def test_single_attn_tp_rank_makes_the_two_gathers_agree(self):
        """At attn_tp=1 there is nothing to sum, so both gathers must coincide."""
        sizes = [3, 7, 1, 5]
        partials = self._partials(sizes, attn_tp_size=1)
        torch.testing.assert_close(
            self._gather(partials, sizes, is_partial=True),
            self._gather(partials, sizes, is_partial=False),
        )


class TestDpGatherStagingFusion(CustomTestCase):
    """The two launches per sum_len partial gather that C1 removes.

    ``_dp_gather_via_all_reduce`` used to issue four launches: ``fill_(0)``, a
    memcpy of this rank's rows into its slot, the all-reduce, and a copy of the
    (out-of-place) all-reduce result back into the caller's buffer. The fill now
    folds into the memcpy, and the copy-back is skipped by callers that take the
    returned tensor. Both are meant to be *exactly* the old behaviour, so pin the
    fused kernel against ``fill_(0) + memcpy`` and pin what the gather returns
    for an out-of-place and an in-place collective.
    """

    @staticmethod
    @contextmanager
    def _sum_len_all_reduce(all_reduce):
        """Drive the sum_len gather path on host tensors.

        Patches out everything that needs a live NCCL group: the WORLD-gather
        switch, the attention-TP rank, the world size, the collective itself and
        the memcpy dispatch (CPU tensors cannot go through Triton).
        """
        from unittest import mock

        from sglang.srt.layers import dp_attention as dpa

        with mock.patch.object(
            dpa, "memcpy_scatter_zero_rest_func", dpa.memcpy_scatter_zero_rest_cpu
        ), mock.patch.object(
            dpa, "world_dp_gather_enabled", lambda: False
        ), mock.patch.object(
            dpa, "get_attn_tensor_model_parallel_rank", lambda: 0
        ), mock.patch.object(
            dpa, "get_tensor_model_parallel_world_size", lambda: 1
        ), mock.patch.object(
            dpa, "tensor_model_parallel_all_reduce", all_reduce
        ):
            yield

    @staticmethod
    def _forward_batch(sizes: List[int]):
        from sglang.srt.layers.dp_attention import DpPaddingMode

        return SimpleNamespace(
            dp_local_start_pos=None,
            dp_local_num_tokens=None,
            global_num_tokens_gpu=torch.tensor(sizes, dtype=torch.int32),
            dp_padding_mode=DpPaddingMode.SUM_LEN,
        )

    def test_fused_scatter_zero_matches_fill_then_memcpy(self):
        """The fused staging write must equal ``fill_(0)`` then ``memcpy``."""
        from sglang.srt.layers.dp_attention import (
            memcpy_cpu,
            memcpy_scatter_zero_rest_cpu,
        )

        torch.manual_seed(3)
        rows, hidden = 16, 5
        # sz == 0 (idle replica) and sz < src rows (a src padded for cuda graph)
        # are the two cases where the fill is the only thing zeroing a row.
        for offset, sz, src_rows in ((0, 4, 4), (4, 3, 3), (12, 4, 4), (7, 0, 2)):
            with self.subTest(offset=offset, sz=sz):
                src = torch.randn(src_rows, hidden)
                # A dirty destination: the fill is what makes the untouched rows
                # zero, so starting from zeros would hide a missing zero-fill.
                dirty = torch.randn(rows, hidden)
                ref = dirty.clone()
                ref.fill_(0)
                memcpy_cpu(
                    ref,
                    src,
                    0,
                    torch.tensor(offset),
                    torch.tensor(sz),
                    False,
                )
                got = dirty.clone()
                memcpy_scatter_zero_rest_cpu(
                    got, src, 0, torch.tensor(offset), torch.tensor(sz)
                )
                self.assertTrue(torch.equal(got, ref))

    @unittest.skipUnless(torch.cuda.is_available(), "Triton memcpy needs a GPU")
    def test_fused_scatter_zero_matches_on_device(self):
        """Same equivalence for the Triton kernel the serving path uses."""
        from sglang.kernels.ops.memory.memcpy_triton import (
            memcpy_scatter_zero_rest_triton,
            memcpy_triton,
        )

        dev = "cuda"
        torch.manual_seed(4)
        cases = (
            # (dst shape, src shape, offset, sz)
            ((16, 5), (4, 5), 4, 4),
            ((16, 5), (8, 5), 0, 3),
            ((16, 5), (4, 5), 12, 4),
            ((16, 5), (4, 5), 7, 0),
            # 1-D int32 input ids: chunk_size collapses to 1.
            ((16,), (4,), 8, 4),
        )
        for dst_shape, src_shape, offset, sz in cases:
            for dtype in (torch.bfloat16, torch.int32):
                with self.subTest(dst=dst_shape, off=offset, sz=sz, dt=dtype):
                    if dtype.is_floating_point:
                        src = torch.randn(src_shape, device=dev).to(dtype)
                        dirty = torch.randn(dst_shape, device=dev).to(dtype)
                    else:
                        src = torch.randint(1, 99, src_shape, device=dev, dtype=dtype)
                        dirty = torch.randint(1, 99, dst_shape, device=dev, dtype=dtype)
                    off_t = torch.tensor(offset, device=dev, dtype=torch.int32)
                    sz_t = torch.tensor(sz, device=dev, dtype=torch.int32)

                    ref = dirty.clone()
                    ref.fill_(0)
                    memcpy_triton(ref, src, 0, off_t, sz_t, False)
                    got = dirty.clone()
                    memcpy_scatter_zero_rest_triton(got, src, 0, off_t, sz_t)
                    self.assertTrue(torch.equal(got, ref))

    def test_out_variant_returns_the_collective_output(self):
        """An out-of-place all-reduce: the result is the collective's buffer."""
        from sglang.srt.layers.dp_attention import dp_gather_partial_out

        sizes = [3, 7, 1, 5]
        hidden = 4
        marker = torch.full((sum(sizes), hidden), 7.0)

        def all_reduce(x):
            # Mimic the custom all-reduce: a fresh buffer, input untouched.
            return x + marker

        local = torch.randn(sizes[1], hidden)
        staging = torch.randn(sum(sizes), hidden)
        fb = self._forward_batch(sizes)
        with _dp_layout(sizes, 1, is_max_len=False):
            with self._sum_len_all_reduce(all_reduce):
                out = dp_gather_partial_out(staging, local, fb)

        expected = torch.zeros(sum(sizes), hidden)
        expected[3 : 3 + 7] = local
        self.assertIsNot(out, staging)
        torch.testing.assert_close(out, expected + marker)
        # The staging buffer is left holding the pre-reduce content, which is
        # what makes the copy-back removable: nothing reads it afterwards.
        torch.testing.assert_close(staging, expected)

    def test_out_variant_returns_the_buffer_for_an_in_place_collective(self):
        """An in-place all-reduce returns its input, so the buffer is the result."""
        from sglang.srt.layers.dp_attention import dp_gather_partial_out

        sizes = [2, 2]
        hidden = 3

        def all_reduce(x):
            x.add_(1.0)
            return x

        local = torch.randn(sizes[0], hidden)
        staging = torch.randn(sum(sizes), hidden)
        fb = self._forward_batch(sizes)
        with _dp_layout(sizes, 0, is_max_len=False):
            with self._sum_len_all_reduce(all_reduce):
                out = dp_gather_partial_out(staging, local, fb)

        expected = torch.zeros(sum(sizes), hidden)
        expected[:2] = local
        self.assertIs(out, staging)
        torch.testing.assert_close(out, expected + 1.0)

    def test_in_place_wrapper_still_fills_the_caller_buffer(self):
        """``dp_gather_partial`` keeps its in-place contract for every caller.

        This is what makes C1 a no-op for the models that did not opt into the
        returning variant: the copy back into ``global_tokens`` is still there
        when (and only when) the collective handed back a different tensor.
        """
        from sglang.srt.layers.dp_attention import dp_gather_partial

        sizes = [2, 2]
        hidden = 3

        def all_reduce(x):
            return x + 5.0

        local = torch.randn(sizes[1], hidden)
        buf = torch.randn(sum(sizes), hidden)
        fb = self._forward_batch(sizes)
        with _dp_layout(sizes, 1, is_max_len=False):
            with self._sum_len_all_reduce(all_reduce):
                dp_gather_partial(buf, local, fb)

        expected = torch.zeros(sum(sizes), hidden)
        expected[2:] = local
        torch.testing.assert_close(buf, expected + 5.0)


def _reference_extend_conv(
    qk: torch.Tensor,
    lag_now: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    conv_state: torch.Tensor,
    lag_state: torch.Tensor,
    seq_lens: List[int],
    slot_ids: List[int],
    has_prefix: List[bool],
    total_padding: int,
    groups: int,
):
    """Per-request torch reference for the fused prefill conv.

    Deliberately the naive formulation -- one ``F.conv1d`` per request with an
    explicitly built left pad -- rather than a copy of ``cca_extend``, so the two
    agree only if the varlen indexing is right.
    """
    qk_out = torch.empty_like(qk)
    lag_out = torch.empty_like(lag_now)
    new_conv_state = conv_state.clone()
    new_lag_state = lag_state.clone()

    start = 0
    for i, seq_len in enumerate(seq_lens):
        end = start + seq_len
        slot = slot_ids[i]
        cur = qk[start:end].transpose(0, 1).unsqueeze(0)  # [1, C, S]
        if has_prefix[i]:
            left = conv_state[slot].unsqueeze(0).to(cur.dtype)
        else:
            left = cur.new_zeros((1, cur.shape[1], total_padding))
        padded = torch.cat([left, cur], dim=-1)
        out = torch.nn.functional.conv1d(padded, weight, bias, groups=groups)
        qk_out[start:end] = out.squeeze(0).transpose(0, 1)
        new_conv_state[slot] = (
            padded[..., -total_padding:].squeeze(0).to(conv_state.dtype)
        )

        lag_cur = lag_now[start:end]
        if has_prefix[i]:
            first = lag_state[slot].squeeze(-1).to(lag_cur.dtype).unsqueeze(0)
        else:
            first = lag_cur.new_zeros((1, lag_cur.shape[-1]))
        lag_out[start:end] = torch.cat([first, lag_cur[:-1]], dim=0)
        new_lag_state[slot] = lag_cur[-1].unsqueeze(-1).to(lag_state.dtype)
        start = end

    return qk_out, lag_out, new_conv_state, new_lag_state


class TestCCAFusedPrefillConv(CustomTestCase):
    """Fused varlen prefill conv vs the per-request torch reference.

    The fused kernel resolves each token's request, start offset, pool slot and
    prefix flag from device tensors instead of a host loop. Every one of those is a
    silent-corruption path: a token attributed to the wrong request reads another
    request's conv history, and nothing downstream notices.
    """

    GROUPS = 3
    CG = 16  # tl.dot needs a tile at least 16 wide
    # The lag stream is the projected val_proj2 value, not the hidden state, so
    # it is narrower than the conv channel count -- keep the test shape that way
    # too rather than reusing a hidden_size-shaped stream.
    LAG_DIM = 8
    TOTAL_PADDING = 2

    def _inputs(self, seq_lens: List[int], has_prefix: List[bool], seed: int = 0):
        torch.manual_seed(seed)
        channels = self.GROUPS * self.CG
        taps = self.TOTAL_PADDING + 1
        total = sum(seq_lens)
        num_slots = len(seq_lens) + 2
        dev = "cuda"
        dt = torch.bfloat16
        return dict(
            qk=torch.randn(total, channels, device=dev, dtype=dt),
            lag_now=torch.randn(total, self.LAG_DIM, device=dev, dtype=dt),
            weight=torch.randn(channels, self.CG, taps, device=dev, dtype=dt) * 0.1,
            bias=torch.randn(channels, device=dev, dtype=dt) * 0.1,
            conv_state=torch.randn(
                num_slots, channels, self.TOTAL_PADDING, device=dev, dtype=dt
            ),
            lag_state=torch.randn(num_slots, self.LAG_DIM, 1, device=dev, dtype=dt),
            seq_lens=seq_lens,
            has_prefix=has_prefix,
        )

    def _run(self, seq_lens, has_prefix, slot_ids=None, seed=0):
        from sglang.kernels.ops.attention import cca_conv1d

        args = self._inputs(seq_lens, has_prefix, seed)
        slot_ids = slot_ids or list(range(len(seq_lens)))
        offsets = [0]
        for s_len in seq_lens:
            offsets.append(offsets[-1] + s_len)
        cu = torch.tensor(offsets, dtype=torch.int32, device="cuda")
        slots = torch.tensor(slot_ids, dtype=torch.int64, device="cuda")
        prefix = torch.tensor(has_prefix, dtype=torch.bool, device="cuda")

        ref_qk, ref_lag, ref_cs, ref_ls = _reference_extend_conv(
            args["qk"],
            args["lag_now"],
            args["weight"],
            args["bias"],
            args["conv_state"],
            args["lag_state"],
            seq_lens,
            slot_ids,
            has_prefix,
            self.TOTAL_PADDING,
            self.GROUPS,
        )

        conv_state = args["conv_state"].clone()
        lag_state = args["lag_state"].clone()
        self.assertTrue(
            cca_conv1d.covered(
                args["qk"],
                args["lag_now"],
                args["weight"],
                args["bias"],
                conv_state,
                lag_state,
                cu,
                prefix,
                slots,
                self.TOTAL_PADDING,
                self.GROUPS,
            )
        )
        got_qk, got_lag = cca_conv1d.cca_conv1d_fn(
            args["qk"],
            args["lag_now"],
            args["weight"],
            args["bias"],
            conv_state,
            lag_state,
            cu,
            prefix,
            slots,
            self.TOTAL_PADDING,
            self.GROUPS,
        )
        return (got_qk, got_lag, conv_state, lag_state), (
            ref_qk,
            ref_lag,
            ref_cs,
            ref_ls,
        )

    def _assert_matches(self, got, ref):
        names = ("qk_out", "lag_prev", "conv_state", "lag_state")
        for name, g, r in zip(names, got, ref):
            torch.testing.assert_close(
                g.float(), r.float(), rtol=2e-2, atol=2e-2, msg=f"{name} mismatch"
            )

    @unittest.skipUnless(torch.cuda.is_available(), "fused conv is a GPU kernel")
    def test_fresh_prefill_multi_request(self):
        got, ref = self._run([5, 3, 8], [False, False, False])
        self._assert_matches(got, ref)

    @unittest.skipUnless(torch.cuda.is_available(), "fused conv is a GPU kernel")
    def test_resumed_prefix_reads_carried_state(self):
        """Mixed prefix flags: the halo taps must come from each slot's history."""
        got, ref = self._run([6, 4, 7], [True, False, True])
        self._assert_matches(got, ref)

    @unittest.skipUnless(torch.cuda.is_available(), "fused conv is a GPU kernel")
    def test_chunk_shorter_than_the_conv_window(self):
        """A 1-token chunk with a prefix: the outgoing window is mostly carried.

        This is the branch where the new conv_state is not fully determined by the
        current chunk, so the tail write has to shift the incoming history rather
        than just copy the last tokens.
        """
        got, ref = self._run([1, 2, 1], [True, True, True])
        self._assert_matches(got, ref)

    @unittest.skipUnless(torch.cuda.is_available(), "fused conv is a GPU kernel")
    def test_non_identity_slot_mapping(self):
        """Slots are pool indices, not request indices, and need not be ordered."""
        got, ref = self._run([4, 4, 4], [True, True, True], slot_ids=[3, 0, 2])
        self._assert_matches(got, ref)

    @unittest.skipUnless(torch.cuda.is_available(), "fused conv is a GPU kernel")
    def test_request_longer_than_one_token_tile(self):
        """Tokens are tiled in blocks of 64; a request must span tiles correctly."""
        got, ref = self._run([200, 17], [True, False])
        self._assert_matches(got, ref)

    @unittest.skipUnless(torch.cuda.is_available(), "fused conv is a GPU kernel")
    def test_single_request(self):
        """One request exercises the degenerate binary search (a single step)."""
        got, ref = self._run([37], [True])
        self._assert_matches(got, ref)

    @unittest.skipUnless(torch.cuda.is_available(), "fused conv is a GPU kernel")
    def test_chunked_prefill_boundary_carries_the_projected_value(self):
        """Two chunks of one request must equal a single chunk over both.

        The fused path's chunk boundary lives in ``_boundary_state_kernel``: it
        parks the chunk's last lag row in the pool slot and, next chunk, reads it
        back as the row preceding the first token. That row is the PROJECTED
        val_proj2 value; if the boundary ever carried the raw hidden state
        instead, the resumed tokens would read a wrong v2 with no shape or dtype
        error -- the output would just get worse.

        Splitting a request and comparing against the unsplit run pins the carry
        in both directions (what is written, and what is read back), which the
        single-shot ``test_resumed_prefix_reads_carried_state`` case cannot: it
        seeds the slot with random values rather than with the kernel's own
        output.
        """
        from sglang.kernels.ops.attention import cca_conv1d

        dev, dt = "cuda", torch.bfloat16
        channels = self.GROUPS * self.CG
        taps = self.TOTAL_PADDING + 1
        s0, s1 = 5, 7
        total = s0 + s1
        torch.manual_seed(31)
        qk = torch.randn(total, channels, device=dev, dtype=dt)
        lag_now = torch.randn(total, self.LAG_DIM, device=dev, dtype=dt)
        weight = torch.randn(channels, self.CG, taps, device=dev, dtype=dt) * 0.1
        bias = torch.randn(channels, device=dev, dtype=dt) * 0.1
        slots = torch.tensor([1], dtype=torch.int64, device=dev)

        def _fresh_pools():
            return (
                torch.zeros(4, channels, self.TOTAL_PADDING, device=dev, dtype=dt),
                torch.zeros(4, self.LAG_DIM, 1, device=dev, dtype=dt),
            )

        def _run(qk_c, lag_c, cs, ls, prefix):
            cu = torch.tensor([0, qk_c.shape[0]], dtype=torch.int32, device=dev)
            pf = torch.tensor([prefix], dtype=torch.bool, device=dev)
            self.assertTrue(
                cca_conv1d.covered(
                    qk_c,
                    lag_c,
                    weight,
                    bias,
                    cs,
                    ls,
                    cu,
                    pf,
                    slots,
                    self.TOTAL_PADDING,
                    self.GROUPS,
                )
            )
            return cca_conv1d.cca_conv1d_fn(
                qk_c,
                lag_c,
                weight,
                bias,
                cs,
                ls,
                cu,
                pf,
                slots,
                self.TOTAL_PADDING,
                self.GROUPS,
            )

        cs_one, ls_one = _fresh_pools()
        one_qk, one_lag = _run(qk, lag_now, cs_one, ls_one, False)

        cs_two, ls_two = _fresh_pools()
        a_qk, a_lag = _run(qk[:s0], lag_now[:s0], cs_two, ls_two, False)
        b_qk, b_lag = _run(qk[s0:], lag_now[s0:], cs_two, ls_two, True)

        torch.testing.assert_close(
            torch.cat([a_qk, b_qk]).float(), one_qk.float(), rtol=2e-2, atol=2e-2
        )
        # The lag stream is pure data movement, so require exact equality: any
        # boundary bug shows as a wrong row, never as rounding.
        self.assertTrue(torch.equal(torch.cat([a_lag, b_lag]), one_lag))
        self.assertTrue(torch.equal(cs_two, cs_one))
        self.assertTrue(torch.equal(ls_two, ls_one))
        # The carried row really is this chunk's last projected value.
        self.assertTrue(torch.equal(ls_two[1].squeeze(-1), lag_now[-1]))

    @unittest.skipUnless(torch.cuda.is_available(), "fused conv is a GPU kernel")
    def test_no_lag_stream_leaves_the_conv_half_identical(self):
        """A rank with no lag drops two of the four launches.

        Ranks whose K heads all come from ``val_proj1`` never read the lag, so
        they pass ``None`` and the shifted copy plus the boundary kernel are
        skipped. The conv half must be untouched by that, and the lag pool must
        stay exactly as handed over.
        """
        from sglang.kernels.ops.attention import cca_conv1d

        args = self._inputs([6, 3], [True, False], seed=12)
        cu = torch.tensor([0, 6, 9], dtype=torch.int32, device="cuda")
        slots = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
        prefix = torch.tensor([True, False], dtype=torch.bool, device="cuda")

        cs_ref = args["conv_state"].clone()
        ls_ref = args["lag_state"].clone()
        ref_qk, _ = cca_conv1d.cca_conv1d_fn(
            args["qk"],
            args["lag_now"],
            args["weight"],
            args["bias"],
            cs_ref,
            ls_ref,
            cu,
            prefix,
            slots,
            self.TOTAL_PADDING,
            self.GROUPS,
        )

        cs_got = args["conv_state"].clone()
        ls_got = args["lag_state"].clone()
        self.assertTrue(
            cca_conv1d.covered(
                args["qk"],
                None,
                args["weight"],
                args["bias"],
                cs_got,
                None,
                cu,
                prefix,
                slots,
                self.TOTAL_PADDING,
                self.GROUPS,
            )
        )
        got_qk, got_lag = cca_conv1d.cca_conv1d_fn(
            args["qk"],
            None,
            args["weight"],
            args["bias"],
            cs_got,
            None,
            cu,
            prefix,
            slots,
            self.TOTAL_PADDING,
            self.GROUPS,
        )
        self.assertIsNone(got_lag)
        self.assertTrue(torch.equal(got_qk, ref_qk))
        self.assertTrue(torch.equal(cs_got, cs_ref))
        # Never handed the lag pool, so it cannot have moved.
        self.assertTrue(torch.equal(ls_got, args["lag_state"]))

    def test_covered_rejects_a_half_specified_lag_pair(self):
        """One lag tensor without the other is refused, not guessed at.

        Guessing "no lag" would skip the chunk-boundary carry, and the next chunk
        would silently resume from a stale slot. This runs on CPU: ``covered()``
        rejects the pair before it ever reaches the device checks.
        """
        from sglang.kernels.ops.attention import cca_conv1d

        qk = torch.randn(4, self.GROUPS * self.CG)
        lag = torch.randn(4, self.LAG_DIM)
        weight = torch.randn(self.GROUPS * self.CG, self.CG, self.TOTAL_PADDING + 1)
        bias = torch.randn(self.GROUPS * self.CG)
        cs = torch.randn(3, self.GROUPS * self.CG, self.TOTAL_PADDING)
        ls = torch.randn(3, self.LAG_DIM, 1)
        cu = torch.tensor([0, 4], dtype=torch.int32)
        pf = torch.zeros(1, dtype=torch.bool)
        slots = torch.zeros(1, dtype=torch.int64)
        for lag_now, lag_state in ((lag, None), (None, ls)):
            with self.subTest(lag_now=lag_now is not None):
                self.assertFalse(
                    cca_conv1d.covered(
                        qk,
                        lag_now,
                        weight,
                        bias,
                        cs,
                        lag_state,
                        cu,
                        pf,
                        slots,
                        self.TOTAL_PADDING,
                        self.GROUPS,
                    )
                )

    @unittest.skipUnless(torch.cuda.is_available(), "fused conv is a GPU kernel")
    def test_covered_rejects_the_unfolded_two_stage_conv(self):
        """Without the folded weight there is nothing for this kernel to apply."""
        from sglang.kernels.ops.attention import cca_conv1d

        args = self._inputs([4], [False])
        cu = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
        self.assertFalse(
            cca_conv1d.covered(
                args["qk"],
                args["lag_now"],
                None,
                args["bias"],
                args["conv_state"],
                args["lag_state"],
                cu,
                torch.zeros(1, dtype=torch.bool, device="cuda"),
                torch.zeros(1, dtype=torch.int64, device="cuda"),
                self.TOTAL_PADDING,
                self.GROUPS,
            )
        )


class TestCCAFusedDecodeConv(CustomTestCase):
    """Fused decode conv (window + shift + matmul) vs the existing two-launch path.

    The reference here is the unfused chain the model runs today, so this pins the
    claim that folding the grouped matmul into the window build changes nothing
    numerically -- and that the in-place history shift still reads each tap before
    it is overwritten.
    """

    GROUPS = 3
    CG = 16
    LAG_DIM = 8
    TOTAL_PADDING = 2

    def _run(self, num_tokens: int, slot_ids: List[int]):
        from sglang.kernels.ops.attention import cca_conv1d_update

        torch.manual_seed(0)
        channels = self.GROUPS * self.CG
        taps = self.TOTAL_PADDING + 1
        dev, dt = "cuda", torch.bfloat16
        num_slots = max(s for s in slot_ids) + 2

        qk = torch.randn(num_tokens, channels, device=dev, dtype=dt)
        lag_now = torch.randn(num_tokens, self.LAG_DIM, device=dev, dtype=dt)
        # The einsum path's weight spans taps + 1 inputs per channel; the extra
        # column is the bias, which this kernel does not read (it adds the bias
        # itself). Fill it so a kernel that mistakenly indexed it would show up.
        coeffs = torch.randn(self.GROUPS, self.CG, self.CG, taps, device=dev, dtype=dt)
        coeffs *= 0.1
        weight = torch.randn(
            self.GROUPS, self.CG, self.CG, taps + 1, device=dev, dtype=dt
        )
        weight[..., :taps] = coeffs
        weight = weight.reshape(self.GROUPS, self.CG, self.CG * (taps + 1)).contiguous()
        bias = torch.randn(self.GROUPS, self.CG, device=dev, dtype=dt) * 0.1
        conv_state = torch.randn(
            num_slots, channels, self.TOTAL_PADDING, device=dev, dtype=dt
        )
        lag_state = torch.randn(num_slots, self.LAG_DIM, 1, device=dev, dtype=dt)
        slots = torch.tensor(slot_ids, dtype=torch.int64, device=dev)

        # Reference: build the window, apply the einsum over the real taps only,
        # add the bias, shift the pools.
        live = [s for s in slot_ids if s >= 0]
        ref_cs, ref_ls = conv_state.clone(), lag_state.clone()
        left = conv_state.index_select(0, slots.clamp(min=0))
        window = torch.cat([left, qk.unsqueeze(-1)], dim=-1)
        grouped = window.reshape(num_tokens, self.GROUPS, -1)
        packed = coeffs.reshape(self.GROUPS, self.CG, self.CG * taps)
        ref_qk = (
            torch.einsum("tgk,gok->tgo", grouped.float(), packed.float()) + bias.float()
        ).reshape(num_tokens, -1)
        ref_prev = lag_state.index_select(0, slots.clamp(min=0)).squeeze(-1)
        for i, s in enumerate(slot_ids):
            if s >= 0:
                ref_cs[s] = window[i, :, -self.TOTAL_PADDING :]
                ref_ls[s] = lag_now[i].unsqueeze(-1)

        got_cs, got_ls = conv_state.clone(), lag_state.clone()
        self.assertTrue(
            cca_conv1d_update.covered(
                qk,
                lag_now,
                weight,
                bias,
                got_cs,
                got_ls,
                slots,
                self.TOTAL_PADDING,
                self.GROUPS,
            )
        )
        got_qk, got_prev = cca_conv1d_update.cca_conv1d_update(
            qk,
            lag_now,
            weight,
            bias,
            got_cs,
            got_ls,
            slots,
            self.TOTAL_PADDING,
            self.GROUPS,
        )
        torch.testing.assert_close(got_qk.float(), ref_qk.float(), rtol=2e-2, atol=2e-2)
        if live:
            idx = torch.tensor(live, device=dev)
            torch.testing.assert_close(
                got_cs.index_select(0, idx).float(),
                ref_cs.index_select(0, idx).float(),
            )
            torch.testing.assert_close(
                got_ls.index_select(0, idx).float(),
                ref_ls.index_select(0, idx).float(),
            )
        return got_prev, ref_prev

    @unittest.skipUnless(torch.cuda.is_available(), "fused conv is a GPU kernel")
    def test_matches_the_unfused_chain(self):
        got, ref = self._run(24, list(range(24)))
        torch.testing.assert_close(got.float(), ref.float())

    @unittest.skipUnless(torch.cuda.is_available(), "fused conv is a GPU kernel")
    def test_spans_multiple_token_tiles(self):
        got, ref = self._run(80, list(range(80)))
        torch.testing.assert_close(got.float(), ref.float())

    @unittest.skipUnless(torch.cuda.is_available(), "fused conv is a GPU kernel")
    def test_padded_rows_leave_the_pool_untouched(self):
        """Batch padding writes negative slot ids; those rows must touch no state."""
        from sglang.kernels.ops.attention import cca_conv1d_update

        torch.manual_seed(1)
        channels = self.GROUPS * self.CG
        taps = self.TOTAL_PADDING + 1
        dev, dt = "cuda", torch.bfloat16
        qk = torch.randn(4, channels, device=dev, dtype=dt)
        lag_now = torch.randn(4, self.LAG_DIM, device=dev, dtype=dt)
        weight = (
            torch.randn(
                self.GROUPS, self.CG, self.CG * (taps + 1), device=dev, dtype=dt
            )
            * 0.1
        )
        bias = torch.randn(self.GROUPS, self.CG, device=dev, dtype=dt) * 0.1
        conv_state = torch.randn(3, channels, self.TOTAL_PADDING, device=dev, dtype=dt)
        lag_state = torch.randn(3, self.LAG_DIM, 1, device=dev, dtype=dt)
        before_cs, before_ls = conv_state.clone(), lag_state.clone()
        slots = torch.tensor([-1, -1, -1, -1], dtype=torch.int64, device=dev)

        cca_conv1d_update.cca_conv1d_update(
            qk,
            lag_now,
            weight,
            bias,
            conv_state,
            lag_state,
            slots,
            self.TOTAL_PADDING,
            self.GROUPS,
        )
        self.assertTrue(torch.equal(conv_state, before_cs))
        self.assertTrue(torch.equal(lag_state, before_ls))


class TestZayaMoDReachability(CustomTestCase):
    """``ZayaRouter.fold_mod_reachability`` decides whether MOD is live.

    ``balancing_biases`` is added to a *softmax probability*, not a logit, so the
    skip slot's score is bounded above by ``1 + b_skip`` while every real expert's
    is at least ``b_j``. When ``1 + b_skip < max_j b_j`` the skip slot is strictly
    below the best real slot for every possible input and no tie-breaking rule can
    pick it -- so the two MOD kernels per MoE layer are dead work.

    ZAYA1-74B ships ``b_skip = -1.0`` on all 60 MoE layers, so this decides the
    real checkpoint. Pinned because the branch is a silent 120-launch-per-step
    cost when it is wrong in one direction, and silently wrong OUTPUT when it is
    wrong in the other.
    """

    def _router(self, *, use_mod=True):
        from sglang.srt.models.zaya import ZayaRouter

        config = _make_swa_config(num_hidden_layers=4, swa_layers=[0, 0, 0, 0])
        config.zaya_use_mod = use_mod
        return ZayaRouter(
            config=config,
            num_moe_experts=4,
            moe_router_topk=1,
            mlp_expansion=8,
            layer_id=1,
        )

    def test_skip_bias_of_minus_one_makes_mod_dead(self):
        # The shipped ZAYA1-74B layout: skip at -1.0, real biases straddling 0.
        router = self._router()
        with torch.no_grad():
            router.balancing_biases.copy_(
                torch.tensor([0.03, -0.06, 0.02, -0.01, -1.0])
            )
        router.fold_mod_reachability()
        self.assertFalse(router.mod_reachable)

    def test_all_real_biases_negative_keeps_mod_live(self):
        # 1 + b_skip == 0 is NOT below max_j b_j here, so the proof does not hold
        # and the branch must stay -- the check is conservative by design.
        router = self._router()
        with torch.no_grad():
            router.balancing_biases.copy_(
                torch.tensor([-0.03, -0.06, -0.02, -0.01, -1.0])
            )
        router.fold_mod_reachability()
        self.assertTrue(router.mod_reachable)

    def test_a_reachable_skip_bias_keeps_mod_live(self):
        # Distinct from the case above: there the real biases fail the test, here
        # the SKIP bias does. Catches a predicate that ignores b_skip entirely.
        router = self._router()
        with torch.no_grad():
            router.balancing_biases.copy_(torch.tensor([0.03, 0.01, 0.02, 0.0, 0.5]))
        router.fold_mod_reachability()
        self.assertTrue(router.mod_reachable)

    def test_boundary_is_strict(self):
        # 1 + b_skip exactly equal to max_j b_j must stay live: torch.argmax
        # tie-breaking is unspecified, so equality is not a proof of anything.
        router = self._router()
        with torch.no_grad():
            router.balancing_biases.copy_(torch.tensor([0.25, 0.1, 0.0, 0.0, -0.75]))
        router.fold_mod_reachability()
        self.assertTrue(router.mod_reachable)

    def test_mod_disabled_in_config_is_never_reachable(self):
        # Without the use_mod early return there is no skip slot at all, so
        # ``biases[-1]`` is a real expert's bias and the comparison reads garbage
        # -- which could mark MOD live on a model that has no skip path.
        router = self._router(use_mod=False)
        router.fold_mod_reachability()
        self.assertFalse(router.mod_reachable)


# ---------------------------------------------------------------------------
# ZayaRouter: fused tail (softmax + bias + top-1 + gather + id munging)
# ---------------------------------------------------------------------------


def _make_router_config(num_moe_experts: int, mlp_expansion: int = 8, **overrides):
    from sglang.srt.configs.zaya import ZayaConfig

    kwargs = dict(
        hidden_size=16,
        ffn_hidden_size=32,
        num_hidden_layers=2,
        num_experts=num_moe_experts,
        num_attention_heads=4,
        num_query_groups=2,
        num_key_value_heads=2,
        head_dim=8,
        cca_time0=2,
        cca_time1=2,
        max_position_embeddings=64,
        moe_router_topk=1,
        zaya_mlp_expansion=mlp_expansion,
        attention_bias=False,
    )
    kwargs.update(overrides)
    return ZayaConfig(**kwargs)


def _make_tiny_router(
    num_moe_experts: int = 4,
    mlp_expansion: int = 8,
    seed: int = 0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    std: float = 0.5,
    **config_overrides,
):
    """A ``ZayaRouter`` with random weights, on ``device`` at ``dtype``.

    ``std`` matters once ``mlp_expansion`` is realistic: three stacked dense
    stages at std 0.5 and width 256 amplify by ~500x, which saturates the
    downstream softmax and makes a numerical comparison meaningless. Pass
    ``1 / sqrt(mlp_expansion)`` for the wide cases.

    Deliberately does NOT call ``_ensure_dist_initialized()``. ZayaRouter is
    built entirely from ``ReplicatedLinear`` and ``RMSNorm``, neither of which
    reads model-parallel state, so it constructs and runs a full forward with
    ``torch.distributed`` never initialized -- unlike CCA, which queries the TP
    rank in ``__init__`` and does need the helper.

    That is not a tidy-up. Calling the helper here aborted the process on
    gfx950: see the warning on ``_ensure_dist_initialized`` for why. The abort
    landed inside ``ReplicatedLinear.__init__``, which is simply where the CPU
    thread was, and cost a round of hunting through Triton kernels that were
    never at fault. Do not add it back.
    """
    from sglang.srt.models.zaya import ZayaRouter

    config = _make_router_config(num_moe_experts, mlp_expansion, **config_overrides)
    torch.manual_seed(seed)
    router = ZayaRouter(
        config=config,
        layer_id=0,
        num_moe_experts=num_moe_experts,
        moe_router_topk=int(config.moe_router_topk),
        mlp_expansion=mlp_expansion,
    )
    router.eval()
    with torch.no_grad():
        for p in router.parameters():
            p.data.normal_(mean=0.0, std=std)
    router = router.to(device=device, dtype=dtype)
    # ``.to(dtype)`` drags the fp32 balancing-bias buffer along with the weights.
    # The real weight loader leaves it fp32, so put it back rather than testing a
    # configuration that never ships.
    with torch.no_grad():
        router.balancing_biases = router.balancing_biases.float()

    if torch.device(device).type == "cpu":
        # sglang's fused ops resolve their implementation from the PLATFORM,
        # once, and cache it -- never from the device of the tensors they are
        # later handed (BaseFusedOp._resolve_forward_method, and RMSNorm.__init__
        # which pins forward_aiter outright when _use_aiter). So on a machine
        # that HAS a GPU, this CPU-only router still holds a GPU RMSNorm, and
        # running its reference forward drives a device kernel down host
        # pointers. That is not a supported configuration, and it does not fail
        # cleanly: it corrupts the heap and the process aborts at some later
        # allocation -- for us, inside the nn.GELU two ops downstream, which
        # reads as a fault in code that never touched a GPU.
        #
        # Pin the native path. On a CPU-only machine this is exactly what
        # dispatch resolves to anyway, so CI behaviour is unchanged; it only
        # stops the GPU boxes from running the GPU kernel on host memory.
        norm = router.rmsnorm_eda
        norm._forward_method = norm.forward_native
    return router, config


# Exactly representable in fp32 and bf16 (mantissa 1.0), so it round-trips
# through a tensor and back to a Python float unchanged. 1e30 does NOT: float32
# stores it as 1.0000000150474662e+30, and assertEqual against the literal 1e30
# then fails. That is not a hypothetical -- it is the second of two setup bugs
# that made the probes below go red on hardware and read, both times, as a
# kernel fault.
_POISON = 2.0**100


def _poisoned_row_view(num_rows, num_cols, padded_cols, device, dtype, seed=0):
    """A ``num_cols``-wide view of a ``padded_cols``-wide poisoned buffer.

    Returns ``(view, backing)``. The view is what the kernel is handed: unit
    innermost stride and exactly ``num_cols`` wide, but with a row stride that
    steps over ``padded_cols``, so the columns a padded lane would reach are
    real, mapped, and unmistakable.
    """
    torch.manual_seed(seed)
    backing = torch.empty(num_rows, padded_cols, device=device, dtype=dtype)
    backing[:, num_cols:] = _POISON
    backing[:, :num_cols] = torch.randn(num_rows, num_cols, device=device).to(dtype)
    return backing[:, :num_cols], backing


def _poisoned_bias_view(num_experts, padded, device):
    """A contiguous ``num_experts`` prefix of a wider, poisoned bias vector.

    The real values go in AFTER the poison, and they are a per-index ramp so
    that smearing one lane onto another changes the answer rather than
    cancelling out. Index ``num_experts - 1`` carries the MOD skip slot's -1,
    which is the value the first version of this fixture clobbered by writing
    ``backing[-1]`` -- the last element of the *backing* buffer, not of the view.
    """
    backing = torch.empty(padded, dtype=torch.float32, device=device)
    backing[num_experts:] = _POISON
    real = torch.arange(num_experts, dtype=torch.float32, device=device) * 0.001
    real[-1] = -1.0
    backing[:num_experts] = real
    return backing[:num_experts], backing


class _SyncEachTest:
    """Synchronise the device on both sides of every test in the class.

    Attribution scaffolding, not a behavioural check. A HIP fault is reported
    asynchronously and aborts the process wherever the CPU thread happens to be,
    so without a sync bracketing each test body the abort surfaces in an
    unrelated test -- which is how the first report of this fault came in
    pointing at ReplicatedLinear.__init__ inside a CPU-only test.

    Syncing in BOTH setUp and tearDown is what makes it airtight: between any two
    consecutive syncs there is exactly one test body. An abort in tearDown blames
    that test; an abort in setUp blames the window since the previous sync. A
    mixin rather than a copied method so a class cannot silently be left out --
    which is the first thing to suspect when an abort seems to cross a class
    boundary.
    """

    # ``is_initialized``, not ``is_available``: synchronise a context that
    # already exists, never create one. A CPU-only test that initializes the
    # device driver just to sync is a confound, and while chasing this fault it
    # was one -- it made "no prior GPU work" untrue of every run.

    def setUp(self):
        super().setUp()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()

    def tearDown(self):
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        super().tearDown()


class TestZayaRouterTailKernel(_SyncEachTest, CustomTestCase):
    """``zaya_router_tail`` on its own, with no sglang module anywhere near it.

    Deliberately first in the file and deliberately module-free: every tensor
    here is a raw ``torch.empty`` / ``torch.randn``. Nothing constructs a
    ``ZayaRouter``, so nothing runs ``ReplicatedLinear`` or sglang's ``RMSNorm``.

    That separation is the point. The gfx950 fault this class was written for
    surfaced as a SIGABRT at an unrelated synchronisation point several tests
    after the offending launch, which makes "the suite aborts" useless as
    evidence about *which* kernel is at fault. If this class passes and a later
    one aborts, the tail kernel is exonerated and the fault belongs to something
    the model path pulls in; if this class aborts, it is the tail kernel. Keep it
    module-free.
    """

    # ZAYA1-74B: 24 experts plus the MOD skip slot. BLOCK pads to 32, so 7 of
    # 32 column lanes are padding -- the configuration a power-of-two-padded
    # mask bug hides in, and the one the smaller shapes cannot expose.
    NUM_MOE_EXPERTS = 24
    NUM_EXPERTS = 25

    def _biases(self, num_experts, mod=True):
        biases = torch.zeros(num_experts, dtype=torch.float32, device="cuda")
        if mod:
            biases[-1] = -1.0  # what ZayaRouter.__init__ writes
        return biases

    def _reference(self, logits, biases):
        """The torch chain, in-line, so this class stays module-free.

        The argmax is spelled out as an explicit lowest-index-wins reduction
        rather than ``torch.argmax``. bf16 carries only 256 values per octave, so
        25 columns drawn from it collide often, and on an exact tie
        ``torch.argmax`` may return any of the winners while the kernel promises
        the lowest. Using torch.argmax here would make this test flaky in the one
        direction the kernel actually pins.
        """
        num_experts = logits.shape[1]
        prob = torch.softmax(logits.float(), dim=-1)
        biased = prob + biases
        best = biased.max(dim=-1, keepdim=True).values
        cols = torch.arange(num_experts, device=logits.device)
        choice = (
            torch.where(biased == best, cols, num_experts)
            .min(dim=-1, keepdim=True)
            .values
        )
        return prob.gather(1, choice), choice

    def _check(self, logits, biases, max_expert_id, out_dtype=torch.float32):
        from sglang.kernels.ops.moe import zaya_router_tail as kernel

        num_experts = logits.shape[1]
        self.assertTrue(
            kernel.covered(
                logits,
                biases,
                num_experts=num_experts,
                max_expert_id=max_expert_id,
                topk=1,
                out_dtype=out_dtype,
            )
        )
        weight, moe_ids, route_prob, skip_ids = kernel.router_tail(
            logits,
            biases,
            num_experts=num_experts,
            max_expert_id=max_expert_id,
            softmax_fp32=True,
            out_dtype=out_dtype,
        )
        # Sync on the launch itself, before any other kernel runs, so a fault
        # is attributed to this call and -- inside a subTest loop -- to this
        # subtest's shape rather than to whatever ran next.
        torch.cuda.synchronize()

        ref_weight, ref_choice = self._reference(logits, biases)

        # Bounds first: an id outside the expert range leaves this kernel
        # silently and faults far away, inside FusedMoE's expert indexing.
        self.assertGreaterEqual(int(moe_ids.min()), 0)
        self.assertLessEqual(int(moe_ids.max()), max_expert_id)
        self.assertGreaterEqual(int(skip_ids.min()), 0)
        self.assertLessEqual(int(skip_ids.max()), num_experts - 1)

        self.assertTrue(
            torch.equal(skip_ids.long(), ref_choice),
            "unclamped choice must be bit-identical to torch",
        )
        self.assertTrue(
            torch.equal(moe_ids.long(), ref_choice.clamp(max=max_expert_id)),
            "clamped choice must be bit-identical to torch",
        )
        torch.testing.assert_close(weight, ref_weight, rtol=1e-5, atol=1e-6)
        return weight, moe_ids, route_prob, skip_ids

    @unittest.skipUnless(torch.cuda.is_available(), "fused tail is a GPU kernel")
    def test_real_zaya_expert_count_across_token_counts(self):
        """25 experts -- the shipping shape -- at every interesting T."""
        biases = self._biases(self.NUM_EXPERTS)
        for num_tokens in (1, 2, 7, 31, 32, 33, 64, 129, 1024):
            for dtype in (torch.bfloat16, torch.float32):
                with self.subTest(tokens=num_tokens, dt=dtype):
                    torch.manual_seed(num_tokens)
                    logits = torch.randn(
                        num_tokens, self.NUM_EXPERTS, device="cuda", dtype=dtype
                    )
                    self._check(
                        logits,
                        biases,
                        self.NUM_MOE_EXPERTS - 1,
                        out_dtype=dtype,
                    )

    def test_poison_fixtures_are_built_correctly(self):
        """CPU-checkable: the probe fixtures themselves.

        The two GPU probes below have gone red on hardware twice, and both times
        the cause was entirely in these few lines of arithmetic -- first ``[-1]``
        indexing the backing buffer instead of the view, then a float32
        round-trip of 1e30 compared against the Python literal. Both times the
        red test read as a kernel fault and cost a hardware round trip to
        disprove. Constructing the fixtures needs no GPU, so validating them
        should not either. This test is the reason a third one cannot happen.
        """
        logits, backing = _poisoned_row_view(4, 25, 32, "cpu", torch.float32, seed=1)
        self.assertEqual(tuple(logits.shape), (4, 25))
        self.assertEqual(logits.stride(), (32, 1))
        self.assertEqual(float(backing[0, 25]), _POISON)
        self.assertTrue(bool(torch.isfinite(logits).all()))
        self.assertLess(float(logits.abs().max()), 100.0)

        biases, bias_backing = _poisoned_bias_view(25, 32, "cpu")
        self.assertTrue(biases.is_contiguous())
        self.assertEqual(biases.numel(), 25)
        # The view's last element is the MOD skip slot, not the buffer's last.
        self.assertEqual(float(biases[-1]), -1.0)
        self.assertEqual(float(bias_backing[25]), _POISON)
        self.assertEqual(float(bias_backing[-1]), _POISON)
        # Distinct per index, so a smear between lanes cannot cancel out.
        self.assertEqual(len({float(v) for v in biases}), 25)
        # And the sentinel survives a float32 round trip exactly, which is the
        # whole reason it is a power of two rather than 1e30.
        stored = torch.tensor([_POISON], dtype=torch.float32)
        self.assertEqual(float(stored[0]), _POISON)

    @unittest.skipUnless(torch.cuda.is_available(), "fused tail is a GPU kernel")
    def test_padding_columns_cannot_influence_the_result(self):
        """A padded-mask overread shows up here as a wrong answer, not a crash.

        BLOCK pads 25 up to 32, so 7 column lanes are padding. Hand the kernel a
        25-column view of a 32-column buffer whose tail columns hold a sentinel.
        Every address those padding lanes could form is mapped memory, so a
        dropped or mis-vectorized mask corrupts the softmax instead of faulting
        -- the only way to probe for it without taking the suite down.
        """
        poisoned, _ = _poisoned_row_view(
            8, self.NUM_EXPERTS, 32, "cuda", torch.float32, seed=3
        )
        self.assertEqual(poisoned.stride(), (32, 1))

        biases = self._biases(self.NUM_EXPERTS)
        weight_poisoned, ids_poisoned, _, _ = self._check(
            poisoned, biases, self.NUM_MOE_EXPERTS - 1
        )
        # The same values in a tight buffer must give the identical answer.
        weight_clean, ids_clean, _, _ = self._check(
            poisoned.contiguous(), biases, self.NUM_MOE_EXPERTS - 1
        )
        self.assertTrue(torch.equal(ids_poisoned, ids_clean))
        torch.testing.assert_close(weight_poisoned, weight_clean, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "fused tail is a GPU kernel")
    def test_padding_bias_lanes_cannot_influence_the_result(self):
        """The bias vector is 25 floats read through a 32-wide block.

        Scope, stated honestly. The poison can only reach the output if BOTH the
        masked load AND the ``tl.where`` that forces every padding lane's biased
        score to -inf fail; a padding bias on its own has no path to the answer.
        So this guards the combination rather than probing the load, and it could
        never have detected a load-mask failure by itself. It earns its place as
        a regression guard: remove either mask later and it goes red.

        The fixture is built by ``_poisoned_bias_view`` and checked on CPU by
        ``test_poison_fixtures_are_built_correctly``, so a failure here is the
        kernel.
        """
        torch.manual_seed(4)
        logits = torch.randn(6, self.NUM_EXPERTS, device="cuda")
        biases, _ = _poisoned_bias_view(self.NUM_EXPERTS, 32, "cuda")

        weight_poisoned, ids_poisoned, _, _ = self._check(
            logits, biases, self.NUM_MOE_EXPERTS - 1
        )
        # Kernel against kernel as well as against torch: the same 25 values in
        # a tight buffer must give bit-identical results.
        weight_clean, ids_clean, _, _ = self._check(
            logits, biases.clone(), self.NUM_MOE_EXPERTS - 1
        )
        self.assertTrue(torch.equal(ids_poisoned, ids_clean))
        torch.testing.assert_close(weight_poisoned, weight_clean, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "fused tail is a GPU kernel")
    def test_outputs_are_exactly_t_rows_and_ids_are_distinct_tensors(self):
        """Shapes, and that the clamped and raw ids are separate buffers.

        Under MOD they must not alias: the clamp would then overwrite the skip
        slot it exists to preserve.
        """
        from sglang.kernels.ops.moe import zaya_router_tail as kernel

        num_tokens = 5
        logits = torch.randn(num_tokens, self.NUM_EXPERTS, device="cuda")
        biases = self._biases(self.NUM_EXPERTS)
        _, moe_ids, _, skip_ids = kernel.router_tail(
            logits,
            biases,
            num_experts=self.NUM_EXPERTS,
            max_expert_id=self.NUM_MOE_EXPERTS - 1,
            softmax_fp32=True,
            out_dtype=torch.float32,
        )
        self.assertEqual(tuple(moe_ids.shape), (num_tokens, 1))
        self.assertEqual(tuple(skip_ids.shape), (num_tokens, 1))
        self.assertIsNot(moe_ids, skip_ids)

    @unittest.skipUnless(torch.cuda.is_available(), "fused tail is a GPU kernel")
    def test_all_experts_tie_at_the_real_shape(self):
        """The degenerate tie, at 25 experts with the real MOD bias vector.

        With every logit equal the softmax is exactly uniform, so the 24 real
        experts all tie and the skip slot loses by its -1 bias. Index 0 must win.
        """
        from sglang.kernels.ops.moe import zaya_router_tail as kernel

        logits = torch.zeros(4, self.NUM_EXPERTS, device="cuda")
        biases = self._biases(self.NUM_EXPERTS)
        _, moe_ids, _, skip_ids = kernel.router_tail(
            logits,
            biases,
            num_experts=self.NUM_EXPERTS,
            max_expert_id=self.NUM_MOE_EXPERTS - 1,
            softmax_fp32=True,
            out_dtype=torch.float32,
        )
        self.assertEqual(moe_ids.flatten().tolist(), [0, 0, 0, 0])
        self.assertEqual(skip_ids.flatten().tolist(), [0, 0, 0, 0])

    @unittest.skipUnless(torch.cuda.is_available(), "fused tail is a GPU kernel")
    def test_nan_logits_still_produce_an_in_range_id(self):
        """A NaN logit must not let an out-of-range id escape to FusedMoE.

        With a NaN in the row, no lane compares equal to the max and the
        tie-break reduction returns its ``NUM_EXPERTS`` sentinel. The value that
        leaves this kernel is what FusedMoE indexes expert weights with, so the
        contract is "always in range", not "always the same as torch" -- torch's
        own argmax is unspecified on NaN. Only the bound is asserted.
        """
        from sglang.kernels.ops.moe import zaya_router_tail as kernel

        logits = torch.randn(4, self.NUM_EXPERTS, device="cuda")
        logits[1, 7] = float("nan")
        logits[3, :] = float("nan")
        biases = self._biases(self.NUM_EXPERTS)
        _, moe_ids, _, skip_ids = kernel.router_tail(
            logits,
            biases,
            num_experts=self.NUM_EXPERTS,
            max_expert_id=self.NUM_MOE_EXPERTS - 1,
            softmax_fp32=True,
            out_dtype=torch.float32,
        )
        self.assertGreaterEqual(int(moe_ids.min()), 0)
        self.assertLessEqual(int(moe_ids.max()), self.NUM_MOE_EXPERTS - 1)
        self.assertGreaterEqual(int(skip_ids.min()), 0)
        self.assertLessEqual(int(skip_ids.max()), self.NUM_EXPERTS - 1)

    def test_router_construction_needs_no_distributed_init(self):
        """``_make_tiny_router`` must never require a process group.

        The regression guard for the gfx950 abort. Initializing sglang's
        model-parallel groups on a GPU box builds CUDA device communicators over
        a gloo process group and kills the process, so the router factory has to
        stay free of it. Meaningful exactly when the router classes are run on
        their own, which is how they are run while this is being chased; skipped
        rather than faked when something earlier in the file has already brought
        distributed up.
        """
        if torch.distributed.is_initialized():
            self.skipTest("distributed already initialized earlier in this file")
        _make_tiny_router(num_moe_experts=4, seed=1)
        self.assertFalse(
            torch.distributed.is_initialized(),
            "building a ZayaRouter must not initialize torch.distributed",
        )

    def test_block_padding_is_never_narrower_than_the_minimum(self):
        """CPU-checkable: the launch never emits a sub-16-wide column block.

        A 2- or 4-element 1-D tensor on a 64-lane wavefront is a degenerate
        layout that no other kernel in this tree exercises, and it was one of the
        two structural oddities in the version of this kernel that faulted. The
        padding lanes are masked and their addresses clamped, so widening to 16
        costs nothing and keeps the layout on a well-travelled path.
        """
        from sglang.kernels.ops.moe import zaya_router_tail as kernel

        self.assertEqual(kernel.block_size(1), 16)
        self.assertEqual(kernel.block_size(2), 16)
        self.assertEqual(kernel.block_size(4), 16)
        self.assertEqual(kernel.block_size(16), 16)
        # The shipping shape: 25 experts in a 32-wide block, 7 lanes of padding.
        self.assertEqual(kernel.block_size(25), 32)
        self.assertEqual(kernel.block_size(64), 64)
        self.assertEqual(kernel.block_size(65), 128)

    @unittest.skipUnless(torch.cuda.is_available(), "fused tail is a GPU kernel")
    def test_narrow_expert_counts_still_match_torch(self):
        """The sub-block expert counts, now that they all pad up to 16."""
        for num_experts in (2, 4, 5, 8):
            with self.subTest(experts=num_experts):
                torch.manual_seed(num_experts)
                logits = torch.randn(3, num_experts, device="cuda")
                self._check(logits, self._biases(num_experts), num_experts - 2)


class TestZayaRouterFusedTail(_SyncEachTest, CustomTestCase):
    """The fused router tail must reproduce the torch chain it replaces.

    Derived property. ``zaya_router_tail`` collapses nine launches -- softmax,
    the balancing-bias add, argmax, the probability gather, the model-dtype
    round, ZayaBlock's int32/clamp/int32 expert-id munging, and the MoE runner's
    opening ``topk_weights.to(float32)`` -- into one kernel that re-derives all
    of it from scratch. Each piece can be subtly wrong and still produce
    plausible tensors: a bias folded into the gathered probability instead of
    only into the comparison, a clamp that erases the MOD skip slot, a tie broken
    the other way.

    The acceptance bar is asymmetric on purpose. The selected expert must be
    bit-identical, because a flipped expert is a different model. The
    probabilities are only ``assert_close``: the fp32 max/sum reduction order
    here does not match torch's tree reduction, so ~1 ULP of drift is expected.
    """

    def _router(self, **kwargs):
        """A CUDA router whose only fused kernel is the tail under test.

        Two deliberate choices, both learned from the gfx950 fault hunt:

        ``mlp_expansion`` defaults to a realistic 256 rather than the 8 this
        file's other factories use. At width 8 sglang's RMSNorm runs at a size
        nothing else in the tree exercises on GPU, which made it a second
        suspect and muddied the evidence about which kernel was faulting.

        And ``fused_router_mlp_ok`` is switched off, so a failure in this class
        can only be the tail kernel -- not the MLP kernel that a realistic
        expansion would otherwise switch on underneath it.
        """
        kwargs.setdefault("mlp_expansion", 256)
        kwargs.setdefault("std", 1.0 / 16.0)
        router, config = _make_tiny_router(device="cuda", **kwargs)
        router.fused_router_mlp_ok = False
        return router, config

    def _reference(self, router, hidden):
        """Run the unfused chain over the same logits the fused path sees."""
        hs, _ = router.down_proj(hidden)
        logits = router._router_logits(router.rmsnorm_eda(hs))
        return logits, router._routing_reference(logits, hidden.dtype, hs)

    @unittest.skipUnless(torch.cuda.is_available(), "fused tail is a GPU kernel")
    def test_ids_bit_identical_and_probs_close(self):
        from sglang.kernels.ops.moe import zaya_router_tail as kernel

        for num_moe_experts, num_tokens, dtype in (
            (24, 1, torch.bfloat16),  # ZAYA1-74B at decode, bs=1
            (24, 129, torch.bfloat16),  # a prefill-shaped batch
            (4, 7, torch.float32),
            (1, 3, torch.bfloat16),  # one real expert plus the skip slot
        ):
            with self.subTest(experts=num_moe_experts, tokens=num_tokens, dt=dtype):
                router, config = self._router(
                    num_moe_experts=num_moe_experts,
                    seed=num_moe_experts,
                    dtype=dtype,
                )
                torch.manual_seed(num_tokens)
                hidden = torch.randn(
                    num_tokens, config.hidden_size, device="cuda", dtype=dtype
                )
                with torch.no_grad():
                    fused = router(hidden)
                    logits, ref = self._reference(router, hidden)

                # Without this the test could be comparing the fallback against
                # itself and passing for the wrong reason.
                self.assertTrue(
                    kernel.covered(
                        logits,
                        router.balancing_biases,
                        num_experts=router.num_experts,
                        max_expert_id=router.num_moe_experts - 1,
                        topk=1,
                        out_dtype=dtype,
                    ),
                    "the fused path must be the one under test",
                )
                self.assertEqual(fused.moe_ids.dtype, torch.int32)
                self.assertEqual(fused.moe_weight.dtype, torch.float32)
                self.assertEqual(fused.route_prob.dtype, dtype)

                self.assertTrue(
                    torch.equal(fused.moe_ids.long(), ref.moe_ids.long()),
                    "selected expert must be bit-identical",
                )
                self.assertTrue(
                    torch.equal(fused.skip_ids.long(), ref.skip_ids.long()),
                    "unclamped choice must be bit-identical",
                )
                torch.testing.assert_close(
                    fused.moe_weight, ref.moe_weight, rtol=1e-5, atol=1e-6
                )
                torch.testing.assert_close(
                    fused.route_prob.float(),
                    ref.route_prob.float(),
                    rtol=1e-2,
                    atol=1e-2,
                )

    @unittest.skipUnless(torch.cuda.is_available(), "fused tail is a GPU kernel")
    def test_exact_ties_pick_the_lowest_index(self):
        """torch.argmax leaves ties unspecified; the kernel pins lowest-wins.

        Ties are reachable in production, not only in tests: ``balancing_biases``
        is a constant vector, so two experts with equal logits have exactly equal
        biased scores in fp32.
        """
        from sglang.kernels.ops.moe import zaya_router_tail as kernel

        num_experts = 8
        args = dict(num_experts=num_experts, max_expert_id=num_experts - 1)
        biases = torch.zeros(num_experts, dtype=torch.float32, device="cuda")

        logits = torch.full((4, num_experts), -5.0, device="cuda")
        logits[:, 3] = 2.0  # columns 3 and 6 tie for the maximum, exactly
        logits[:, 6] = 2.0
        self.assertTrue(
            kernel.covered(logits, biases, topk=1, out_dtype=torch.float32, **args)
        )
        _, moe_ids, _, _ = kernel.router_tail(
            logits, biases, softmax_fp32=True, out_dtype=torch.float32, **args
        )
        self.assertEqual(moe_ids.flatten().tolist(), [3, 3, 3, 3])

        # An all-equal row is the degenerate tie: every index wins the max, so
        # index 0 must be the one reported.
        flat = torch.zeros((2, num_experts), device="cuda")
        _, flat_ids, _, _ = kernel.router_tail(
            flat, biases, softmax_fp32=True, out_dtype=torch.float32, **args
        )
        self.assertEqual(flat_ids.flatten().tolist(), [0, 0])

    @unittest.skipUnless(torch.cuda.is_available(), "fused tail is a GPU kernel")
    def test_skip_slot_survives_the_clamp(self):
        """``moe_ids`` is clamped; ``skip_ids`` must still name the skip slot.

        This is the trap the separate raw-id output exists for. The MOD skip slot
        is id ``num_moe_experts``, and the clamp folds it onto real expert
        ``num_moe_experts - 1``. Were the clamped ids the only ones emitted,
        every skipped token would be indistinguishable from one genuinely routed
        to the last expert and MOD would silently stop skipping.
        """
        num_moe_experts = 6
        router, config = self._router(num_moe_experts=num_moe_experts, seed=5)
        self.assertTrue(router.use_mod)
        self.assertEqual(router.num_experts, num_moe_experts + 1)
        with torch.no_grad():
            router.balancing_biases.zero_()
            router.balancing_biases[-1] = 10.0  # the skip slot always wins
            hidden = torch.randn(5, config.hidden_size, device="cuda")
            fused = router(hidden)
            _, ref = self._reference(router, hidden)

        self.assertEqual(fused.skip_ids.flatten().tolist(), [num_moe_experts] * 5)
        self.assertEqual(fused.moe_ids.flatten().tolist(), [num_moe_experts - 1] * 5)
        self.assertTrue(torch.equal(fused.skip_ids.long(), ref.skip_ids.long()))
        self.assertTrue(torch.equal(fused.moe_ids.long(), ref.moe_ids.long()))
        # Two distinct tensors, so the clamp cannot corrupt the skip predicate.
        self.assertIsNot(fused.moe_ids, fused.skip_ids)

    @unittest.skipUnless(torch.cuda.is_available(), "fused tail is a GPU kernel")
    def test_without_mod_the_id_tensors_alias(self):
        """No skip slot means the clamp is a no-op, so no second store is made."""
        from sglang.kernels.ops.moe import zaya_router_tail as kernel

        num_experts = 4
        logits = torch.randn(3, num_experts, device="cuda")
        biases = torch.zeros(num_experts, dtype=torch.float32, device="cuda")
        _, moe_ids, _, skip_ids = kernel.router_tail(
            logits,
            biases,
            num_experts=num_experts,
            max_expert_id=num_experts - 1,
            softmax_fp32=True,
            out_dtype=torch.float32,
        )
        self.assertIs(moe_ids, skip_ids)

    @unittest.skipUnless(torch.cuda.is_available(), "fused tail is a GPU kernel")
    def test_zero_tokens_is_a_no_op(self):
        """Idle DP-attention forwards arrive with T == 0 and must not launch."""
        from sglang.kernels.ops.moe import zaya_router_tail as kernel

        num_experts = 5
        logits = torch.randn(0, num_experts, device="cuda")
        biases = torch.zeros(num_experts, dtype=torch.float32, device="cuda")
        outs = kernel.router_tail(
            logits,
            biases,
            num_experts=num_experts,
            max_expert_id=num_experts - 2,
            softmax_fp32=True,
            out_dtype=torch.bfloat16,
        )
        for t in outs:
            self.assertEqual(tuple(t.shape), (0, 1))

    @unittest.skipUnless(torch.cuda.is_available(), "control needs a GPU")
    def test_reference_path_at_the_narrow_norm_width(self):
        """Control: the same router shapes with NO fused kernel at all.

        The other half of the gfx950 localization. Builds the router at
        ``mlp_expansion=8`` -- the width this file used everywhere before, and a
        width nothing else in the tree runs sglang's RMSNorm at on GPU -- and
        takes the torch reference path end to end, then synchronises. If THIS
        aborts, the fault was never in either fused kernel and the search should
        move to the aiter RMSNorm at that width.
        """
        router, config = _make_tiny_router(
            num_moe_experts=24, mlp_expansion=8, seed=9, device="cuda"
        )
        router.fused_router_mlp_ok = False
        hidden = torch.randn(5, config.hidden_size, device="cuda")
        with torch.no_grad():
            hs, _ = router.down_proj(hidden)
            logits = router._router_logits_reference(router.rmsnorm_eda(hs))
            ref = router._routing_reference(logits, hidden.dtype, hs)
        torch.cuda.synchronize()
        self.assertEqual(tuple(ref.moe_ids.shape), (5, 1))
        self.assertEqual(tuple(logits.shape), (5, 25))

    def test_uncovered_inputs_fall_back(self):
        """``covered()`` is the only guard between an unsupported input and a
        wrong-answer launch, so its negative branches must hold."""
        from sglang.kernels.ops.moe import zaya_router_tail as kernel

        num_experts = 5
        logits = torch.randn(4, num_experts)
        biases = torch.zeros(num_experts, dtype=torch.float32)
        ok = dict(
            num_experts=num_experts,
            max_expert_id=num_experts - 2,
            topk=1,
            out_dtype=torch.bfloat16,
        )
        # CPU tensors are not served by the Triton path.
        self.assertFalse(kernel.covered(logits, biases, **ok))
        # top-k > 1 needs the cumulative-skip rewrite the kernel does not do.
        self.assertFalse(kernel.covered(logits, biases, **{**ok, "topk": 2}))
        # An expert axis wider than one Triton block.
        self.assertFalse(kernel.covered(logits, biases, **{**ok, "num_experts": 4096}))
        # A clamp ceiling outside the expert range.
        self.assertFalse(
            kernel.covered(logits, biases, **{**ok, "max_expert_id": num_experts})
        )
        # Logits that disagree with the declared expert count.
        self.assertFalse(kernel.covered(torch.randn(4, num_experts + 1), biases, **ok))
        # A bias vector of the wrong dtype or the wrong length.
        self.assertFalse(kernel.covered(logits, biases.double(), **ok))
        self.assertFalse(kernel.covered(logits, torch.zeros(num_experts + 1), **ok))
        # Non-unit innermost stride: the kernel indexes columns as +1.
        transposed = torch.randn(num_experts, 4).t()
        self.assertEqual(transposed.shape, (4, num_experts))
        self.assertNotEqual(transposed.stride(-1), 1)
        self.assertFalse(kernel.covered(transposed, biases, **ok))

    def test_cpu_router_takes_the_reference_path(self):
        """The covered()/fallback split, seen from the model's side.

        On CPU the fused branch is skipped before the kernel module is even
        imported, and the reference chain must still honour the same contract:
        fp32 weights, int32 ids clamped into the real-expert range, and an
        unclamped ``skip_ids`` that still names the skip slot.
        """
        num_moe_experts = 4
        router, config = _make_tiny_router(num_moe_experts=num_moe_experts, seed=7)
        with torch.no_grad():
            router.balancing_biases.zero_()
            router.balancing_biases[-1] = 10.0
            routing = router(torch.randn(6, config.hidden_size))

        self.assertEqual(routing.moe_weight.dtype, torch.float32)
        self.assertEqual(routing.moe_ids.dtype, torch.int32)
        self.assertEqual(routing.route_prob.dtype, torch.float32)
        self.assertEqual(tuple(routing.moe_ids.shape), (6, 1))
        self.assertEqual(routing.skip_ids.flatten().tolist(), [num_moe_experts] * 6)
        self.assertEqual(routing.moe_ids.flatten().tolist(), [num_moe_experts - 1] * 6)
        self.assertGreaterEqual(int(routing.moe_ids.min()), 0)
        self.assertLessEqual(int(routing.moe_ids.max()), num_moe_experts - 1)

    def test_eda_recursion_publishes_the_post_eda_pre_norm_state(self):
        """Chain two MoE layers and pin what the second one folds in.

        Getting this wrong -- publishing the *normalized* router state, or the
        pre-EDA one -- changes routing in all 60 MoE layers of the 74B with no
        crash and no shape error, so it is worth a test rather than a comment.
        The in-place addcmul that replaced ``hs + prev * scale`` makes the
        aliasing here load-bearing.
        """
        first, config = _make_tiny_router(num_moe_experts=4, seed=11)
        second, _ = _make_tiny_router(num_moe_experts=4, seed=12)
        self.assertTrue(first.use_eda)

        hidden = torch.randn(5, config.hidden_size)
        with torch.no_grad():
            r1 = first(hidden)
            # The first MoE layer has no previous state, so it publishes
            # down_proj's output unchanged -- pre-norm.
            expected_1, _ = first.down_proj(hidden)
            torch.testing.assert_close(r1.hidden_states_next, expected_1)

            r2 = second(hidden, r1.hidden_states_next)
            base_2, _ = second.down_proj(hidden)
            expected_2 = base_2 + r1.hidden_states_next * second.router_states_scale
            # addcmul fuses the multiply-add, so this is close, not identical.
            torch.testing.assert_close(
                r2.hidden_states_next, expected_2, rtol=1e-6, atol=1e-6
            )
            # And emphatically not the normalized tensor.
            normed = second.rmsnorm_eda(expected_2)
            self.assertFalse(
                torch.allclose(r2.hidden_states_next, normed),
                "the EDA recursion must thread the pre-norm state",
            )

    def test_eda_add_does_not_disturb_the_previous_layers_state(self):
        """The addcmul writes into down_proj's buffer, and only that one.

        The in-place form is only safe while ``hs`` is exclusively owned, so pin
        the two observable halves of that: the previous layer's published state
        is read-only here, and what this layer publishes carries the EDA term
        rather than the raw projection.
        """
        router, config = _make_tiny_router(num_moe_experts=4, seed=13)
        hidden = torch.randn(3, config.hidden_size)
        prev = torch.randn(3, router.mlp_expansion)
        prev_before = prev.clone()
        with torch.no_grad():
            routing = router(hidden, prev)
            raw, _ = router.down_proj(hidden)
        self.assertTrue(torch.equal(prev, prev_before))
        self.assertFalse(torch.allclose(routing.hidden_states_next, raw))


# ---------------------------------------------------------------------------
# ZayaRouter: fused router MLP (3 dense stages + 2 GELUs in one kernel)
# ---------------------------------------------------------------------------


class TestZayaRouterFusedMLP(_SyncEachTest, CustomTestCase):
    """The fused router MLP must reproduce the ``nn.Sequential`` it replaces.

    Derived property. Five launches (Linear, GELU, Linear, GELU, Linear) become
    one kernel that re-derives the weight transposes, the bias placement, the
    rounding points and the activation from scratch. The activation is the
    dangerous one: ``nn.GELU()`` is ``approximate='none'``, the erf form, and the
    tanh approximation agrees with it to ~5e-4 absolute -- close enough to pass
    for numerical noise, far enough to flip routing on near-ties in all 60 MoE
    layers of the 74B. So the erf-vs-tanh guard gets tests of its own, on CPU,
    where they run on every commit rather than only on GPU hardware.
    """

    def test_structural_guard_accepts_the_real_router(self):
        from sglang.srt.models.zaya import fusable_router_mlp

        router, _ = _make_tiny_router(num_moe_experts=4, seed=21)
        self.assertTrue(fusable_router_mlp(router.router_mlp))
        self.assertTrue(router.fused_router_mlp_ok)

    def test_structural_guard_rejects_the_tanh_approximation(self):
        """The one failure mode with no numerical signature loud enough to catch.

        ``covered()`` sees only tensors, so nothing downstream can tell which
        GELU the eager path was built with. If this guard regressed, the fused
        kernel would compute the erf form while the module held the tanh one --
        or the reverse -- and the only symptom would be slightly different
        expert choices.
        """
        from sglang.srt.models.zaya import fusable_router_mlp

        for idx in (1, 3):
            for approximate in ("tanh",):
                with self.subTest(activation_index=idx, approximate=approximate):
                    router, _ = _make_tiny_router(num_moe_experts=4, seed=22)
                    stages = list(router.router_mlp)
                    stages[idx] = torch.nn.GELU(approximate=approximate)
                    self.assertFalse(fusable_router_mlp(torch.nn.Sequential(*stages)))

        # A different activation entirely is also refused.
        router, _ = _make_tiny_router(num_moe_experts=4, seed=22)
        stages = list(router.router_mlp)
        stages[1] = torch.nn.SiLU()
        self.assertFalse(fusable_router_mlp(torch.nn.Sequential(*stages)))

        # And the erf form -- spelled out explicitly -- is still accepted.
        router, _ = _make_tiny_router(num_moe_experts=4, seed=22)
        stages = list(router.router_mlp)
        stages[1] = torch.nn.GELU(approximate="none")
        self.assertTrue(fusable_router_mlp(torch.nn.Sequential(*stages)))

    def test_structural_guard_rejects_a_reshaped_mlp(self):
        from sglang.srt.models.zaya import fusable_router_mlp

        # A final projection carrying a bias the kernel would silently drop.
        router, _ = _make_tiny_router(num_moe_experts=4, seed=23)
        router.router_mlp[4].bias = torch.nn.Parameter(torch.zeros(router.num_experts))
        self.assertFalse(fusable_router_mlp(router.router_mlp))

        # A stage that returns its bias instead of applying it.
        router, _ = _make_tiny_router(num_moe_experts=4, seed=23)
        router.router_mlp[0].skip_bias_add = True
        self.assertFalse(fusable_router_mlp(router.router_mlp))

        # A different number of stages.
        router, _ = _make_tiny_router(num_moe_experts=4, seed=23)
        stages = list(router.router_mlp)[:3]
        self.assertFalse(fusable_router_mlp(torch.nn.Sequential(*stages)))

    def test_erf_and_tanh_gelu_actually_differ(self):
        """Pin that the guard above is guarding something real.

        Were the two GELU flavours interchangeable to within bf16 rounding there
        would be no reason to check which one the module holds, and the guard
        would be ceremony. They are not.
        """
        x = torch.linspace(-4.0, 4.0, 401)
        erf_form = torch.nn.functional.gelu(x, approximate="none")
        tanh_form = torch.nn.functional.gelu(x, approximate="tanh")
        gap = (erf_form - tanh_form).abs().max().item()
        # The magnitude the kernel docstring quotes, so the two stay in sync.
        self.assertGreater(gap, 1e-4)
        self.assertLess(gap, 1e-2)

    def test_uncovered_inputs_fall_back(self):
        from sglang.kernels.ops.moe import zaya_router_mlp as kernel

        expansion, num_experts = 32, 5
        dt = torch.bfloat16
        x = torch.randn(6, expansion, dtype=dt)
        w1 = torch.randn(expansion, expansion, dtype=dt)
        w2 = torch.randn(expansion, expansion, dtype=dt)
        w3 = torch.randn(num_experts, expansion, dtype=dt)
        b1 = torch.zeros(expansion, dtype=dt)
        b2 = torch.zeros(expansion, dtype=dt)
        args = (x, w1, b1, w2, b2, w3)

        # CPU tensors are not served by the Triton path.
        self.assertFalse(kernel.covered(*args, num_experts=num_experts))
        # A missing bias (bias=False on a stage the kernel assumes has one).
        self.assertFalse(
            kernel.covered(x, w1, None, w2, b2, w3, num_experts=num_experts)
        )
        # fp32 is excluded: tl.dot would pick a reduced-precision fp32 mode.
        self.assertFalse(
            kernel.covered(
                x.float(),
                w1.float(),
                b1.float(),
                w2.float(),
                b2.float(),
                w3.float(),
                num_experts=num_experts,
            )
        )
        # An expansion beyond the cap the K split imposes.
        wide = 512
        self.assertFalse(
            kernel.covered(
                torch.randn(6, wide, dtype=dt),
                torch.randn(wide, wide, dtype=dt),
                torch.zeros(wide, dtype=dt),
                torch.randn(wide, wide, dtype=dt),
                torch.zeros(wide, dtype=dt),
                torch.randn(num_experts, wide, dtype=dt),
                num_experts=num_experts,
            )
        )
        # An odd expansion, which cannot be split in halves.
        odd = 33
        self.assertFalse(
            kernel.covered(
                torch.randn(6, odd, dtype=dt),
                torch.randn(odd, odd, dtype=dt),
                torch.zeros(odd, dtype=dt),
                torch.randn(odd, odd, dtype=dt),
                torch.zeros(odd, dtype=dt),
                torch.randn(num_experts, odd, dtype=dt),
                num_experts=num_experts,
            )
        )
        # A shape disagreement between the stages.
        self.assertFalse(
            kernel.covered(
                x,
                w1,
                b1,
                torch.randn(expansion, expansion + 2, dtype=dt),
                b2,
                w3,
                num_experts=num_experts,
            )
        )
        # A final projection that does not match the declared expert count.
        self.assertFalse(kernel.covered(*args, num_experts=num_experts + 1))
        # Non-unit innermost stride on the activation.
        self.assertFalse(
            kernel.covered(
                torch.randn(expansion, 6, dtype=dt).t(),
                w1,
                b1,
                w2,
                b2,
                w3,
                num_experts=num_experts,
            )
        )

    def test_cpu_router_takes_the_reference_mlp(self):
        """On CPU the fused branch is skipped before the kernel is imported."""
        router, _ = _make_tiny_router(num_moe_experts=4, seed=24)
        hs_norm = torch.randn(4, router.mlp_expansion)
        with torch.no_grad():
            got = router._router_logits(hs_norm)
            ref = router._router_logits_reference(hs_norm)
        self.assertTrue(torch.equal(got, ref))

    def _operands(self, router, hs_norm):
        first, second, last = (router.router_mlp[i] for i in (0, 2, 4))
        return (
            hs_norm,
            first.weight,
            first.bias,
            second.weight,
            second.bias,
            last.weight,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "fused MLP is a GPU kernel")
    def test_matches_the_unfused_sequential(self):
        from sglang.kernels.ops.moe import zaya_router_mlp as kernel

        # 256 / 24(+1) is ZAYA1-74B exactly; 32 / 4(+1) exercises a K split
        # right at the tl.dot minimum and a padded expert axis.
        for expansion, num_moe_experts, num_tokens in (
            (256, 24, 1),
            (256, 24, 33),
            (256, 24, 130),
            (32, 4, 7),
        ):
            with self.subTest(d=expansion, e=num_moe_experts, t=num_tokens):
                router, _ = _make_tiny_router(
                    num_moe_experts=num_moe_experts,
                    mlp_expansion=expansion,
                    seed=expansion,
                    device="cuda",
                    dtype=torch.bfloat16,
                    std=1.0 / (expansion**0.5),
                )
                torch.manual_seed(num_tokens)
                hs_norm = torch.randn(
                    num_tokens, expansion, device="cuda", dtype=torch.bfloat16
                )
                operands = self._operands(router, hs_norm)
                self.assertTrue(
                    kernel.covered(*operands, num_experts=router.num_experts),
                    "the fused path must be the one under test",
                )
                with torch.no_grad():
                    got = kernel.router_mlp(*operands, num_experts=router.num_experts)
                    ref = router._router_logits_reference(hs_norm)

                self.assertEqual(got.shape, ref.shape)
                self.assertEqual(got.dtype, ref.dtype)
                # Two bf16 GEMM chains with different K-reduction orders, so
                # close but not identical. bf16 carries ~3 decimal digits, so
                # 3e-2 relative is a couple of ULP on these logits.
                torch.testing.assert_close(
                    got.float(), ref.float(), rtol=3e-2, atol=3e-2
                )

    @unittest.skipUnless(torch.cuda.is_available(), "fused MLP is a GPU kernel")
    def test_selected_expert_is_unchanged_end_to_end(self):
        """The bar that matters: same expert, through the MLP *and* the tail.

        assert_close on the logits is necessary but not sufficient. The logits
        feed an argmax, and the point of the campaign is that greedy routing does
        not move, so compare the fully fused router against the fully unfused
        chain on a ZAYA1-shaped router.
        """
        router, config = _make_tiny_router(
            num_moe_experts=24,
            mlp_expansion=256,
            seed=31,
            device="cuda",
            dtype=torch.bfloat16,
            std=1.0 / 16.0,
        )
        torch.manual_seed(31)
        hidden = torch.randn(
            64, config.hidden_size, device="cuda", dtype=torch.bfloat16
        )
        with torch.no_grad():
            fused = router(hidden)
            hs, _ = router.down_proj(hidden)
            logits = router._router_logits_reference(router.rmsnorm_eda(hs))
            ref = router._routing_reference(logits, hidden.dtype, hs)

        self.assertTrue(
            torch.equal(fused.moe_ids.long(), ref.moe_ids.long()),
            "greedy expert choice must survive both fusions",
        )
        self.assertTrue(torch.equal(fused.skip_ids.long(), ref.skip_ids.long()))
        torch.testing.assert_close(
            fused.moe_weight, ref.moe_weight, rtol=3e-2, atol=3e-2
        )

    @unittest.skipUnless(torch.cuda.is_available(), "fused MLP is a GPU kernel")
    def test_zero_tokens_is_a_no_op(self):
        """Idle DP-attention forwards arrive with T == 0 and must not launch."""
        from sglang.kernels.ops.moe import zaya_router_mlp as kernel

        expansion, num_experts = 32, 5
        dev, dt = "cuda", torch.bfloat16
        out = kernel.router_mlp(
            torch.randn(0, expansion, device=dev, dtype=dt),
            torch.randn(expansion, expansion, device=dev, dtype=dt),
            torch.zeros(expansion, device=dev, dtype=dt),
            torch.randn(expansion, expansion, device=dev, dtype=dt),
            torch.zeros(expansion, device=dev, dtype=dt),
            torch.randn(num_experts, expansion, device=dev, dtype=dt),
            num_experts=num_experts,
        )
        self.assertEqual(tuple(out.shape), (0, num_experts))

    @unittest.skipUnless(torch.cuda.is_available(), "fused MLP is a GPU kernel")
    def test_partial_row_tile_matches(self):
        """T not a multiple of BLOCK_M: the masked rows must not corrupt output."""
        from sglang.kernels.ops.moe import zaya_router_mlp as kernel

        expansion, num_moe_experts = 32, 4
        router, _ = _make_tiny_router(
            num_moe_experts=num_moe_experts,
            mlp_expansion=expansion,
            seed=41,
            device="cuda",
            dtype=torch.bfloat16,
            std=1.0 / (expansion**0.5),
        )
        for num_tokens in (1, 31, 32, 33):
            with self.subTest(tokens=num_tokens):
                torch.manual_seed(num_tokens)
                hs_norm = torch.randn(
                    num_tokens, expansion, device="cuda", dtype=torch.bfloat16
                )
                operands = self._operands(router, hs_norm)
                with torch.no_grad():
                    got = kernel.router_mlp(*operands, num_experts=router.num_experts)
                    ref = router._router_logits_reference(hs_norm)
                self.assertEqual(tuple(got.shape), tuple(ref.shape))
                torch.testing.assert_close(
                    got.float(), ref.float(), rtol=3e-2, atol=3e-2
                )


if __name__ == "__main__":
    unittest.main()
