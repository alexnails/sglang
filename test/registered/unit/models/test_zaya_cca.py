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
4. Multi-request prefills update only the conv state and ``prev_hs`` slots for
   each request and leave unused slots zero.
5. A simulated tensor-parallel (TP=2) CCA produces per-rank q / k / v slices
   that match the corresponding head slices of a TP=1 reference, both for
   prefill (``_forward_extend``) and for decode (``_forward_decode``).

All tests run on CPU with a tiny configuration so they stay fast and have no
GPU dependency. State is stored in a mock centralized pool that mirrors the
``HybridReqToTokenPool`` / ``MambaPool`` interface used at serving time.
"""

import os
import unittest
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
    ``conv[0]`` state. ``conv[1]`` (prev_hs) is replicated and stays at full
    ``hidden_size``.
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
            num_layers, pool_size + 1, cca_config.hidden_size, 1
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
        prev_hs_state = layer_cache.conv[1]

        self.assertTrue(torch.any(conv_state[2] != 0))
        self.assertTrue(torch.any(conv_state[5] != 0))

        torch.testing.assert_close(
            prev_hs_state[2].squeeze(-1).to(torch.float32),
            hs0[-1].to(torch.float32),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            prev_hs_state[5].squeeze(-1).to(torch.float32),
            hs1[-1].to(torch.float32),
            atol=1e-5,
            rtol=1e-5,
        )

        for idx in (0, 1, 3, 4):
            self.assertTrue(torch.all(conv_state[idx] == 0))
            self.assertTrue(torch.all(prev_hs_state[idx] == 0))

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
        for num_channels, hidden_size, total_padding in ((64, 32, 2), (48, 16, 3)):
            for num_tokens in (1, 6):
                with self.subTest(c=num_channels, h=hidden_size, p=total_padding):
                    slots = (
                        torch.randperm(32, device=dev)[:num_tokens].to(torch.long) + 1
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

    def _run(self, num_q_heads: int, num_k_heads: int, num_tokens: int):
        import torch as _torch

        from sglang.kernels.ops.attention import cca_qk_mix as kernel

        head_dim = 32
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
        # degenerate group where the k blend reduces to a single q head.
        for num_q_heads, num_k_heads in ((8, 1), (8, 2), (2, 2)):
            for num_tokens in (1, 5):
                with self.subTest(q=num_q_heads, k=num_k_heads, t=num_tokens):
                    self._run(num_q_heads, num_k_heads, num_tokens)

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
        taps = cca.total_padding + 1
        torch.manual_seed(7)
        window = torch.randn(T, cca.in_out_ch, taps, dtype=torch.float32) * 0.3
        with torch.no_grad():
            # Reference: the real two-stage conv, which yields exactly one step.
            ref = cca.conv_qk(window)
            self.assertEqual(ref.shape, (T, cca.in_out_ch, 1))
            ref = ref.squeeze(-1)

            cca.fold_decode_conv()
            grouped = window.reshape(T, cca.decode_conv_groups, -1)
            got = (
                torch.einsum("tgk,gok->tgo", grouped, cca.decode_conv_weight)
                + cca.decode_conv_bias
            ).reshape(T, -1)
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)

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


def _reference_extend_conv(
    qk: torch.Tensor,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    conv_state: torch.Tensor,
    prev_hs_state: torch.Tensor,
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
    v2_out = torch.empty_like(hidden_states)
    new_conv_state = conv_state.clone()
    new_prev_hs = prev_hs_state.clone()

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

        hs_cur = hidden_states[start:end]
        if has_prefix[i]:
            first = prev_hs_state[slot].squeeze(-1).to(hs_cur.dtype).unsqueeze(0)
        else:
            first = hs_cur.new_zeros((1, hs_cur.shape[-1]))
        v2_out[start:end] = torch.cat([first, hs_cur[:-1]], dim=0)
        new_prev_hs[slot] = hs_cur[-1].unsqueeze(-1).to(prev_hs_state.dtype)
        start = end

    return qk_out, v2_out, new_conv_state, new_prev_hs


class TestCCAFusedPrefillConv(CustomTestCase):
    """Fused varlen prefill conv vs the per-request torch reference.

    The fused kernel resolves each token's request, start offset, pool slot and
    prefix flag from device tensors instead of a host loop. Every one of those is a
    silent-corruption path: a token attributed to the wrong request reads another
    request's conv history, and nothing downstream notices.
    """

    GROUPS = 3
    CG = 16  # tl.dot needs a tile at least 16 wide
    HIDDEN = 8
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
            hidden_states=torch.randn(total, self.HIDDEN, device=dev, dtype=dt),
            weight=torch.randn(channels, self.CG, taps, device=dev, dtype=dt) * 0.1,
            bias=torch.randn(channels, device=dev, dtype=dt) * 0.1,
            conv_state=torch.randn(
                num_slots, channels, self.TOTAL_PADDING, device=dev, dtype=dt
            ),
            prev_hs_state=torch.randn(num_slots, self.HIDDEN, 1, device=dev, dtype=dt),
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

        ref_qk, ref_v2, ref_cs, ref_ph = _reference_extend_conv(
            args["qk"],
            args["hidden_states"],
            args["weight"],
            args["bias"],
            args["conv_state"],
            args["prev_hs_state"],
            seq_lens,
            slot_ids,
            has_prefix,
            self.TOTAL_PADDING,
            self.GROUPS,
        )

        conv_state = args["conv_state"].clone()
        prev_hs_state = args["prev_hs_state"].clone()
        self.assertTrue(
            cca_conv1d.covered(
                args["qk"],
                args["hidden_states"],
                args["weight"],
                args["bias"],
                conv_state,
                prev_hs_state,
                cu,
                prefix,
                slots,
                self.TOTAL_PADDING,
                self.GROUPS,
            )
        )
        got_qk, got_v2 = cca_conv1d.cca_conv1d_fn(
            args["qk"],
            args["hidden_states"],
            args["weight"],
            args["bias"],
            conv_state,
            prev_hs_state,
            cu,
            prefix,
            slots,
            self.TOTAL_PADDING,
            self.GROUPS,
        )
        return (got_qk, got_v2, conv_state, prev_hs_state), (
            ref_qk,
            ref_v2,
            ref_cs,
            ref_ph,
        )

    def _assert_matches(self, got, ref):
        names = ("qk_out", "v2_input", "conv_state", "prev_hs_state")
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
    def test_covered_rejects_the_unfolded_two_stage_conv(self):
        """Without the folded weight there is nothing for this kernel to apply."""
        from sglang.kernels.ops.attention import cca_conv1d

        args = self._inputs([4], [False])
        cu = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
        self.assertFalse(
            cca_conv1d.covered(
                args["qk"],
                args["hidden_states"],
                None,
                args["bias"],
                args["conv_state"],
                args["prev_hs_state"],
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
    HIDDEN = 8
    TOTAL_PADDING = 2

    def _run(self, num_tokens: int, slot_ids: List[int]):
        from sglang.kernels.ops.attention import cca_conv1d_update

        torch.manual_seed(0)
        channels = self.GROUPS * self.CG
        taps = self.TOTAL_PADDING + 1
        dev, dt = "cuda", torch.bfloat16
        num_slots = max(s for s in slot_ids) + 2

        qk = torch.randn(num_tokens, channels, device=dev, dtype=dt)
        hs = torch.randn(num_tokens, self.HIDDEN, device=dev, dtype=dt)
        weight = (
            torch.randn(self.GROUPS, self.CG, self.CG * taps, device=dev, dtype=dt)
            * 0.1
        )
        bias = torch.randn(self.GROUPS, self.CG, device=dev, dtype=dt) * 0.1
        conv_state = torch.randn(
            num_slots, channels, self.TOTAL_PADDING, device=dev, dtype=dt
        )
        prev_hs = torch.randn(num_slots, self.HIDDEN, 1, device=dev, dtype=dt)
        slots = torch.tensor(slot_ids, dtype=torch.int64, device=dev)

        # Reference: build the window, apply the einsum, shift the pools.
        live = [s for s in slot_ids if s >= 0]
        ref_cs, ref_ph = conv_state.clone(), prev_hs.clone()
        left = conv_state.index_select(0, slots.clamp(min=0))
        window = torch.cat([left, qk.unsqueeze(-1)], dim=-1)
        grouped = window.reshape(num_tokens, self.GROUPS, -1)
        ref_qk = (
            torch.einsum("tgk,gok->tgo", grouped.float(), weight.float()) + bias.float()
        ).reshape(num_tokens, -1)
        ref_prev = prev_hs.index_select(0, slots.clamp(min=0)).squeeze(-1)
        for i, s in enumerate(slot_ids):
            if s >= 0:
                ref_cs[s] = window[i, :, -self.TOTAL_PADDING :]
                ref_ph[s] = hs[i].unsqueeze(-1)

        got_cs, got_ph = conv_state.clone(), prev_hs.clone()
        self.assertTrue(
            cca_conv1d_update.covered(
                qk,
                hs,
                weight,
                bias,
                got_cs,
                got_ph,
                slots,
                self.TOTAL_PADDING,
                self.GROUPS,
            )
        )
        got_qk, got_prev = cca_conv1d_update.cca_conv1d_update(
            qk,
            hs,
            weight,
            bias,
            got_cs,
            got_ph,
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
                got_ph.index_select(0, idx).float(),
                ref_ph.index_select(0, idx).float(),
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
        hs = torch.randn(4, self.HIDDEN, device=dev, dtype=dt)
        weight = (
            torch.randn(self.GROUPS, self.CG, self.CG * taps, device=dev, dtype=dt)
            * 0.1
        )
        bias = torch.randn(self.GROUPS, self.CG, device=dev, dtype=dt) * 0.1
        conv_state = torch.randn(3, channels, self.TOTAL_PADDING, device=dev, dtype=dt)
        prev_hs = torch.randn(3, self.HIDDEN, 1, device=dev, dtype=dt)
        before_cs, before_ph = conv_state.clone(), prev_hs.clone()
        slots = torch.tensor([-1, -1, -1, -1], dtype=torch.int64, device=dev)

        cca_conv1d_update.cca_conv1d_update(
            qk,
            hs,
            weight,
            bias,
            conv_state,
            prev_hs,
            slots,
            self.TOTAL_PADDING,
            self.GROUPS,
        )
        self.assertTrue(torch.equal(conv_state, before_cs))
        self.assertTrue(torch.equal(prev_hs, before_ph))


if __name__ == "__main__":
    unittest.main()
