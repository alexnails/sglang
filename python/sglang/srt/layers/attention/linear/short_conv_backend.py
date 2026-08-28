# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Short-convolution attention backend.

Several hybrid models interleave a *causal short conv with per-request conv
state* (stored in the centralized ``MambaPool``) with softmax attention layers:

* **LFM2** (:class:`Lfm2ShortConv <sglang.srt.models.lfm2.Lfm2ShortConv>`) --
  a depthwise gated short conv (``causal_conv1d_fn`` / ``causal_conv1d_update``)
  as a standalone token mixer on its own conv layers.
* **ZAYA1** (:class:`CCA <sglang.srt.models.zaya.CCA>`) -- a two-stage grouped
  conv plus a one-token ``prev_hs`` lag, preprocessing q/k for the layer's
  softmax attention.

These share the *state plumbing* -- resolving the per-request slot indices, the
``has_initial_state`` prefix mask, the ``query_start_loc`` cu-seqlens, and the
cuda-graph static index buffers, all once per forward step -- but NOT the conv
kernel itself. ``ShortConvAttnBackend`` owns only the plumbing and hands it out
via :meth:`conv_state_metadata` as a :class:`ShortConvMetadata`; each model runs
its own conv kernel against that handle, so the model definition holds no pool
access.

The backend is a *sidecar*: it is invoked directly by the model (through
:class:`ShortConvHybridAttnBackend
<sglang.srt.layers.attention.hybrid_linear_attn_backend.ShortConvHybridAttnBackend>`),
never through the full-vs-linear ``forward_decode`` / ``forward_extend``
dispatch. Metadata + cuda-graph capture/replay come from
:class:`MambaAttnBackendBase`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, NamedTuple, Optional, Sequence

import torch

from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
    track_mamba_states_if_needed,
)
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_server_args

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class ShortConvMetadata(NamedTuple):
    """Per-(layer, step) conv-state handle handed to a model's conv kernel.

    ``layer_cache`` exposes the per-layer pool views (``conv[0]`` = conv state,
    ``conv[1]`` = an optional second state such as ZAYA1's ``prev_hs``,
    ``temporal`` = SSM state, unused by pure short convs). The device tensors are
    cuda-graph-static on the decode/replay path; the ``*_cpu`` host mirrors are
    built once per step only for models whose extend path runs a host loop
    (e.g. ZAYA1 v1) and are ``None`` on decode.
    """

    layer_cache: Any
    cache_indices: torch.Tensor
    # cu-seqlens for the varlen prefill conv (device, int32). None on decode.
    query_start_loc: Optional[torch.Tensor] = None
    # Per-request "resumes a cached prefix" mask (device bool). None on decode.
    has_initial_state: Optional[torch.Tensor] = None
    # Host mirror of cache_indices for extend host loops. None on decode.
    slot_ids_cpu: Optional[List[int]] = None
    # Host mirror of has_initial_state for extend host loops. None on decode.
    has_prefix_cpu: Optional[List[bool]] = None


class ShortConvAttnBackend(MambaAttnBackendBase):
    """Owns the short-conv per-request state plumbing (see module docstring)."""

    # State IO is index-driven; no host seq-lens plumbing required from the
    # runner. (The extend path reads ``extend_*_cpu`` off the batch, which is
    # always populated for extend regardless of this flag.)
    needs_cpu_seq_lens: bool = False

    # Pure short conv: ``temporal`` is a zero-element tensor, so the radix
    # track never snapshots an SSM state and the base skips building its
    # (host-synchronizing) SSM track indices.
    has_temporal_state: bool = False

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        mamba_cache = self.req_to_token_pool.mamba_pool.mamba_cache
        # conv[0] == conv_state: [n_layers, n_slots, conv_dim, conv_kernel - 1]
        self.conv_states_shape = mamba_cache.conv[0].shape
        # Sliding-window length of EVERY conv entry (its trailing axis). ZAYA1
        # has two: conv[0] is the conv_qk left padding (window ==
        # total_padding) and conv[1] is the one-token ``prev_hs`` lag (window
        # == 1). LFM2 has one (window == conv_L_cache - 1). Each entry's state
        # at length L is exactly that entry's last ``window`` INPUT rows, which
        # is what makes the extend-side snapshot a plain gather.
        self.conv_window_lens: List[int] = [int(c.shape[-1]) for c in mamba_cache.conv]

        # Per-step state, resolved ONCE per step in init_forward_metadata /
        # init_forward_metadata_out_graph (never per conv layer). The extend host
        # mirrors drive the extend loop; ``_cache_indices`` is the int64 slot
        # index view shared by all conv layers within the step.
        self._has_initial_state: Optional[torch.Tensor] = None
        self._slot_ids_cpu: Optional[List[int]] = None
        self._has_prefix_cpu: Optional[List[bool]] = None
        self._cache_indices: Optional[torch.Tensor] = None
        self._cache_indices_buf: Optional[torch.Tensor] = None

        # --- mamba radix cache, extra_buffer strategy ---------------------
        # Per-step extend track state: one flattened-token index tensor per
        # conv entry ([n_tracked, window]) plus the destination track slots.
        # Both None unless this step actually tracks something.
        self._track_conv_indices: Optional[List[torch.Tensor]] = None
        self._track_dst: Optional[torch.Tensor] = None
        # Decode-side, all-layers-at-once track plumbing (see
        # _init_track_state); None when the strategy is not extra_buffer.
        self._track_layer_row_base: Optional[torch.Tensor] = None
        self._track_pairs: Optional[List[tuple]] = None
        self.enable_mamba_extra_buffer = (
            model_runner.server_args.enable_mamba_extra_buffer()
        )
        if self.enable_mamba_extra_buffer:
            self._init_track_state(model_runner.server_args, mamba_cache)

    def _init_track_state(self, server_args, mamba_cache) -> None:
        """Validate + precompute the radix track plumbing (extra_buffer only).

        The decode snapshot is a pure row copy (live slot -> track slot) that
        every conv layer needs on the same step. ZAYA1 runs dozens of conv
        layers and its decode is launch-bound, so instead of one per layer this
        flattens the pool's ``[n_layers, n_slots, ...]`` conv tensors to
        ``[n_layers * n_slots, ...]`` and does the whole model in ONE launch,
        with row ids ``layer * n_slots + slot``. ``_track_layer_row_base`` is
        the constant ``[n_layers, 1]`` column of ``layer * n_slots``.
        """
        # Speculative decoding needs a per-draft-token track (the snapshot has
        # to land on the accepted step, not the last verified one), and the
        # decode cuda-graph runner disables its mamba-track buffers outright
        # when a spec algorithm is set -- the snapshot would then silently
        # never happen. Refuse the combination instead.
        if getattr(server_args, "speculative_algorithm", None) is not None:
            raise NotImplementedError(
                "mamba extra_buffer for short-conv models does not support "
                "speculative decoding; use --mamba-radix-cache-strategy "
                "no_buffer."
            )
        # The extend-side snapshot is a gather whose row count is
        # mamba_track_mask.sum() -- data dependent, so a captured prefill graph
        # would bake in whatever count capture happened to see (zero, since the
        # capture mask buffer is all-False) and never snapshot again. Unlike
        # the decode side there is no inert-buffer form of this, so refuse the
        # combination instead of shipping a graph that silently drops every
        # prefill checkpoint.
        prefill_graph = getattr(
            getattr(server_args, "cuda_graph_config", None), "prefill", None
        )
        if getattr(prefill_graph, "backend", "disabled") != "disabled":
            raise NotImplementedError(
                "mamba extra_buffer for short-conv models is not supported "
                "together with a prefill CUDA graph: the extend track gather "
                "has a data-dependent shape. Use "
                "--mamba-radix-cache-strategy no_buffer, or disable the "
                "prefill CUDA graph."
            )
        chunk = server_args.mamba_cache_chunk_size
        max_window = max(self.conv_window_lens)
        # The extend snapshot gathers the ``window`` input rows ending at the
        # chunk-aligned track position, which is >= mamba_cache_chunk_size into
        # the current extend (mamba_track_mask is only set when the extend is
        # at least one chunk long). A window that long would have to reach back
        # into the cached prefix, which the gather cannot express.
        assert max_window < chunk, (
            f"short-conv extra_buffer needs every conv window "
            f"({self.conv_window_lens}) < mamba_cache_chunk_size ({chunk}); "
            f"the minimum viable chunk here is {max_window + 1}. This is "
            "derived in ServerArgs.mamba_cache_chunk_size, which must not "
            "take a conv-only model's mamba_chunk_size (its scan length, 1) "
            "as the caching granularity."
        )
        assert server_args.mamba_track_interval >= chunk, (
            f"mamba_track_interval ({server_args.mamba_track_interval}) must be "
            f">= mamba_cache_chunk_size ({chunk})"
        )

        num_layers, num_slots = mamba_cache.conv[0].shape[:2]
        entries = []
        for conv in mamba_cache.conv:
            assert tuple(conv.shape[:2]) == (num_layers, num_slots), (
                "all conv entries must share the pool's [n_layers, n_slots] "
                f"leading dims, got {tuple(conv.shape[:2])}"
            )
            if not conv.is_contiguous():
                # Page-major / envelope conv views are strided, so the
                # flatten(0, 1) row addressing below is invalid. Fail loudly
                # rather than silently snapshotting the wrong bytes.
                raise NotImplementedError(
                    "mamba extra_buffer for short-conv models requires "
                    "contiguous conv state; the page-major envelope layout is "
                    "not supported yet."
                )
            entries.append(conv.flatten(0, 1))
        if len(entries) % 2 == 1:
            # The shared track kernel copies exactly two state tensors per
            # launch. Short-conv models carry a zero-element ``temporal``, so
            # it pairs with an odd conv entry at no cost.
            temporal = mamba_cache.temporal.flatten(0, 1)
            assert temporal[0].numel() == 0, (
                "short-conv backend expects an empty temporal state; a real "
                "SSM state would be silently dropped from the radix snapshot"
            )
            entries.append(temporal)
        self._track_pairs = [
            (entries[i], entries[i + 1]) for i in range(0, len(entries), 2)
        ]
        self._track_layer_row_base = (
            torch.arange(num_layers, dtype=torch.int64, device=self.device) * num_slots
        ).unsqueeze(1)

    def _reset_step_state(self):
        self._has_initial_state = None
        self._slot_ids_cpu = None
        self._has_prefix_cpu = None
        self._track_conv_indices = None
        self._track_dst = None

    def _alloc_cache_indices_buf(self, max_bs: int):
        # Persistent int64 index buffer, refilled in place per step so the
        # captured (cuda or cpu) graph reads a stable address.
        self._cache_indices_buf = torch.empty(
            max_bs, dtype=torch.int64, device=self.device
        )

    def _refresh_cache_indices(self):
        # Resolve the int64 slot-index view ONCE per step, shared by every conv
        # layer. When a graph index buffer is allocated and large enough, refill
        # it IN PLACE and hand out a view -- the captured graph then reads a
        # stable address that this (pre-replay) hook keeps current, so it is
        # cuda- and cpu-graph safe. Otherwise (eager, or bs beyond the buffer)
        # a fresh cast is fine.
        md = self.forward_metadata
        idx = md.mamba_cache_indices if md is not None else None
        buf = self._cache_indices_buf
        # Batch padding poisons unused rows' slot ids to -1 (see
        # MambaAttnBackendBase._forward_metadata: "padded rows are then poisoned
        # to -1"). Padding appears under cuda-graph bs rounding and, notably,
        # whenever DP attention pads a replica's batch. Clamp ONCE per step, here
        # where the shared int64 view is resolved, so every conv layer's
        # index_select / index_copy_ is in bounds without a per-layer clamp:
        # MambaPool reserves slot 0 (MambaSlotAllocator hands out 1..size), so
        # padded rows land on that scratch slot -- they can neither read out of
        # bounds nor clobber a live request's state, and the model discards their
        # outputs anyway. An unclamped -1 is an out-of-bounds device gather; on
        # ROCm it aborts the queue with HSA_STATUS_ERROR_EXCEPTION 0x1016 rather
        # than raising, which surfaced as a hard crash on ZAYA1 at attn_tp > 1
        # under DP attention.
        if idx is None:
            self._cache_indices = None
        elif buf is not None and idx.shape[0] <= buf.shape[0]:
            n = idx.shape[0]
            buf[:n].copy_(idx)
            buf[:n].clamp_(min=0)
            self._cache_indices = buf[:n]
        else:
            # ``clamp`` (not ``clamp_``): ``to(torch.long)`` is a no-op alias when
            # idx is already int64, so an in-place clamp would mutate the
            # backend's own mamba_cache_indices.
            self._cache_indices = idx.to(torch.long).clamp(min=0)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        super().init_cuda_graph_state(max_bs, max_num_tokens)
        self._alloc_cache_indices_buf(max_bs)

    def init_cpu_graph_state(self, max_bs: int, max_num_tokens: int):
        super().init_cpu_graph_state(max_bs, max_num_tokens)
        self._alloc_cache_indices_buf(max_bs)

    def _init_track_conv_indices(
        self, query_start_loc: torch.Tensor, forward_batch: ForwardBatch
    ) -> List[torch.Tensor]:
        """Flattened input positions to snapshot, ONE tensor per conv entry.

        Overrides the single-conv base implementation: a short-conv model may
        carry several conv entries with different window lengths (ZAYA1:
        ``[total_padding, 1]``), and each entry's snapshot is its own window of
        its own input tensor. The window for every entry ENDS at the same
        chunk-aligned track position, so ``indices[j][:, -1]`` is the same
        column for every ``j``.

        Returned tensors are ``[n_tracked, window_j]`` and index the flattened
        token axis; rows are restricted to ``mamba_track_mask``.
        """
        lens_to_track = (
            forward_batch.mamba_track_seqlens - forward_batch.extend_prefix_lens
        )
        chunk = get_server_args().mamba_cache_chunk_size
        aligned_len = (lens_to_track // chunk) * chunk
        # One past the last token whose input belongs in the snapshot.
        end = (query_start_loc[:-1] + aligned_len)[forward_batch.mamba_track_mask]
        last = query_start_loc[-1] - 1
        out: List[torch.Tensor] = []
        for window in self.conv_window_lens:
            starts = end - window
            offsets = torch.arange(window, device=self.device, dtype=starts.dtype)
            out.append((starts.unsqueeze(-1) + offsets).clamp(0, last))
        return out

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # Eager path (also the CPU-graph replay path). Builds
        # self.forward_metadata and runs the deferred mamba clear/COW ops.
        super().init_forward_metadata(forward_batch)
        self._reset_step_state()
        self._refresh_cache_indices()
        mode = forward_batch.forward_mode
        if (
            mode.is_extend()
            and not mode.is_target_verify()
            and not mode.is_draft_extend_v2()
        ):
            self._has_initial_state = forward_batch.extend_prefix_lens > 0
            if self._cache_indices is not None:
                self._slot_ids_cpu = self._cache_indices.tolist()
                self._has_prefix_cpu = [
                    int(p) > 0 for p in forward_batch.extend_prefix_lens_cpu
                ]
        # Extend-side radix track: the base only populates track_conv_indices
        # on the plain-extend branch and only when some row is tracked, so its
        # presence is the gate. mamba_track_indices was translated
        # virtual->physical in place by _forward_metadata.
        md = self.forward_metadata
        if md is not None and md.track_conv_indices is not None:
            self._track_conv_indices = md.track_conv_indices
            self._track_dst = forward_batch.mamba_track_indices[
                forward_batch.mamba_track_mask
            ]

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        # Decode cuda-graph capture + replay path -- no extend prefix state.
        super().init_forward_metadata_out_graph(forward_batch, in_capture)
        self._reset_step_state()
        self._refresh_cache_indices()

    def init_forward_metadata_capture_cpu_graph(self, *args, **kwargs):
        # Decode CPU-graph capture path. The base fills forward_metadata but not
        # the int64 view; without this the conv layers would capture a ``None``
        # index (crash / corrupt state). Replay goes through init_forward_metadata
        # and refills the SAME buffer, so the captured cpu graph reads a stable
        # address kept current at replay.
        super().init_forward_metadata_capture_cpu_graph(*args, **kwargs)
        self._reset_step_state()
        self._refresh_cache_indices()

    def conv_state_metadata(
        self, layer_id: int, forward_batch: ForwardBatch
    ) -> ShortConvMetadata:
        """Return the conv-state handle for ``layer_id`` at the current step.

        The per-step fields are already resolved on ``self.forward_metadata`` /
        ``self._*`` (in ``init_forward_metadata`` / ``_out_graph``);
        ``forward_batch`` is accepted for interface parity with the unit-test
        mock and is not otherwise required here.
        """
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        md = self.forward_metadata

        # Slot indices are cached ONCE per step in init_forward_metadata /
        # init_forward_metadata_out_graph (int64). Hand back the cached view -- no
        # per-layer recompute. Decode is cuda-graph-safe because that view is a
        # persistent buffer refilled in place before each replay.
        return ShortConvMetadata(
            layer_cache=layer_cache,
            cache_indices=self._cache_indices,
            query_start_loc=md.query_start_loc,
            has_initial_state=self._has_initial_state,
            slot_ids_cpu=self._slot_ids_cpu,
            has_prefix_cpu=self._has_prefix_cpu,
        )

    # ------------------------------------------------------------------
    # Mamba radix cache, extra_buffer strategy: the track snapshot
    # ------------------------------------------------------------------
    # Under `no_buffer` the radix tree is handed the request's LIVE state slot,
    # which is only ever current at the exact token count the scheduler last
    # saw -- hence page_size == 1 and no overlap schedule. `extra_buffer` gives
    # each request one or two extra pool slots (the ping-pong track buffer) and
    # snapshots the state into them at KNOWN, chunk-aligned sequence lengths.
    # That snapshot is what lets the cached key length and the cached state
    # agree while the scheduler runs a step ahead of the GPU, and what lets the
    # cached prefix be trimmed to a page boundary. Without it, enabling
    # extra_buffer would insert `token_ids[:mamba_last_track_seqlen]` against a
    # never-written slot -- a prefix hit that restores garbage conv state.

    def track_conv_states_extend(
        self, layer_cache: Any, conv_inputs: Sequence[torch.Tensor]
    ) -> None:
        """Snapshot this layer's conv entries at the chunk-aligned track point.

        ``conv_inputs[j]`` is the ``[T, C_j]`` tensor whose last ``window_j``
        rows ARE ``layer_cache.conv[j]`` after the conv runs, in the flattened
        token layout. Call once per conv layer on the extend path; the state
        slot the conv itself writes is a different row, so before-or-after the
        conv is equivalent. A no-op unless this step tracks something.
        """
        index_list = self._track_conv_indices
        if index_list is None:
            return
        dst = self._track_dst
        assert (
            len(conv_inputs) == len(index_list) == len(layer_cache.conv)
        ), f"expected {len(index_list)} conv inputs, got {len(conv_inputs)}"
        for conv_state, x, indices in zip(layer_cache.conv, conv_inputs, index_list):
            # [C, T] -> [C, n_tracked, window] -> [n_tracked, C, window]
            window = x.transpose(0, 1)[:, indices].transpose(0, 1)
            conv_state[dst] = window.to(conv_state.dtype)

    def track_conv_states_decode(self, forward_batch: ForwardBatch) -> None:
        """Snapshot EVERY conv layer's state into the track slots (one launch).

        Call once per decode step, after the last conv layer has updated its
        state: ``mamba_track_mask`` is built from the POST-increment seq_lens,
        so the row is tracked on the step whose output makes the length a
        multiple of ``mamba_track_interval``.

        CUDA-graph contract. Every tensor read here is either a persistent
        buffer refilled in place before replay (``_cache_indices`` and
        ``forward_metadata.mamba_track_indices``, both backend-owned; and
        ``forward_batch.mamba_track_mask``, the graph registry slot) or a
        constant allocated at init. Capture therefore MUST reach this call and
        record the scatter: during capture the mask buffer is all-False, so the
        kernel is inert and copies nothing, but the launch is in the graph and
        the refilled mask makes it fire at replay. Skipping the launch at
        capture time because "nothing is tracked right now" would silently drop
        every snapshot for the life of the graph. The intermediate index
        tensors below are allocated inside the captured region, so they come
        from the graph's private pool and are reused verbatim on replay.
        """
        if self._track_pairs is None:
            return
        if not forward_batch.forward_mode.is_decode_or_idle():
            return
        md = self.forward_metadata
        src = self._cache_indices
        mask = forward_batch.mamba_track_mask
        dst = md.mamba_track_indices if md is not None else None
        if src is None or mask is None or dst is None:
            return
        bs = src.shape[0]
        if bs == 0:
            return

        row_ok = mask[:bs]
        if self.enable_unified_memory:
            # The unified pool's v2p translate tombstones freed slots with -1;
            # folded into the mask (not left to the kernel's own check) because
            # `layer_base + -1` would alias the previous layer's last slot.
            # `_cache_indices` has already clamped its own -1s away, so the
            # source sentinel is read off the untouched metadata tensor.
            raw_src = md.mamba_cache_indices
            row_ok = row_ok & (dst[:bs] >= 0) & (raw_src[:bs] >= 0)
        base = self._track_layer_row_base  # [n_layers, 1]
        num_layers = base.shape[0]
        # Row id of (layer, slot) in the flattened [n_layers * n_slots, ...] view.
        src_rows = (base + src).reshape(-1)
        dst_rows = (base + dst[:bs]).reshape(-1)
        mask_rows = row_ok.expand(num_layers, bs).reshape(-1)
        total_rows = num_layers * bs
        for state_a, state_b in self._track_pairs:
            track_mamba_states_if_needed(
                state_a,
                state_b,
                src_rows,
                mask_rows,
                dst_rows,
                total_rows,
                # Invalid rows are already masked off above.
                check_freed_slots=False,
            )

    # The short-conv layers are invoked via conv_state_metadata + the model's own
    # conv kernel, never through the HybridLinearAttnBackend full-vs-linear
    # dispatch. Mirror Mamba2AttnBackend and guard the routed entrypoints.
    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError(
            "ShortConvAttnBackend is invoked via conv_state_metadata; "
            "it does not run through forward_decode."
        )

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError(
            "ShortConvAttnBackend is invoked via conv_state_metadata; "
            "it does not run through forward_extend."
        )
