from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


class TorchNativeAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        pass

    def _run_sdpa_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
        seq_lens_cpu: Optional[torch.Tensor] = None,
        extend_prefix_lens_cpu: Optional[List[int]] = None,
        extend_seq_lens_cpu: Optional[List[int]] = None,
    ):
        """Run the extend forward by using torch native sdpa op.

        Pre-computes loop indices on CPU to avoid per-iteration GPU->CPU syncs.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_kv_heads, head_size]
            v_cache: [max_total_num_tokens, num_kv_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            extend_prefix_lens: [num_seqs]
            extend_seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool
            seq_lens_cpu: optional CPU tensor to avoid GPU sync
            extend_prefix_lens_cpu: optional CPU list to avoid GPU sync
            extend_seq_lens_cpu: optional CPU list to avoid GPU sync

        Returns:
            output: [num_tokens, num_heads, head_size]
        """
        num_seqs = seq_lens.shape[0]
        if num_seqs == 0:
            return output

        assert num_seqs == extend_prefix_lens.shape[0]
        assert num_seqs == extend_seq_lens.shape[0]

        # Pre-compute all loop-control values on CPU (one sync instead of N)
        if extend_seq_lens_cpu is None:
            extend_seq_lens_cpu = extend_seq_lens.tolist()
        if extend_prefix_lens_cpu is None:
            extend_prefix_lens_cpu = extend_prefix_lens.tolist()
        if seq_lens_cpu is not None:
            seq_lens_list = (
                seq_lens_cpu.tolist()
                if isinstance(seq_lens_cpu, torch.Tensor)
                else list(seq_lens_cpu)
            )
        else:
            seq_lens_list = seq_lens.tolist()
        req_pool_indices_list = req_pool_indices.tolist()

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q = 0
        for seq_idx in range(num_seqs):
            ext_len = extend_seq_lens_cpu[seq_idx]
            prefix_len = extend_prefix_lens_cpu[seq_idx]
            seq_len = seq_lens_list[seq_idx]
            req_idx = req_pool_indices_list[seq_idx]

            end_q = start_q + ext_len

            per_req_query = query[:, start_q:end_q, :]
            per_req_query_redudant = torch.empty(
                (per_req_query.shape[0], seq_len, per_req_query.shape[2]),
                dtype=per_req_query.dtype,
                device=per_req_query.device,
            )

            per_req_query_redudant[:, prefix_len:, :] = per_req_query

            per_req_tokens = req_to_token[req_idx, :seq_len]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            per_req_out_redudant = (
                scaled_dot_product_attention(
                    per_req_query_redudant.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out_redudant[prefix_len:, :, :]
            start_q = end_q
        return output

    def _run_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
        seq_lens_cpu: Optional[torch.Tensor] = None,
    ):
        """Batched decode forward: single vectorized SDPA call for all requests.

        Instead of looping per-request (N separate SDPA calls with GPU syncs),
        gathers all K/V into padded tensors and runs one batched SDPA.

        Args:
            query: [batch_size, num_q_heads, qk_head_dim]
            output: [batch_size, num_q_heads, v_head_dim]
            k_cache: [max_total_num_tokens, num_kv_heads, qk_head_dim]
            v_cache: [max_total_num_tokens, num_kv_heads, v_head_dim]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [batch_size]
            seq_lens: [batch_size]
            scaling: float or None
            enable_gqa: bool
            causal: bool (unused, decode is never causal)
            seq_lens_cpu: optional CPU tensor to avoid GPU sync

        Returns:
            output: [batch_size, num_q_heads, v_head_dim]
        """
        bs = query.shape[0]
        if bs == 0:
            return output

        device = query.device

        # Compute max KV length from CPU tensor to avoid a GPU->CPU sync
        if seq_lens_cpu is not None:
            max_kv_len = int(seq_lens_cpu.max().item())
        else:
            max_kv_len = int(seq_lens.max().item())

        if max_kv_len == 0:
            return output

        # Vectorized KV index gathering: select the req_to_token rows for
        # all requests at once, then truncate to max_kv_len.
        gather_indices = req_to_token[req_pool_indices, :max_kv_len]  # [bs, max_kv_len]

        # Mask: True for valid KV positions, False for padding
        arange = torch.arange(max_kv_len, device=device)
        valid_mask = arange.unsqueeze(0) < seq_lens.unsqueeze(1)  # [bs, max_kv_len]

        # Zero out padding indices so gather doesn't hit out-of-bounds
        gather_indices = gather_indices * valid_mask.long()

        # Batched gather: fetch all K/V in one operation
        flat_idx = gather_indices.reshape(-1)  # [bs * max_kv_len]
        num_kv_heads = k_cache.shape[1]
        qk_head_dim = k_cache.shape[2]
        v_head_dim = v_cache.shape[2]

        # [bs * max_kv_len, num_kv_heads, head_dim] -> [bs, num_kv_heads, max_kv_len, head_dim]
        gathered_k = (
            k_cache[flat_idx]
            .reshape(bs, max_kv_len, num_kv_heads, qk_head_dim)
            .transpose(1, 2)
        )
        gathered_v = (
            v_cache[flat_idx]
            .reshape(bs, max_kv_len, num_kv_heads, v_head_dim)
            .transpose(1, 2)
        )

        if gathered_k.dtype != query.dtype:
            gathered_k = gathered_k.to(query.dtype)
            gathered_v = gathered_v.to(query.dtype)

        # Query: [bs, num_q_heads, qk_head_dim] -> [bs, num_q_heads, 1, qk_head_dim]
        q = query.unsqueeze(2)

        # Attention mask: [bs, 1, 1, max_kv_len]
        attn_mask = valid_mask.unsqueeze(1).unsqueeze(1)

        # Single batched SDPA call for ALL requests
        attn_out = scaled_dot_product_attention(
            q,
            gathered_k,
            gathered_v,
            attn_mask=attn_mask,
            enable_gqa=enable_gqa,
            scale=scaling,
            is_causal=False,
        )

        # [bs, num_q_heads, 1, v_head_dim] -> [bs, num_q_heads, v_head_dim]
        output[:] = attn_out.squeeze(2)
        return output

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

        self._run_sdpa_forward_extend(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_prefix_lens,
            forward_batch.extend_seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=causal,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
            extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        )
        return o

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        self._run_sdpa_forward_decode(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=False,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
        )

        return o

    def init_cpu_graph_state(self, max_bs: int, max_num_tokens: int):
        pass

    def init_forward_metadata_capture_cpu_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices,
        seq_lens,
        encoder_lens,
        forward_mode,
        spec_info,
    ):
        pass

    def get_cpu_graph_seq_len_fill_value(self):
        return 1

    def support_triton(self):
        return False
