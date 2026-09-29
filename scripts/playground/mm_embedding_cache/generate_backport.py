"""Backport of #36595 (20a491d1d3) onto v0.5.17.

Upstream landed in python/sglang/srt/managers/mm_schedule.py, which does not
exist in 0.5.17 -- the multimodal scheduling code was split out of mm_utils.py
after that release (#32415). This applies the same five changes to mm_utils.py.

Usage: python3 backport_36595.py <path-to-mm_utils.py>
"""

import sys

path = sys.argv[1]
src = open(path).read()

if "_discard_mismatched_cached_embedding" in src:
    print("already backported:", path)
    sys.exit(0)


def sub(old, new, why):
    global src
    if old not in src:
        raise SystemExit(f"ANCHOR NOT FOUND ({why}):\n{old[:200]}")
    if src.count(old) != 1:
        raise SystemExit(f"ANCHOR NOT UNIQUE ({why}): {src.count(old)} matches")
    src = src.replace(old, new, 1)
    print("  applied:", why)


# ---- 1. helpers -------------------------------------------------------------
sub(
    "def _can_skip_pre_embed_feature_move(data_embedding_func: DataEmbeddingFunc) -> bool:",
    '''def _embedding_token_count(embedding: torch.Tensor) -> int:
    """Return the number of multimodal tokens represented by an embedding."""
    # Vision encoders may return [tokens, hidden] or a higher-rank tensor.  The
    # scheduler always consumes the flattened token dimension.
    return embedding.reshape(-1, embedding.shape[-1]).shape[0]


def _discard_mismatched_cached_embedding(
    cache_key: Optional[int],
    expected_token_count: int,
    cached_token_count: int,
) -> None:
    """Log and remove a cache entry that cannot serve the current item."""
    logger.warning(
        "Discarding cached multimodal embedding due to a token-count mismatch: "
        "cache_key=%s, expected_tokens=%d, cached_tokens=%d. Recomputing embedding.",
        cache_key,
        expected_token_count,
        cached_token_count,
    )
    embedding_cache.free(cache_key, None)


def _can_skip_pre_embed_feature_move(data_embedding_func: DataEmbeddingFunc) -> bool:''',
    "add _embedding_token_count + _discard_mismatched_cached_embedding",
)

# ---- 2. guard the full (combined) path --------------------------------------
sub(
    """    embedding_per_req = embedding_cache.get(item_hashes)

    if embedding_per_req is None:""",
    """    embedding_per_req = embedding_cache.get(item_hashes)

    # A compact feature hash can collide for inputs with different token
    # counts.  Never feed a stale cache entry into the scheduler: the length
    # mismatch would otherwise surface much later in _adjust_embedding_length
    # as an unrecoverable prefill crash.
    if embedding_per_req is not None and not isinstance(
        embedding_per_req, EVSEmbeddingResult
    ):
        expected_token_count = sum(end - start + 1 for start, end in items_offset)
        cached_token_count = _embedding_token_count(embedding_per_req.embedding)
        if cached_token_count != expected_token_count:
            _discard_mismatched_cached_embedding(
                embedding_items_hash, expected_token_count, cached_token_count
            )
            embedding_per_req = None

    if embedding_per_req is None:""",
    "guard _get_chunked_embedding_full",
)

# ---- 3. batched per-image path: key on (hash, token_count) -------------------
sub(
    """) -> Dict[int, torch.Tensor]:
    \"\"\"
    Collect cache misses across ALL per-image requests, deduplicate by hash,
    encode in a single ViT call, and populate the cache.

    Returns:
        hash_to_embedding: mapping from item.hash to its full embedding tensor.
    \"\"\"
    unique_misses: Dict[int, Tuple[MultimodalDataItem, int]] = {}
    hash_to_embedding: Dict[int, torch.Tensor] = {}""",
    """) -> Dict[Tuple[Optional[int], int], torch.Tensor]:
    \"\"\"
    Collect cache misses across ALL per-image requests, deduplicate by hash and
    expected token count, encode in a single ViT call, and populate the cache.

    Returns:
        hash_to_embedding: mapping from (item.hash, token_count) to its full
            embedding tensor.  Including the token count prevents two
            colliding hashes with different placeholder spans from being
            deduplicated within the same batch.
    \"\"\"
    unique_misses: Dict[Tuple[Optional[int], int], Tuple[MultimodalDataItem, int]] = {}
    hash_to_embedding: Dict[Tuple[Optional[int], int], torch.Tensor] = {}""",
    "retype _batch_encode_per_image_misses",
)

sub(
    """        for _idx, item, start, end in overlapping:
            if item.hash in hash_to_embedding:
                continue
            cached = embedding_cache.get_single(item.hash)
            if cached is not None:
                hash_to_embedding[item.hash] = cached.embedding
            elif item.hash not in unique_misses:
                token_count = end - start + 1
                unique_misses[item.hash] = (item, token_count)""",
    """        for _idx, item, start, end in overlapping:
            expected_token_count = end - start + 1
            cache_key = (item.hash, expected_token_count)
            if cache_key in hash_to_embedding:
                continue
            cached = embedding_cache.get_single(item.hash)
            if cached is not None:
                cached_embedding = cached.embedding
                cached_token_count = _embedding_token_count(cached_embedding)
                if cached_token_count == expected_token_count:
                    hash_to_embedding[cache_key] = cached_embedding
                else:
                    _discard_mismatched_cached_embedding(
                        item.hash, expected_token_count, cached_token_count
                    )
                    unique_misses[cache_key] = (item, expected_token_count)
            elif cache_key not in unique_misses:
                unique_misses[cache_key] = (item, expected_token_count)""",
    "guard batched cache reads",
)

sub(
    """        ordered_hashes = list(unique_misses.keys())
        miss_items = [unique_misses[h][0] for h in ordered_hashes]
        token_counts = [unique_misses[h][1] for h in ordered_hashes]""",
    """        ordered_cache_keys = list(unique_misses.keys())
        miss_items = [unique_misses[key][0] for key in ordered_cache_keys]
        token_counts = [unique_misses[key][1] for key in ordered_cache_keys]""",
    "rename ordered_hashes -> ordered_cache_keys",
)

sub(
    """        for h, emb in zip(ordered_hashes, split_embeddings):
            embedding_cache.set(h, EmbeddingResult(embedding=emb))
            # Keep a local ref (no extra GPU memory) so assembly never fails due to LRU eviction.
            hash_to_embedding[h] = emb""",
    """        for cache_key, emb in zip(ordered_cache_keys, split_embeddings):
            embedding_cache.set(cache_key[0], EmbeddingResult(embedding=emb))
            # Keep a local ref (no extra GPU memory) so assembly never fails due to LRU eviction.
            hash_to_embedding[cache_key] = emb""",
    "store under (hash, token_count)",
)

# ---- 4. per-item path -------------------------------------------------------
sub(
    """    for idx, item, start, end in overlapping:
        cached = embedding_cache.get_single(item.hash)
        if cached is not None:
            cached_embeddings[idx] = cached.embedding
            _acknowledge_deferred_cuda_ipc_cache_hits([item])
        else:
            miss_items.append((idx, item, start, end))""",
    """    for idx, item, start, end in overlapping:
        expected_token_count = end - start + 1
        cached = embedding_cache.get_single(item.hash)
        if cached is not None:
            cached_embedding = cached.embedding
            cached_token_count = _embedding_token_count(cached_embedding)
            if cached_token_count == expected_token_count:
                cached_embeddings[idx] = cached_embedding
                _acknowledge_deferred_cuda_ipc_cache_hits([item])
            else:
                _discard_mismatched_cached_embedding(
                    item.hash, expected_token_count, cached_token_count
                )
                miss_items.append((idx, item, start, end))
        else:
            miss_items.append((idx, item, start, end))""",
    "guard _get_chunked_embedding_by_item",
)

# ---- 5. assembly lookup -----------------------------------------------------
sub(
    """    hash_to_embedding: Dict[int, torch.Tensor],
    extend_prefix_len: int,
    extend_seq_len: int,
) -> Optional[torch.Tensor]:""",
    """    hash_to_embedding: Dict[Tuple[Optional[int], int], torch.Tensor],
    extend_prefix_len: int,
    extend_seq_len: int,
) -> Optional[torch.Tensor]:""",
    "retype _assemble_per_image_chunk",
)

sub(
    """        emb = hash_to_embedding[item.hash]  # shape: (end - start + 1, hidden)""",
    """        cache_key = (item.hash, end - start + 1)
        emb = hash_to_embedding[cache_key]  # shape: (end - start + 1, hidden)""",
    "look up by (hash, token_count)",
)

sub(
    """    # Phase 1: batch encode all per-image cache misses in ONE ViT call
    hash_to_embedding: Dict[int, torch.Tensor] = {}""",
    """    # Phase 1: batch encode all per-image cache misses in ONE ViT call
    hash_to_embedding: Dict[Tuple[Optional[int], int], torch.Tensor] = {}""",
    "retype local in _get_chunked_prefill_embedding",
)

open(path, "w").write(src)
print("backported:", path)
