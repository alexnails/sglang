# Multimodal embedding-cache length mismatch

Diagnostic artifacts for the crash:

```
RuntimeError: Insufficient multimodal embedding length:
num_mm_tokens_in_input_ids=279 vs num_mm_tokens_in_embedding=260. This is an internal error
```

## Cause

`MultiModalStaticCache` is keyed on bare `item.hash`, a 64-bit digest of the
*flattened* feature bytes. Neither the shape nor the number of placeholder
tokens the item occupies in `input_ids` is part of the key, so one key can map
to entries of different row counts.

Before #36595 the three cache-read sites returned the entry without checking
its length against the item's span. A short entry is then absorbed silently:
the assembly slices `emb[local_start:local_end]` using bounds derived from the
placeholder offsets, and torch slicing clamps rather than raising. The
shortfall only surfaces later in `_adjust_embedding_length`, which can trim an
over-long embedding but has no recovery path for a short one.

The asymmetry matters: an over-long cached entry slices back to exactly the
span and is invisible. Only an under-long entry is fatal.

Fixed by #36595 (`20a491d1d3`), first released in **v0.5.19** — not in v0.5.18.

## Reproducing

```
python3 scripts/playground/mm_embedding_cache/repro_length_mismatch.py
```

CPU only; no engine, GPU or model weights. Resolves the scheduling module at
import time, so it runs against both the pre-split layout (`mm_utils`) and the
current one (`mm_schedule`). Exits 1 if the crash reproduces, 0 if the build
discards and re-encodes the mismatched entry.

## Backporting to v0.5.17

#36595 landed in `python/sglang/srt/managers/mm_schedule.py`, which does not
exist in v0.5.17 — the multimodal scheduling code was split out of
`mm_utils.py` afterwards (#32415). A plain cherry-pick will not apply.

`backport-36595-onto-v0.5.17.patch` is the equivalent change against
`mm_utils.py` (82 insertions, 22 deletions), applied with:

```
git apply scripts/playground/mm_embedding_cache/backport-36595-onto-v0.5.17.patch
```

`generate_backport.py` regenerates it from a clean v0.5.17 tree; its anchors
are asserted, so it fails loudly rather than misapplying against a different
base. No other commits are needed — `EVSEmbeddingResult` is already imported in
v0.5.17, and `MultiModalStaticCache.free()` has the same signature and ignores
the allocator argument.

## Stopgap if a version bump is blocked

`SGLANG_VLM_CACHE_SIZE_MB=0` disables the cache entirely (`set()` admits
nothing at `max_size=0`). Measured cost on gemma-4-31B-it / H200, 256-token
images, `max_tokens=1`, n=30, median ms:

| workload | radix cache | 100 MB | 0 | delta |
|---|---|---|---|---|
| repeated image | off | 63.0 | 89.8 | +26.8 ms |
| repeated image | on | 59.4 | 59.6 | none |
| unique images | off | 89.2 | 89.4 | none |
| unique images | on | 90.6 | 89.4 | none |

~27 ms is the ViT re-encode for one image, and it is the upper bound: it is
paid only where the embedding cache would otherwise have hit. With a radix
prefix hit, or with unique images, disabling is free.

This does not fully close the hole on v0.5.17 — the in-batch dedup in
`_batch_encode_per_image_misses` still keys on bare `item.hash`, so two
colliding items within one batch can still mismatch.
