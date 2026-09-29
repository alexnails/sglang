"""Reproduce the multimodal embedding-cache length mismatch.

Serves a cache entry whose row count is shorter than the item's placeholder
span, then drives the chunked-prefill assembly and the length check that
consumes it. On an unguarded build this raises:

    RuntimeError: Insufficient multimodal embedding length:
    num_mm_tokens_in_input_ids=279 vs num_mm_tokens_in_embedding=260

On a build carrying #36595 the entry is discarded and re-encoded instead.

CPU only; no engine, no GPU, no model weights. Resolves the scheduling module
at import time, so it runs against both the pre-split layout (mm_utils) and
the current one (mm_schedule).

    python3 scripts/playground/mm_embedding_cache/repro_length_mismatch.py

Exit code 0 if every path handled the mismatch, 1 if any path crashed.
"""

import inspect
import logging
import sys
import traceback

import torch

try:
    from sglang.srt.managers import mm_schedule as sched

    LAYOUT = "mm_schedule"
except ImportError:
    from sglang.srt.managers import mm_utils as sched

    LAYOUT = "mm_utils"

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem

HIDDEN = 16
ITEM_OFFSETS = [(2, 101), (110, 209), (220, 298)]
TOTAL_LEN = 320
POISONED_ROWS = 81
EXPECTED_TOKENS = sum(end - start + 1 for start, end in ITEM_OFFSETS)
CPU = torch.device("cpu")


def num_tokens(item):
    start, end = item.offsets[0]
    return end - start + 1


def item_embedding(item):
    generator = torch.Generator().manual_seed(item.hash)
    return torch.randn(num_tokens(item), HIDDEN, generator=generator)


def encoder(items):
    return [item_embedding(item) for item in items]


def make_items():
    return [
        MultimodalDataItem(
            modality=Modality.IMAGE,
            hash=1000 + index,
            feature=torch.zeros(1),
            offsets=[offsets],
        )
        for index, offsets in enumerate(ITEM_OFFSETS)
    ]


def adjust_embedding_length(embedding, expected_tokens):
    parameters = list(inspect.signature(sched._adjust_embedding_length).parameters)
    if parameters[1] == "mask":
        mask = torch.zeros(TOTAL_LEN, dtype=torch.bool)
        for start, end in ITEM_OFFSETS:
            mask[start : end + 1] = True
        assert int(mask.sum()) == expected_tokens, int(mask.sum())
        second_arg = mask.unsqueeze(-1)
    else:
        second_arg = expected_tokens
    return sched._adjust_embedding_length(
        embedding, second_arg, logging.getLogger("repro")
    )


def assemble_by_item(items):
    return sched._get_chunked_embedding_by_item(
        encoder, items, ITEM_OFFSETS, 0, TOTAL_LEN, CPU
    )


def assemble_batched(items):
    request = sched.PerImageRequestInfo(
        req_idx=0,
        items=items,
        items_offset=ITEM_OFFSETS,
        extend_prefix_len=0,
        extend_seq_len=TOTAL_LEN,
    )
    embeddings = sched._batch_encode_per_image_misses(encoder, [request], CPU)
    return sched._assemble_per_image_chunk(
        request.overlapping, embeddings, 0, TOTAL_LEN
    )


def run_case(name, assemble):
    print(f"\n--- {name} ---")
    sched.init_mm_embedding_cache(1 << 30)
    items = make_items()

    sched.embedding_cache.set(
        items[0].hash,
        sched.EmbeddingResult(embedding=torch.zeros(POISONED_ROWS, HIDDEN)),
    )
    print(
        f"poisoned cache: item0 span={num_tokens(items[0])} tokens, "
        f"cached entry={POISONED_ROWS} rows"
    )

    try:
        chunk = assemble(items)
    except Exception:
        print("assembly raised:")
        traceback.print_exc(file=sys.stdout)
        return "assembly-error"

    print(f"assembled rows = {chunk.shape[0]}   placeholders = {EXPECTED_TOKENS}")

    try:
        adjusted = adjust_embedding_length(chunk, EXPECTED_TOKENS)
    except RuntimeError as exc:
        print(f"CRASH: RuntimeError: {exc}")
        return "crash"
    print(f"handled; _adjust_embedding_length returned {adjusted.shape[0]} rows")
    return "ok"


def main():
    has_guard = hasattr(sched, "_discard_mismatched_cached_embedding")
    print("=" * 72)
    print(f"module : {sched.__file__}")
    print(f"layout : {LAYOUT}")
    print(f"cache length-guard present (#36595): {has_guard}")
    print("=" * 72)

    sched._acknowledge_deferred_cuda_ipc_cache_hits = lambda _items: None

    results = {
        "per-item (_get_chunked_embedding_by_item)": run_case(
            "per-item path", assemble_by_item
        ),
        "batched (_batch_encode_per_image_misses)": run_case(
            "batched path", assemble_batched
        ),
    }

    print("\n" + "=" * 72)
    print("SUMMARY")
    for path, outcome in results.items():
        print(f"  {path}: {outcome}")
    print("=" * 72)

    return 1 if any(outcome != "ok" for outcome in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
