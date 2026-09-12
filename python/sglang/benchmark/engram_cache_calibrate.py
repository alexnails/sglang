from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from sglang.srt.layers.attention.dsv4.torch_quant import FP8_BLOCK_SIZE
from sglang.srt.layers.engram import (
    build_compressed_token_map,
    build_engram_layout,
    compute_hash_multipliers,
)

logger = logging.getLogger(__name__)

_CHUNK = 1 << 22


def _layout_tables(config, vocab_size: int):
    layout = build_engram_layout(config)
    flat = [[p for per_ngram in layer for p in per_ngram] for layer in layout.primes]
    offsets = np.array([np.cumsum([0, *sizes[:-1]]) for sizes in flat])
    multipliers = compute_hash_multipliers(
        layout.layer_ids, layout.max_ngram_size, vocab_size
    ).numpy()
    return layout, np.array(layout.primes), offsets, multipliers


def _lookback(ids: np.ndarray, n: int, carry: Optional[np.ndarray]):
    """Column s is the s-th predecessor; blocked marks look-back off the start.

    `carry` is the previous chunk's tail so a chunk boundary does not
    manufacture sequence starts that the server would never see.
    """
    if carry is None:
        pad = np.zeros(n - 1, dtype=ids.dtype)
        blocked_head = n - 1
    else:
        pad, blocked_head = carry, 0
    joined = np.concatenate([pad, ids])
    tokens = np.empty((ids.shape[0], n), dtype=np.int64)
    for shift in range(n):
        lo = (n - 1) - shift
        tokens[:, shift] = joined[lo : lo + ids.shape[0]]
    blocked = np.zeros_like(tokens, dtype=bool)
    if blocked_head:
        pos = np.arange(ids.shape[0])
        for shift in range(n):
            blocked[:, shift] = pos < shift - (n - 1 - blocked_head)
    return tokens, blocked, joined[-(n - 1) :]


def rolling_groups(
    streams: Iterator[np.ndarray],
    config,
    tokenizer,
) -> tuple[list[list[np.ndarray]], int, int]:
    """Group ids per (engram layer, n-gram size).

    The n-gram's heads share one rolling hash and differ only in which prime
    they mod, so the rolling value is the group identity: one lookup group of
    `n_heads` rows.
    """
    token_map, compressed_vocab = build_compressed_token_map(tokenizer)
    assert compressed_vocab == config.engram_compressed_vocab_size, (
        f"the tokenizer normalizes to {compressed_vocab} distinct tokens but the "
        f"config expects {config.engram_compressed_vocab_size}; every hash "
        "multiplier depends on it"
    )
    layout, _, _, multipliers = _layout_tables(config, compressed_vocab)
    token_map = np.asarray(token_map, dtype=np.int64)
    pad_row = token_map[config.engram_pad_token_id]
    n_layers, n_ngram = len(layout.layer_ids), layout.max_ngram_size - 1
    parts: list[list[list[np.ndarray]]] = [
        [[] for _ in range(n_ngram)] for _ in range(n_layers)
    ]
    total = 0
    for stream in streams:
        carry = None
        for lo in range(0, stream.shape[0], _CHUNK):
            chunk = stream[lo : lo + _CHUNK]
            tokens, blocked, carry = _lookback(chunk, layout.max_ngram_size, carry)
            compressed = np.where(blocked, pad_row, token_map[tokens])
            products = compressed[:, None, :] * multipliers[None, :, :]
            rolling = products[:, :, 0]
            for j in range(n_ngram):
                rolling = np.bitwise_xor(rolling, products[:, :, j + 1])
                for layer in range(n_layers):
                    parts[layer][j].append(rolling[:, layer].copy())
            total += chunk.shape[0]
    joined = [
        [np.concatenate(parts[layer][j]) for j in range(n_ngram)]
        for layer in range(n_layers)
    ]
    return joined, total, compressed_vocab


def count_exact(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact (value, count) by sorting; the group space is too large to bin."""
    values = np.sort(values)
    if values.shape[0] == 0:
        return values, np.zeros(0, dtype=np.int64)
    edges = np.flatnonzero(np.diff(values)) + 1
    starts = np.concatenate([[0], edges])
    counts = np.diff(np.concatenate([starts, [values.shape[0]]]))
    return values[starts], counts


def plan_for_layer(
    groups_by_ngram: list[np.ndarray],
    primes_layer: np.ndarray,
    offsets_layer: np.ndarray,
    n_heads: int,
    max_groups: int,
) -> tuple[np.ndarray, dict]:
    """Frequency-ordered member row ids, [groups, n_heads].

    Every group costs the same `n_heads` rows, so ordering the three n-gram
    sizes together by count is exactly the allocation that covers the most
    lookups for a given budget.
    """
    vals, cnts, sizes = [], [], []
    for j, stream in enumerate(groups_by_ngram):
        v, c = count_exact(stream)
        vals.append(v)
        cnts.append(c)
        sizes.append(np.full(v.shape[0], j, dtype=np.int8))
    vals, cnts, sizes = (
        np.concatenate(vals),
        np.concatenate(cnts),
        np.concatenate(sizes),
    )
    lookups = int(cnts.sum())
    keep = min(max_groups, vals.shape[0])
    if keep == 0:
        return np.zeros((0, n_heads), dtype=np.int64), {"lookups": lookups}
    top = np.argpartition(cnts, -keep)[-keep:]
    top = top[np.argsort(cnts[top])[::-1]]
    rolling, size, count = vals[top], sizes[top], cnts[top]
    members = np.empty((keep, n_heads), dtype=np.int64)
    for j in range(len(groups_by_ngram)):
        sel = size == j
        if not sel.any():
            continue
        members[sel] = (
            rolling[sel][:, None] % primes_layer[j][None, :]
            + offsets_layer[j * n_heads : (j + 1) * n_heads][None, :]
        )
    return members, {
        "lookups": lookups,
        "distinct_groups": int(vals.shape[0]),
        "groups": keep,
        "covered": int(count.sum()),
        "hit_rate": float(count.sum()) / lookups if lookups else 0.0,
        "per_ngram_groups": {
            str(j): int((size == j).sum()) for j in range(len(groups_by_ngram))
        },
    }


def hit_rate_curve(groups_by_ngram: list[np.ndarray], group_bytes: int) -> list[dict]:
    counts = np.concatenate([count_exact(s)[1] for s in groups_by_ngram])
    order = np.sort(counts)[::-1].astype(np.int64)
    total = int(order.sum())
    cum = np.cumsum(order)
    out, k = [], 1 << 14
    while k < order.shape[0]:
        out.append(
            {
                "groups": int(k),
                "gib": k * group_bytes / 2**30,
                "hit_rate": float(cum[k - 1]) / total,
            }
        )
        k <<= 1
    out.append(
        {
            "groups": int(order.shape[0]),
            "gib": order.shape[0] * group_bytes / 2**30,
            "hit_rate": 1.0,
        }
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Calibrate the DeepSeek-V4.1 engram HBM cache hot set."
    )
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="Plan .npz to write.")
    ap.add_argument("--max-gib", type=float, default=32.0)
    args = ap.parse_args()

    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

    config = ModelConfig(args.model_path, trust_remote_code=True).hf_text_config
    tokenizer = get_tokenizer(args.model_path)
    layout, primes, offsets, _ = _layout_tables(
        config, config.engram_compressed_vocab_size
    )
    n_heads = layout.n_heads
    group_bytes = n_heads * (layout.head_dim + layout.head_dim // FP8_BLOCK_SIZE)
    max_groups = int(args.max_gib * 2**30) // group_bytes

    def streams():
        for path in args.corpus:
            text = Path(path).read_text(errors="replace")
            yield np.asarray(
                tokenizer(text, add_special_tokens=False)["input_ids"],
                dtype=np.int64,
            )

    grouped, tokens, _ = rolling_groups(streams(), config, tokenizer)

    arrays, stats, curves = {}, {}, {}
    for idx, layer_id in enumerate(layout.layer_ids):
        members, st = plan_for_layer(
            grouped[idx], primes[idx], offsets[idx], n_heads, max_groups
        )
        arrays[f"layer_{layer_id}"] = members
        stats[f"layer_{layer_id}"] = st
        curves[f"layer_{layer_id}"] = hit_rate_curve(grouped[idx], group_bytes)

    np.savez(args.out, **arrays)
    Path(args.out).with_suffix(".json").write_text(
        json.dumps(
            {
                "model_path": args.model_path,
                "corpus": list(args.corpus),
                "corpus_tokens": tokens,
                "n_heads": n_heads,
                "group_bytes": group_bytes,
                "max_gib": args.max_gib,
                "layers": stats,
                "curves": curves,
            },
            indent=2,
        )
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
