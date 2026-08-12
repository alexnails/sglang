# Copyright 2023-2024 SGLang Team
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
"""Reproduce every issue found while auditing --tokenizer-backend=gigatoken.

Seven numbered cases. 1-2 are startup failures a user can hit; 3-5 are the
divergences that justify how the integration is built (each is already handled,
and the script shows both the raw failure and the handled behavior); 6-7 are
claim/performance corrections.

    python scripts/playground/gigatoken_repro_issues.py

Requires `pip install 'sglang[gigatoken]'` and either network access or a warm
HuggingFace cache for DeepSeek-V3, bge-reranker-v2-m3, llama-tokenizer and
Qwen2.5-1.5B-Instruct. Takes ~40s; case 6 is a benchmark, so run it on an idle
machine if you care about the numbers.

Three environment traps, none of them bugs in the code under test:
  * On macOS without triton, every sglang import dies in
    `torch/_inductor/runtime/triton_heuristics.py` because a few modules apply
    `@torch.compile` at module scope. Run under a launcher that no-ops
    `torch.compile` before importing sglang.
  * With HF_HUB_OFFLINE=1 and a partially cached repo, transformers silently
    returns a `vocab_size=1` tokenizer, and gigatoken then fails with the
    misleading `no single-byte vocab entry for byte 0x00`. Warm the cache first.
  * Case 3 needs network even with a warm cache: `hf-internal-testing/
    llama-tokenizer` ships no config.json, so the lookup cannot be satisfied
    offline.

Expected output, gigatoken 0.10.0 / transformers 5.12.1 / M4 Max:
  1 REPRODUCED  DeepSeek-V3 raises at startup (sglang's loader only)
  2 REPRODUCED  bge-reranker-v2-m3 raises at startup (Unigram)
  3 REPRODUCED  14/66 partial-decode windows differ; gate turns decode off
  4 REPRODUCED  deepcopy of gigatoken's own shim raises
  5 REPRODUCED  xgrammar and apply_chat_template both fail on the shim
  6 ~48-56x delivered vs 185-235x raw API
  7 ~26% of the fast path wasted on an empty-affix double list copy
"""

import copy
import time
import warnings

warnings.filterwarnings("ignore")

import gigatoken
from transformers import AutoTokenizer

from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

FAIL, OK = "REPRODUCED", "not reproduced"


def head(n, title):
    print(f"\n{'='*76}\n#{n}  {title}\n{'='*76}")


def bench(fn, n=100):
    fn()
    start = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - start) / n * 1e6


def case_1_deepseek_startup_failure():
    """sglang resolves DeepSeek-V3 to a tokenizer gigatoken refuses to load."""
    head(1, "DeepSeek-V3 cannot start with --tokenizer-backend=gigatoken")
    try:
        get_tokenizer("deepseek-ai/DeepSeek-V3", tokenizer_backend="gigatoken")
        print(f"  {OK}: it loaded")
    except RuntimeError as e:
        print(f"  {FAIL}: RuntimeError at startup")
        print(f"    cause: {str(e.__cause__).splitlines()[0]}")

    sglang_tok = get_tokenizer("deepseek-ai/DeepSeek-V3")
    auto_tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
    print(
        f"  sglang loads {type(sglang_tok).__name__}, "
        f"AutoTokenizer loads {type(auto_tok).__name__}"
    )
    print("    gigatoken on AutoTokenizer's object: ", end="")
    try:
        gigatoken.Tokenizer(auto_tok)
        print("loads fine  <-- the failure is specific to sglang's loader")
    except Exception as e:  # noqa: BLE001 - reporting whatever it raises
        print(f"also fails: {e}")
    # Not a bug: _fix_v5_add_bos_eos_token restores add_bos_token for
    # LlamaTokenizer classes, which is why the two objects differ at all.
    print(
        f"  the two tokenizers differ: sglang={sglang_tok.encode('Hello world')} "
        f"auto={auto_tok.encode('Hello world')}  (_fix_v5_add_bos_eos_token, intended)"
    )


def case_2_unigram_unsupported():
    head(2, "bge-reranker-v2-m3 cannot start either (Unigram unsupported)")
    try:
        get_tokenizer("BAAI/bge-reranker-v2-m3", tokenizer_backend="gigatoken")
        print(f"  {OK}")
    except RuntimeError as e:
        print(f"  {FAIL}: {str(e.__cause__).splitlines()[0]}")


def case_3_byte_fallback_decode():
    """One U+FFFD per undecodable byte (HF) vs one per truncated run (gigatoken)."""
    head(3, "Byte-fallback partial-decode divergence (undocumented upstream)")
    tokenizer = get_tokenizer("hf-internal-testing/llama-tokenizer")
    backend = gigatoken.Tokenizer(tokenizer)
    ids = tokenizer.encode("ok 日本語 🚀 end", add_special_tokens=False)

    differing = []
    for read in range(1, len(ids) + 1):
        for surr in range(read):
            window = ids[surr:read]
            expected = tokenizer.decode(window)
            got = backend.decode(window).decode("utf-8", "replace")
            if expected != got:
                differing.append((surr, read, expected, got))

    total = len(ids) * (len(ids) + 1) // 2
    print(f"  {FAIL if differing else OK}: {len(differing)}/{total} windows differ")
    for surr, read, expected, got in differing[:3]:
        print(f"    [{surr}:{read}]  hf={expected!r}  gigatoken={got!r}")
    same_text = all(
        e.replace("�", "") == g.replace("�", "") for _, _, e, g in differing
    )
    print(f"    U+FFFD count only, same real text: {same_text}")

    accelerated = get_tokenizer(
        "hf-internal-testing/llama-tokenizer", tokenizer_backend="gigatoken"
    )
    print(
        "  handled: the load-time probe turns decode off for this tokenizer -> "
        f"_gigatoken_decode_ok={accelerated._gigatoken_decode_ok}"
    )


def case_4_deepcopy():
    """gigatoken defines no __getstate__/__reduce__/__deepcopy__ anywhere."""
    head(4, "gigatoken objects are unpicklable -> deepcopy breaks (library-wide)")
    hf = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    try:
        copy.deepcopy(gigatoken.Tokenizer(hf).as_hf())
        print(f"  {OK}")
    except TypeError as e:
        print(f"  {FAIL} on gigatoken's own as_hf() shim: {e}")

    accelerated = get_tokenizer(
        "Qwen/Qwen2.5-1.5B-Instruct", tokenizer_backend="gigatoken"
    )
    copy.deepcopy(accelerated)
    print("  handled: __deepcopy__ shares the backend -> deepcopy OK")


def case_5_whole_object_swap():
    """Why the integration patches methods instead of replacing the object."""
    head(5, "Whole-object swap would break xgrammar + chat templating")
    import xgrammar

    hf = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    shim = gigatoken.Tokenizer(hf).as_hf()
    messages = [{"role": "user", "content": "hi"}]

    try:
        xgrammar.TokenizerInfo.from_huggingface(shim, vocab_size=hf.vocab_size)
        print(f"  {OK}: xgrammar accepted the shim")
    except Exception as e:  # noqa: BLE001 - reporting whatever it raises
        print(f"  {FAIL}: xgrammar on shim -> {type(e).__name__}: {str(e)[:60]}")
    try:
        shim.apply_chat_template(messages, tokenize=False)
        print(f"  {OK}: shim has apply_chat_template")
    except AttributeError as e:
        print(f"  {FAIL}: chat template on shim -> AttributeError: {e}")

    accelerated = get_tokenizer(
        "Qwen/Qwen2.5-1.5B-Instruct", tokenizer_backend="gigatoken"
    )
    xgrammar.TokenizerInfo.from_huggingface(accelerated, vocab_size=hf.vocab_size)
    accelerated.apply_chat_template(messages, tokenize=False)
    print("  handled: patched tokenizer keeps both -> xgrammar OK, chat template OK")


def case_6_and_7_speed():
    """Delivered speedup, and the empty-affix copy inside it."""
    base = get_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")
    accelerated = get_tokenizer(
        "Qwen/Qwen2.5-1.5B-Instruct", tokenizer_backend="gigatoken"
    )
    raw = gigatoken.Tokenizer(base)
    doc = "The quick brown fox jumps over the lazy dog. " * 2800

    head(6, "Delivered speedup is ~50x, not the ~200x of the raw numpy API")
    hf_us = bench(lambda: base.encode(doc))
    patched_us = bench(lambda: accelerated.encode(doc))
    raw_us = bench(lambda: raw.encode(doc))
    print(
        f"  {len(base.encode(doc))} tokens: transformers {hf_us:.0f}us | "
        f"patched {patched_us:.0f}us ({hf_us/patched_us:.0f}x) | "
        f"raw API {raw_us:.0f}us ({hf_us/raw_us:.0f}x)"
    )
    print(
        "  the gap is .tolist(): sglang needs Python lists, the raw API returns numpy"
    )

    head(7, "~25% of the fast path is a wasted double list copy (empty affixes)")
    ids = raw.encode(doc)
    prefix = suffix = []
    tolist_us = bench(lambda: ids.tolist())
    concat_us = bench(lambda: prefix + ids.tolist() + suffix)
    print(
        f"  affixes here: prefix={accelerated._gigatoken_prefix_ids} "
        f"suffix={accelerated._gigatoken_suffix_ids}"
    )
    print(
        f"  tolist() {tolist_us:.0f}us -> '[] + tolist() + []' {concat_us:.0f}us = "
        f"{concat_us-tolist_us:.0f}us wasted "
        f"({(concat_us-tolist_us)/patched_us*100:.0f}% of the {patched_us:.0f}us fast path)"
    )


if __name__ == "__main__":
    case_1_deepseek_startup_failure()
    case_2_unigram_unsupported()
    case_3_byte_fallback_decode()
    case_4_deepcopy()
    case_5_whole_object_swap()
    case_6_and_7_speed()
