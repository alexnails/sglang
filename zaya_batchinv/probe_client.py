#!/usr/bin/env python3
"""Batch-invariance experiments for ZAYA1-74B. Client side, no server changes.

Subcommands, cheapest first:

  distinct   Reproduce the original measurement and extend it: distinct-output
             count, serial vs parallel, across prompts of varying decisiveness.
             Tells you whether "4 of 8" is a property of the SERVER or of the
             PROMPT.  ~2 min.

  epsilon    THE INSTRUMENT.  Score one FIXED token sequence twice -- alone,
             then inside a batch of N copies -- and diff the per-position
             logprobs.  Turns a 1-bit "did the text change" readout into a
             real number: the logit perturbation in nats, per position, over
             hundreds of positions.  Also reports the top-2 vocab gap, so you
             can predict the flip rate instead of inferring it.  ~3 min.

  dprank     Send the same prompt to each DP rank in turn, one at a time
             (batch size 1 everywhere).  If outputs differ, the variable is
             the RANK, not the batch size -- which would mean the four
             distinct outputs were four DP replicas, not four batch effects.
             ~2 min.

  sweep      Batch-size sweep with all requests pinned to one DP rank, so
             token count is varied with rank held fixed.  ~5 min.

  flips      Drive the in-server router probe (see zaya_router_probe.py):
             arm the sentinel, fire one request, disarm.  Run once serial and
             once parallel, then compare with analyze_router.py.  ~3 min.

Usage:
    python3 probe_client.py distinct --url http://127.0.0.1:30000
    python3 probe_client.py epsilon  --url ... --batch 8 --tokens 128
    python3 probe_client.py dprank   --url ... --dp-size 4
    python3 probe_client.py sweep    --url ... --sizes 1,2,4,8,16,32
    python3 probe_client.py flips    --url ... --probe-dir /data/zaya_probe/serial --batch 1
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import statistics
import sys

import requests

# Prompts ordered by how DECISIVE the continuation is. If the divergence is
# ordinary float noise being resolved by a genuinely near-tied next-token
# choice, DECISIVE collapses to 1 distinct output while AMBIGUOUS does not.
PROMPTS = {
    "ambiguous": (
        "Count from one to twenty, then name three colours, then explain "
        "why the sky is blue."
    ),
    "decisive": (
        "Repeat exactly, with no extra words: "
        "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
    ),
    "factual": (
        "The capital of France is Paris. The capital of Japan is Tokyo. "
        "The capital of Italy is"
    ),
    "arith": "2+2=4. 3+3=6. 4+4=8. 5+5=10. 6+6=",
}


def gen(url, *, text=None, input_ids=None, max_new_tokens=64, logprob=False,
        start_len=-1, top_logprobs=0, dp_rank=None, timeout=600):
    body = {
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
    }
    if input_ids is not None:
        body["input_ids"] = input_ids
    else:
        body["text"] = text
    if logprob:
        body.update(
            {
                "return_logprob": True,
                "return_text_in_logprobs": False,
                "logprob_start_len": start_len,
                "top_logprobs_num": top_logprobs,
            }
        )
    if dp_rank is not None:
        body["routed_dp_rank"] = dp_rank
    r = requests.post(url + "/generate", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def flush(url):
    requests.post(url + "/flush_cache", params={"timeout": 30}, timeout=60)


def digest(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def first_divergence(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1


# --------------------------------------------------------------------------
def cmd_distinct(args):
    url = args.url
    print(f"{'prompt':<12} {'mode':<9} {'n':>3} {'distinct':>8}  first-divergence-char")
    for name, prompt in PROMPTS.items():
        if args.prompts and name not in args.prompts.split(","):
            continue
        results = {}
        # serial
        flush(url)
        serial = []
        for _ in range(args.n):
            serial.append(gen(url, text=prompt, max_new_tokens=args.tokens)["text"])
        results["serial"] = serial
        # parallel
        flush(url)
        with cf.ThreadPoolExecutor(max_workers=args.n) as ex:
            futs = [
                ex.submit(gen, url, text=prompt, max_new_tokens=args.tokens)
                for _ in range(args.n)
            ]
            results["parallel"] = [f.result()["text"] for f in futs]

        for mode, outs in results.items():
            uniq = sorted(set(outs))
            div = (
                first_divergence(results["serial"][0], outs[0])
                if mode == "parallel"
                else -1
            )
            print(f"{name:<12} {mode:<9} {len(outs):>3} {len(uniq):>8}  {div}")
        if args.verbose:
            for mode, outs in results.items():
                for i, o in enumerate(sorted(set(outs))):
                    print(f"    [{mode} #{i}] {digest(o)} {o[:110]!r}")
    print(
        "\nREAD: if 'decisive'/'factual'/'arith' give distinct=1 in BOTH modes while\n"
        "'ambiguous' does not, the perturbation is ordinary-sized and the original\n"
        "measurement was dominated by a near-tied next-token choice on a weakly\n"
        "instruction-tuned checkpoint.  If EVERY prompt gives distinct>1 in\n"
        "parallel, the perturbation is large and something is genuinely wrong."
    )


# --------------------------------------------------------------------------
def cmd_epsilon(args):
    """Measure the logit perturbation directly, at fixed token positions.

    Method (borrowed from python/sglang/test/kl_test_utils.py): generate once
    to fix a concrete token sequence, then RE-SCORE that same sequence with
    max_new_tokens=0 and logprob_start_len=0 -- once alone, once inside a
    batch of N identical copies.  Both runs score identical tokens, so the
    per-position logprob difference is a clean measurement of the perturbation
    with no divergence contamination.
    """
    url = args.url
    prompt = PROMPTS[args.prompt]

    flush(url)
    seed = gen(url, text=prompt, max_new_tokens=args.tokens)
    ids = seed["meta_info"].get("input_ids") or []
    if not ids:
        # older builds do not echo input_ids; re-tokenize via a 0-token call
        probe = gen(url, text=prompt, max_new_tokens=0, logprob=True, start_len=0)
        ids = [t[1] for t in probe["meta_info"]["input_token_logprobs"]]
    full = list(ids) + list(seed["output_ids"])
    print(f"prompt={args.prompt}  prompt_tokens={len(ids)}  total_tokens={len(full)}")

    def score(batch_n):
        flush(url)
        if batch_n == 1:
            res = [
                gen(
                    url,
                    input_ids=full,
                    max_new_tokens=0,
                    logprob=True,
                    start_len=0,
                    top_logprobs=args.top_logprobs,
                )
            ]
        else:
            with cf.ThreadPoolExecutor(max_workers=batch_n) as ex:
                futs = [
                    ex.submit(
                        gen,
                        url,
                        input_ids=full,
                        max_new_tokens=0,
                        logprob=True,
                        start_len=0,
                        top_logprobs=args.top_logprobs,
                    )
                    for _ in range(batch_n)
                ]
                res = [f.result() for f in futs]
        return res

    alone = score(1)[0]
    batched = score(args.batch)

    lp_alone = [x[0] for x in alone["meta_info"]["input_token_logprobs"]]
    tops_alone = alone["meta_info"].get("input_top_logprobs") or []

    # top-2 vocab gap: how tied is each next-token decision?
    gaps = []
    for entry in tops_alone:
        if entry and len(entry) >= 2:
            gaps.append(abs(entry[0][0] - entry[1][0]))
    if gaps:
        gaps_sorted = sorted(gaps)
        q = lambda p: gaps_sorted[min(len(gaps_sorted) - 1, int(p * len(gaps_sorted)))]
        print(
            f"\ntop-2 vocab gap (nats) over {len(gaps)} positions: "
            f"p01={q(0.01):.4g} p05={q(0.05):.4g} p10={q(0.10):.4g} "
            f"p50={q(0.50):.4g}"
        )
        for thr in (1e-4, 1e-3, 1e-2, 1e-1):
            frac = sum(g < thr for g in gaps) / len(gaps)
            print(f"    P(gap < {thr:g}) = {frac:.4f}  -> ~1 flip every "
                  f"{(1/frac):.0f} tokens if eps={thr:g}")

    print(f"\nper-position |delta logprob| vs the alone run, batch={args.batch}:")
    all_d = []
    for i, res in enumerate(batched):
        lp = [x[0] for x in res["meta_info"]["input_token_logprobs"]]
        n = min(len(lp), len(lp_alone))
        d = [
            abs(lp[k] - lp_alone[k])
            for k in range(n)
            if lp[k] is not None and lp_alone[k] is not None
        ]
        if not d:
            continue
        all_d.extend(d)
        nz = sum(x > 0 for x in d)
        print(
            f"  copy {i}: n={len(d)} nonzero={nz} ({nz/len(d):.1%}) "
            f"mean={statistics.fmean(d):.3g} p99={sorted(d)[int(0.99*len(d))]:.3g} "
            f"max={max(d):.3g}"
        )
    if all_d:
        nz = sum(x > 0 for x in all_d)
        print(
            f"\nEPSILON (the number this whole investigation needs): "
            f"nonzero {nz/len(all_d):.1%} of positions, "
            f"mean |dlogprob| = {statistics.fmean(all_d):.4g} nats, "
            f"max = {max(all_d):.4g} nats"
        )
        print(
            "READ: compare EPSILON against the top-2 gap percentiles above.\n"
            "  eps ~1e-4..1e-3  -> ordinary bf16 batch noise; greedy-id equality was\n"
            "                     always a coin flip and the gate was never sound.\n"
            "  eps >~1e-1       -> NOT float noise.  Something is routing tokens to\n"
            "                     different experts wholesale; chase it."
        )
    # bit-identity check among the batched copies themselves
    texts = {digest(json.dumps([x[0] for x in r['meta_info']['input_token_logprobs']]))
             for r in batched}
    print(f"distinct logprob vectors among the {args.batch} batched copies: {len(texts)}")


# --------------------------------------------------------------------------
def cmd_dprank(args):
    url = args.url
    prompt = PROMPTS[args.prompt]
    print("Same prompt, batch size 1 everywhere, one DP rank at a time.")
    outs = {}
    for rank in range(args.dp_size):
        flush(url)
        o = gen(url, text=prompt, max_new_tokens=args.tokens, dp_rank=rank)["text"]
        outs[rank] = o
        print(f"  rank {rank}: {digest(o)} {o[:90]!r}")
    print(f"\ndistinct across {args.dp_size} DP ranks at batch size 1: "
          f"{len(set(outs.values()))}")
    print(
        "READ: >1 here means the four distinct outputs were four DP REPLICAS, not a\n"
        "batch-size effect at all -- a much narrower and more suspicious finding,\n"
        "since replicas run identical code on identical weights.  ==1 means DP rank\n"
        "is innocent and token count is the variable."
    )


# --------------------------------------------------------------------------
def cmd_sweep(args):
    url = args.url
    prompt = PROMPTS[args.prompt]
    sizes = [int(x) for x in args.sizes.split(",")]
    ref = None
    print(f"{'batch':>6} {'pin':>5} {'distinct':>8}  {'matches-bs1':>11}")
    for pin in ([args.dp_rank] if args.dp_rank is not None else [None]):
        for n in sizes:
            flush(url)
            if n == 1:
                outs = [gen(url, text=prompt, max_new_tokens=args.tokens,
                            dp_rank=pin)["text"]]
            else:
                with cf.ThreadPoolExecutor(max_workers=n) as ex:
                    futs = [
                        ex.submit(gen, url, text=prompt,
                                  max_new_tokens=args.tokens, dp_rank=pin)
                        for _ in range(n)
                    ]
                    outs = [f.result()["text"] for f in futs]
            if ref is None:
                ref = outs[0]
            m = sum(o == ref for o in outs)
            print(f"{n:>6} {str(pin):>5} {len(set(outs)):>8}  {m}/{len(outs):<11}")
    print(
        "\nREAD: with pin set, DP rank is constant and only the token count moves.\n"
        "A step change at a particular batch size points at a size-keyed kernel or\n"
        "collective threshold (see the ranked list in the report)."
    )


# --------------------------------------------------------------------------
def cmd_flips(args):
    """Arm the in-server router probe around one request."""
    os.makedirs(args.probe_dir, exist_ok=True)
    sentinel = os.path.join(args.probe_dir, "ON")
    prompt = PROMPTS[args.prompt]
    flush(args.url)
    open(sentinel, "w").close()
    try:
        if args.batch == 1:
            gen(args.url, text=prompt, max_new_tokens=args.tokens)
        else:
            with cf.ThreadPoolExecutor(max_workers=args.batch) as ex:
                futs = [
                    ex.submit(gen, args.url, text=prompt, max_new_tokens=args.tokens)
                    for _ in range(args.batch)
                ]
                [f.result() for f in futs]
    finally:
        os.remove(sentinel)
    print(f"recorded into {args.probe_dir}; now run analyze_router.py")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:30000")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("distinct")
    d.add_argument("--n", type=int, default=8)
    d.add_argument("--tokens", type=int, default=64)
    d.add_argument("--prompts", default=None, help="comma list; default all")
    d.add_argument("--verbose", action="store_true")
    d.set_defaults(fn=cmd_distinct)

    e = sub.add_parser("epsilon")
    e.add_argument("--batch", type=int, default=8)
    e.add_argument("--tokens", type=int, default=128)
    e.add_argument("--prompt", default="ambiguous", choices=list(PROMPTS))
    e.add_argument("--top-logprobs", type=int, default=4)
    e.set_defaults(fn=cmd_epsilon)

    r = sub.add_parser("dprank")
    r.add_argument("--dp-size", type=int, default=4)
    r.add_argument("--tokens", type=int, default=64)
    r.add_argument("--prompt", default="ambiguous", choices=list(PROMPTS))
    r.set_defaults(fn=cmd_dprank)

    s = sub.add_parser("sweep")
    s.add_argument("--sizes", default="1,2,4,8,16,32")
    s.add_argument("--tokens", type=int, default=48)
    s.add_argument("--dp-rank", type=int, default=0)
    s.add_argument("--prompt", default="ambiguous", choices=list(PROMPTS))
    s.set_defaults(fn=cmd_sweep)

    f = sub.add_parser("flips")
    f.add_argument("--probe-dir", required=True)
    f.add_argument("--batch", type=int, default=1)
    f.add_argument("--tokens", type=int, default=1)
    f.add_argument("--prompt", default="ambiguous", choices=list(PROMPTS))
    f.set_defaults(fn=cmd_flips)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
