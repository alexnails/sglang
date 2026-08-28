#!/usr/bin/env python3
"""Analyse zaya_router_probe dumps: flip margin distribution + direct flip count.

Two questions, both answered from the dumps:

  1. HOW TIED IS THE TOP-1 ROUTER?  Pool the biased-score top1-top2 gap over
     every (layer, token) and report the distribution.  This is the number the
     whole "is top-1 routing the dominant term" argument rests on, and it is
     measurable from ONE forward pass -- no A/B needed.

  2. DID TOKENS ACTUALLY FLIP EXPERTS?  Diff the recorded expert ids between a
     serial pass (T = P rows) and a parallel pass (T = B*P rows, reshaped to
     [B, P]).  This is the direct measurement that
     `--enable-return-routed-experts` cannot give you for ZAYA1.

Usage:
    # gap distribution only, from a single run
    python3 analyze_router.py gap /data/zaya_probe/serial

    # flip count, serial vs parallel
    python3 analyze_router.py flips /data/zaya_probe/serial \
                                    /data/zaya_probe/parallel --batch 8

Both accept --rank (default 0) and --pass-idx (default 0: the FIRST recorded
pass, which is the prefill).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np


def load(dirpath, rank, pass_idx):
    pat = os.path.join(dirpath, f"rank{rank}_pass{pass_idx:03d}.npz")
    hits = glob.glob(pat)
    if not hits:
        avail = sorted(os.path.basename(p) for p in glob.glob(
            os.path.join(dirpath, "rank*_pass*.npz")))
        sys.exit(f"no {pat}\navailable: {avail}")
    return np.load(hits[0])


def layers_of(z, key):
    out = {}
    for name in z.files:
        if name.startswith(key + "/l"):
            out[int(name.split("/l")[1])] = z[name]
    return dict(sorted(out.items()))


def cmd_gap(args):
    z = load(args.dir, args.rank, args.pass_idx)
    gaps = layers_of(z, "gap")
    if not gaps:
        sys.exit("no gap arrays in dump")
    pooled = np.concatenate([v.ravel() for v in gaps.values()])
    print(f"{len(gaps)} MoE layers, {len(pooled)} (layer,token) routing decisions")
    qs = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50]
    print("\nbiased-score top1-top2 gap percentiles:")
    for q in qs:
        print(f"  p{q*100:>5.1f} = {np.quantile(pooled, q):.3e}")

    print("\nP(gap < eps)  ->  expected flips per token per 60-layer forward:")
    for eps in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        p = float((pooled < eps).mean())
        print(f"  eps={eps:.0e}: P={p:.3e}   E[flips/token] = {p*len(gaps):.4f}   "
              f"P(>=1 flip) = {1-(1-p)**len(gaps):.4f}")
    print(
        "\nREAD: find the row whose 'P(>=1 flip)' is near 1.  That eps is the\n"
        "smallest router-score perturbation that would explain a divergence on\n"
        "essentially every token.  Compare it against the measured EPSILON from\n"
        "`probe_client.py epsilon` -- but note EPSILON is in logprob nats and\n"
        "this is in probability units, so convert: d(prob) ~ p * d(logit), with\n"
        "p ~ 1/24 = 0.042 for a diffuse router."
    )

    per_layer = np.array([np.quantile(v, 0.01) for v in gaps.values()])
    worst = np.argsort(per_layer)[:8]
    ids = list(gaps.keys())
    print("\n8 tightest layers by p1 gap (most flip-prone):")
    for i in worst:
        print(f"  layer {ids[i]:>3}: p1 gap = {per_layer[i]:.3e}")

    ids_arr = layers_of(z, "ids")
    if ids_arr:
        allids = np.concatenate([v.ravel() for v in ids_arr.values()])
        counts = np.bincount(allids.astype(np.int64), minlength=25)
        nz = counts[counts > 0]
        print(f"\nexpert load: {len(nz)} experts used, "
              f"min={counts[counts>0].min()} max={counts.max()} "
              f"(perfect balance would be {len(allids)//max(1,len(nz))})")
        if counts[24] if len(counts) > 24 else 0:
            print(f"  WARNING: {counts[24]} tokens chose the MOD skip slot (id 24), "
                  "which fold_mod_reachability claims is unreachable")


def cmd_flips(args):
    za = load(args.a, args.rank, args.pass_idx)
    zb = load(args.b, args.rank, args.pass_idx)
    ia, ib = layers_of(za, "ids"), layers_of(zb, "ids")
    common = sorted(set(ia) & set(ib))
    if not common:
        sys.exit("no common layers")

    ta, tb = len(ia[common[0]]), len(ib[common[0]])
    print(f"serial rows T={ta}, parallel rows T={tb}, layers={len(common)}")
    if tb % ta != 0:
        sys.exit(
            f"parallel rows {tb} is not a multiple of serial rows {ta}.\n"
            "The two passes did not see the same per-request token set (chunked\n"
            "prefill, DP padding, or the prefill split across steps). Re-run with\n"
            "--tokens 1, --disable-radix-cache, and a prompt short enough to\n"
            "prefill in one step, or lower --batch."
        )
    b = tb // ta
    print(f"inferred {b} copies per parallel pass")

    total = flips = 0
    per_layer = {}
    flip_gaps, keep_gaps = [], []
    ga, gb = layers_of(za, "gap"), layers_of(zb, "gap")
    first_logit_diff = None
    la, lb = layers_of(za, "logits"), layers_of(zb, "logits")

    for lid in common:
        a = ia[lid].astype(np.int32)
        bb = ib[lid].astype(np.int32).reshape(b, ta)
        eq = bb == a[None, :]
        n = eq.size
        f = int((~eq).sum())
        total += n
        flips += f
        if f:
            per_layer[lid] = f
        if lid in ga:
            g = np.repeat(ga[lid][None, :], b, axis=0)
            flip_gaps.append(g[~eq])
            keep_gaps.append(g[eq])
        if first_logit_diff is None and lid in la and lid in lb:
            x = la[lid]
            y = lb[lid].reshape(b, *x.shape)
            if not np.array_equal(np.repeat(x[None], b, 0), y):
                first_logit_diff = lid

    print(f"\nEXPERT FLIPS: {flips} / {total} routing decisions "
          f"({flips/total:.3%})")
    print(f"  -> E[flips per token per forward] = {flips/total*len(common):.3f}")
    print(f"  -> P(>=1 flip per token) = {1-(1-flips/total)**len(common):.4f}")
    if first_logit_diff is not None:
        print(f"\nFIRST MoE layer whose router LOGITS differ at all: {first_logit_diff}")
        print("  (logits differing but ids matching = the perturbation is present but"
              " has not yet flipped an argmax)")
    else:
        print("\nrouter logits are BIT-IDENTICAL in every layer -> the two passes are"
              " numerically identical; nothing to explain in the router.")

    if per_layer:
        print("\nflips by layer (first 12 with any):")
        for lid, f in list(per_layer.items())[:12]:
            print(f"  layer {lid:>3}: {f}")

    if flip_gaps:
        fg = np.concatenate(flip_gaps)
        kg = np.concatenate(keep_gaps)
        if len(fg):
            print(f"\ngap AT flipped decisions:   median={np.median(fg):.3e} "
                  f"p90={np.quantile(fg,0.9):.3e} max={fg.max():.3e}")
        print(f"gap at UNflipped decisions: median={np.median(kg):.3e} "
              f"p01={np.quantile(kg,0.01):.3e}")
        if len(fg):
            print(
                "\nREAD: if the flipped decisions all sit in the extreme low tail of\n"
                "the gap distribution, top-1 argmax is behaving exactly as the theory\n"
                "predicts -- ordinary noise resolving genuine near-ties.  If flips\n"
                "occur at LARGE gaps, the perturbation is not float noise and you\n"
                "have a real bug: look at the first-differing-logits layer above."
            )

    hsa = sorted(k for k in za.files if k.startswith("hs/"))
    hsb = sorted(k for k in zb.files if k.startswith("hs/"))
    if hsa and hsb:
        for k in hsa:
            if k not in zb.files:
                continue
            x, y = za[k], zb[k].reshape(b, *za[k].shape)
            if not np.array_equal(np.repeat(x[None], b, 0), y):
                print(f"\nFIRST layer-entry norm whose output differs: {k} "
                      f"(call index {int(k.split('/c')[1])} of {len(hsa)}; "
                      "even index = attention layer, odd = MoE layer)")
                break
        else:
            print("\nall layer-entry norm fingerprints identical")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--pass-idx", type=int, default=0)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gap")
    g.add_argument("dir")
    g.set_defaults(fn=cmd_gap)
    f = sub.add_parser("flips")
    f.add_argument("a")
    f.add_argument("b")
    f.set_defaults(fn=cmd_flips)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
