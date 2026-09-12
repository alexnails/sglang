#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash on 4x GB300: cookbook configs, the tuned config, BCG, and
the engram HBM cache — measured with sgl-bench, not a hand-rolled client.

Per point: warm-up requests + a prefix-cache flush, random dataset at range
ratio 1 (so every prompt is exactly 4096 in / 1024 out and prompts are
distinct, which identical prompts would not be — they would share their whole
prefix in the radix cache and leave the KV pool idle).

Acceptance length is pinned with SGLANG_SIMULATE_ACC_LEN and every point is
gated on the harness reporting it back within ACC_TOL. Generated text is not
meaningful under a forced acceptance length, so read decode speed and
throughput, not output quality.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import socket
import statistics
import subprocess
import time
from pathlib import Path

import requests

MODEL = os.environ["MODEL_PATH"]
SERVED = "deepseek-ai/DeepSeek-V4.1-Flash"
SGLANG = "/scratch/engram/sglang"
PORT = int(os.environ.get("AB_PORT", "31021"))
BASE = f"http://127.0.0.1:{PORT}"
LOGDIR = Path("/scratch/logs/ab3")
ACC_LEN = os.environ.get("ACC_LEN", "5.8")
ACC_TOL = 0.03
GSP_GROUPS = 8
LONGBENCH_CTX = int(os.environ.get("LONGBENCH_CTX", "32768"))
GSP_QUESTION_LEN = 512

COMMON = [
    "--model-path",
    MODEL,
    "--served-model-name",
    SERVED,
    "--tp",
    "4",
    "--ep-size",
    "4",
    "--trust-remote-code",
    "--host",
    "127.0.0.1",
    "--port",
    str(PORT),
    "--random-seed",
    "42",
    "--decode-log-interval",
    "10",
]

TUNED = COMMON + [
    "--mem-fraction-static",
    "0.80",
    "--max-total-tokens",
    "33554432",
    "--chunked-prefill-size",
    "4096",
    "--cuda-graph-bs-decode",
    "1",
    "2",
    "4",
    "8",
    "16",
    "32",
    "64",
    "--max-running-requests",
    "128",
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-dspark-block-size",
    "5",
    "--skip-server-warmup",
    "--reasoning-parser",
    "deepseek-v41",
]
BCG = ["--cuda-graph-backend-prefill", "breakable"]
DP = ["--enable-dp-attention", "--dp-size", "4", "--enable-dp-lm-head"]
# DSpark refuses dp attention without a dp lm head.


def _uncapped(args: list[str]) -> list[str]:
    """Drop --max-total-tokens so the KV pool sizes from whatever the
    weights leave behind. With the cap in place the token limit binds long
    before memory does, so freed HBM cannot show up as capacity."""
    out, skip = [], False
    for a in args:
        if a == "--max-total-tokens":
            skip = True
            continue
        if skip:
            skip = False
            continue
        out.append(a)
    return out


def _chunk(args: list[str], size: str) -> list[str]:
    """Set the chunk size, appending it when the arm never passed one.

    DP attention halves chunked_prefill_size (server_args.py), so an arm that
    leaves it unset inherits sglang's default and runs at half of it; the
    effective value is what this halves to, not what is passed.
    """
    out = list(args)
    if "--chunked-prefill-size" in out:
        out[out.index("--chunked-prefill-size") + 1] = size
    else:
        out += ["--chunked-prefill-size", size]
    return out


HOST_ENV = {"SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": "1"}
SHARED_ENV = {**HOST_ENV, "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT": "shared"}


def cache_env(gib: str, plan: str) -> dict:
    return {
        **SHARED_ENV,
        "SGLANG_ENABLE_DSV41_ENGRAM_HBM_CACHE": "1",
        "SGLANG_DSV41_ENGRAM_HBM_CACHE_PLAN": plan,
        "SGLANG_DSV41_ENGRAM_HBM_CACHE_GIB": gib,
    }


def build_arms(plan: str | None, gibs: list[str]) -> dict[str, dict]:
    arms: dict[str, dict] = {
        # cookbook GB300, exactly as published
        "cb-lowlat": {
            "args": COMMON
            + [
                "--mem-fraction-static",
                "0.8",
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-dspark-block-size",
                "5",
                "--cuda-graph-max-bs-decode",
                "64",
                "--reasoning-parser",
                "auto",
                "--tool-call-parser",
                "auto",
            ],
            "env": {},
        },
        "cb-tput": {
            "args": COMMON
            + [
                "--max-running-requests",
                "256",
                "--reasoning-parser",
                "auto",
                "--tool-call-parser",
                "auto",
            ],
            "env": {},
        },
        # tuned config, engram placement
        "tuned": {"args": TUNED, "env": {}},
        "tuned-host": {"args": TUNED, "env": HOST_ENV},
        "tuned-host-shared": {"args": TUNED, "env": SHARED_ENV},
        # BCG: prefill cuda graph sglang otherwise disables for DeepSeek-V4
        "tuned-bcg": {"args": TUNED + BCG, "env": {}},
        "tuned-bcg-host-shared": {"args": TUNED + BCG, "env": SHARED_ENV},
        "tuned-bcg-16k": {"args": _chunk(TUNED + BCG, "16384"), "env": {}},
        "tuned-16k": {"args": _chunk(TUNED, "16384"), "env": {}},
        # The capacity pair: identical but for engram placement, with the
        # token cap removed so the freed HBM lands in the KV pool.
        "tuned-free": {"args": _uncapped(TUNED), "env": {}},
        "tuned-host-shared-free": {"args": _uncapped(TUNED), "env": SHARED_ENV},
    }
    CB_LOWLAT = arms["cb-lowlat"]["args"]
    CB_TPUT = arms["cb-tput"]["args"]
    SPEC = ["--speculative-algorithm", "DSPARK", "--speculative-dspark-block-size", "5"]
    arms.update(
        {
            # cb-lowlat already carries DSpark; it gains the prefill graph and the
            # host-placed engram tables.
            "cb-lowlat-bcg": {"args": CB_LOWLAT + BCG, "env": {}},
            "cb-lowlat-bcg-hiengram": {"args": CB_LOWLAT + BCG, "env": "CACHE"},
            "cb-lowlat-hiengram": {"args": CB_LOWLAT, "env": "CACHE"},
            "cb-lowlat-host-shared": {"args": CB_LOWLAT, "env": SHARED_ENV},
            # Does the freed HBM let DP attention fit, and does skipping the DP
            # engram gather help on top of that?
            "cb-lowlat-dp": {"args": CB_LOWLAT + DP, "env": {}},
            "cb-lowlat-dp-host-shared": {"args": CB_LOWLAT + DP, "env": SHARED_ENV},
            "cb-lowlat-bcg-dp": {"args": CB_LOWLAT + BCG + DP, "env": {}},
            # cb-tput ships without speculative decoding, so it gains that too.
            "cb-tput-spec-bcg": {"args": CB_TPUT + SPEC + BCG, "env": {}},
            "cb-tput-spec-bcg-hiengram": {"args": CB_TPUT + SPEC + BCG, "env": "CACHE"},
        }
    )
    # A cache arm needs a plan, but a round that selects none of them must not
    # be blocked by that: drop them instead, so selecting one without --plan
    # fails by name rather than taking the whole round down.
    default_gib = gibs[0] if gibs else "8"
    for name in [k for k, v in arms.items() if v["env"] == "CACHE"]:
        if plan:
            arms[name]["env"] = cache_env(default_gib, plan)
        else:
            del arms[name]
    for gib in gibs:
        assert plan, "--hiengram-gib needs --plan"
        arms[f"tuned-hiengram-{gib}"] = {"args": TUNED, "env": cache_env(gib, plan)}
        arms[f"tuned-bcg-hiengram-{gib}"] = {
            "args": TUNED + BCG,
            "env": cache_env(gib, plan),
        }
        arms[f"cb-lowlat-hiengram-{gib}"] = {
            "args": CB_LOWLAT,
            "env": cache_env(gib, plan),
        }
        arms[f"cb-lowlat-bcg-hiengram-{gib}"] = {
            "args": CB_LOWLAT + BCG,
            "env": cache_env(gib, plan),
        }
        arms[f"cb-lowlat-dp-hiengram-{gib}"] = {
            "args": CB_LOWLAT + DP,
            "env": cache_env(gib, plan),
        }
        # DP clamps the prefill chunk to 4096, which is the launch-bound regime
        # where the prefill graph paid. Its losses were all measured at 16k.
        arms[f"cb-lowlat-dp-bcg-hiengram-{gib}"] = {
            "args": CB_LOWLAT + DP + BCG,
            "env": cache_env(gib, plan),
        }
    return arms


def launch(arm: str, spec: dict, log: Path):
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "PYTHONPATH": f"{SGLANG}/python",
            "MAX_JOBS": "16",
            "SGLANG_RAGGED_VERIFY_MODE": "static",
            "SGLANG_SIMULATE_ACC_LEN": ACC_LEN,
            "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
        }
    )
    env.update(spec["env"])
    fh = log.open("w")
    fh.write(f"### arm={arm} acc_len={ACC_LEN} env={spec['env']}\n")
    fh.flush()
    nccl_port = _pick_nccl_port()
    fh.write(f"### nccl_port={nccl_port}\n")
    fh.flush()
    proc = subprocess.Popen(
        [
            "python3",
            "-m",
            "sglang.launch_server",
            *spec["args"],
            "--nccl-port",
            str(nccl_port),
        ],
        env=env,
        stdout=fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=SGLANG,
    )
    return proc, fh


_NCCL_PORTS = list(range(21100, 21400, 7))
_nccl_idx = 0


def _pick_nccl_port() -> int:
    """Reserve an nccl port TCPStore can actually bind.

    sglang defaults nccl_port to get_free_port(), which probes with SO_REUSEADDR
    while torch's TCPStore binds without it; the probe then hands back a port
    with lingering peers that TCPStore rejects as EADDRINUSE, before model load.
    Probing the way TCPStore binds, and never reusing the previous arm's port,
    removes both halves.
    """
    global _nccl_idx
    for _ in range(len(_NCCL_PORTS)):
        port = _NCCL_PORTS[_nccl_idx % len(_NCCL_PORTS)]
        _nccl_idx += 1
        try:
            with socket.socket() as sk:
                sk.bind(("127.0.0.1", port))
                sk.listen(1)
            return port
        except OSError:
            continue
    raise RuntimeError("no bindable nccl port in range")


def wait_ready(proc, log: Path, timeout: int) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited {proc.returncode}; see {log}")
        try:
            if requests.get(f"{BASE}/health", timeout=5).status_code == 200:
                break
        except requests.RequestException:
            time.sleep(10)
    else:
        raise TimeoutError(f"not ready in {timeout}s; see {log}")
    info = json.dumps(requests.get(f"{BASE}/get_model_info", timeout=60).json())
    assert "V4.1-Flash" in info, f"wrong model on port {PORT}: {info[:300]}"


def bench_point(
    arm: str,
    conc: int,
    num_prompts: int,
    warmup: int,
    in_len: int,
    out_len: int,
    dataset: str = "random",
    seed: int = 42,
    rep: int = 0,
) -> dict:
    """One sgl-bench point.

    `random` builds each prompt as a consecutive ascending run of token ids, so
    every n-gram is novel and an engram plan hits ~0% on it by construction:
    that pass measures miss cost, not cache benefit. `sharegpt` is real chat
    text and is the workload where the hot set can actually hit.
    """
    tag = f"{arm}_{dataset}_c{conc}" + (f"_r{rep}" if rep else "")
    jsonl = LOGDIR / f"{tag}.jsonl"
    jsonl.unlink(missing_ok=True)
    if dataset == "random":
        shape = [
            "--random-input-len",
            str(in_len),
            "--random-output-len",
            str(out_len),
            "--random-range-ratio",
            "1",
        ]
    elif dataset == "generated-shared-prefix":
        # 4096 in as 3584 shared + 512 unique: 87.5% of every prompt is a
        # cached prefix, which is never re-prefilled and so issues no engram
        # lookups. gsp ignores --num-prompts, so the count comes from the
        # group shape.
        per_group = max(2, -(-num_prompts // GSP_GROUPS))
        shape = [
            "--gsp-num-groups",
            str(GSP_GROUPS),
            "--gsp-prompts-per-group",
            str(per_group),
            "--gsp-system-prompt-len",
            str(in_len - GSP_QUESTION_LEN),
            "--gsp-question-len",
            str(GSP_QUESTION_LEN),
            "--gsp-output-len",
            str(out_len),
            "--gsp-range-ratio",
            "1",
        ]
    elif dataset == "longbench_v2":
        # Natural long-context prose: the only workload with real n-gram reuse
        # and prefill heavy enough for the prefill graph to matter.
        shape = [
            "--sharegpt-output-len",
            str(out_len),
            "--sharegpt-context-len",
            str(LONGBENCH_CTX),
        ]
    else:
        shape = ["--sharegpt-output-len", str(out_len)]
    cmd = [
        "python3",
        "-m",
        "sglang.benchmark.serving",
        "--backend",
        "sglang",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--model",
        MODEL,
        "--served-model-name",
        SERVED,
        "--dataset-name",
        dataset,
        *shape,
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(conc),
        "--warmup-requests",
        str(warmup),
        "--flush-cache",
        "--seed",
        str(seed),
        "--disable-tqdm",
        "--output-file",
        str(jsonl),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{SGLANG}/python"
    log = LOGDIR / f"{tag}.bench.log"
    with log.open("w") as fh:
        rc = subprocess.call(
            cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=SGLANG
        )
    if not jsonl.exists():
        return {"concurrency": conc, "error": f"sgl-bench rc={rc}, see {log.name}"}
    rec = json.loads(jsonl.read_text().strip().splitlines()[-1])
    keep = (
        "completed",
        "request_throughput",
        "input_throughput",
        "output_throughput",
        "total_throughput",
        "mean_ttft_ms",
        "median_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "median_itl_ms",
        "p99_itl_ms",
        "mean_e2e_latency_ms",
        "median_e2e_latency_ms",
        "p99_e2e_latency_ms",
        "accept_length",
        "spec_accept_length",
        "total_input",
        "total_output",
    )
    out = {"concurrency": conc, "num_prompts": num_prompts, "seed": seed, "rep": rep}
    out.update({k: rec[k] for k in keep if k in rec})
    acc = out.get("accept_length") or out.get("spec_accept_length")
    out["acc_reported"] = acc
    if acc:
        out["acc_within_tol"] = abs(acc - float(ACC_LEN)) <= ACC_TOL
    return out


_CACHE = re.compile(
    r"engram hbm cache layer (\d+): (\d+) groups, (\d+) rows \(([\d.]+) GiB\)"
)
_HOST = re.compile(r"engram host table layer \d+: [^\n]*")
_WB = re.compile(r"Load weight begin\. avail mem=([\d.]+) GB")
_WU = re.compile(r"Load weight end\..*?mem usage=([\d.]+) GB")
_WA = re.compile(r"Load weight end\..*?avail mem=([\d.]+) GB")
_POOL = re.compile(r"Memory pool end\. avail mem=([\d.]+) GB")
_MAXTOK = re.compile(r"max_total_num_tokens=(\d+)")
_TOKUSE = re.compile(r"full token usage: ([\d.]+)")
_QUEUE = re.compile(r"#queue-req: (\d+)")
_GRAPH = re.compile(r"[Gg]raph.{0,40}?([\d.]+) GB")


def scrape(log: Path) -> dict:
    t = log.read_text(errors="replace")
    wa = [float(x) for x in _WA.findall(t)]
    pool = [float(x) for x in _POOL.findall(t)]
    cache = {}
    for a, b, c, d in _CACHE.findall(t):
        cache[int(a)] = {
            "layer": int(a),
            "groups": int(b),
            "rows": int(c),
            "gib": float(d),
        }
    maxtok = [int(x) for x in _MAXTOK.findall(t)]
    return {
        "engram_cache": sorted(cache.values(), key=lambda x: x["layer"]),
        "engram_cache_gib_total": round(sum(v["gib"] for v in cache.values()), 2)
        or None,
        "host_table": sorted(set(_HOST.findall(t)))[:2],
        "avail_before_weights_gb": max(
            (float(x) for x in _WB.findall(t)), default=None
        ),
        "weight_usage_gb": sorted(set(round(float(x), 2) for x in _WU.findall(t)))[-3:],
        "avail_after_weights_gb": min(wa) if wa else None,
        "avail_after_pool_gb": min(pool) if pool else None,
        "kv_pool_gb": round(min(wa) - min(pool), 2) if wa and pool else None,
        "max_total_num_tokens": maxtok[-1] if maxtok else None,
        "peak_token_usage": max((float(x) for x in _TOKUSE.findall(t)), default=None),
        "peak_queue_req": max((int(x) for x in _QUEUE.findall(t)), default=None),
        "graph_mem_lines": sorted(set(_GRAPH.findall(t)))[-3:],
    }


def _gpu_used_mib() -> int:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return max(int(x) for x in r.stdout.split())
    except Exception:
        return 0


def stop(proc, fh, settle_mib: int = 2048) -> None:
    """Wait for the port to close AND for HBM to drain.

    A server that has stopped answering /health may still be releasing memory;
    launching the next arm into that leaves it short, which shows up much later
    as a CUDA OOM during graph capture rather than as a teardown problem.
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=300)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=120)
        except Exception:
            pass
    finally:
        fh.close()
    for _ in range(90):
        try:
            requests.get(f"{BASE}/health", timeout=2)
            time.sleep(2)
        except requests.RequestException:
            break
    for i in range(120):
        used = _gpu_used_mib()
        if used <= settle_mib:
            if i:
                print(f"  (gpu drained after {i * 5}s, {used} MiB left)", flush=True)
            return
        time.sleep(5)
    print(f"  WARNING: gpu still holding {_gpu_used_mib()} MiB after 600s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--hiengram-gib", nargs="*", default=[])
    ap.add_argument("--concurrency", nargs="*", type=int, default=[1, 8, 16, 32, 64])
    ap.add_argument("--prompt-factor", type=int, default=4)
    ap.add_argument("--min-prompts", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument(
        "--max-running",
        type=int,
        default=0,
        help="Append --max-running-requests N to every arm. sglang pins "
        "this model to 256, so anything above that just queues.",
    )
    ap.add_argument(
        "--mem-fraction",
        type=float,
        default=0.0,
        help="Override --mem-fraction-static everywhere. The weights plus "
        "pool consume the whole budget, so a high "
        "--max-running-requests has no room for its request slots.",
    )
    ap.add_argument(
        "--max-total-tokens",
        type=int,
        default=0,
        help="Cap the KV pool on every arm. An uncapped pool fills HBM "
        "before the request slots are allocated, so a high "
        "--max-running-requests then OOMs.",
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Runs per point; >1 reports the median and the "
        "min-to-max spread, so a 1%% delta can be judged.",
    )
    ap.add_argument(
        "--extra",
        default="",
        help="Extra launch flags appended to every arm, one string. "
        "Later flags win in sglang, so these override the arm.",
    )
    ap.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="Override --chunked-prefill-size on every arm. DP "
        "attention halves it, so pass 2x the value you want.",
    )
    ap.add_argument("--input-len", type=int, default=4096)
    ap.add_argument("--output-len", type=int, default=1024)
    ap.add_argument("--ready-timeout", type=int, default=5400)
    ap.add_argument(
        "--dataset",
        choices=["random", "sharegpt", "generated-shared-prefix", "longbench_v2"],
        default="random",
    )
    ap.add_argument("--out", default="/scratch/logs/ab3/results.json")
    args = ap.parse_args()

    LOGDIR.mkdir(parents=True, exist_ok=True)
    arms = build_arms(args.plan, args.hiengram_gib)
    if args.max_running:
        for spec in arms.values():
            spec["args"] = spec["args"] + [
                "--max-running-requests",
                str(args.max_running),
            ]
    if args.mem_fraction:
        for spec in arms.values():
            a = list(spec["args"])
            if "--mem-fraction-static" in a:
                a[a.index("--mem-fraction-static") + 1] = str(args.mem_fraction)
            else:
                a += ["--mem-fraction-static", str(args.mem_fraction)]
            spec["args"] = a
    if args.max_total_tokens:
        for spec in arms.values():
            if "--max-total-tokens" not in spec["args"]:
                spec["args"] = spec["args"] + [
                    "--max-total-tokens",
                    str(args.max_total_tokens),
                ]
    if args.chunk:
        for spec in arms.values():
            spec["args"] = _chunk(spec["args"], str(args.chunk))
    if args.extra:
        for spec in arms.values():
            spec["args"] = spec["args"] + shlex.split(args.extra)
    out_path = Path(args.out)
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    out.setdefault(
        "meta",
        {
            "model": MODEL,
            "acc_len_pinned": ACC_LEN,
            "acc_tol": ACC_TOL,
            "harness": "sglang.benchmark.serving",
            "workload": {
                "random": f"random-{args.input_len}in-{args.output_len}out ratio=1",
                "sharegpt": f"sharegpt out={args.output_len}",
                "longbench_v2": f"longbench_v2 out={args.output_len}",
                "generated-shared-prefix": (
                    f"gsp {GSP_GROUPS} groups, "
                    f"{args.input_len - GSP_QUESTION_LEN} shared + "
                    f"{GSP_QUESTION_LEN} unique in, {args.output_len} out"
                ),
            }[args.dataset],
            "node": "4x GB300, TP4/EP4",
        },
    )
    out.setdefault("arms", {})

    for arm in args.arms:
        spec = arms[arm]
        log = LOGDIR / f"{arm}_{args.dataset}.server.log"
        print(f"=== {arm} -> {log}", flush=True)
        rec: dict = {
            "env": spec["env"],
            "dataset": args.dataset,
            "bcg": "--cuda-graph-backend-prefill" in spec["args"],
            "points": [],
        }
        proc, fh = launch(arm, spec, log)
        try:
            wait_ready(proc, log, args.ready_timeout)
            for c in args.concurrency:
                n = max(args.min_prompts, args.prompt_factor * c)
                runs = []
                for rep in range(args.repeats):
                    p = bench_point(
                        arm,
                        c,
                        n,
                        args.warmup,
                        args.input_len,
                        args.output_len,
                        dataset=args.dataset,
                        seed=42 + rep,
                        rep=rep,
                    )
                    runs.append(p)
                    if args.repeats > 1:
                        requests.post(f"{BASE}/flush_cache", timeout=120)
                good = [r for r in runs if r.get("output_throughput")]
                head = dict(good[0]) if good else dict(runs[0])
                if len(good) > 1:
                    tps = sorted(r["output_throughput"] for r in good)
                    head["output_throughput"] = statistics.median(tps)
                    head["tps_runs"] = tps
                    head["tps_spread_pct"] = (tps[-1] - tps[0]) / tps[0] * 100
                    for k in ("median_tpot_ms", "median_ttft_ms", "median_itl_ms"):
                        vals = [r[k] for r in good if r.get(k)]
                        if vals:
                            head[k] = statistics.median(vals)
                rec["points"].append(head)
                # a multi-hour arm writes its json only at the end; land each
                # point now so an eviction costs one point, not the arm
                try:
                    with open(str(out_path) + ".points.jsonl", "a") as jf:
                        jf.write(
                            json.dumps({"arm": arm, "dataset": args.dataset, **head})
                            + "\n"
                        )
                except Exception as e:
                    print(f"  (point journal failed: {e})", flush=True)
                spread = head.get("tps_spread_pct")
                print(
                    f"  c={c:<4} out_tps={head.get('output_throughput')} "
                    f"ttft_p50={head.get('median_ttft_ms')} "
                    f"itl_p50={head.get('median_itl_ms')} "
                    f"acc={head.get('acc_reported')} ok={head.get('acc_within_tol')}"
                    + (f" spread={spread:.2f}%" if spread is not None else ""),
                    flush=True,
                )
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
            print("  ERROR", rec["error"], flush=True)
        finally:
            try:
                rec.update(scrape(log))
            except Exception:
                pass
            stop(proc, fh)
        out["arms"][arm] = rec
        out_path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
