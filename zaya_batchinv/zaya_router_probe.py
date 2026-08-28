"""Direct measurement of ZAYA1 top-1 router behaviour, without touching source.

`--enable-return-routed-experts` does NOT work for ZAYA1 (two independent
reasons, both verified in the tree):

  * the capture hook is `capture_routed_experts_if_allowed` at
    python/sglang/srt/layers/moe/topk.py:1971, reached only from
    `select_experts` / `build_precomputed_topk_output`. `ZayaBlock.forward`
    builds `StandardTopKOutput` by hand at
    python/sglang/srt/models/zaya.py:2027-2031 and never calls either, so the
    capturer would record all-zeros.
  * `RoutedExpertsCapturer.__init__` reads
    `model_config.hf_text_config.num_experts_per_tok`
    (python/sglang/srt/state_capturer/routed_experts.py:70). `ZayaConfig`
    (python/sglang/srt/configs/zaya.py) never defines it, so the server raises
    AttributeError at startup.

So this monkeypatch replaces it. It wraps `ZayaRouter._routing_reference`,
which is the routing path whenever `SGLANG_OPT_ZAYA_FUSED_ROUTER` is off (the
default, and the campaign's banked config), and records per (forward pass,
MoE layer, row):

  ids       int16  the argmax expert
  runnerup  int16  the second-best expert
  gap       fp32   biased_score[top1] - biased_score[top2]   <-- the flip margin
  prob      fp32   the gathered softmax probability of the winner
  logits    fp32   the raw [T, 25] router logits (bit-comparable across runs)

plus, per layer-entry, a fingerprint of the normed hidden state so a numerical
divergence can be localised to the layer that first produces it.

Nothing here changes a single value the model computes; it only reads tensors.

Usage
-----
    export PYTHONPATH=/path/to/zaya_batchinv:$PYTHONPATH
    export ZAYA_PROBE_DIR=/data/zaya_probe/run_serial
    python3 -m sglang.launch_server ... --disable-cuda-graph

Recording is OFF until the sentinel file `$ZAYA_PROBE_DIR/ON` exists, so the
long warmup/capture phase is not recorded. `probe_client.py` creates and
removes it around each measured request. Output: one
`$ZAYA_PROBE_DIR/rank<N>_pass<K>.npz` per recorded forward pass per rank.

Env knobs
---------
    ZAYA_PROBE_DIR         output directory (required; also arms sitecustomize)
    ZAYA_PROBE_MAX_PASSES  stop after this many recorded passes (default 16)
    ZAYA_PROBE_LOGITS      1 to store full logits (default 1)
    ZAYA_PROBE_HS          1 to store hidden-state fingerprints (default 1)
"""

from __future__ import annotations

import atexit
import os
import sys
import threading

_installed = False
_lock = threading.Lock()


def _log(msg: str) -> None:
    print(f"[zaya-probe] {msg}", file=sys.stderr, flush=True)


class _Recorder:
    def __init__(self, out_dir: str) -> None:
        self.out_dir = out_dir
        self.on_file = os.path.join(out_dir, "ON")
        self.max_passes = int(os.environ.get("ZAYA_PROBE_MAX_PASSES", "16"))
        self.want_logits = os.environ.get("ZAYA_PROBE_LOGITS", "1") == "1"
        self.want_hs = os.environ.get("ZAYA_PROBE_HS", "1") == "1"
        self.rank = self._rank()
        self.pass_idx = 0
        self.last_layer = 1 << 30
        self.cur: dict[str, list] = {}
        self.hs_calls: list = []
        os.makedirs(out_dir, exist_ok=True)
        atexit.register(self.flush)

    @staticmethod
    def _rank() -> int:
        for key in ("SGLANG_TP_RANK", "RANK", "LOCAL_RANK"):
            if key in os.environ:
                try:
                    return int(os.environ[key])
                except ValueError:
                    pass
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return dist.get_rank()
        except Exception:
            pass
        return os.getpid() % 100000

    def armed(self) -> bool:
        if self.pass_idx >= self.max_passes:
            return False
        return os.path.exists(self.on_file)

    # -- pass bookkeeping ---------------------------------------------------
    def _maybe_roll(self, layer_id: int) -> None:
        """A layer id that does not increase means a new forward pass began."""
        if layer_id <= self.last_layer and self.cur:
            self.flush()
        self.last_layer = layer_id

    def flush(self) -> None:
        if not self.cur and not self.hs_calls:
            return
        import numpy as np

        payload = {}
        for key, entries in self.cur.items():
            for layer_id, arr in entries:
                payload[f"{key}/l{layer_id:03d}"] = arr
        for i, arr in enumerate(self.hs_calls):
            payload[f"hs/c{i:03d}"] = arr
        path = os.path.join(
            self.out_dir, f"rank{self.rank}_pass{self.pass_idx:03d}.npz"
        )
        try:
            np.savez_compressed(path, **payload)
            _log(f"wrote {path} ({len(payload)} arrays)")
        except Exception as exc:
            _log(f"flush failed: {exc!r}")
        self.cur = {}
        self.hs_calls = []
        self.pass_idx += 1
        self.last_layer = 1 << 30

    def add(self, key: str, layer_id: int, arr) -> None:
        self.cur.setdefault(key, []).append((layer_id, arr))


_rec: _Recorder | None = None


def _capturing() -> bool:
    """True while a CUDA graph is being captured -- any host sync would abort."""
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    except Exception:
        return True  # fail closed: never risk breaking capture


def install() -> None:
    global _installed, _rec
    with _lock:
        if _installed:
            return
        out_dir = os.environ.get("ZAYA_PROBE_DIR")
        if not out_dir:
            return

        import torch

        from sglang.srt.models import zaya as zmod

        _rec = _Recorder(out_dir)
        rec = _rec

        orig_routing = zmod.ZayaRouter._routing_reference
        orig_norm = zmod._residual_scale_norm

        def patched_routing(self, logits, model_dtype, router_hidden_states_next):
            out = orig_routing(self, logits, model_dtype, router_hidden_states_next)
            if rec.armed() and not _capturing():
                try:
                    _record_router(rec, self, logits, out)
                except Exception as exc:  # never take the server down
                    _log(f"record failed: {exc!r}")
            return out

        def patched_norm(res_scale, norm, residual, hidden_states, target_dtype):
            normed, new_res = orig_norm(
                res_scale, norm, residual, hidden_states, target_dtype
            )
            if rec.want_hs and rec.armed() and not _capturing():
                try:
                    # Two fingerprints: a low-order one that changes on any bit
                    # difference anywhere in the row, and the first 8 channels
                    # verbatim so a diff can be eyeballed.
                    f = normed.detach().float()
                    rec.hs_calls.append(
                        torch.stack(
                            [f.sum(-1), f.abs().sum(-1), f[:, 0], f[:, 1]], dim=-1
                        )
                        .cpu()
                        .numpy()
                    )
                except Exception as exc:
                    _log(f"hs record failed: {exc!r}")
            return normed, new_res

        zmod.ZayaRouter._routing_reference = patched_routing
        zmod._residual_scale_norm = patched_norm
        # ZayaDecoder*Layer.forward resolved _residual_scale_norm as a module
        # global at call time, so rebinding the module attribute is enough.

        _installed = True
        _log(f"installed, out_dir={out_dir}, rank={rec.rank}")


def _record_router(rec: "_Recorder", router, logits, routing) -> None:
    import torch

    layer_id = int(getattr(router, "layer_id", -1))
    rec._maybe_roll(layer_id)

    if router.router_softmax_fp32:
        prob = torch.softmax(logits, dim=-1, dtype=torch.float32)
    else:
        prob = torch.softmax(logits, dim=-1).float()
    biased = prob + router.balancing_biases

    k = min(2, biased.shape[-1])
    topv, topi = torch.topk(biased, k, dim=-1)
    gap = (topv[:, 0] - topv[:, 1]) if k == 2 else torch.zeros_like(topv[:, 0])

    rec.add("ids", layer_id, topi[:, 0].to(torch.int16).cpu().numpy())
    rec.add(
        "runnerup",
        layer_id,
        (topi[:, 1] if k == 2 else topi[:, 0]).to(torch.int16).cpu().numpy(),
    )
    rec.add("gap", layer_id, gap.float().cpu().numpy())
    rec.add("prob", layer_id, routing.route_prob.detach().float().reshape(-1).cpu().numpy())
    if rec.want_logits:
        rec.add("logits", layer_id, logits.detach().float().cpu().numpy())
