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
"""Generate a Rust ``clap``-derive ``ServerArgs`` from the Python dataclass.

The Python ``ServerArgs`` dataclass (``server_args.py``) is the single source
of truth for the SGLang CLI. During the Rust frontend migration we need the
*same flags* available in both frontends; rather than hand-maintaining a second
list of ~400 flags, this generator reflects over the exact same
``dataclasses.fields()`` + ``Arg(...)`` metadata that ``add_cli_args_from_dataclass``
uses to build the argparse CLI, and emits a Rust ``clap`` struct instead.

Usage::

    # Regenerate the committed Rust file in place:
    python -m sglang.srt.arg_groups.gen_rust_cli --write

    # Fail (non-zero) if the committed file is stale — used by the drift gate:
    python -m sglang.srt.arg_groups.gen_rust_cli --check

What is auto-derived vs. hand-maintained
-----------------------------------------
* **Auto:** every field carrying ``A[T, Arg(...)]`` / ``A[T, "help"]`` metadata,
  with its type, default, choices, and ``aliases``. This is the bulk (~400).
* **Manual tail** (``_MANUAL_FIELDS`` / ``_DEPRECATED_ALIASES`` /
  ``_DEPRECATED_FLAGS`` below): exactly mirrors the special cases that
  ``add_cli_args`` registers by hand — dynamic-choice flags, the ``--config``
  meta-arg, and deprecated aliases. This is the same hand-maintained surface
  that already exists in Python, kept in one place here.

Exotic field types (``Dict``, ``Union``, custom classes — see the 7 fields that
take a string + custom ``type_parser`` on the CLI) map to ``Option<String>``:
clap accepts the raw string and the value is parsed downstream, which is exactly
what argparse does via ``type_parser``. This preserves *flag parity*, which is
the contract during the migration.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path
from typing import List, get_args, get_origin

from sglang.srt.arg_groups.arg_utils import (
    _field_default,
    _field_to_cli_name,
    _unwrap_annotated,
    _unwrap_literal,
    _unwrap_optional,
)
from sglang.srt.server_args import ServerArgs

_MISSING = dataclasses.MISSING

# Resolve repo-root/rust/sglang-grpc/src/server_args_generated.rs robustly.
# __file__ = <root>/python/sglang/srt/arg_groups/gen_rust_cli.py
_REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_PATH = _REPO_ROOT / "rust" / "sglang-grpc" / "src" / "server_args_generated.rs"

# Rust 2024 reserved words — field names that collide get a raw-identifier prefix.
_RUST_KEYWORDS = {
    "as",
    "break",
    "const",
    "continue",
    "crate",
    "dyn",
    "else",
    "enum",
    "extern",
    "false",
    "fn",
    "for",
    "if",
    "impl",
    "in",
    "let",
    "loop",
    "match",
    "mod",
    "move",
    "mut",
    "pub",
    "ref",
    "return",
    "self",
    "Self",
    "static",
    "struct",
    "super",
    "trait",
    "true",
    "type",
    "unsafe",
    "use",
    "where",
    "while",
    "async",
    "await",
    "gen",
}


# ---------------------------------------------------------------------------
# Hand-maintained tail — mirrors the manual registrations in
# ServerArgs.add_cli_args (server_args.py). Keep in sync when those change;
# the drift check does NOT cover this block (it is not derivable from metadata).
# ---------------------------------------------------------------------------

# Deprecated flags whose argparse `dest` IS a real ServerArgs field: register
# the old flag as a clap `alias` on the canonical field. {field_name: [old_flags]}
_DEPRECATED_ALIASES: dict[str, list[str]] = {
    "dsa_prefill_backend": ["nsa-prefill-backend"],
    "dsa_decode_backend": ["nsa-decode-backend"],
    "speculative_draft_window_size": ["speculative-dflash-draft-window-size"],
    "mamba_radix_cache_strategy": ["mamba-scheduler-strategy"],
    "cuda_graph_max_bs_decode": ["cuda-graph-max-bs"],
    "cuda_graph_bs_decode": ["cuda-graph-bs"],
    "cuda_graph_bs_prefill": ["piecewise-cuda-graph-tokens"],
    "cuda_graph_tc_compiler": ["piecewise-cuda-graph-compiler"],
    "cuda_graph_max_bs_prefill": ["piecewise-cuda-graph-max-tokens"],
}

# Dynamic-choice fields + the --config meta-arg. These ARE special-cased in
# add_cli_args because their choices come from runtime registries; Rust stays
# permissive (server validates). Emitted verbatim as struct fields.
_MANUAL_FIELDS: list[str] = [
    "    /// Specify the parser for reasoning models (dynamic choices; server-validated).",
    '    #[arg(long = "reasoning-parser")]',
    "    pub reasoning_parser: Option<String>,",
    "",
    "    /// Specify the parser for tool-call interactions (dynamic choices; server-validated).",
    '    #[arg(long = "tool-call-parser")]',
    "    pub tool_call_parser: Option<String>,",
    "",
    "    /// Check the real KV-cache in the canary.",
    '    #[arg(long = "kv-canary-real-data", value_parser = ["none", "partial", "all"], default_value = "none")]',
    "    pub kv_canary_real_data: String,",
    "",
    "    /// Read CLI options from a YAML config file.",
    '    #[arg(long = "config")]',
    "    pub config: Option<String>,",
]

# Deprecated flags that do NOT map 1:1 onto a field via simple rename
# (store_true / store_const onto another field, or pure no-op warnings).
# Accepted-but-hidden so flag parity holds; the remap stays in Python for now.
_DEPRECATED_FLAGS: list[str] = [
    "    // --- Deprecated flags (hidden; accepted for parity, remapped in Python) ---",
    '    #[arg(long = "stream-output", hide = true, action = clap::ArgAction::SetTrue)]',
    "    pub dep_stream_output: bool,",
    '    #[arg(long = "prefill-round-robin-balance", hide = true, action = clap::ArgAction::SetTrue)]',
    "    pub dep_prefill_round_robin_balance: bool,",
    '    #[arg(long = "collect-tokens-histogram", hide = true, action = clap::ArgAction::SetTrue)]',
    "    pub dep_collect_tokens_histogram: bool,",
    '    #[arg(long = "disable-cuda-graph", hide = true, action = clap::ArgAction::SetTrue)]',
    "    pub dep_disable_cuda_graph: bool,",
    '    #[arg(long = "enable-breakable-cuda-graph", hide = true, action = clap::ArgAction::SetTrue)]',
    "    pub dep_enable_breakable_cuda_graph: bool,",
    '    #[arg(long = "disable-piecewise-cuda-graph", hide = true, action = clap::ArgAction::SetTrue)]',
    "    pub dep_disable_piecewise_cuda_graph: bool,",
    '    #[arg(long = "enforce-piecewise-cuda-graph", hide = true, action = clap::ArgAction::SetTrue)]',
    "    pub dep_enforce_piecewise_cuda_graph: bool,",
    '    #[arg(long = "enable-dsa-prefill-context-parallel", hide = true, action = clap::ArgAction::SetTrue)]',
    "    pub dep_enable_dsa_prefill_context_parallel: bool,",
    '    #[arg(long = "enable-nsa-prefill-context-parallel", hide = true, action = clap::ArgAction::SetTrue)]',
    "    pub dep_enable_nsa_prefill_context_parallel: bool,",
    '    #[arg(long = "enable-prefill-context-parallel", hide = true, action = clap::ArgAction::SetTrue)]',
    "    pub dep_enable_prefill_context_parallel: bool,",
    '    #[arg(long = "enable-flashinfer-allreduce-fusion", hide = true, action = clap::ArgAction::SetTrue)]',
    "    pub dep_enable_flashinfer_allreduce_fusion: bool,",
    "    // Deprecated cp-mode aliases -> --cp-strategy (old value space differs,",
    "    // so taken as raw strings rather than clap aliases on cp_strategy).",
    '    #[arg(long = "dsa-prefill-cp-mode", hide = true)]',
    "    pub dep_dsa_prefill_cp_mode: Option<String>,",
    '    #[arg(long = "nsa-prefill-cp-mode", hide = true)]',
    "    pub dep_nsa_prefill_cp_mode: Option<String>,",
    '    #[arg(long = "prefill-cp-mode", hide = true)]',
    "    pub dep_prefill_cp_mode: Option<String>,",
]

# Fields with no Arg metadata that the auto-loop will skip. The dynamic-choice
# ones are re-emitted in _MANUAL_FIELDS; anything else here is intentionally
# CLI-less and excluded from both frontends.
_KNOWN_SKIPPED = {
    "reasoning_parser",
    "tool_call_parser",
    "kv_canary_real_data",
    # Intentionally CLI-less (no Arg metadata): a callable and a logger list.
    "custom_sigquit_handler",
    "stat_loggers",
}


# ---------------------------------------------------------------------------
# Emit helpers
# ---------------------------------------------------------------------------


def _rust_ident(name: str) -> str:
    return f"r#{name}" if name in _RUST_KEYWORDS else name


def _rust_str_lit(s: str) -> str:
    """A Rust double-quoted string literal (escaped, single line)."""
    s = " ".join(str(s).split())  # collapse whitespace/newlines
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def _scalar_rust_type(inner) -> str | None:
    return {str: "String", int: "i64", float: "f64", bool: "bool"}.get(inner)


def _ordered_choices(choices) -> list:
    """Deterministic choice order.

    Some choice sources are ``set`` (e.g. ``SAMPLING_BACKEND_CHOICES``), whose
    iteration order varies across processes under hash randomization. Sort those
    so generated output is stable (otherwise the drift check is flaky). Ordered
    sources (list/tuple, Literal members) keep their meaningful order.
    """
    if isinstance(choices, (set, frozenset)):
        return sorted(choices, key=str)
    return list(choices)


def _float_lit(v: float) -> str:
    if not math.isfinite(v):
        return "f64::INFINITY" if v > 0 else "f64::NEG_INFINITY"
    return f"{v!r}f64"


def _emit_field(field, hints) -> list[str] | None:
    """Return Rust lines for one dataclass field, or None to skip it."""
    hint = hints.get(field.name)
    if hint is None:
        return None
    raw_type, meta = _unwrap_annotated(hint)
    if meta is None or meta.no_cli:
        return None

    inner_type, is_optional = _unwrap_optional(raw_type)
    literal_vals = _unwrap_literal(inner_type)
    origin = get_origin(inner_type)
    default = _field_default(field)

    # A field defaulting to None is optional even if its annotation isn't
    # Union[..., None] — e.g. Literal[None, "mooncake", "nixl"]. Python's
    # argparse marks such fields not-required (default is not MISSING), so a
    # truly-required field is only one with no default at all (e.g. model_path).
    effective_optional = is_optional or default is None

    cli_name = (meta.cli_name or _field_to_cli_name(field.name)).lstrip("-")
    aliases = [a.lstrip("-") for a in (meta.aliases or [])]
    aliases += _DEPRECATED_ALIASES.get(field.name, [])

    arg_parts = [f'long = "{cli_name}"']
    for a in aliases:
        arg_parts.append(f'visible_alias = "{a}"')

    rust_type: str
    is_list = (origin in (list, List)) and meta.type_parser is None

    if literal_vals is not None:
        # Literal[...] -> choices on a String field. Drop a bare None member
        # (it encodes optionality, not a selectable CLI value).
        rust_type = "String"
        choices = [
            c for c in _ordered_choices(meta.choices or literal_vals) if c is not None
        ]
        if choices and all(isinstance(c, str) for c in choices):
            arg_parts.append(
                "value_parser = [" + ", ".join(_rust_str_lit(c) for c in choices) + "]"
            )
    elif is_list:
        elem_args = get_args(inner_type)
        elem = elem_args[0] if elem_args else str
        elem_rust = _scalar_rust_type(elem) or "String"
        rust_type = f"Vec<{elem_rust}>"
        arg_parts.append("num_args = 1..")
        if meta.choices and elem_rust == "String":
            arg_parts.append(
                "value_parser = ["
                + ", ".join(_rust_str_lit(c) for c in _ordered_choices(meta.choices))
                + "]"
            )
    else:
        scalar = _scalar_rust_type(inner_type)
        if scalar is None:
            # Exotic (Dict / Union / custom class): take the raw CLI string.
            rust_type = "Option<String>"
        elif scalar == "bool":
            rust_type = "bool"
            arg_parts.append("action = clap::ArgAction::SetTrue")
        else:
            rust_type = scalar
            if meta.choices and scalar == "String":
                arg_parts.append(
                    "value_parser = ["
                    + ", ".join(
                        _rust_str_lit(c) for c in _ordered_choices(meta.choices)
                    )
                    + "]"
                )

    # Default handling (clap carries the authoritative CLI default).
    # `default_value_t` is typed and only safe on a non-Optional numeric field;
    # for Option<num>-with-default (and everything else) use the string
    # `default_value`, which clap parses into the field type (Some(..) for Option).
    if (
        rust_type != "bool"
        and not is_list
        and default is not _MISSING
        and default is not None
    ):
        scalar = _scalar_rust_type(inner_type)
        if scalar == "i64" and not effective_optional:
            arg_parts.append(f"default_value_t = {int(default)}i64")
        elif scalar == "f64" and not effective_optional:
            arg_parts.append(f"default_value_t = {_float_lit(float(default))}")
        else:
            arg_parts.append(f"default_value = {_rust_str_lit(default)}")

    # Wrap scalar in Option when optional (lists/bools stay as-is).
    if effective_optional and not is_list and rust_type in ("String", "i64", "f64"):
        rust_type = f"Option<{rust_type}>"

    lines = []
    if meta.help:
        lines.append(f"    /// {' '.join(meta.help.split())}")
    lines.append(f"    #[arg({', '.join(arg_parts)})]")
    lines.append(f"    pub {_rust_ident(field.name)}: {rust_type},")

    # argparse.BooleanOptionalAction auto-registers a `--no-<flag>` negation;
    # mirror it as a hidden companion bool so the flag set matches exactly.
    if meta.action is argparse.BooleanOptionalAction:
        lines.append(f"    /// Negation of --{cli_name} (BooleanOptionalAction).")
        lines.append(
            f'    #[arg(long = "no-{cli_name}", action = clap::ArgAction::SetTrue, hide = true)]'
        )
        lines.append(f"    pub no_{field.name}: bool,")
    return lines


_HEADER = """\
// ============================================================================
// @generated by python/sglang/srt/arg_groups/gen_rust_cli.py — DO NOT EDIT.
//
// Source of truth: ServerArgs in python/sglang/srt/server_args.py.
// Regenerate:  python -m sglang.srt.arg_groups.gen_rust_cli --write
// A pre-commit hook + CI drift check fail the build if this file is stale.
// ============================================================================
#![allow(clippy::all)]

/// Server-wide CLI configuration, mirroring the Python `ServerArgs` flags.
///
/// During the Rust frontend migration this guarantees both frontends accept
/// the same flags. Exotic-typed flags are taken as raw strings and parsed
/// downstream (matching argparse `type_parser`).
#[derive(Debug, Clone, clap::Parser, serde::Serialize, serde::Deserialize, Default)]
#[serde(default)]
#[command(name = "sglang-server")]
pub struct ServerArgs {
"""


def generate() -> str:
    from typing import get_type_hints

    hints = get_type_hints(ServerArgs, include_extras=True)
    out: list[str] = [_HEADER]

    skipped: list[str] = []
    first = True
    for field in dataclasses.fields(ServerArgs):
        lines = _emit_field(field, hints)
        if lines is None:
            hint = hints.get(field.name)
            if hint is not None:
                _, meta = _unwrap_annotated(hint)
                if meta is None and field.name not in _KNOWN_SKIPPED:
                    skipped.append(field.name)
            continue
        if not first:
            out.append("")
        first = False
        out.extend(lines)

    out.append("")
    out.append(
        "    // ===== Hand-maintained tail (mirrors add_cli_args specials) ====="
    )
    out.extend(_MANUAL_FIELDS)
    out.append("")
    out.extend(_DEPRECATED_FLAGS)
    out.append("}")
    out.append("")

    if skipped:
        print(
            "WARNING: fields skipped (no Arg metadata, not in _KNOWN_SKIPPED): "
            + ", ".join(skipped),
            file=sys.stderr,
        )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true", help="Regenerate the file in place.")
    g.add_argument("--check", action="store_true", help="Fail if the file is stale.")
    g.add_argument("--stdout", action="store_true", help="Print to stdout.")
    args = ap.parse_args()

    content = generate()

    if args.stdout:
        sys.stdout.write(content)
        return 0
    if args.write:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(content)
        print(f"Wrote {OUTPUT_PATH}")
        return 0
    if args.check:
        current = OUTPUT_PATH.read_text() if OUTPUT_PATH.exists() else ""
        if current != content:
            print(
                f"DRIFT: {OUTPUT_PATH} is out of date with ServerArgs.\n"
                "Run: python -m sglang.srt.arg_groups.gen_rust_cli --write",
                file=sys.stderr,
            )
            return 1
        print("OK: Rust ServerArgs is in sync.")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
