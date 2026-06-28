#!/usr/bin/env python3
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
"""Fail if the generated Rust ``ServerArgs`` is out of sync with the Python one.

The Python ``ServerArgs`` dataclass is the single source of truth; the Rust
``clap`` struct (``rust/sglang-grpc/src/server_args_generated.rs``) is generated
from it so both frontends accept the same flags. This guard fails when the two
drift, so a stale Rust file cannot merge.

Used in two places:

* **pre-commit hook** — runs locally on every commit that touches the relevant
  files. Importing ``ServerArgs`` requires the sglang/torch runtime; in a bare
  environment (e.g. the lint CI image that only installs pre-commit) that import
  fails, so this script *skips gracefully* (exit 0) there. Real enforcement for
  such environments is the dedicated CI workflow below.
* **CI workflow** (``.github/workflows/check-server-args-parity.yml``) — installs
  the deps, so the import succeeds and drift is a hard failure that blocks merge.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from sglang.srt.arg_groups.gen_rust_cli import OUTPUT_PATH, generate
    except (ImportError, ModuleNotFoundError) as e:
        # sglang/torch not importable here (bare lint env) — skip, don't fail.
        print(
            f"[rust-cli-drift] skipped: cannot import ServerArgs "
            f"({type(e).__name__}: {e})."
        )
        print(
            "[rust-cli-drift] expected in envs without the sglang runtime; "
            "the check-server-args-parity CI workflow is the authoritative gate."
        )
        return 0

    want = generate()
    have = OUTPUT_PATH.read_text() if OUTPUT_PATH.exists() else ""
    if want != have:
        sys.stderr.write(
            "\nERROR: the Rust ServerArgs is out of sync with the Python "
            "ServerArgs.\n"
            f"  {OUTPUT_PATH}\n"
            "Regenerate it and commit the result:\n"
            "  python -m sglang.srt.arg_groups.gen_rust_cli --write\n\n"
        )
        return 1

    print("[rust-cli-drift] OK: Rust ServerArgs matches the Python source of truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
