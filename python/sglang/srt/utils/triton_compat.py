"""Compatibility layer for optional triton dependency.

Provides ``triton_jit`` (drop-in for ``@triton.jit``) and a ``tl`` stub so
that kernel *definitions* are silently skipped when triton is not installed
while ``tl.constexpr`` annotations still resolve at function-definition time.

Usage in any kernel file::

    from sglang.srt.utils.triton_compat import tl, triton, triton_jit

    @triton_jit
    def my_kernel(x_ptr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        ...
"""

from __future__ import annotations

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False
    triton = None  # type: ignore[assignment]

    class _TritonLanguageStub:
        """Minimal stub so ``tl.constexpr``, ``tl.uint32``, etc. resolve in annotations."""

        def __getattr__(self, name: str):
            return None

    tl = _TritonLanguageStub()  # type: ignore[assignment]


def triton_jit(fn):
    """Drop-in replacement for ``@triton.jit``.

    When triton is installed this is identical to ``triton.jit``.
    When triton is absent the decorated function is returned as-is (calling it
    will fail inside the body when it tries to use ``tl`` operations, which is
    the desired behavior — GPU kernels should never be called on CPU).
    """
    if _HAS_TRITON:
        return triton.jit(fn)
    return fn


def has_triton() -> bool:
    return _HAS_TRITON
