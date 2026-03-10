# sglang-mps

Apple MPS (Metal Performance Shaders) platform plugin for [SGLang](https://github.com/sgl-project/sglang).

Enables LLM inference on Apple Silicon (M1/M2/M3/M4) Macs using the Metal GPU backend.

## Installation

```bash
# Install SGLang core first
pip install -e python/

# Install the MPS plugin
pip install -e sglang-mps/
```

The plugin registers itself automatically via Python entry points. No configuration needed -- SGLang discovers it at startup.

## Usage

```bash
# Auto-detect MPS on Apple Silicon (if no CUDA/XPU/etc. available)
python -m sglang.launch_server --model Qwen/Qwen2.5-0.5B

# Explicitly select MPS
python -m sglang.launch_server --model Qwen/Qwen2.5-0.5B --device mps
```

## How It Works

This package provides `MPSSRTPlatform`, a platform plugin that extends `SRTPlatform`. It is discovered by SGLang's `discover_platforms()` via the `sglang.platforms` entry point group.

The platform configures:

- **Attention**: `torch_native` backend (PyTorch SDPA)
- **Sampling**: `pytorch` backend
- **Distributed**: `gloo` backend
- **Graph runner**: `CPUGraphRunner` (torch.compile-based, eager by default)
- **Memory**: Uses `torch.mps.recommended_max_memory()` for budget, conservative `mem_fraction_static=0.50` since MPS shares unified memory with the OS

## Supported Models

Any model that runs on pure PyTorch ops (no custom CUDA kernels required):

- Small LLMs: Qwen2.5-0.5B, Qwen2.5-1.5B, Llama-3.2-1B, Phi-3-mini, etc.
- Medium LLMs: Models up to ~8B parameters (depending on available unified memory)
- Unquantized models only

## Limitations

- **No quantization**: AWQ, GPTQ, FP8, and other quantization methods require CUDA kernels. Only unquantized models are supported.
- **No CUDA graphs**: MPS has no graph capture API. Inference runs in eager mode by default.
- **torch.compile**: Experimental on MPS ([PyTorch #150121](https://github.com/pytorch/pytorch/issues/150121)). Disabled by default; opt in with `--enable-torch-compile`.
- **Single device**: `tp_size` and `dp_size` are enforced to 1. Multi-node setups (e.g., chained Mac Minis) are future work.
- **No float64**: Metal does not support float64 tensors. The platform auto-converts to float32 with a warning.
- **No Triton/FlashAttention/FlashInfer**: These are CUDA-only. All ops go through PyTorch native implementations.
- **Unified memory**: MPS shares system RAM. The default `mem_fraction_static=0.50` is conservative to avoid starving the OS. Adjust with `--mem-fraction-static` if needed.

## Architecture

```
sglang-mps/
  pyproject.toml              # entry_points: sglang.platforms -> mps
  sglang_mps/
    __init__.py
    platform.py               # MPSSRTPlatform(SRTPlatform)
```

This follows the same external plugin pattern as `sglang-npu`:

```toml
[project.entry-points."sglang.platforms"]
mps = "sglang_mps.platform:MPSSRTPlatform"
```

## License

Apache-2.0
