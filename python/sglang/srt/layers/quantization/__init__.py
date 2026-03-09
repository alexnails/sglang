# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/v0.5.5/vllm/model_executor/layers/quantization/__init__.py
from __future__ import annotations

import builtins
import importlib
import inspect
from typing import TYPE_CHECKING, Dict, Optional, Type

import torch


# Define empty classes as placeholders when vllm is not available
class DummyConfig:
    def override_quantization_method(self, *args, **kwargs):
        return None


CompressedTensorsConfig = DummyConfig

from sglang.srt.layers.quantization.base_config import QuantizationConfig

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKOutput

# Lazy quantization config registry: maps method name -> (module_path, class_name)
# Configs are only imported when get_quantization_config() is called.
_LAZY_QUANTIZATION_METHODS: Dict[str, tuple[str, str]] = {
    "fp8": ("sglang.srt.layers.quantization.fp8", "Fp8Config"),
    "mxfp8": ("sglang.srt.layers.quantization.fp8", "Fp8Config"),
    "blockwise_int8": (
        "sglang.srt.layers.quantization.blockwise_int8",
        "BlockInt8Config",
    ),
    "modelopt": (
        "sglang.srt.layers.quantization.modelopt_quant",
        "ModelOptFp8Config",
    ),
    "modelopt_fp8": (
        "sglang.srt.layers.quantization.modelopt_quant",
        "ModelOptFp8Config",
    ),
    "modelopt_fp4": (
        "sglang.srt.layers.quantization.modelopt_quant",
        "ModelOptFp4Config",
    ),
    "w8a8_int8": ("sglang.srt.layers.quantization.w8a8_int8", "W8A8Int8Config"),
    "w8a8_fp8": ("sglang.srt.layers.quantization.w8a8_fp8", "W8A8Fp8Config"),
    "awq": ("sglang.srt.layers.quantization.awq", "AWQConfig"),
    "awq_marlin": ("sglang.srt.layers.quantization.awq", "AWQMarlinConfig"),
    "bitsandbytes": (
        "sglang.srt.layers.quantization.bitsandbytes",
        "BitsAndBytesConfig",
    ),
    "gguf": ("sglang.srt.layers.quantization.gguf", "GGUFConfig"),
    "gptq": ("sglang.srt.layers.quantization.gptq", "GPTQConfig"),
    "gptq_marlin": ("sglang.srt.layers.quantization.gptq", "GPTQMarlinConfig"),
    "moe_wna16": ("sglang.srt.layers.quantization.moe_wna16", "MoeWNA16Config"),
    "compressed-tensors": (
        "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors",
        "CompressedTensorsConfig",
    ),
    "qoq": ("sglang.srt.layers.quantization.qoq", "QoQConfig"),
    "w4afp8": ("sglang.srt.layers.quantization.w4afp8", "W4AFp8Config"),
    "petit_nvfp4": ("sglang.srt.layers.quantization.petit", "PetitNvFp4Config"),
    "fbgemm_fp8": ("sglang.srt.layers.quantization.fpgemm_fp8", "FBGEMMFp8Config"),
    "quark": ("sglang.srt.layers.quantization.quark.quark", "QuarkConfig"),
    "auto-round": ("sglang.srt.layers.quantization.auto_round", "AutoRoundConfig"),
    "modelslim": (
        "sglang.srt.layers.quantization.modelslim.modelslim",
        "ModelSlimConfig",
    ),
    "quark_int4fp8_moe": (
        "sglang.srt.layers.quantization.quark_int4fp8_moe",
        "QuarkInt4Fp8Config",
    ),
    "mxfp4": ("sglang.srt.layers.quantization.mxfp4", "Mxfp4Config"),
}

_resolved_cache: Dict[str, Type[QuantizationConfig]] = {}


def _resolve(method: str) -> Type[QuantizationConfig]:
    if method in _resolved_cache:
        return _resolved_cache[method]
    if method not in _LAZY_QUANTIZATION_METHODS:
        raise ValueError(
            f"Invalid quantization method: {method}. "
            f"Available methods: {list(_LAZY_QUANTIZATION_METHODS.keys())}"
        )
    module_path, class_name = _LAZY_QUANTIZATION_METHODS[method]
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    _resolved_cache[method] = cls
    return cls


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    return _resolve(quantization)


# Backwards-compat: some code does `from sglang.srt.layers.quantization import Fp8Config`
# We support that via __getattr__ lazy loading.
_ATTR_TO_METHOD = {
    "Fp8Config": "fp8",
    "BlockInt8Config": "blockwise_int8",
    "ModelOptFp8Config": "modelopt_fp8",
    "ModelOptFp4Config": "modelopt_fp4",
    "W8A8Int8Config": "w8a8_int8",
    "W8A8Fp8Config": "w8a8_fp8",
    "AWQConfig": "awq",
    "AWQMarlinConfig": "awq_marlin",
    "BitsAndBytesConfig": "bitsandbytes",
    "GGUFConfig": "gguf",
    "GPTQConfig": "gptq",
    "GPTQMarlinConfig": "gptq_marlin",
    "MoeWNA16Config": "moe_wna16",
    "CompressedTensorsConfig": "compressed-tensors",
    "QoQConfig": "qoq",
    "W4AFp8Config": "w4afp8",
    "PetitNvFp4Config": "petit_nvfp4",
    "FBGEMMFp8Config": "fbgemm_fp8",
    "QuarkConfig": "quark",
    "AutoRoundConfig": "auto-round",
    "ModelSlimConfig": "modelslim",
    "QuarkInt4Fp8Config": "quark_int4fp8_moe",
    "Mxfp4Config": "mxfp4",
}


def __getattr__(name: str):
    if name in _ATTR_TO_METHOD:
        return _resolve(_ATTR_TO_METHOD[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Keep QUANTIZATION_METHODS as a lazy-resolving dict for backwards compat
class _LazyQuantDict(dict):
    def __missing__(self, key):
        if key in _LAZY_QUANTIZATION_METHODS:
            cls = _resolve(key)
            self[key] = cls
            return cls
        raise KeyError(key)

    def __contains__(self, key):
        return key in _LAZY_QUANTIZATION_METHODS or super().__contains__(key)

    def keys(self):
        return _LAZY_QUANTIZATION_METHODS.keys()

    def items(self):
        return ((k, self[k]) for k in _LAZY_QUANTIZATION_METHODS)

    def values(self):
        return (self[k] for k in _LAZY_QUANTIZATION_METHODS)


QUANTIZATION_METHODS = _LazyQuantDict()
BASE_QUANTIZATION_METHODS = QUANTIZATION_METHODS


original_isinstance = builtins.isinstance
