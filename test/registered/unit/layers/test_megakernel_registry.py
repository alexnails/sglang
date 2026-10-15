import unittest
from types import SimpleNamespace

from sglang.kernels.spec import CapabilityRequirement
from sglang.srt.layers.megakernel import (
    NO_MEGAKERNEL,
    MegaKernelCapability,
    MegaKernelProvider,
    MegaKernelScope,
    StepMode,
    bind_layer,
    on_weight_loaded,
    register_megakernel,
    registry,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context, reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

FFN = MegaKernelScope.LAYER_FFN
ATTN = MegaKernelScope.LAYER_ATTN_FRONT
HF_CONFIG = SimpleNamespace(architectures=["ToyForCausalLM"])


class _ToyProvider(MegaKernelProvider):
    capability = MegaKernelCapability(
        architectures=frozenset({"ToyForCausalLM"}),
        tp_sizes=frozenset({1, 2}),
        step_modes=frozenset({StepMode.DECODE, StepMode.TARGET_VERIFY}),
        max_rows=8,
    )

    def __init__(self, name):
        super().__init__(name)
        self.loaded = []
        self.allow_step = True

    def bind_layer(self, layer, scope):
        if scope is FFN or layer.has_attn_front:
            return lambda: (self.name, layer.idx, scope)
        return None

    def on_weight_loaded(self, layers, name, tensor):
        self.loaded.append(name)

    def eligible(self, forward_batch, rows):
        return self.allow_step


class _CudaOnlyProvider(_ToyProvider):
    capability = MegaKernelCapability(devices=frozenset({CapabilityRequirement.CUDA}))


def _layer(idx, has_attn_front=True):
    return SimpleNamespace(idx=idx, has_attn_front=has_attn_front)


def _batch(mode):
    return SimpleNamespace(forward_mode=mode)


class TestMegaKernelRegistry(CustomTestCase):
    def setUp(self):
        registry._ACTIVE.clear()
        self._factories = dict(registry._FACTORIES)
        register_megakernel("toy", _ToyProvider)
        register_megakernel("toy_cuda", _CudaOnlyProvider)
        register_megakernel("toy_lazy", f"{__name__}:_ToyProvider")

    def tearDown(self):
        registry._ACTIVE.clear()
        registry._FACTORIES.clear()
        registry._FACTORIES.update(self._factories)
        reset_context()

    def _bind(self, backend, layer=None, **fields):
        with get_context().override_server_args(megakernel_backend=backend, **fields):
            return bind_layer(
                layer or _layer(0), hf_config=HF_CONFIG, scopes=(FFN, ATTN)
            )

    def test_no_backend_keeps_every_step_on_the_normal_path(self):
        bound = self._bind(None)
        self.assertIs(bound, NO_MEGAKERNEL)
        self.assertIsNone(
            bound.launch(FFN, forward_batch=_batch(ForwardMode.DECODE), rows=1)
        )

    def test_launch_only_on_eligible_steps(self):
        bound = self._bind("toy")
        self.assertTrue(bound.has(FFN) and bound.has(ATTN))
        decode, verify = _batch(ForwardMode.DECODE), _batch(ForwardMode.TARGET_VERIFY)
        self.assertEqual(
            bound.launch(FFN, forward_batch=decode, rows=8)(), ("toy", 0, FFN)
        )
        self.assertIsNotNone(bound.launch(ATTN, forward_batch=verify, rows=1))
        self.assertIsNone(bound.launch(FFN, forward_batch=decode, rows=9))
        self.assertIsNone(bound.launch(FFN, forward_batch=decode, rows=0))
        for mode in (ForwardMode.EXTEND, ForwardMode.MIXED, ForwardMode.IDLE):
            self.assertIsNone(bound.launch(FFN, forward_batch=_batch(mode), rows=1))

    def test_provider_step_gate_routes_to_normal_path(self):
        bound = self._bind("toy")
        bound.provider.allow_step = False
        self.assertIsNone(
            bound.launch(FFN, forward_batch=_batch(ForwardMode.DECODE), rows=1)
        )

    def test_one_provider_instance_serves_every_layer(self):
        with get_context().override_server_args(megakernel_backend="toy"):
            a = bind_layer(_layer(0), hf_config=HF_CONFIG, scopes=(FFN,))
            b = bind_layer(_layer(1), hf_config=HF_CONFIG, scopes=(FFN,))
            on_weight_loaded(layers=None, name="w", tensor=None)
        self.assertIs(a.provider, b.provider)
        self.assertEqual(a.provider.loaded, ["w"])

    def test_lazy_import_path_factory(self):
        self.assertIsInstance(self._bind("toy_lazy").provider, _ToyProvider)

    def test_unknown_backend_names_the_registered_ones(self):
        with self.assertRaisesRegex(ValueError, r"registered: .*'toy'"):
            self._bind("nope")

    def test_startup_rejections_say_why(self):
        cases = {
            "toy_cuda": ({}, "does not run on"),
            "toy": ({"tp_size": 4}, "tensor parallel size"),
        }
        for backend, (fields, reason) in cases.items():
            registry._ACTIVE.clear()
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(
                    ValueError, f"--megakernel-backend {backend} .*{reason}"
                ):
                    self._bind(backend, **fields)

    def test_architecture_mismatch_is_rejected(self):
        with get_context().override_server_args(megakernel_backend="toy"):
            with self.assertRaisesRegex(ValueError, "supports \\['ToyForCausalLM'\\]"):
                bind_layer(
                    _layer(0),
                    hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
                    scopes=(FFN,),
                )


if __name__ == "__main__":
    unittest.main()
