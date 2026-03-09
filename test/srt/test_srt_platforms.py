"""Unit tests for the SRT platform plugin infrastructure.

These tests validate the discovery, registry, and platform class behaviour
without requiring any accelerator hardware.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import torch


class TestPlatformEnums(unittest.TestCase):
    def test_platform_enum_members(self):
        from sglang.platforms.interface import PlatformEnum

        for name in (
            "CUDA",
            "ROCM",
            "CPU",
            "NPU",
            "XPU",
            "MUSA",
            "HPU",
            "MPS",
            "TPU",
            "OOT",
            "UNSPECIFIED",
        ):
            self.assertTrue(hasattr(PlatformEnum, name))

    def test_cpu_arch_enum(self):
        from sglang.platforms.interface import CpuArchEnum

        self.assertTrue(hasattr(CpuArchEnum, "X86"))
        self.assertTrue(hasattr(CpuArchEnum, "ARM"))
        self.assertTrue(hasattr(CpuArchEnum, "UNSPECIFIED"))

    def test_device_capability(self):
        from sglang.platforms.interface import DeviceCapability

        cap = DeviceCapability(major=8, minor=9)
        self.assertEqual(cap.as_version_str(), "8.9")
        self.assertEqual(cap.to_int(), 89)


class TestDiscoverPlatforms(unittest.TestCase):
    def setUp(self):
        import sglang.srt.platforms as mod

        mod._discovered = None
        mod._instances.clear()

    def tearDown(self):
        import sglang.srt.platforms as mod

        mod._discovered = None
        mod._instances.clear()

    def test_builtins_always_present(self):
        from sglang.srt.platforms import discover_platforms
        from sglang.srt.platforms.cpu import CPUSRTPlatform
        from sglang.srt.platforms.cuda import CUDASRTPlatform

        platforms = discover_platforms()
        self.assertIs(platforms["cuda"], CUDASRTPlatform)
        self.assertIs(platforms["cpu"], CPUSRTPlatform)

    def test_stubs_present(self):
        from sglang.srt.platforms import discover_platforms

        platforms = discover_platforms()
        for name in ("npu", "xpu", "musa", "hpu"):
            self.assertIn(name, platforms)

    def test_entry_points_loaded(self):
        """Verify that entry_points in group 'sglang.platforms' are scanned."""
        from sglang.srt.platforms import discover_platforms
        from sglang.srt.platforms.cpu import CPUSRTPlatform

        fake_ep = MagicMock()
        fake_ep.name = "test_device"
        fake_ep.load.return_value = CPUSRTPlatform

        with patch("importlib.metadata.entry_points", return_value=[fake_ep]):
            platforms = discover_platforms()

        self.assertIs(platforms["test_device"], CPUSRTPlatform)

    def test_env_var_override(self):
        """SGLANG_PLATFORM_PLUGIN should register a platform by qualname."""
        import sglang.srt.platforms as mod

        mod._discovered = None

        env = {
            "SGLANG_PLATFORM_PLUGIN": "mydev:sglang.srt.platforms.cpu.CPUSRTPlatform"
        }
        with patch.dict(os.environ, env):
            from sglang.srt.platforms import discover_platforms
            from sglang.srt.platforms.cpu import CPUSRTPlatform

            platforms = discover_platforms()
            self.assertIs(platforms["mydev"], CPUSRTPlatform)


class TestGetPlatform(unittest.TestCase):
    def setUp(self):
        import sglang.srt.platforms as mod

        mod._discovered = None
        mod._instances.clear()

    def tearDown(self):
        import sglang.srt.platforms as mod

        mod._discovered = None
        mod._instances.clear()

    def test_get_cpu_platform(self):
        from sglang.srt.platforms import get_platform
        from sglang.srt.platforms.cpu import CPUSRTPlatform

        plat = get_platform("cpu")
        self.assertIsInstance(plat, CPUSRTPlatform)

    def test_get_cuda_platform(self):
        from sglang.srt.platforms import get_platform
        from sglang.srt.platforms.cuda import CUDASRTPlatform

        plat = get_platform("cuda")
        self.assertIsInstance(plat, CUDASRTPlatform)

    def test_device_with_index_normalised(self):
        from sglang.srt.platforms import get_platform
        from sglang.srt.platforms.cuda import CUDASRTPlatform

        plat = get_platform("cuda:0")
        self.assertIsInstance(plat, CUDASRTPlatform)

    def test_instances_are_cached(self):
        from sglang.srt.platforms import get_platform

        a = get_platform("cpu")
        b = get_platform("cpu")
        self.assertIs(a, b)

    def test_unknown_device_falls_back(self):
        from sglang.srt.platforms import get_platform
        from sglang.srt.platforms.cuda import CUDASRTPlatform

        plat = get_platform("unknown_device_xyz")
        self.assertIsInstance(plat, CUDASRTPlatform)


class TestCPUSRTPlatform(unittest.TestCase):
    def setUp(self):
        from sglang.srt.platforms.cpu import CPUSRTPlatform

        self.plat = CPUSRTPlatform()

    def test_detection_flags(self):
        self.assertTrue(self.plat.is_cpu())
        self.assertFalse(self.plat.is_cuda())
        self.assertFalse(self.plat.is_hip())
        self.assertFalse(self.plat.is_npu())

    def test_get_device(self):
        dev = self.plat.get_device(0)
        self.assertEqual(dev, torch.device("cpu"))

    def test_get_distributed_backend(self):
        self.assertEqual(self.plat.get_distributed_backend(), "gloo")

    def test_graph_runner_class(self):
        cls = self.plat.get_graph_runner_class()
        from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner

        self.assertIs(cls, CPUGraphRunner)

    def test_graph_backend_name(self):
        self.assertEqual(self.plat.graph_backend_name, "cpu graph")

    def test_supports_torch_compile(self):
        self.assertTrue(self.plat.supports_torch_compile())

    def test_configure_server_args_sets_defaults(self):
        from sglang.srt.utils.common import cpu_has_amx_support

        class FakeArgs:
            attention_backend = None
            sampling_backend = None

        args = FakeArgs()
        self.plat.configure_server_args(args)
        if cpu_has_amx_support():
            self.assertEqual(args.attention_backend, "intel_amx")
        else:
            self.assertEqual(args.attention_backend, "torch_native")
        self.assertEqual(args.sampling_backend, "pytorch")

    def test_configure_server_args_preserves_existing_attention(self):
        class FakeArgs:
            attention_backend = "torch_native"
            sampling_backend = None

        args = FakeArgs()
        self.plat.configure_server_args(args)
        self.assertEqual(args.attention_backend, "torch_native")
        self.assertEqual(args.sampling_backend, "pytorch")

    def test_available_memory_returns_tuple(self):
        free, total = self.plat.get_available_memory()
        self.assertIsInstance(free, int)
        self.assertIsInstance(total, int)
        self.assertGreater(total, 0)
        self.assertGreater(free, 0)

    def test_total_memory(self):
        total = self.plat.get_device_total_memory()
        self.assertGreater(total, 0)

    def test_profiler_activities(self):
        import torch.profiler

        activities = self.plat.get_profiler_activities()
        self.assertIn(torch.profiler.ProfilerActivity.CPU, activities)


class TestCUDASRTPlatform(unittest.TestCase):
    def setUp(self):
        from sglang.srt.platforms.cuda import CUDASRTPlatform

        self.plat = CUDASRTPlatform()

    def test_detection_flags(self):
        self.assertTrue(self.plat.is_cuda())
        self.assertFalse(self.plat.is_cpu())
        self.assertFalse(self.plat.is_npu())

    def test_is_cuda_alike(self):
        self.assertTrue(self.plat.is_cuda_alike())

    def test_get_distributed_backend(self):
        self.assertEqual(self.plat.get_distributed_backend(), "nccl")

    def test_graph_runner_class(self):
        try:
            cls = self.plat.get_graph_runner_class()
        except ImportError:
            self.skipTest("CudaGraphRunner deps (triton, etc.) not available")
        from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner

        self.assertIs(cls, CudaGraphRunner)

    def test_supports_torch_compile(self):
        self.assertTrue(self.plat.supports_torch_compile())

    def test_graph_backend_name(self):
        self.assertEqual(self.plat.graph_backend_name, "cuda graph")


class TestStubPlatforms(unittest.TestCase):
    def test_npu_stub(self):
        from sglang.srt.platforms._stubs import NPUSRTPlatformStub

        plat = NPUSRTPlatformStub()
        self.assertTrue(plat.is_npu())
        try:
            dev = plat.get_device(3)
            self.assertEqual(dev, torch.device("npu:3"))
        except RuntimeError:
            pass  # torch.device("npu:...") needs torch_npu extension
        self.assertEqual(plat.get_distributed_backend(), "hccl")
        self.assertEqual(plat.graph_backend_name, "npu graph")
        self.assertFalse(plat.supports_torch_compile())

    def test_xpu_stub(self):
        from sglang.srt.platforms._stubs import XPUSRTPlatformStub

        plat = XPUSRTPlatformStub()
        self.assertTrue(plat.is_xpu())
        self.assertEqual(plat.get_device(1), torch.device("xpu:1"))
        self.assertEqual(plat.get_distributed_backend(), "xccl")

    def test_musa_stub(self):
        from sglang.srt.platforms._stubs import MUSASRTPlatformStub

        plat = MUSASRTPlatformStub()
        self.assertTrue(plat.is_musa())
        try:
            dev = plat.get_device(0)
            self.assertEqual(dev, torch.device("musa:0"))
        except RuntimeError:
            pass  # torch.device("musa:...") needs torch musa extension
        self.assertEqual(plat.get_distributed_backend(), "mccl")

    def test_hpu_stub(self):
        from sglang.srt.platforms._stubs import HPUSRTPlatformStub

        plat = HPUSRTPlatformStub()
        self.assertTrue(plat.is_hpu())
        self.assertEqual(plat.get_device(2), torch.device("hpu:2"))
        self.assertEqual(plat.get_distributed_backend(), "hccl")


class TestBasePlatformDetection(unittest.TestCase):
    def test_cpu_architecture_returns_valid_enum(self):
        from sglang.platforms.interface import BasePlatform, CpuArchEnum

        arch = BasePlatform.get_cpu_architecture()
        self.assertIn(arch, list(CpuArchEnum))


if __name__ == "__main__":
    unittest.main()
