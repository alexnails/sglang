"""End-to-end test for CPU-only inference.

Usage:
    python -m pytest test/srt/test_cpu_inference.py -v

Requires:
    - A CPU-only environment (no CUDA/XPU/NPU required)
    - Network access to download the model on first run
    - The model weights cached locally for subsequent runs

The test launches a real server with --device cpu and validates
that a generate request returns a non-empty response.
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

MODEL = "Qwen/Qwen3-1.7B"
TIMEOUT = 600  # CPU model loading is slow


class TestCPUInference(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            MODEL,
            cls.base_url,
            timeout=TIMEOUT,
            device="cpu",
            other_args=[
                "--device",
                "cpu",
                "--max-total-tokens",
                "512",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_generate_basic(self):
        """Server should return a non-empty generated text."""
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                },
            },
            timeout=120,
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("text", data)
        self.assertTrue(len(data["text"]) > 0, "Generated text should not be empty")

    def test_health_endpoint(self):
        """Health endpoint should respond 200."""
        response = requests.get(self.base_url + "/health", timeout=10)
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
