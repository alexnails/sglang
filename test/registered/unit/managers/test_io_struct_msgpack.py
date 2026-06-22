import unittest
import warnings
from array import array
from typing import Any

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (  # noqa: E402
    BaseReq,
    PickleWrapper,
    TokenizedGenerateReqInput,
    UpdateWeightFromDiskReqInput,
    _msgpack_decoder,
    hook_custom_types,
    msgpack_decode,
    msgpack_encode,
    unwrap_from_pickle,
    wrap_as_pickle,
)
from sglang.srt.observability.req_time_stats import (  # noqa: E402
    APIServerReqTimeStats,
)
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class MsgpackPayload(BaseReq, kw_only=True):
    tensor: torch.Tensor
    np_array: np.ndarray
    int_array: array
    np_scalar: Any


class UnsupportedNestedPayload(BaseReq, kw_only=True):
    value: Any


hook_custom_types(MsgpackPayload, UnsupportedNestedPayload)


class TestIoStructMsgpack(CustomTestCase):
    def test_supported_custom_payload_round_trip(self):
        payload = MsgpackPayload(
            tensor=torch.arange(12, dtype=torch.float32).reshape(3, 4).t()[1:],
            np_array=np.arange(12, dtype=np.float32).reshape(3, 4).T[1:],
            int_array=array("i", [1, 2, 3]),
            np_scalar=np.float32(1.25),
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given buffer is not writable",
                category=UserWarning,
            )
            rebuilt = msgpack_decode(msgpack_encode(payload))

        self.assertIsInstance(rebuilt, MsgpackPayload)
        self.assertTrue(torch.equal(rebuilt.tensor, payload.tensor))
        self.assertTrue(np.array_equal(rebuilt.np_array, payload.np_array))
        self.assertEqual(rebuilt.int_array, payload.int_array)
        self.assertEqual(rebuilt.np_scalar, 1.25)

    def test_top_level_fallback_uses_pickle_wrapper(self):
        encoded = msgpack_encode(SamplingParams(stop_token_ids=[1, 2]))

        self.assertIsInstance(_msgpack_decoder.decode(encoded), PickleWrapper)
        self.assertEqual(msgpack_decode(encoded).stop_token_ids, {1, 2})

    def test_unsupported_nested_object_fails_fast(self):
        with self.assertRaisesRegex(TypeError, "PickleWrapper"):
            msgpack_encode(UnsupportedNestedPayload(value=object()))

    def test_any_payload_struct_uses_pickle_transport(self):
        req = UpdateWeightFromDiskReqInput(
            model_path="/tmp/model",
            manifest={"non_msgpack_value": 1 + 2j},
        )
        encoded = msgpack_encode(req)

        self.assertIsInstance(_msgpack_decoder.decode(encoded), PickleWrapper)
        rebuilt = msgpack_decode(encoded)
        self.assertIsInstance(rebuilt, UpdateWeightFromDiskReqInput)
        self.assertEqual(rebuilt.manifest, {"non_msgpack_value": 1 + 2j})

    def test_explicit_pickle_wrapper_field_round_trip(self):
        time_stats = APIServerReqTimeStats()
        req = TokenizedGenerateReqInput(
            rid="rid-trace",
            input_text="hello",
            input_ids=array("l", [1, 2, 3]),
            mm_inputs=None,
            sampling_params=SamplingParams(max_new_tokens=4),
            return_logprob=False,
            logprob_start_len=0,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            time_stats=wrap_as_pickle(time_stats),
        )

        rebuilt = msgpack_decode(msgpack_encode(req))

        self.assertIsInstance(rebuilt.time_stats, PickleWrapper)
        self.assertIsInstance(
            unwrap_from_pickle(rebuilt.time_stats), APIServerReqTimeStats
        )


if __name__ == "__main__":
    unittest.main()
