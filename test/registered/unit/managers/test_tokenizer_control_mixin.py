import unittest

from pydantic import TypeAdapter

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput  # noqa: E402
from sglang.srt.utils import (  # noqa: E402
    MultiprocessingSerializer,
    normalize_serialized_named_tensor_payloads,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestUpdateWeightsFromTensorPayloadDecoding(CustomTestCase):
    def test_base64_and_raw_payloads_normalize_to_pickle_bytes(self):
        payload = [("weight", [1, 2, 3])]
        raw = MultiprocessingSerializer.serialize(payload)
        encoded = MultiprocessingSerializer.serialize(payload, output_str=True)
        req = TypeAdapter(UpdateWeightsFromTensorReqInput).validate_python(
            {"serialized_named_tensors": [encoded, raw]}
        )

        normalized = normalize_serialized_named_tensor_payloads(
            req.serialized_named_tensors
        )

        self.assertEqual(normalized, [raw, raw])
        self.assertEqual(MultiprocessingSerializer.deserialize(normalized[0]), payload)


if __name__ == "__main__":
    unittest.main()
