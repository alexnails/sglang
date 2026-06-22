"""Unit tests for the generated msgpack codec (srt/grpc/messages.py).

The proto (proto/sglang/runtime/v1/sglang.proto) is the IDL; this codec and the
Rust one (rust/sglang-grpc/src/msgpack.rs) are generated from it and must speak the
same wire. The golden vectors below are shared verbatim with the Rust test
(`golden_vector_*`) so the two codecs can never silently drift apart.
"""

import unittest

from sglang.srt.grpc import messages as M
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Canonical wire bytes, identical to the constants in rust/sglang-grpc/src/msgpack.rs.
GV_GENERATE = bytes.fromhex(
    "83a9696e7075745f69647393010203af73616d706c696e675f706172616d73"
    "83ae6d61785f6e65775f746f6b656e7310ae73746f705f746f6b656e5f6964"
    "739102a16e01a3726964a3616263"
)
GV_OPENAI = bytes.fromhex(
    "82a96a736f6e5f626f6479c4077b2261223a317dad74726163655f68656164" "65727381a178a179"
)
GV_TOKENIZE_RESPONSE = bytes.fromhex(
    "84a6746f6b656e7393010203a5636f756e7403ad6d61785f6d6f64656c5f6c656e"
    "cc80aa696e7075745f74657874a26869"
)
# IPC wire (ipc.proto): a msgpack tagged positional array mirroring io_struct's
# #28688 TokenizedGenerateReqInput. Float-free, so byte-identical cross-language.
GV_TOK_GENERATE = bytes.fromhex(
    "9eb9546f6b656e697a656447656e6572617465526571496e707574a3616263"
    "c0a2686992a169c40c010000000200000003000000c0c0c083ae6d61785f6e"
    "65775f746f6b656e7310ae73746f705f746f6b656e5f6964739102a16e01c2"
    "000090c3"
)


class TestMsgpackCodec(CustomTestCase):
    def test_golden_vector_generate_request(self):
        """Float-free message must be byte-identical to the Rust codec's output."""
        req = M.GenerateRequest(
            input_ids=[1, 2, 3],
            rid="abc",
            sampling_params=M.SamplingParams(
                n=1, max_new_tokens=16, stop_token_ids=[2]
            ),
        )
        self.assertEqual(M.encode(req), GV_GENERATE)
        self.assertEqual(M.decode(GV_GENERATE, M.GenerateRequest), req)

    def test_golden_vector_openai_request(self):
        """`bytes` field must be encoded as msgpack bin and round-trip."""
        req = M.OpenAIRequest(json_body=b'{"a":1}', trace_headers={"x": "y"})
        self.assertEqual(M.encode(req), GV_OPENAI)
        self.assertEqual(M.decode(GV_OPENAI, M.OpenAIRequest), req)
        # The bytes field is real msgpack bin (0xc4/0xc5/0xc6), not an int array.
        self.assertTrue(any(b in GV_OPENAI for b in (0xC4, 0xC5, 0xC6)))

    def test_golden_vector_tokenize_response_scalars(self):
        """Implicit proto3 scalars must be byte-identical to the Rust codec."""
        resp = M.TokenizeResponse(
            tokens=[1, 2, 3],
            count=3,
            max_model_len=128,
            input_text="hi",
        )
        self.assertEqual(M.encode(resp), GV_TOKENIZE_RESPONSE)
        self.assertEqual(M.decode(GV_TOKENIZE_RESPONSE, M.TokenizeResponse), resp)

    def test_round_trip_all_field_shapes(self):
        """Nested message + map + repeated + optional all survive a round-trip."""
        req = M.TextGenerateRequest(
            text="Hello, msgpack!",
            sampling_params=M.SamplingParams(
                temperature=0.7, top_p=0.95, max_new_tokens=128, stop=["</s>"]
            ),
            stream=True,
            rid="req-123",
            trace_headers={"traceparent": "00-abc-def-01"},
        )
        self.assertEqual(M.decode(M.encode(req), M.TextGenerateRequest), req)

    def test_unset_optionals_are_omitted(self):
        """omit_defaults: an all-default struct encodes to an empty msgpack map."""
        self.assertEqual(M.encode(M.SamplingParams()), b"\x80")

    def test_implicit_scalar_defaults_are_omitted(self):
        """Implicit proto3 scalar defaults are omitted, matching Rust serde attrs."""
        self.assertEqual(M.encode(M.TextGenerateResponse()), b"\x80")
        self.assertEqual(M.encode(M.TokenizeResponse()), b"\x80")
        self.assertEqual(M.encode(M.OpenAIResponse()), b"\x80")

    def test_decode_is_tolerant(self):
        """Absent keys take defaults; unknown keys are ignored (proto-evolution safe)."""
        # Missing every field -> all defaults.
        empty = M.decode(b"\x80", M.GenerateRequest)
        self.assertEqual(empty.input_ids, [])
        self.assertIsNone(empty.rid)
        self.assertIsNone(empty.sampling_params)
        # An unknown field (from a newer proto) must not raise.
        import msgspec

        blob = msgspec.msgpack.encode({"rid": "x", "field_from_the_future": 7})
        self.assertEqual(M.decode(blob, M.GenerateRequest).rid, "x")


class TestIpcWireCodec(CustomTestCase):
    """ipc.proto types must encode to #28688's msgpack tagged-array wire and be
    decodable by the real Scheduler codec (io_struct)."""

    def _sample_req(self):
        from array import array

        return M.TokenizedGenerateReqInput(
            rid="abc",
            input_text="hi",
            input_ids=M.TokenArray(typecode="i", data=array("i", [1, 2, 3]).tobytes()),
            sampling_params=M.SamplingParams(
                max_new_tokens=16, n=1, stop_token_ids=[2]
            ),
            return_logprob=False,
            logprob_start_len=0,
            top_logprobs_num=0,
            stream=True,
        )

    def test_tokenized_generate_golden_vector(self):
        """Float-free; byte-identical to the Rust codec (cross-language drift lock)."""
        self.assertEqual(M.encode(self._sample_req()), GV_TOK_GENERATE)

    def test_array_like_tagged_shape(self):
        """0x9e = fixarray(14): tag + 13-field faithful prefix, not a field map."""
        blob = M.encode(self._sample_req())
        self.assertEqual(blob[0], 0x9E)
        back = M.decode(GV_TOK_GENERATE, M.TokenizedGenerateReqInput)
        self.assertEqual(back.rid, "abc")
        self.assertEqual(back.input_ids.typecode, "i")

    def test_scheduler_decodes_generated_wire(self):
        """The real Scheduler codec must accept our bytes and rebuild a correct
        TokenizedGenerateReqInput, filling the omitted trailing fields."""
        import sglang.srt.managers.io_struct as io_struct

        d = io_struct.msgpack_decode(M.encode(self._sample_req()))
        self.assertEqual(type(d).__name__, "TokenizedGenerateReqInput")
        self.assertEqual(d.rid, "abc")
        self.assertEqual(list(d.input_ids), [1, 2, 3])
        self.assertEqual(d.input_ids.typecode, "i")
        self.assertEqual(d.sampling_params.max_new_tokens, 16)
        self.assertEqual(d.sampling_params.stop_token_ids, {2})
        self.assertIs(d.stream, True)
        self.assertIs(d.return_hidden_states, False)  # trailing default filled


if __name__ == "__main__":
    unittest.main()
