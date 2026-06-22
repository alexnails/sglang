//! msgpack (de)serialization for the proto messages, for the ZMQ transport path
//! (RFC #22558 Phase 4+). The proto is the IDL; `build.rs` derives serde on every
//! generated message. This module pins the wire convention so the Python `msgspec`
//! codec generated from the same proto interoperates byte-for-byte where it counts.
//!
//! Wire convention (must match `python/sglang/srt/grpc/messages.py`):
//!   * each message is a msgpack **map keyed by proto field name** (`with_struct_map`),
//!   * `bytes` fields are msgpack **bin**, not int arrays (`BytesMode::ForceAll`),
//!   * default-valued fields are omitted (skip attrs from `build.rs`),
//!   * decoding is tolerant — unknown keys are ignored, absent keys take defaults.
//!
//! Note: proto `float` is 32-bit but Python emits 64-bit doubles, so float fields
//! are not byte-identical across languages; both decoders coerce numerically and the
//! effective precision is float32 (the proto's declared width).

use serde::{Serialize, de::DeserializeOwned};

pub(crate) fn is_default_bool(value: &bool) -> bool {
    !*value
}

pub(crate) fn is_default_i32(value: &i32) -> bool {
    *value == 0
}

#[allow(dead_code)]
pub(crate) fn is_default_i64(value: &i64) -> bool {
    *value == 0
}

#[allow(dead_code)]
pub(crate) fn is_default_u32(value: &u32) -> bool {
    *value == 0
}

#[allow(dead_code)]
pub(crate) fn is_default_u64(value: &u64) -> bool {
    *value == 0
}

#[allow(dead_code)]
pub(crate) fn is_default_f32(value: &f32) -> bool {
    *value == 0.0
}

#[allow(dead_code)]
pub(crate) fn is_default_f64(value: &f64) -> bool {
    *value == 0.0
}

/// Serialize a generated proto message to msgpack bytes for the ZMQ wire.
pub fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, rmp_serde::encode::Error> {
    let mut buf = Vec::new();
    let mut serializer = rmp_serde::Serializer::new(&mut buf)
        .with_struct_map()
        .with_bytes(rmp_serde::config::BytesMode::ForceAll);
    value.serialize(&mut serializer)?;
    Ok(buf)
}

/// Deserialize a generated proto message from msgpack bytes received on the ZMQ wire.
pub fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, rmp_serde::decode::Error> {
    rmp_serde::from_slice(bytes)
}

// ---------------------------------------------------------------------------
// IPC array_like+tag encoder (matches sglang.srt.managers.io_struct, #28688).
//
// The Scheduler's tagged-union decoder expects each request as a msgpack
// *positional array* whose first element is the type tag (the struct name),
// not a field-name map. rmp-serde's struct-as-map is a single global serializer
// setting, so we can't emit "outer array + inner SamplingParams map" in one
// serde pass; instead we hand-write the outer array with low-level `rmp` and
// reuse `encode` (with_struct_map) for the nested SamplingParams map. Only the
// faithful prefix (through the last required field `stream`) is emitted; msgspec
// fills the remaining defaulted fields. Byte-identical to the Python codec's
// `messages.encode` (shared golden vector below / in test_msgpack_codec.py).
// ---------------------------------------------------------------------------

fn put_nil(buf: &mut Vec<u8>) {
    rmp::encode::write_nil(buf).expect("vec write is infallible");
}

fn put_str(buf: &mut Vec<u8>, s: &str) {
    rmp::encode::write_str(buf, s).expect("vec write is infallible");
}

fn put_opt_str(buf: &mut Vec<u8>, s: Option<&str>) {
    match s {
        Some(v) => put_str(buf, v),
        None => put_nil(buf),
    }
}

fn put_opt_bin(buf: &mut Vec<u8>, b: Option<&[u8]>) {
    match b {
        Some(v) => {
            rmp::encode::write_bin(buf, v).expect("vec write is infallible");
        }
        None => put_nil(buf),
    }
}

/// Minimal msgpack int, matching msgspec: non-negative as unsigned, else signed.
fn put_int(buf: &mut Vec<u8>, v: i64) {
    if v >= 0 {
        rmp::encode::write_uint(buf, v as u64).expect("vec write is infallible");
    } else {
        rmp::encode::write_sint(buf, v).expect("vec write is infallible");
    }
}

/// Encode an `ipc::TokenizedGenerateReqInput` to the Scheduler IPC wire.
pub fn encode_tokenized_generate(
    req: &crate::proto::ipc::TokenizedGenerateReqInput,
) -> Result<Vec<u8>, rmp_serde::encode::Error> {
    let mut buf = Vec::new();
    // tag + 13 prefix fields = a 14-element array.
    rmp::encode::write_array_len(&mut buf, 14).expect("vec write is infallible");
    put_str(&mut buf, "TokenizedGenerateReqInput");
    put_opt_str(&mut buf, req.rid.as_deref());
    put_opt_str(&mut buf, req.http_worker_ipc.as_deref());
    put_opt_str(&mut buf, req.input_text.as_deref());
    // input_ids: [typecode, raw_bytes] (io_struct dec_hook -> array.array), or nil.
    match &req.input_ids {
        Some(ta) => {
            rmp::encode::write_array_len(&mut buf, 2).expect("vec write is infallible");
            put_str(&mut buf, &ta.typecode);
            rmp::encode::write_bin(&mut buf, &ta.data).expect("vec write is infallible");
        }
        None => put_nil(&mut buf),
    }
    // None on the text/token generate path (multimodal/embeds not yet wired).
    put_opt_bin(&mut buf, req.input_embeds.as_deref());
    put_opt_bin(&mut buf, req.mm_inputs.as_deref());
    put_opt_bin(&mut buf, req.token_type_ids.as_deref());
    // sampling_params stays a field-name map (io_struct keeps it non-array_like).
    match &req.sampling_params {
        Some(sp) => buf.extend_from_slice(&encode(sp)?),
        None => put_nil(&mut buf),
    }
    rmp::encode::write_bool(&mut buf, req.return_logprob).expect("vec write is infallible");
    put_int(&mut buf, req.logprob_start_len as i64);
    put_int(&mut buf, req.top_logprobs_num as i64);
    rmp::encode::write_array_len(&mut buf, req.token_ids_logprob.len() as u32)
        .expect("vec write is infallible");
    for v in &req.token_ids_logprob {
        put_int(&mut buf, *v as i64);
    }
    rmp::encode::write_bool(&mut buf, req.stream).expect("vec write is infallible");
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::*;

    #[test]
    fn round_trip_text_generate_request() {
        let mut trace_headers = std::collections::HashMap::new();
        trace_headers.insert("traceparent".to_string(), "00-abc-def-01".to_string());
        let req = TextGenerateRequest {
            text: "Hello, msgpack!".to_string(),
            sampling_params: Some(SamplingParams {
                temperature: Some(0.7),
                top_p: Some(0.95),
                max_new_tokens: Some(128),
                stop: vec!["</s>".to_string()],
                ..Default::default()
            }),
            stream: Some(true),
            rid: Some("req-123".to_string()),
            trace_headers,
            ..Default::default()
        };
        let bytes = encode(&req).unwrap();
        let back: TextGenerateRequest = decode(&bytes).unwrap();
        assert_eq!(req, back);
    }

    #[test]
    fn unset_optionals_are_omitted() {
        // Empty SamplingParams must encode as an empty msgpack map (0x80), matching
        // the Python codec exactly — proof the skip_serializing_if attrs are applied.
        let bytes = encode(&SamplingParams::default()).unwrap();
        assert_eq!(bytes, vec![0x80]);
    }

    #[test]
    fn bytes_field_is_msgpack_bin() {
        // proto `bytes` must be msgpack bin (0xc4/0xc5/0xc6) so Python's msgspec
        // decodes it as `bytes`, not as a list of ints.
        let req = OpenAiRequest {
            json_body: br#"{"model":"x"}"#.to_vec(),
            ..Default::default()
        };
        let bytes = encode(&req).unwrap();
        assert!(bytes.iter().any(|&b| matches!(b, 0xc4 | 0xc5 | 0xc6)));
        let back: OpenAiRequest = decode(&bytes).unwrap();
        assert_eq!(req.json_body, back.json_body);
    }

    #[test]
    fn repeated_int_round_trip() {
        let req = GenerateRequest {
            input_ids: vec![1, 2, 300, 40000],
            ..Default::default()
        };
        let back: GenerateRequest = decode(&encode(&req).unwrap()).unwrap();
        assert_eq!(req.input_ids, back.input_ids);
    }

    fn from_hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    // Canonical wire vectors minted from the Python `msgspec` codec generated from
    // the same proto (see test/registered/unit/grpc/test_msgpack_codec.py). These
    // must stay byte-identical so the two codecs can never silently drift apart.
    const GV_GENERATE: &str = "83a9696e7075745f69647393010203af73616d706c696e675f706172616d7383ae6d61785f6e65775f746f6b656e7310ae73746f705f746f6b656e5f6964739102a16e01a3726964a3616263";
    const GV_OPENAI: &str =
        "82a96a736f6e5f626f6479c4077b2261223a317dad74726163655f6865616465727381a178a179";
    const GV_TOKENIZE_RESPONSE: &str = "84a6746f6b656e7393010203a5636f756e7403ad6d61785f6d6f64656c5f6c656ecc80aa696e7075745f74657874a26869";
    // IPC array_like+tag wire; identical to GV_TOK_GENERATE in
    // test/registered/unit/grpc/test_msgpack_codec.py (float-free => byte-stable).
    const GV_TOK_GENERATE: &str = "9eb9546f6b656e697a656447656e6572617465526571496e707574a3616263c0a2686992a169c40c010000000200000003000000c0c0c083ae6d61785f6e65775f746f6b656e7310ae73746f705f746f6b656e5f6964739102a16e01c2000090c3";

    #[test]
    fn golden_vector_generate_request() {
        let expect = from_hex(GV_GENERATE);
        let req = GenerateRequest {
            input_ids: vec![1, 2, 3],
            rid: Some("abc".into()),
            sampling_params: Some(SamplingParams {
                n: Some(1),
                max_new_tokens: Some(16),
                stop_token_ids: vec![2],
                ..Default::default()
            }),
            ..Default::default()
        };
        // Byte-identical with Python (no float fields here).
        assert_eq!(encode(&req).unwrap(), expect);
        assert_eq!(decode::<GenerateRequest>(&expect).unwrap(), req);
    }

    #[test]
    fn golden_vector_openai_request() {
        let expect = from_hex(GV_OPENAI);
        let mut trace_headers = std::collections::HashMap::new();
        trace_headers.insert("x".to_string(), "y".to_string());
        let req = OpenAiRequest {
            json_body: br#"{"a":1}"#.to_vec(),
            trace_headers,
        };
        assert_eq!(encode(&req).unwrap(), expect);
        assert_eq!(decode::<OpenAiRequest>(&expect).unwrap(), req);
    }

    #[test]
    fn implicit_scalar_defaults_are_omitted() {
        // Covers proto3 implicit scalars (string/bool/int/bytes), not just
        // optionals/repeated/maps. This must match msgspec's omit_defaults=True.
        assert_eq!(
            encode(&TextGenerateResponse::default()).unwrap(),
            vec![0x80]
        );
        assert_eq!(encode(&TokenizeResponse::default()).unwrap(), vec![0x80]);
        assert_eq!(encode(&OpenAiResponse::default()).unwrap(), vec![0x80]);
    }

    #[test]
    fn golden_vector_tokenize_response_scalars() {
        let expect = from_hex(GV_TOKENIZE_RESPONSE);
        let resp = TokenizeResponse {
            tokens: vec![1, 2, 3],
            count: 3,
            max_model_len: 128,
            input_text: "hi".to_string(),
        };
        assert_eq!(encode(&resp).unwrap(), expect);
        assert_eq!(decode::<TokenizeResponse>(&expect).unwrap(), resp);
    }

    #[test]
    fn golden_vector_tokenized_generate_ipc() {
        // The IPC tagged-array wire must be byte-identical to the Python codec
        // (messages.encode) so the Scheduler's io_struct decoder accepts it.
        let expect = from_hex(GV_TOK_GENERATE);
        let req = crate::proto::ipc::TokenizedGenerateReqInput {
            rid: Some("abc".into()),
            input_text: Some("hi".into()),
            input_ids: Some(crate::proto::ipc::TokenArray {
                typecode: "i".into(),
                // array("i", [1, 2, 3]).tobytes() — int32 little-endian.
                data: vec![1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0],
            }),
            sampling_params: Some(SamplingParams {
                max_new_tokens: Some(16),
                n: Some(1),
                stop_token_ids: vec![2],
                ..Default::default()
            }),
            stream: true,
            ..Default::default()
        };
        assert_eq!(encode_tokenized_generate(&req).unwrap(), expect);
    }
}
